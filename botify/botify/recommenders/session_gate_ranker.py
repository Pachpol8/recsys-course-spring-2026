"""
Session‑aware RandomForest ranker with conservative gating.
Replaces baseline recommendation only when predicted probability
of a "good" listen is sufficiently higher.
"""
import json
import os
import logging
import numpy as np
import joblib
from .recommender import Recommender
from ..data.redis import get_redis_connection

logger = logging.getLogger(__name__)

class SessionGateRanker(Recommender):
    def __init__(self, sasrec_recommender, config):
        self.sasrec = sasrec_recommender
        self.config = config

        # Путь к модели (лежит в корне контейнера /app)
        model_path = os.path.join(os.path.dirname(__file__), "../../session_gate_rf_bundle.joblib")
        if not os.path.exists(model_path):

            model_path = "/app/session_gate_rf_bundle.joblib"
        
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_cols = bundle['feature_cols']
        self.params = bundle['params']
        self.popular_tracks = bundle.get('popular_tracks', [])

        self.redis_client = get_redis_connection()

        # Консервативные пороги
        self.min_prev_time = 0.5          # предыдущий трек слушали >0.5 сек
        self.improvement_threshold = 0.15 # на 15% выше вероятности первого кандидата
        self.min_prob = 0.55              # абсолютная вероятность хорошего прослушивания

    def _get_user_history(self, user_id, limit=20):
        key = f"user:{user_id}:listens"
        raw = self.redis_client.lrange(key, -limit, -1)
        events = []
        for item in raw:
            try:
                ev = json.loads(item)
                events.append(ev)
            except Exception:
                pass
        return events

    def _recent_stats(self, history, window=5):
        if not history:
            return 0.0, 0.0, 0.0, 0.0
        times = [ev.get("time", 0.0) for ev in history[-window:]]
        avg_time = float(np.mean(times))
        last_time = float(times[-1])
        good_frac = float(np.mean([t >= self.params.get("good_time", 0.8) for t in times]))
        skip_frac = float(np.mean([t < 0.25 for t in times]))
        return avg_time, last_time, good_frac, skip_frac

    def _build_features(self, baseline_candidates, cand):
        """baseline_candidates – список ID треков от SasRec (порядок важен)"""
        if cand in baseline_candidates:
            rank = baseline_candidates.index(cand) + 1
            rr = 1.0 / rank
            hit = 1.0
        else:
            rank = 99.0
            rr = 0.0
            hit = 0.0
        
        # popular_rank – если трек в списке популярных
        popular_rank = 99.0
        is_popular = 0.0
        if cand in self.popular_tracks:
            popular_rank = float(self.popular_tracks.index(cand) + 1)
            is_popular = 1.0 if popular_rank <= 20 else 0.0

        # Для остальных признаков (история) мы будем передавать отдельно,
        # но для упрощения пока используем значения по умолчанию.
        # В продакшене можно расширить.
        features = {
            "rank_in_sasrec": rank,
            "reciprocal_rank": rr,
            "hit": hit,
            "is_popular": is_popular,
            "popular_rank": popular_rank,
            "hist_len": 0.0,          # позже заполним в recommend_next
            "recent_avg_time": 0.0,
            "recent_last_time": 0.0,
            "recent_good_frac": 0.0,
            "recent_skip_frac": 0.0,
            "same_as_prev": 0.0,
        }
        ordered = [features[col] for col in self.feature_cols]
        return np.array(ordered)

    def recommend_next(self, user, prev_track, prev_time):
        # Получаем историю пользователя
        history = self._get_user_history(user)
        if len(history) < 1:
            # Нет истории – отдаём baseline
            return self.sasrec.recommend_next(user, prev_track, prev_time)

        # Базовые рекомендации от SasRec
        baseline_recs = self.sasrec.recommend_next(user, prev_track, prev_time)
        if not baseline_recs:
            return []

        # Кандидаты – первые 20 из baseline + до 10 популярных (если нет дублей)
        candidates = baseline_recs[:20]
        for pt in self.popular_tracks[:10]:
            if pt not in candidates and len(candidates) < 30:
                candidates.append(pt)

        # Вычисляем признаки истории (один раз для всех кандидатов)
        avg_time, last_time, good_frac, skip_frac = self._recent_stats(history, self.params.get("anchor_window", 5))
        hist_len = len(history)
        same_as_prev = 1.0 if prev_track in candidates else 0.0  # упрощённо

        scores = {}
        for cand in candidates:
            feats_dict = {
                "rank_in_sasrec": (baseline_recs.index(cand) + 1) if cand in baseline_recs else 99.0,
                "reciprocal_rank": 1.0 / (baseline_recs.index(cand) + 1) if cand in baseline_recs else 0.0,
                "hit": 1.0 if cand in baseline_recs else 0.0,
                "is_popular": 1.0 if cand in self.popular_tracks[:20] else 0.0,
                "popular_rank": float(self.popular_tracks.index(cand) + 1) if cand in self.popular_tracks else 99.0,
                "hist_len": hist_len,
                "recent_avg_time": avg_time,
                "recent_last_time": last_time,
                "recent_good_frac": good_frac,
                "recent_skip_frac": skip_frac,
                "same_as_prev": same_as_prev,
            }
            feats = [feats_dict[col] for col in self.feature_cols]
            feats_scaled = self.scaler.transform([feats])
            prob = self.model.predict_proba(feats_scaled)[0, 1]
            scores[cand] = prob

        best_cand = max(scores, key=scores.get)
        best_prob = scores[best_cand]
        first_cand = baseline_recs[0]
        first_prob = scores.get(first_cand, best_prob)

        # Решаем, заменять ли первый трек
        if (prev_time > self.min_prev_time and
            best_prob > self.min_prob and
            best_prob > first_prob + self.improvement_threshold):
            logger.info(f"Replace {first_cand} ({first_prob:.3f}) with {best_cand} ({best_prob:.3f})")
            result = [best_cand] + [c for c in baseline_recs if c != best_cand][:9]
            return result
        else:
            return baseline_recs
