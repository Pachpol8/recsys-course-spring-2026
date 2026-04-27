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
import redis
from .recommender import Recommender
from ..data.redis import get_redis_connection

logger = logging.getLogger(__name__)

class SessionGateRanker(Recommender):
    def __init__(self, sasrec_recommender, config):
        self.sasrec = sasrec_recommender
        self.config = config

        # Путь к модели (относительно корня проекта)
        model_path = os.path.join(os.path.dirname(__file__), "../../session_gate_rf_bundle.joblib")
        bundle = joblib.load(model_path)

        self.model = bundle['model']          # RandomForestClassifier
        self.scaler = bundle['scaler']        # StandardScaler
        self.feature_cols = bundle['feature_cols']
        self.params = bundle['params']
        self.popular_tracks = bundle.get('popular_tracks', [])

        self.redis_client = get_redis_connection()

        # Консервативные пороги 
        self.min_prev_time = 0.5               
        self.improvement_threshold = 0.15      
        self.min_prob = 0.55                   

    def _get_user_history(self, user_id, limit=20):
        """Извлекает последние события пользователя из Redis."""
        key = f"user:{user_id}:history"
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
        """Вычисляет статистику по последним window трекам."""
        if not history:
            return 0.0, 0.0, 0.0, 0.0
        times = [ev.get("time", 0.0) for ev in history[-window:]]
        avg_time = float(np.mean(times))
        last_time = float(times[-1])
        good_frac = float(np.mean([t >= self.params["good_time"] for t in times]))
        skip_frac = float(np.mean([t < 0.25 for t in times]))
        return avg_time, last_time, good_frac, skip_frac

    def _rank_score(self, neighbors, cand):
        """Возвращает (rank, reciprocal_rank)."""
        if cand in neighbors:
            rank = neighbors.index(cand) + 1
            return rank, 1.0 / rank
        return 99, 0.0

    def _build_features(self, history, prev_track, cand):
        """Строит вектор признаков, совпадающий с обучением."""
        # 1. Статистика истории
        avg_time, last_time, good_frac, skip_frac = self._recent_stats(history, self.params["anchor_window"])
        hist_len = len(history)

        # 2. Вспомогательные данные
        same_as_prev = 1.0 if cand == prev_track else 0.0
        # популярность кандидата (берём из предвычисленного словаря, но его нет в runtime – используем константу 0)
        # Для реального сервиса можно загрузить popularity из Redis или файла
        popularity = 0.0

        # 3. SasRec признаки для последних anchor_window треков
        sasrec_ranks = []
        sasrec_rr = []
        for ev in history[-self.params["anchor_window"]:]:
            start = ev["track"]
            neighbors = self.sasrec.get_recommendations(start, n=50) if hasattr(self.sasrec, "get_recommendations") else []
            rank, rr = self._rank_score(neighbors, cand)
            sasrec_ranks.append(rank)
            sasrec_rr.append(rr)
        sasrec_min_rank = float(min(sasrec_ranks))
        sasrec_avg_rr = float(np.mean(sasrec_rr))
        sasrec_hit = 1.0 if sasrec_min_rank < 99 else 0.0

        # 4. Популярные треки
        popular_rank = 99.0
        is_popular = 0.0
        if cand in self.popular_tracks:
            popular_rank = float(self.popular_tracks.index(cand) + 1)
            is_popular = 1.0 if popular_rank <= 20 else 0.0

        # 5. Артисты – пока заглушка (в runtime у нас нет метаданных)
        same_artist = 0.0
        artist_repeat = 0.0

        features = {
            "hist_len": hist_len,
            "recent_avg_time": avg_time,
            "recent_last_time": last_time,
            "recent_good_frac": good_frac,
            "recent_skip_frac": skip_frac,
            "same_as_prev": same_as_prev,
            "cand_popularity": popularity,
            "sasrec_min_rank": sasrec_min_rank,
            "sasrec_avg_rr": sasrec_avg_rr,
            "sasrec_hit": sasrec_hit,
            "popular_rank": popular_rank,
            "is_popular": is_popular,
            "same_artist": same_artist,
            "artist_repeat": artist_repeat,
        }
        # Гарантируем порядок признаков
        ordered = [features[col] for col in self.feature_cols]
        return np.array(ordered)

    def next(self, request):
        user = request.user_id
        history = self._get_user_history(user)
        if len(history) < 1:
            return self.sasrec.next(request)

        last_event = history[-1]
        last_track = last_event["track"]
        last_time = last_event.get("time", 0.0)

        # Получаем baseline рекомендации от SasRec
        baseline_recs = self.sasrec.next(request)
        if not baseline_recs:
            return []

        # Берём топ-20 кандидатов от SasRec 
        candidates = baseline_recs[:20]
        # Добавляем 5 популярных треков, которых нет в candidates
        for pt in self.popular_tracks[:10]:
            if pt not in candidates and len(candidates) < 30:
                candidates.append(pt)

        # Оцениваем каждого кандидата моделью
        scores = {}
        for cand in candidates:
            feats = self._build_features(history, last_track, cand)
            feats_scaled = self.scaler.transform([feats])
            prob = self.model.predict_proba(feats_scaled)[0, 1]   # вероятность "хорошего" класса
            scores[cand] = prob

        best_candidate = max(scores, key=scores.get)
        best_prob = scores[best_candidate]
        first_candidate = baseline_recs[0]
        first_prob = scores.get(first_candidate, best_prob)

        # Консервативное правило
        if (last_time > self.min_prev_time and
            best_prob > self.min_prob and
            best_prob > first_prob + self.improvement_threshold):
            logger.info(f"Replace {first_candidate} ({first_prob:.3f}) with {best_candidate} ({best_prob:.3f})")
            result = [best_candidate] + [c for c in baseline_recs if c != best_candidate][:9]
            return result
        else:
            return baseline_recs
