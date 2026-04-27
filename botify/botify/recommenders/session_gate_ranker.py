"""
SessionGateRanker – реранкер на основе RandomForest.
Загружает модель, scaler, список популярных треков и параметры.
Для каждого кандидата вычисляет признаки, включая ранг в выдаче SasRec,
и решает, заменять ли первый трек.
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

        # Путь к модели (относительно этого файла)
        model_path = os.path.join(os.path.dirname(__file__), "../../session_gate_rf_bundle.joblib")
        if not os.path.exists(model_path):
            model_path = "/app/botify/botify/session_gate_rf_bundle.joblib"
        if not os.path.exists(model_path):
            model_path = "/app/session_gate_rf_bundle.joblib"

        logger.info(f"Loading model from {model_path}")
        bundle = joblib.load(model_path)

        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_cols = bundle['feature_cols']
        self.params = bundle['params']
        self.popular_tracks = bundle['popular_tracks']

        self.redis_client = get_redis_connection()

        # Пороги замены (можно подкрутить, но оставим разумные)
        self.min_prev_time = 0.5
        self.improvement_threshold = 0.15
        self.min_prob = 0.55

        # Параметры из обучения
        self.good_time = self.params.get('good_time', 0.8)
        self.anchor_window = self.params.get('anchor_window', 5)
        self.topk_sasrec = self.params.get('topk_sasrec', 12)
        self.topk_popular = self.params.get('topk_popular', 8)
        self.max_candidates = self.params.get('max_candidates', 30)

    def _get_user_history(self, user_id, limit=20):
        """Возвращает список (track, time) из Redis"""
        key = f"user:{user_id}:listens"
        raw = self.redis_client.lrange(key, -limit, -1)
        events = []
        for item in raw:
            try:
                ev = json.loads(item)
                events.append((ev["track"], ev["time"]))
            except Exception:
                pass
        return events

    def _recent_stats(self, history):
        """Статистика по последним anchor_window трекам"""
        if not history:
            return 0.0, 0.0, 0.0, 0.0
        times = [t for _, t in history[-self.anchor_window:]]
        avg_time = float(np.mean(times))
        last_time = float(times[-1])
        good_frac = float(np.mean([t >= self.good_time for t in times]))
        skip_frac = float(np.mean([t < 0.25 for t in times]))
        return avg_time, last_time, good_frac, skip_frac

    def _build_features(self, history, prev_track, prev_time, cand, baseline_recs):
        """
        Вычисляет 14 признаков для кандидата.
        baseline_recs – список ID треков от SasRec (порядок важен)
        """
        # 1. Статистика истории
        avg_time, last_time, good_frac, skip_frac = self._recent_stats(history)
        hist_len = len(history)

        # 2. Признаки взаимодействия
        same_as_prev = 1.0 if cand == prev_track else 0.0
        cand_popularity = 0.0   # в рантайме нет глобальной популярности, можно игнорировать

        # 3. SasRec‑признаки на основе позиции в baseline_recs
        if cand in baseline_recs:
            rank = baseline_recs.index(cand) + 1
            rr = 1.0 / rank
            hit = 1.0
        else:
            rank = 99
            rr = 0.0
            hit = 0.0
        sasrec_min_rank = float(rank)
        sasrec_avg_rr = float(rr)
        sasrec_hit = hit

        # 4. Популярность из заранее сохранённого списка
        popular_rank = 99.0
        is_popular = 0.0
        if cand in self.popular_tracks:
            popular_rank = float(self.popular_tracks.index(cand) + 1)
            is_popular = 1.0 if popular_rank <= 20 else 0.0

        # 5. Признаки артистов (нет данных, заполняем нулями)
        same_artist = 0.0
        artist_repeat = 0.0

        # Собираем признаки в порядке, заданном model.feature_cols
        feats_dict = {
            'hist_len': hist_len,
            'recent_avg_time': avg_time,
            'recent_last_time': last_time,
            'recent_good_frac': good_frac,
            'recent_skip_frac': skip_frac,
            'same_as_prev': same_as_prev,
            'cand_popularity': cand_popularity,
            'sasrec_min_rank': sasrec_min_rank,
            'sasrec_avg_rr': sasrec_avg_rr,
            'sasrec_hit': sasrec_hit,
            'popular_rank': popular_rank,
            'is_popular': is_popular,
            'same_artist': same_artist,
            'artist_repeat': artist_repeat,
        }
        ordered = [feats_dict[col] for col in self.feature_cols]
        return np.array(ordered)

    def recommend_next(self, user, prev_track, prev_time):
        try:
            # 1. История пользователя
            history = self._get_user_history(user)
            # 2. Базовые рекомендации от SasRec
            baseline_recs = self.sasrec.recommend_next(user, prev_track, prev_time)
            if not baseline_recs:
                return []

            # 3. Кандидаты: первые topk_sasrec из baseline + популярные (без дублей)
            candidates = baseline_recs[:self.topk_sasrec]
            for pt in self.popular_tracks[:self.topk_popular]:
                if pt not in candidates and len(candidates) < self.max_candidates:
                    candidates.append(pt)

            # 4. Вычисляем признаки и предсказания
            scores = {}
            for cand in candidates:
                feats = self._build_features(history, prev_track, prev_time, cand, baseline_recs)
                feats_scaled = self.scaler.transform([feats])
                prob = self.model.predict_proba(feats_scaled)[0, 1]
                scores[cand] = prob

            best_cand = max(scores, key=scores.get)
            best_prob = scores[best_cand]
            first_cand = baseline_recs[0]
            first_prob = scores.get(first_cand, best_prob)

            # 5. Решение: заменить первый трек на лучший кандидат, если условия соблюдены
            if (prev_time > self.min_prev_time and
                best_prob > self.min_prob and
                best_prob > first_prob + self.improvement_threshold):
                logger.info(f"Replace {first_cand} ({first_prob:.3f}) with {best_cand} ({best_prob:.3f})")
                result = [best_cand] + [c for c in baseline_recs if c != best_cand][:9]
                return result
            else:
                return baseline_recs

        except Exception as e:
            logger.exception("Error in SessionGateRanker.recommend_next")
            # При любой ошибке возвращаем baseline
            return self.sasrec.recommend_next(user, prev_track, prev_time)
