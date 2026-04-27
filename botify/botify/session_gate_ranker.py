"""
Session‑aware RandomForest ranker with conservative gating.
Replaces baseline recommendation only when predicted probability
of a "good" listen is sufficiently higher.
"""

import json
import pickle
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
        self.redis_client = get_redis_connection()

        # Получаем доступ к Redis, где хранятся рекомендации SasRec (item2item)
        if hasattr(sasrec_recommender, 'i2i_redis'):
            self.i2i_redis = sasrec_recommender.i2i_redis
        else:
            logger.warning("SasRec recommender has no i2i_redis, using fallback for sasrec features")
            self.i2i_redis = None

        # Загружаем модель
        try:
            bundle = joblib.load("session_gate_rf_bundle.joblib")
            self.model = bundle['model']
            self.scaler = bundle['scaler']
            self.feature_cols = bundle['feature_cols']
            self.params = bundle['params']
            self.popular_tracks = bundle.get('popular_tracks', [])
            logger.info("SessionGateRanker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

        # Консервативные пороги
        self.min_prev_time = 0.5
        self.improvement_threshold = 0.15
        self.min_prob = 0.55

    def _get_user_history(self, user_id, limit=20):
        """Загружает историю прослушиваний пользователя из Redis."""
        key = f"user:{user_id}:listens"
        raw = self.redis_client.lrange(key, -limit, -1)
        events = []
        for item in raw:
            try:
                ev = json.loads(item)
                events.append(ev)
            except Exception:
                pass
        return events   # каждое событие: {"track": id, "time": seconds}

    def _recent_stats(self, history, window=5):
        """Средние статистики по последним window трекам."""
        if not history:
            return 0.0, 0.0, 0.0, 0.0
        times = [ev.get("time", 0.0) for ev in history[-window:]]
        avg_time = float(np.mean(times))
        last_time = float(times[-1])
        good_frac = float(np.mean([t >= self.params.get("good_time", 0.8) for t in times]))
        skip_frac = float(np.mean([t < 0.25 for t in times]))
        return avg_time, last_time, good_frac, skip_frac

    def _get_sasrec_neighbors(self, track_id, n=50):
        """Получает список рекомендаций SasRec для трека из Redis (pickle)."""
        if self.i2i_redis is None:
            return []
        data = self.i2i_redis.get(track_id)
        if data is None:
            return []
        try:
            recs = pickle.loads(data)
            if isinstance(recs, list):
                return recs[:n]
            else:
                return []
        except Exception:
            return []

    def _rank_score(self, neighbors, cand):
        """Возвращает (rank, reciprocal_rank) для кандидата в списке соседей."""
        if cand in neighbors:
            rank = neighbors.index(cand) + 1
            return rank, 1.0 / rank
        return 99, 0.0

    def _build_features(self, history, prev_track, cand):
        """Строит вектор признаков, совпадающий с обучением."""
        window = self.params.get("anchor_window", 5)
        avg_time, last_time, good_frac, skip_frac = self._recent_stats(history, window)
        hist_len = len(history)

        same_as_prev = 1.0 if cand == prev_track else 0.0
        popularity = 0.0   # можно заменить на реальную частоту, но для простоты 0

        # SasRec признаки (по каждому из последних window треков)
        sasrec_ranks = []
        sasrec_rr = []
        for ev in history[-window:]:
            start = ev["track"]
            neighbors = self._get_sasrec_neighbors(start, 50)
            rank, rr = self._rank_score(neighbors, cand)
            sasrec_ranks.append(rank)
            sasrec_rr.append(rr)
        sasrec_min_rank = float(min(sasrec_ranks))
        sasrec_avg_rr = float(np.mean(sasrec_rr)) if sasrec_rr else 0.0
        sasrec_hit = 1.0 if sasrec_min_rank < 99 else 0.0

        # Признаки популярности 
        popular_rank = 99.0
        is_popular = 0.0
        if cand in self.popular_tracks:
            popular_rank = float(self.popular_tracks.index(cand) + 1)
            is_popular = 1.0 if popular_rank <= 20 else 0.0

        
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
        # Убеждаемся, что порядок признаков соответствует обучению
        ordered = [features[col] for col in self.feature_cols]
        return np.array(ordered)

    def recommend_next(self, user: int, prev_track: int, prev_time: float) -> int:
        """Основной метод, вызываемый сервером для получения следующего трека."""
        if self.model is None:
            logger.warning("Model not loaded, fallback to SasRec")
            return self.sasrec.recommend_next(user, prev_track, prev_time)

        try:
            history = self._get_user_history(user)
            if len(history) < 1:
                return self.sasrec.recommend_next(user, prev_track, prev_time)

            # Baseline рекомендации от SasRec
            baseline_recs = self.sasrec.recommend_next(user, prev_track, prev_time)
            if not baseline_recs:
                return prev_track

            # Кандидаты: топ-20 SasRec + популярные треки (чтобы расширить пул)
            candidates = baseline_recs[:20]
            for pt in self.popular_tracks[:10]:
                if pt not in candidates and len(candidates) < 30:
                    candidates.append(pt)

            # Оценка каждого кандидата моделью
            scores = {}
            for cand in candidates:
                feats = self._build_features(history, prev_track, cand)
                feats_scaled = self.scaler.transform([feats])
                prob = self.model.predict_proba(feats_scaled)[0, 1]
                scores[cand] = prob

            best_candidate = max(scores, key=scores.get)
            best_prob = scores[best_candidate]
            first_candidate = baseline_recs[0]
            first_prob = scores.get(first_candidate, best_prob)

            # Консервативное правило замены
            if (prev_time > self.min_prev_time and
                best_prob > self.min_prob and
                best_prob > first_prob + self.improvement_threshold):
                logger.info(f"Replace {first_candidate} ({first_prob:.3f}) with {best_candidate} ({best_prob:.3f})")
                return best_candidate
            else:
                return first_candidate
        except Exception as e:
            logger.error(f"Error in recommend_next: {e}, fallback to SasRec")
            return self.sasrec.recommend_next(user, prev_track, prev_time)
