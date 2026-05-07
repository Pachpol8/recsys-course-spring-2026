"""
SessionGateRanker – реранкер на основе RandomForest с gating-логикой.
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

        model_path = "/app/session_gate_rf_bundle.joblib"

        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "../../session_gate_rf_bundle.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model bundle not found at {model_path}."
            )

        logger.info(f"Loading model from {model_path}")
        bundle = joblib.load(model_path)

        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_cols = bundle['feature_cols']
        self.params = bundle['params']
        self.popular_tracks = bundle.get('popular_tracks', [])

        self.redis_client = get_redis_connection()

        self.min_prev_time = 0.5
        self.improvement_threshold = 0.15
        self.min_prob = 0.55

        self.good_time = self.params.get('good_time', 0.8)
        self.anchor_window = self.params.get('anchor_window', 5)
        self.topk_sasrec = self.params.get('topk_sasrec', 12)
        self.topk_popular = self.params.get('topk_popular', 8)
        self.max_candidates = self.params.get('max_candidates', 30)

    def _get_user_history(self, user_id: int, limit: int = 20):
        key = f"user:{user_id}:listens"
        raw = self.redis_client.lrange(key, -limit, -1)
        events = []
        for item in raw:
            try:
                ev = json.loads(item)
                events.append((ev.get("track"), ev.get("time", 0.0)))
            except Exception:
                pass
        return events

    def _recent_stats(self, history):
        if not history:
            return 0.0, 0.0, 0.0, 0.0

        times = [t for _, t in history[-self.anchor_window:]]
        avg_time = float(np.mean(times))
        last_time = float(times[-1]) if times else 0.0
        good_frac = float(np.mean([t >= self.good_time for t in times]))
        skip_frac = float(np.mean([t < 0.25 for t in times]))

        return avg_time, last_time, good_frac, skip_frac

    def _build_features(self, history, prev_track, prev_time, cand, baseline_recs):
        avg_time, last_time, good_frac, skip_frac = self._recent_stats(history)
        hist_len = len(history)

        same_as_prev = 1.0 if cand == prev_track else 0.0

        if cand in baseline_recs:
            rank = baseline_recs.index(cand) + 1
            rr = 1.0 / rank
            hit = 1.0
        else:
            rank = 99
            rr = 0.0
            hit = 0.0

        popular_rank = 99.0
        is_popular = 0.0
        if cand in self.popular_tracks:
            popular_rank = float(self.popular_tracks.index(cand) + 1)
            is_popular = 1.0 if popular_rank <= 20 else 0.0

        same_artist = 0.0
        artist_repeat = 0.0
        cand_popularity = 0.0

        feats_dict = {
            'hist_len': hist_len,
            'recent_avg_time': avg_time,
            'recent_last_time': last_time,
            'recent_good_frac': good_frac,
            'recent_skip_frac': skip_frac,
            'same_as_prev': same_as_prev,
            'cand_popularity': cand_popularity,
            'sasrec_min_rank': float(rank),
            'sasrec_avg_rr': float(rr),
            'sasrec_hit': hit,
            'popular_rank': popular_rank,
            'is_popular': is_popular,
            'same_artist': same_artist,
            'artist_repeat': artist_repeat,
        }

        ordered = [feats_dict[col] for col in self.feature_cols]
        return np.array(ordered, dtype=float)

    def recommend_next(self, user: int, prev_track: int, prev_time: float) -> int:  # ← ВАЖНО: int, не list[int]
        """Основной метод рекомендаций"""
        try:
            history = self._get_user_history(user)

            baseline_recs = self.sasrec.recommend_next(user, prev_track, prev_time)
            if not baseline_recs:
                return int(prev_track) + 1 if prev_track else 1

            if len(history) < 2:
                return int(baseline_recs[0])

            candidates = list(dict.fromkeys(baseline_recs[:self.topk_sasrec]))
            for pt in self.popular_tracks[:self.topk_popular]:
                if pt not in candidates and len(candidates) < self.max_candidates:
                    candidates.append(pt)

            scores = {}
            for cand in candidates:
                feats = self._build_features(history, prev_track, prev_time, cand, baseline_recs)
                feats_scaled = self.scaler.transform([feats])
                prob = self.model.predict_proba(feats_scaled)[0, 1]
                scores[cand] = prob

            if not scores:
                return int(baseline_recs[0])

            best_cand = max(scores, key=scores.get)
            best_prob = scores[best_cand]
            first_cand = baseline_recs[0]
            first_prob = scores.get(first_cand, best_prob)

            if (prev_time > self.min_prev_time and
                best_prob > self.min_prob and
                best_prob > first_prob + self.improvement_threshold):

                logger.info(f"Gate: Replace {first_cand} → {best_cand}")
                return int(best_cand)

            return int(baseline_recs[0])

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            try:
                fallback = self.sasrec.recommend_next(user, prev_track, prev_time)
                return int(fallback[0]) if fallback else int(prev_track) + 1 if prev_track else 1
            except:
                return int(prev_track) + 1 if prev_track else 1

    def next(self, request):
        """Обёртка для совместимости"""
        return self.recommend_next(request.user, request.track, request.time)
