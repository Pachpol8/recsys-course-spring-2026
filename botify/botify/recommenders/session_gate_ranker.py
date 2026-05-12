"""
Session-aware RandomForest reranker.
Uses SasRec as candidate generator and RandomForest as a mild reranker.
"""
import os
import logging
import numpy as np
import joblib
from .recommender import Recommender

logger = logging.getLogger(__name__)


class SessionGateRanker(Recommender):
    def __init__(self, sasrec_recommender, config):
        self.sasrec = sasrec_recommender
        self.config = config

        model_path = "/app/data/session_gate_rf_bundle.joblib"
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../../../data/session_gate_rf_bundle.joblib",
            )

        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, using fallback")
            self.model = None
            self.scaler = None
            self.feature_cols = []
            self.params = {}
            self.popular_tracks = []
            return

        logger.info(f"Loading model from {model_path}")
        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]
        self.params = bundle.get("params", {})
        self.popular_tracks = bundle.get("popular_tracks", [])

        self.min_prev_time = 0.5
        self.improvement_threshold = 0.05
        self.min_prob = 0.50

    def _valid_track(self, track):
        return isinstance(track, (int, float)) and 0 < int(track) <= 16197

    def recommend_next(self, user, prev_track, prev_time) -> int:
        try:
            baseline_recs = self.sasrec.recommend_next(user, prev_track, prev_time)
            if not baseline_recs:
                return int(prev_track) + 1 if prev_track else 100

            valid_baseline = [int(t) for t in baseline_recs if self._valid_track(t)]
            if not valid_baseline:
                return int(prev_track) + 1 if prev_track else 100

            candidates = valid_baseline[:10]
            if len(candidates) < 3:
                candidates = valid_baseline[:min(20, len(valid_baseline))]

            scores = {}
            for cand in candidates:
                rank = float(valid_baseline.index(cand) + 1) if cand in valid_baseline else 99.0
                rr = 1.0 / rank if cand in valid_baseline and rank > 0 else 0.0
                hit = 1.0 if cand in valid_baseline else 0.0
                is_popular = 1.0 if cand in self.popular_tracks[:20] else 0.0
                popular_rank = float(self.popular_tracks.index(cand) + 1) if cand in self.popular_tracks else 99.0

                hist_len = 0.0
                recent_avg_time = float(prev_time)
                recent_last_time = float(prev_time)
                recent_good_frac = 1.0 if prev_time >= self.params.get("good_time", 0.8) else 0.0
                recent_skip_frac = 1.0 if prev_time < 0.25 else 0.0
                same_as_prev = 1.0 if cand == prev_track else 0.0

                feats_dict = {
                    "rank_in_sasrec": rank,
                    "reciprocal_rank": rr,
                    "hit": hit,
                    "is_popular": is_popular,
                    "popular_rank": popular_rank,
                    "hist_len": hist_len,
                    "recent_avg_time": recent_avg_time,
                    "recent_last_time": recent_last_time,
                    "recent_good_frac": recent_good_frac,
                    "recent_skip_frac": recent_skip_frac,
                    "same_as_prev": same_as_prev,
                }

                feats = [feats_dict[col] for col in self.feature_cols]
                feats_scaled = self.scaler.transform([feats])
                prob = self.model.predict_proba(feats_scaled)[0, 1]
                scores[cand] = prob

            if not scores:
                return int(valid_baseline[0])

            best_cand = max(scores, key=scores.get)
            best_prob = scores[best_cand]
            first_cand = valid_baseline[0]
            first_prob = scores.get(first_cand, best_prob)

            if (
                prev_time > self.min_prev_time
                and best_prob > self.min_prob
                and best_prob > first_prob + self.improvement_threshold
            ):
                return int(best_cand)

            return int(first_cand)

        except Exception as e:
            logger.error(f"Error in recommend_next: {e}", exc_info=True)
            try:
                fallback = self.sasrec.recommend_next(user, prev_track, prev_time)
                if fallback:
                    for t in fallback:
                        if self._valid_track(t):
                            return int(t)
            except Exception:
                pass
            return int(prev_track) + 1 if prev_track else 100
