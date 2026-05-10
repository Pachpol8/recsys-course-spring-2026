"""
Session-aware RandomForest ranker (без Redis).
Рекомендует на основе модели и рангов SasRec.
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
            model_path = os.path.join(os.path.dirname(__file__), "../../../data/session_gate_rf_bundle.joblib")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, using fallback")
            self.model = None
            self.scaler = None
            self.feature_cols = None
            self.params = {}
            self.popular_tracks = []
            return
        
        logger.info(f"Loading model from {model_path}")
        bundle = joblib.load(model_path)
        
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_cols = bundle['feature_cols']
        self.params = bundle.get('params', {})
        self.popular_tracks = bundle.get('popular_tracks', [])

        self.min_prev_time = 0.5
        self.improvement_threshold = 0.15
        self.min_prob = 0.55

    def recommend_next(self, user, prev_track, prev_time) -> int:
        try:
            if self.model is None or self.scaler is None:
                logger.warning("Model not loaded, using SasRec fallback")
                baseline = self.sasrec.recommend_next(user, prev_track, prev_time)
                if baseline:
                    return int(baseline[0])
                return int(prev_track) + 1 if prev_track else 100
            
            baseline_recs = self.sasrec.recommend_next(user, prev_track, prev_time)
            if not baseline_recs:
                return int(prev_track) + 1 if prev_track else 100
            
            if len(baseline_recs) == 1:
                logger.warning(f"DEBUG popular_tracks[:20]: {self.popular_tracks[:20]}")
                logger.warning(f"DEBUG len(popular_tracks): {len(self.popular_tracks)}")
            
            valid_baseline = [int(t) for t in baseline_recs if isinstance(t, (int, float)) and 0 < t <= 16197]
            if not valid_baseline:
                valid_baseline = [int(t) for t in baseline_recs if isinstance(t, (int, float)) and t > 0]
            if not valid_baseline:
                return 100
            
            candidates = valid_baseline[:20]
            
            for pt in self.popular_tracks[:10]:
                if isinstance(pt, (int, float)) and 0 < pt <= 16197:
                    if pt not in candidates and len(candidates) < 30:
                        candidates.append(int(pt))
            
            scores = {}
            for cand in candidates:
                rank = float(valid_baseline.index(cand) + 1) if cand in valid_baseline else 99.0
                rr = 1.0 / rank if cand in valid_baseline else 0.0
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
            
            if (prev_time > self.min_prev_time and
                best_prob > self.min_prob and
                best_prob > first_prob + self.improvement_threshold):
                logger.info(f"Gate: Replace {first_cand} ({first_prob:.3f}) → {best_cand} ({best_prob:.3f})")
                if 0 < best_cand <= 16197:
                    return int(best_cand)
            
            for t in valid_baseline:
                if 0 < t <= 16197:
                    return int(t)
            
            return 100
        
        except Exception as e:
            logger.error(f"Error in recommend_next: {e}", exc_info=True)
            try:
                fallback = self.sasrec.recommend_next(user, prev_track, prev_time)
                if fallback:
                    for t in fallback:
                        if isinstance(t, (int, float)) and 0 < t <= 16197:
                            return int(t)
            except:
                pass
            return 100
