import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Feature order used everywhere below -- keep predict(), predict_proba(),
# and _train_default_model() in sync if you ever change this.
FEATURE_NAMES = [
    "avg_density", "max_density", "avg_motion", "turbulence",
    "audio_panic", "critical_zone_count", "avg_trend_slope", "panic_zone_count",
]


class RiskPredictor:
    """
    Two things changed from a purely reactive classifier:

    1. Trend features (avg_trend_slope): the zone manager now tracks a
       rolling occupancy history per zone and fits a slope to it. A zone
       sitting at 60% but climbing fast is a very different risk than one
       sitting at 60% and flat -- the ORIGINAL feature set couldn't see
       that difference at all, it only ever saw the instantaneous number.
    2. Forward-looking training labels: the synthetic training data below
       labels each sample by what happens LEAD_STEPS later in the
       simulated sequence, not by the current instant. That's what makes
       this a genuine early-warning model rather than a same-frame
       classifier wearing a "predictive" label -- say this distinction
       explicitly to judges, it's the actual technical differentiator.

    Honesty note for your presentation: this is trained on synthetic,
    physically-motivated sequences (no labeled real stampede data exists
    to train on in 36 hours). Say so plainly -- it's the correct answer
    and sets up your "path to product" slide (retrain on real pilot data,
    same code, zero architecture changes).
    """

    def __init__(self, model_path="models/risk_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(["SAFE", "WARNING", "CRITICAL"])

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self._train_default_model()

    def _train_default_model(self, n_sequences=500, seq_len=150, lead_steps=8):
        """
        Simulates per-venue risk sequences (density random-walking with
        occasional surges, motion correlated with surges), then labels
        each timestep by the risk state `lead_steps` INTO THE FUTURE.
        The model therefore learns to recognize the early signature of a
        surge (rising trend + rising motion) before density itself has
        crossed the danger line.
        """
        rng = np.random.default_rng(42)
        X, y = [], []

        for _ in range(n_sequences):
            n_zones = rng.integers(3, 7)
            density = np.zeros((seq_len, n_zones))
            motion = np.zeros(seq_len)
            turbulence = np.zeros(seq_len)
            d = rng.uniform(0.1, 0.4, size=n_zones)

            for t in range(seq_len):
                surge = rng.random() < 0.025
                if surge:
                    d = d + rng.uniform(0.15, 0.45, size=n_zones) * (rng.random(n_zones) < 0.6)
                    motion[t] = rng.uniform(6, 16)
                    turbulence[t] = rng.uniform(3, 9)
                else:
                    d = d + rng.normal(0, 0.015, size=n_zones)
                    motion[t] = max(0, rng.normal(2, 1.5))
                    turbulence[t] = max(0, rng.normal(1, 0.8))
                d = np.clip(d, 0, 1.3)
                density[t] = d

            for t in range(5, seq_len - lead_steps):
                window = density[max(0, t - 10):t + 1]
                avg_density = float(window[-1].mean())
                max_density = float(window[-1].max())
                trend_slope = float(np.polyfit(np.arange(len(window)), window.mean(axis=1), 1)[0])
                critical_zone_count = int((window[-1] >= 0.8).sum())
                panic_zone_count = int(turbulence[t] > 3.5) * min(2, n_zones)
                audio_panic = int(rng.random() < 0.03)

                future_density = density[t + lead_steps]
                future_max = float(future_density.max())
                if future_max >= 0.8:
                    label = "CRITICAL"
                elif future_max >= 0.5:
                    label = "WARNING"
                else:
                    label = "SAFE"

                X.append([
                    avg_density, max_density, motion[t], turbulence[t],
                    audio_panic, critical_zone_count, trend_slope, panic_zone_count,
                ])
                y.append(label)

        X = np.array(X)
        y_encoded = self.label_encoder.transform(y)

        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
        self.model.fit(X, y_encoded)

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[RiskPredictor] Trained on {len(X)} synthetic samples, forecasting {lead_steps} steps ahead.")
        print(f"[RiskPredictor] Saved -> {self.model_path}")

    def _build_features(self, zone_summary, motion_metrics, audio_panic, zone_motion=None):
        densities = [z["density"] for z in zone_summary.values()]
        trends = [z.get("trend_slope", 0.0) for z in zone_summary.values()]
        critical_count = sum(1 for z in zone_summary.values() if z["risk"] == "CRITICAL")

        if zone_motion:
            panic_zone_count = sum(1 for m in zone_motion.values() if m.get("panic_detected"))
        else:
            panic_zone_count = 0

        return np.array([[
            float(np.mean(densities)) if densities else 0.0,
            float(np.max(densities)) if densities else 0.0,
            motion_metrics.get("avg_magnitude", 0.0),
            motion_metrics.get("turbulence", 0.0),
            int(audio_panic),
            critical_count,
            float(np.mean(trends)) if trends else 0.0,
            panic_zone_count,
        ]])

    def predict(self, zone_summary, motion_metrics, audio_panic=False, zone_motion=None):
        """Returns: 'SAFE', 'WARNING', or 'CRITICAL' (forecast, not just current state)."""
        features = self._build_features(zone_summary, motion_metrics, audio_panic, zone_motion)
        pred_encoded = self.model.predict(features)[0]
        return self.label_encoder.classes_[pred_encoded]

    def predict_proba(self, zone_summary, motion_metrics, audio_panic=False, zone_motion=None):
        """Returns probabilities for each class."""
        features = self._build_features(zone_summary, motion_metrics, audio_panic, zone_motion)
        proba = self.model.predict_proba(features)[0]
        return {
            "SAFE": round(float(proba[0]), 3),
            "WARNING": round(float(proba[1]), 3),
            "CRITICAL": round(float(proba[2]), 3),
        }
