import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class RiskPredictor:
    def __init__(self, model_path="models/risk_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(["SAFE", "WARNING", "CRITICAL"])

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self._train_default_model()

    def _train_default_model(self):
        """
        Train a baseline model with synthetic data.
        Replace with real data as you collect it.
        Features: [avg_density, max_density, avg_motion, turbulence, audio_panic, zone_critical_count]
        """
        np.random.seed(42)
        n = 1500

        X = []
        y = []

        # Generate SAFE samples
        for _ in range(600):
            X.append([
                np.random.uniform(0, 0.45),   # avg_density
                np.random.uniform(0, 0.55),   # max_density
                np.random.uniform(0, 3.0),    # avg_motion
                np.random.uniform(0, 2.0),    # turbulence
                0,                             # audio_panic
                0                              # critical zone count
            ])
            y.append("SAFE")

        # Generate WARNING samples
        for _ in range(500):
            X.append([
                np.random.uniform(0.4, 0.7),
                np.random.uniform(0.5, 0.8),
                np.random.uniform(2.0, 6.0),
                np.random.uniform(1.5, 3.5),
                np.random.choice([0, 1], p=[0.85, 0.15]),
                np.random.randint(0, 2)
            ])
            y.append("WARNING")

        # Generate CRITICAL samples
        for _ in range(400):
            X.append([
                np.random.uniform(0.65, 1.0),
                np.random.uniform(0.75, 1.0),
                np.random.uniform(5.0, 15.0),
                np.random.uniform(3.0, 8.0),
                np.random.choice([0, 1], p=[0.4, 0.6]),
                np.random.randint(1, 4)
            ])
            y.append("CRITICAL")

        X = np.array(X)
        y_encoded = self.label_encoder.transform(y)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y_encoded)

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print("[RiskPredictor] Model trained and saved.")

    def predict(self, zone_summary, motion_metrics, audio_panic=False):
        """
        Predict overall risk level.
        Returns: "SAFE", "WARNING", or "CRITICAL"
        """
        densities = [z["density"] for z in zone_summary.values()]
        critical_count = sum(1 for z in zone_summary.values() if z["risk"] == "CRITICAL")

        features = np.array([[
            np.mean(densities),
            np.max(densities),
            motion_metrics.get("avg_magnitude", 0.0),
            motion_metrics.get("turbulence", 0.0),
            int(audio_panic),
            critical_count
        ]])

        pred_encoded = self.model.predict(features)[0]
        return self.label_encoder.classes_[pred_encoded]

    def predict_proba(self, zone_summary, motion_metrics, audio_panic=False):
        """Returns probabilities for each class."""
        densities = [z["density"] for z in zone_summary.values()]
        critical_count = sum(1 for z in zone_summary.values() if z["risk"] == "CRITICAL")

        features = np.array([[
            np.mean(densities),
            np.max(densities),
            motion_metrics.get("avg_magnitude", 0.0),
            motion_metrics.get("turbulence", 0.0),
            int(audio_panic),
            critical_count
        ]])

        proba = self.model.predict_proba(features)[0]
        return {
            "SAFE": round(proba[0], 3),
            "WARNING": round(proba[1], 3),
            "CRITICAL": round(proba[2], 3)
        }