"""
NeuroCrowd Predictive ML Model (XGBoost Risk & Stampede Forecaster)
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

class CrowdRiskPredictor:
    def __init__(self):
        self.model = None
        self.classes = ["SAFE", "MODERATE", "HIGH_RISK", "CRITICAL"]
        self._train_initial_model()

    def _generate_synthetic_training_data(self, n_samples=2500):
        np.random.seed(42)
        
        n_safe = int(n_samples * 0.4)
        density_safe = np.random.uniform(0.1, 1.4, n_safe)
        delta_safe = np.random.uniform(-0.1, 0.15, n_safe)
        occ_safe = density_safe / 4.0
        motion_safe = np.random.uniform(0.5, 2.5, n_safe)
        turb_safe = np.random.uniform(0.1, 0.8, n_safe)
        y_safe = np.zeros(n_safe)

        n_mod = int(n_samples * 0.3)
        density_mod = np.random.uniform(1.5, 2.4, n_mod)
        delta_mod = np.random.uniform(0.05, 0.3, n_mod)
        occ_mod = density_mod / 4.0
        motion_mod = np.random.uniform(0.3, 1.8, n_mod)
        turb_mod = np.random.uniform(0.5, 1.5, n_mod)
        y_mod = np.ones(n_mod)

        n_high = int(n_samples * 0.2)
        density_high = np.random.uniform(2.5, 3.4, n_high)
        delta_high = np.random.uniform(0.2, 0.6, n_high)
        occ_high = density_high / 4.0
        motion_high = np.random.uniform(0.1, 1.2, n_high)
        turb_high = np.random.uniform(1.2, 3.0, n_high)
        y_high = np.full(n_high, 2)

        n_crit = n_samples - (n_safe + n_mod + n_high)
        density_crit = np.random.uniform(3.5, 6.0, n_crit)
        delta_crit = np.random.uniform(0.4, 1.2, n_crit)
        occ_crit = np.minimum(1.0, density_crit / 4.0)
        motion_crit = np.random.uniform(0.05, 4.0, n_crit)
        turb_crit = np.random.uniform(2.5, 5.0, n_crit)
        y_crit = np.full(n_crit, 3)

        X = np.vstack([
            np.hstack([density_safe, density_mod, density_high, density_crit]),
            np.hstack([delta_safe, delta_mod, delta_high, delta_crit]),
            np.hstack([occ_safe, occ_mod, occ_high, occ_crit]),
            np.hstack([motion_safe, motion_mod, motion_high, motion_crit]),
            np.hstack([turb_safe, turb_mod, turb_high, turb_crit])
        ]).T

        y = np.hstack([y_safe, y_mod, y_high, y_crit])
        return X, y

    def _train_initial_model(self):
        X, y = self._generate_synthetic_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def predict(self, density: float, density_delta: float, occupancy_ratio: float,
                motion_mag: float, turbulence: float):
        features = np.array([[density, density_delta, occupancy_ratio, motion_mag, turbulence]])
        probs = self.model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probs)
        confidence = float(probs[predicted_class_idx])
        risk_label = self.classes[predicted_class_idx]

        density_danger = min(100.0, (density / 4.0) * 80.0)
        prob_danger = (probs[1] * 35.0) + (probs[2] * 70.0) + (probs[3] * 100.0)
        danger_index = round(0.5 * density_danger + 0.5 * prob_danger, 1)
        danger_index = min(100.0, max(0.0, danger_index))

        return {
            "forecast_risk": risk_label,
            "confidence": round(confidence * 100, 1),
            "danger_index": danger_index
        }