import os
import pickle
from typing import Any, Dict, List

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from .sklearn_pickle import load_sklearn_pickle, sha256_file


class StressForecasterNN:
    """Autoregressive NN forecaster for short-term stress trajectories."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.model_path = os.path.join(os.path.dirname(__file__), "stress_forecaster_nn.pkl")
        self.model = self._load_or_train_model()

    @staticmethod
    def _sha256_file(path: str) -> str:
        return sha256_file(path)

    def _load_pickle_with_integrity(self, path: str):
        expected_hash = os.getenv("STRESS_FORECASTER_SHA256", "").strip()
        return load_sklearn_pickle(path, expected_hash=expected_hash)

    def _load_or_train_model(self) -> Pipeline:
        if os.path.exists(self.model_path):
            try:
                return self._load_pickle_with_integrity(self.model_path)
            except Exception:
                pass

        model = self._train_synthetic_forecaster()
        with open(self.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        return model

    def _train_synthetic_forecaster(self) -> Pipeline:
        rng = np.random.default_rng(42)
        X = []
        y = []

        for _ in range(5500):
            base = rng.uniform(0.2, 2.8)
            drift = rng.uniform(-0.12, 0.12)
            noise = rng.normal(0, 0.22, size=14)
            seq = []
            current = base
            for t in range(14):
                current = np.clip(current + drift + noise[t], 0.0, 3.0)
                seq.append(current)

            for i in range(self.window_size, len(seq)):
                X.append(seq[i - self.window_size:i])
                y.append(seq[i])

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(96, 48),
                        activation="relu",
                        alpha=1e-3,
                        random_state=42,
                        max_iter=700,
                        learning_rate_init=7e-4,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        )
        model.fit(X_arr, y_arr)
        return model

    def forecast_levels(self, levels: List[float], horizon: int = 3) -> Dict[str, Any]:
        clean = [float(np.clip(x, 0.0, 3.0)) for x in levels]
        if len(clean) < 2:
            return {
                "method": "nn_autoregressive",
                "status": "insufficient_data",
                "required_min_points": 2,
                "predictions": [],
            }

        history = clean[:]
        padded = history[-self.window_size:]
        while len(padded) < self.window_size:
            padded.insert(0, padded[0])

        preds = []
        for step in range(max(1, int(horizon))):
            x = np.array([padded[-self.window_size:]], dtype=float)
            next_val = float(np.clip(self.model.predict(x)[0], 0.0, 3.0))
            preds.append(
                {
                    "step": step + 1,
                    "predicted_level": round(next_val, 3),
                    "predicted_label": self._label_for_level(next_val),
                }
            )
            padded.append(next_val)

        variance = float(np.var(history[-min(8, len(history)) :]))
        confidence = float(np.clip(1.0 - min(variance / 2.5, 0.8), 0.2, 0.95))

        return {
            "method": "nn_autoregressive",
            "status": "ok",
            "horizon": int(horizon),
            "confidence": round(confidence, 3),
            "predictions": preds,
        }

    @staticmethod
    def _label_for_level(level: float) -> str:
        if level < 0.75:
            return "Low"
        if level < 1.75:
            return "Moderate"
        if level < 2.5:
            return "High"
        return "Severe"


stress_forecaster = StressForecasterNN()
