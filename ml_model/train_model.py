import json
import os
import pickle
import hashlib
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - compatibility fallback for older sklearn
    FrozenEstimator = None
from .model_artifacts import (
    ensure_runtime_model_dir,
    stress_meta_path,
    stress_model_path,
    stress_shap_model_path,
)
from .questionnaire_config import (
    EXPECTED_FEATURE_COLUMNS,
    QUESTION_WEIGHTS,
    apply_question_weights,
)

TARGET_COLUMN = "stress_level"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_training_data(n_samples=1000):
    """
    Generate synthetic training data for stress detection.
    Based on 18 CBT questions with responses 1-5.
    """
    np.random.seed(42)

    data = []

    for _ in range(n_samples):
        stress_level = np.random.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15])

        if stress_level == 0:
            responses = np.random.randint(1, 3, size=18).tolist()
        elif stress_level == 1:
            responses = np.random.randint(2, 4, size=18).tolist()
        elif stress_level == 2:
            responses = np.random.randint(3, 5, size=18).tolist()
        else:
            responses = np.random.randint(4, 6, size=18).tolist()

        responses = [min(5, max(1, r + np.random.randint(-1, 2))) for r in responses]
        data.append(responses + [stress_level])

    columns = EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]
    return pd.DataFrame(data, columns=columns)


def load_training_data(dataset_path=None, fallback_samples=1000):
    """
    Load training data from CSV when available.
    Fallback to synthetic data only if dataset is missing.
    """
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading training data from CSV: {dataset_path}")
        df = pd.read_csv(dataset_path)

        expected_columns = EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        df = df[expected_columns]
        print(f"Loaded {len(df)} rows from dataset.")
        return df

    print(
        "Dataset file not found. Falling back to synthetic training data "
        f"({fallback_samples} rows)."
    )
    return generate_training_data(n_samples=fallback_samples)


def _build_calibrator(base_estimator, method: str) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_estimator),
            method=method,
        )
    return CalibratedClassifierCV(estimator=base_estimator, method=method, cv="prefit")


def train_stress_model(dataset_filename="stress_training_dataset_100k.csv"):
    """Train an ensemble model (RF + GBM + LR stacking) for stress detection."""
    model_dir = os.path.dirname(__file__)
    runtime_model_dir = ensure_runtime_model_dir()
    dataset_path = os.path.join(model_dir, dataset_filename) if dataset_filename else None
    df = load_training_data(dataset_path=dataset_path, fallback_samples=1000)
    run_full_cv = _env_flag("STRESS_MODEL_RUN_FULL_CV", default=False)
    enable_calibration = _env_flag("STRESS_MODEL_ENABLE_CALIBRATION", default=True)
    calibration_method = os.getenv("STRESS_MODEL_CALIBRATION_METHOD", "sigmoid").strip() or "sigmoid"
    calibration_size = float(np.clip(_env_float("STRESS_MODEL_CALIBRATION_SIZE", 0.15), 0.05, 0.4))
    rf_n_jobs = max(1, _env_int("STRESS_MODEL_RF_N_JOBS", 1))

    X = df.drop(TARGET_COLUMN, axis=1)
    X_weighted = apply_question_weights(X)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X_weighted, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=rf_n_jobs,
    )
    gbm = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    print("Training ensemble model (RF + GBM + LR)...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("lr", lr)],
        voting="soft",
        weights=[2, 2, 1],
    )
    final_model = ensemble

    if enable_calibration:
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train,
            y_train,
            test_size=calibration_size,
            random_state=42,
            stratify=y_train,
        )
        ensemble.fit(X_fit, y_fit)
        print(
            "Calibrating probabilities using a held-out split "
            f"({len(X_calib)} rows, method={calibration_method})..."
        )
        final_model = _build_calibrator(ensemble, method=calibration_method)
        if FrozenEstimator is not None:
            final_model.fit(X_calib, y_calib)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The `cv='prefit'` option is deprecated",
                    category=FutureWarning,
                )
                final_model.fit(X_calib, y_calib)
    else:
        ensemble.fit(X_train, y_train)
        print("Skipping probability calibration (STRESS_MODEL_ENABLE_CALIBRATION=0).")

    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cv_mean = None
    cv_std = None
    if run_full_cv:
        print("Running 5-fold cross-validation on the base ensemble...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_weighted, y, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

    print("\nModel training complete.")
    print(f"Test Accuracy: {accuracy:.4f}")
    if cv_mean is not None and cv_std is not None:
        print(f"5-Fold CV Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    else:
        print("5-Fold CV Accuracy: skipped (set STRESS_MODEL_RUN_FULL_CV=1 to enable)")
    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Low", "Moderate", "High", "Severe"],
        )
    )

    # Feature importance from the RF sub-model
    rf_model = ensemble.named_estimators_["rf"]
    feature_importance = pd.DataFrame(
        {
            "feature": [f"Question {i+1}" for i in range(18)],
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 5 most important questions:")
    print(feature_importance.head())

    model_path = stress_model_path()
    with open(model_path, "wb") as file:
        pickle.dump(final_model, file)

    shap_model_path = stress_shap_model_path()
    with open(shap_model_path, "wb") as file:
        pickle.dump(rf_model, file)

    model_sha256 = _sha256_file(str(model_path))
    shap_model_sha256 = _sha256_file(str(shap_model_path))

    metadata = {
        "dataset_path": dataset_path if dataset_path and os.path.exists(dataset_path) else None,
        "artifact_dir": str(runtime_model_dir),
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": EXPECTED_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "model_type": (
            f"CalibratedEnsemble(RF+GBM+LR,{calibration_method})"
            if enable_calibration
            else "VotingEnsemble(RF+GBM+LR)"
        ),
        "input_preprocessing": "question_weighted",
        "ensemble_weights": [2, 2, 1],
        "calibration_enabled": bool(enable_calibration),
        "calibration_method": calibration_method if enable_calibration else None,
        "calibration_size": float(calibration_size) if enable_calibration else None,
        "full_cv_enabled": bool(run_full_cv),
        "question_weights": {
            feature: float(QUESTION_WEIGHTS[feature]) for feature in EXPECTED_FEATURE_COLUMNS
        },
        "rf_n_estimators": 150,
        "rf_n_jobs": int(rf_n_jobs),
        "gbm_n_estimators": 150,
        "random_state": 42,
        "accuracy": float(accuracy),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "model_sha256": model_sha256,
        "shap_model_sha256": shap_model_sha256,
        "feature_importance": {
            f"q{i+1}": float(rf_model.feature_importances_[i]) for i in range(18)
        },
    }

    metadata_path = stress_meta_path()
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nEnsemble model saved to: {model_path}")
    print(f"SHAP-compatible RF saved to: {shap_model_path}")
    print(f"Training metadata saved to: {metadata_path}")

    return final_model


if __name__ == "__main__":
    train_stress_model()
