from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from .audio_dataset_tools import build_feature_dataset_from_manifest, build_public_emotion_manifest
    from .audio_features import AUDIO_FEATURE_COLUMNS, FEATURE_GROUP_WEIGHTS, apply_feature_weights
except ImportError:
    from audio_dataset_tools import build_feature_dataset_from_manifest, build_public_emotion_manifest
    from audio_features import AUDIO_FEATURE_COLUMNS, FEATURE_GROUP_WEIGHTS, apply_feature_weights

CLASS_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def build_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=random_state,
            nthread=1,
        )

    return models


def _has_explicit_split(feature_df: pd.DataFrame) -> bool:
    if "split" not in feature_df.columns:
        return False

    normalized = feature_df["split"].fillna("").astype(str).str.lower().str.strip()
    return normalized.isin({"train", "training"}).any() and normalized.isin(
        {"test", "testing", "eval", "evaluation", "val", "validation"}
    ).any()


def _split_dataset(
    feature_df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if _has_explicit_split(feature_df):
        normalized = feature_df["split"].fillna("").astype(str).str.lower().str.strip()
        train_df = feature_df.loc[normalized.isin({"train", "training"})].copy()
        test_df = feature_df.loc[
            normalized.isin({"test", "testing", "eval", "evaluation", "val", "validation"})
        ].copy()
        if not train_df.empty and not test_df.empty:
            return train_df, test_df, "manifest_split"

    if "speaker_id" in feature_df.columns:
        speaker_groups = feature_df["speaker_id"].fillna("").astype(str).str.strip()
        non_empty_speakers = speaker_groups[speaker_groups != ""]
        if non_empty_speakers.nunique() >= 6:
            all_classes = set(feature_df["stress_level"].astype(int).tolist())
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            for train_index, test_index in splitter.split(
                feature_df,
                feature_df["stress_level"],
                groups=speaker_groups,
            ):
                train_df = feature_df.iloc[train_index].copy()
                test_df = feature_df.iloc[test_index].copy()
                if set(train_df["stress_level"].astype(int).tolist()) == all_classes and set(
                    test_df["stress_level"].astype(int).tolist()
                ) == all_classes:
                    return train_df, test_df, "group_shuffle_split"

    train_df, test_df = train_test_split(
        feature_df,
        test_size=0.2,
        stratify=feature_df["stress_level"],
        random_state=random_state,
    )
    return train_df.copy(), test_df.copy(), "stratified_random_split"


def _feature_fill_values(feature_df: pd.DataFrame) -> Dict[str, float]:
    fill_values: Dict[str, float] = {}
    for column in AUDIO_FEATURE_COLUMNS:
        median = pd.to_numeric(feature_df[column], errors="coerce").median()
        fill_values[column] = float(median) if pd.notna(median) else 0.0
    return fill_values


def _prepare_split_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Dict[str, float]]:
    X_train = train_df[AUDIO_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[AUDIO_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    fill_values = _feature_fill_values(X_train)
    X_train_filled = X_train.fillna(fill_values)
    X_test_filled = X_test.fillna(fill_values)
    X_train_array = X_train_filled.to_numpy(dtype=np.float32)
    X_test_array = X_test_filled.to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_array)
    X_test_scaled = scaler.transform(X_test_array)

    X_train_weighted = apply_feature_weights(X_train_scaled, AUDIO_FEATURE_COLUMNS)
    X_test_weighted = apply_feature_weights(X_test_scaled, AUDIO_FEATURE_COLUMNS)

    return X_train_weighted, X_test_weighted, scaler, fill_values


def _fit_model(model: Any, X_train: np.ndarray, y_train: pd.Series, sample_weight: np.ndarray) -> Any:
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _feature_importance(model: Any) -> Dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {column: 0.0 for column in AUDIO_FEATURE_COLUMNS}

    importances = getattr(model, "feature_importances_")
    ranked = sorted(
        zip(AUDIO_FEATURE_COLUMNS, np.asarray(importances, dtype=float)),
        key=lambda item: item[1],
        reverse=True,
    )
    return {name: float(value) for name, value in ranked}


def train_audio_stress_model(
    manifest_csv: str | Path | None = None,
    dataset_root: str | Path | None = None,
    cached_features_csv: str | Path | None = None,
    ravdess_root: str | Path | None = None,
    cremad_root: str | Path | None = None,
    tess_root: str | Path | None = None,
    public_data_dir: str | Path | None = None,
    manifest_out: str | Path | None = None,
    features_out: str | Path | None = None,
    model_out: str | Path | None = None,
    scaler_out: str | Path | None = None,
    metadata_out: str | Path | None = None,
    skip_failed: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    model_dir = Path(__file__).resolve().parent
    manifest_out = Path(manifest_out or model_dir / "public_combined_stress_manifest.csv")
    features_out = Path(features_out or model_dir / "public_combined_stress_features.csv")
    model_out = Path(model_out or model_dir / "audio_stress_model.pkl")
    scaler_out = Path(scaler_out or model_dir / "audio_stress_scaler.pkl")
    metadata_out = Path(metadata_out or model_dir / "audio_stress_model_meta.json")

    feature_source: str
    manifest_source: Optional[str] = None

    if cached_features_csv:
        feature_df = pd.read_csv(cached_features_csv)
        feature_source = str(Path(cached_features_csv).resolve())
    else:
        if manifest_csv:
            manifest_source = str(Path(manifest_csv).resolve())
        else:
            manifest = build_public_emotion_manifest(
                ravdess_root=ravdess_root,
                cremad_root=cremad_root,
                tess_root=tess_root,
                output_csv=manifest_out,
                base_dir=public_data_dir,
                skip_missing=True,
            )
            manifest_csv = manifest_out
            manifest_source = str(manifest_out.resolve())
            if manifest.empty:
                raise ValueError("The combined public manifest is empty.")

        feature_df = build_feature_dataset_from_manifest(
            manifest_csv=manifest_csv,
            output_csv=features_out,
            dataset_root=dataset_root,
            skip_failed=skip_failed,
        )
        feature_source = str(features_out.resolve())

    missing_features = [column for column in AUDIO_FEATURE_COLUMNS if column not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Feature dataset is missing required columns: {missing_features}")

    train_df, test_df, split_method = _split_dataset(feature_df, random_state=random_state)
    X_train, X_test, scaler, fill_values = _prepare_split_features(train_df, test_df)
    y_train = train_df["stress_level"].astype(int)
    y_test = test_df["stress_level"].astype(int)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    candidate_models = build_candidate_models(random_state=random_state)
    candidate_results: Dict[str, Dict[str, float]] = {}

    best_model_name = ""
    best_model: Any | None = None
    best_predictions: Optional[np.ndarray] = None
    best_probabilities: Optional[np.ndarray] = None
    best_recall = -1.0
    best_accuracy = -1.0

    for model_name, candidate_model in candidate_models.items():
        fitted_model = _fit_model(candidate_model, X_train, y_train, sample_weights)
        predictions = fitted_model.predict(X_test)
        probabilities = fitted_model.predict_proba(X_test)

        accuracy = float(accuracy_score(y_test, predictions))
        precision = float(precision_score(y_test, predictions, average="macro", zero_division=0))
        recall = float(recall_score(y_test, predictions, average="macro", zero_division=0))

        candidate_results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

        if recall > best_recall or (recall == best_recall and accuracy > best_accuracy):
            best_model_name = model_name
            best_model = fitted_model
            best_predictions = predictions
            best_probabilities = probabilities
            best_recall = recall
            best_accuracy = accuracy

    if best_model is None or best_predictions is None or best_probabilities is None:
        raise RuntimeError("No audio stress model candidate could be trained.")

    accuracy = float(accuracy_score(y_test, best_predictions))
    precision = float(precision_score(y_test, best_predictions, average="macro", zero_division=0))
    recall = float(recall_score(y_test, best_predictions, average="macro", zero_division=0))
    confusion = confusion_matrix(y_test, best_predictions, labels=[0, 1, 2]).tolist()
    report = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1, 2],
        target_names=[CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]],
        output_dict=True,
        zero_division=0,
    )
    importance = _feature_importance(best_model)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_out)
    joblib.dump(scaler, scaler_out)

    dataset_distribution = (
        feature_df["dataset"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        if "dataset" in feature_df.columns
        else {}
    )
    class_distribution = (
        feature_df["stress_label"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
    )
    speaker_count = (
        feature_df["speaker_id"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()
        if "speaker_id" in feature_df.columns
        else 0
    )

    metadata = {
        "manifest_source": manifest_source,
        "feature_source": feature_source,
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root else None,
        "public_data_dir": str(Path(public_data_dir).resolve()) if public_data_dir else None,
        "selected_model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "model_path": str(model_out.resolve()),
        "scaler_path": str(scaler_out.resolve()),
        "total_rows": int(len(feature_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_method": split_method,
        "speaker_count": int(speaker_count),
        "dataset_distribution": {str(key): int(value) for key, value in dataset_distribution.items()},
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
        "features": list(AUDIO_FEATURE_COLUMNS),
        "feature_weights": FEATURE_GROUP_WEIGHTS,
        "feature_fill_values": fill_values,
        "classes": [0, 1, 2],
        "class_labels": CLASS_LABELS,
        "candidate_results": candidate_results,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        },
        "classification_report": report,
        "confusion_matrix": confusion,
        "top_feature_importance": dict(list(importance.items())[:20]),
        "random_state": random_state,
    }

    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved to: {model_out}")
    print(f"Scaler saved to: {scaler_out}")
    print(f"Metadata saved to: {metadata_out}")
    print(f"Selected model: {best_model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a weighted voice stress classifier.")
    parser.add_argument("--manifest", help="Manifest CSV with audio paths and stress labels.")
    parser.add_argument("--dataset-root", help="Root folder used to resolve relative audio paths in the manifest.")
    parser.add_argument("--features-csv", help="Precomputed feature CSV. Skips feature extraction if provided.")
    parser.add_argument("--ravdess-root", help="Optional explicit RAVDESS root.")
    parser.add_argument("--cremad-root", help="Optional explicit CREMA-D root.")
    parser.add_argument("--tess-root", help="Optional explicit TESS root.")
    parser.add_argument("--public-data-dir", help="Optional parent folder containing public emotion datasets.")
    parser.add_argument("--manifest-out", help="Path to save the auto-generated manifest CSV.")
    parser.add_argument("--features-out", help="Path to save the extracted feature CSV.")
    parser.add_argument("--model-out", help="Path to save the trained model.")
    parser.add_argument("--scaler-out", help="Path to save the fitted scaler.")
    parser.add_argument("--metadata-out", help="Path to save training metadata JSON.")
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip unreadable or unsupported audio files during feature extraction.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_audio_stress_model(
        manifest_csv=args.manifest,
        dataset_root=args.dataset_root,
        cached_features_csv=args.features_csv,
        ravdess_root=args.ravdess_root,
        cremad_root=args.cremad_root,
        tess_root=args.tess_root,
        public_data_dir=args.public_data_dir,
        manifest_out=args.manifest_out,
        features_out=args.features_out,
        model_out=args.model_out,
        scaler_out=args.scaler_out,
        metadata_out=args.metadata_out,
        skip_failed=args.skip_failed,
    )


if __name__ == "__main__":
    main()
