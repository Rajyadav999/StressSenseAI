from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

EMOTION_CODE_TO_NAME = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

STATEMENT_CODE_TO_TEXT = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

# Public emotion datasets are not clinical stress datasets.
# This mapping creates a stress-proxy label so the model can be bootstrapped.
STRESS_PROXY_MAP = {
    ("01", "01"): ("low", 0),       # neutral
    ("02", "01"): ("low", 0),       # calm normal
    ("02", "02"): ("moderate", 1),  # calm strong
    ("03", "01"): ("moderate", 1),  # happy normal
    ("03", "02"): ("moderate", 1),  # happy strong
    ("04", "01"): ("moderate", 1),  # sad normal
    ("04", "02"): ("high", 2),      # sad strong
    ("08", "01"): ("moderate", 1),  # surprised normal
    ("08", "02"): ("high", 2),      # surprised strong
    ("05", "01"): ("high", 2),      # angry normal
    ("05", "02"): ("severe", 3),    # angry strong
    ("06", "01"): ("high", 2),      # fearful normal
    ("06", "02"): ("severe", 3),    # fearful strong
    ("07", "01"): ("high", 2),      # disgust normal
    ("07", "02"): ("severe", 3),    # disgust strong
}


def actor_split(actor_id: int) -> str:
    return "test" if actor_id >= 21 else "train"


def prepare_ravdess_manifest(dataset_root: str | Path, output_csv: str | Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    rows: List[Dict[str, object]] = []
    wav_files = sorted(dataset_root.rglob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found under {dataset_root}")

    for audio_path in wav_files:
        parts = audio_path.stem.split("-")
        if len(parts) != 7:
            continue

        _, _, emotion_code, intensity_code, statement_code, repetition_code, actor_code = parts
        if (emotion_code, intensity_code) not in STRESS_PROXY_MAP:
            continue

        stress_label, stress_level = STRESS_PROXY_MAP[(emotion_code, intensity_code)]
        actor_id = int(actor_code)
        emotion_name = EMOTION_CODE_TO_NAME.get(emotion_code, "unknown")
        intensity_name = "strong" if intensity_code == "02" else "normal"
        statement_text = STATEMENT_CODE_TO_TEXT.get(statement_code, "")

        rows.append(
            {
                "sample_id": audio_path.stem,
                "audio_path": audio_path.resolve().as_posix(),
                "stress_label": stress_label,
                "stress_level": stress_level,
                "speaker_id": f"actor_{actor_id:02d}",
                "split": actor_split(actor_id),
                "language": "en",
                "assignment_id": f"ravdess_statement_{statement_code}",
                "recording_device": "studio",
                "environment": "controlled",
                "transcript": statement_text,
                "notes": f"ravdess_emotion={emotion_name}; intensity={intensity_name}; repetition={repetition_code}",
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No manifest rows were generated from the RAVDESS dataset.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    label_counts = manifest["stress_label"].value_counts().sort_index().to_dict()
    print(f"Saved {len(manifest)} rows to {output_path}")
    print(f"Class distribution: {label_counts}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a stress-proxy manifest from the RAVDESS dataset.")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing the extracted RAVDESS WAV files.")
    parser.add_argument("--output-csv", required=True, help="Path to write the manifest CSV.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    prepare_ravdess_manifest(args.dataset_root, args.output_csv)


if __name__ == "__main__":
    main()
