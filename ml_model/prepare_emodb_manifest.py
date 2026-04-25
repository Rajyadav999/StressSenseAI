from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

EMOTION_CODE_TO_NAME = {
    "N": "neutral",
    "L": "boredom",
    "F": "happiness",
    "T": "sadness",
    "E": "disgust",
    "A": "fear",
    "W": "anger",
}

STATEMENT_CODE_TO_TEXT = {
    "a01": "Der Lappen liegt auf dem Eisschrank.",
    "a02": "Das will sie am Mittwoch abgeben.",
    "a04": "Heute abend koennte ich es ihm sagen.",
    "a05": "Das schwarze Blatt Papier befindet sich da oben neben dem Holzstueck.",
    "a07": "In sieben Stunden wird es soweit sein.",
    "b01": "Was sind denn das fuer Tueten, die da unter dem Tisch stehen.",
    "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
    "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
    "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
    "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
}

STRESS_PROXY_MAP = {
    "N": ("low", 0),
    "L": ("low", 0),
    "F": ("moderate", 1),
    "T": ("moderate", 1),
    "E": ("high", 2),
    "A": ("high", 2),
    "W": ("severe", 3),
}


def speaker_split(speaker_id: str) -> str:
    return "test" if speaker_id in {"15", "16"} else "train"


def prepare_emodb_manifest(dataset_root: str | Path, output_csv: str | Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    wav_root = dataset_root / "wav"
    if not wav_root.exists():
        raise ValueError(f"Expected wav folder under {dataset_root}")

    rows: List[Dict[str, object]] = []
    wav_files = sorted(wav_root.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found under {wav_root}")

    for audio_path in wav_files:
        stem = audio_path.stem
        if len(stem) < 6:
            continue

        speaker_id = stem[:2]
        statement_code = stem[2:5]
        emotion_code = stem[5].upper()
        variation_code = stem[6:] if len(stem) > 6 else ""

        if emotion_code not in STRESS_PROXY_MAP:
            continue

        stress_label, stress_level = STRESS_PROXY_MAP[emotion_code]
        emotion_name = EMOTION_CODE_TO_NAME.get(emotion_code, "unknown")

        rows.append(
            {
                "sample_id": stem,
                "audio_path": audio_path.resolve().as_posix(),
                "stress_label": stress_label,
                "stress_level": stress_level,
                "speaker_id": f"speaker_{speaker_id}",
                "split": speaker_split(speaker_id),
                "language": "de",
                "assignment_id": f"emodb_{statement_code}",
                "recording_device": "studio",
                "environment": "controlled",
                "transcript": STATEMENT_CODE_TO_TEXT.get(statement_code, ""),
                "notes": f"emodb_emotion={emotion_name}; variation={variation_code}",
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No manifest rows were generated from the EMO-DB dataset.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    label_counts = manifest["stress_label"].value_counts().sort_index().to_dict()
    print(f"Saved {len(manifest)} rows to {output_path}")
    print(f"Class distribution: {label_counts}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a stress-proxy manifest from the EMO-DB dataset.")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing the extracted EMO-DB dataset.")
    parser.add_argument("--output-csv", required=True, help="Path to write the manifest CSV.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    prepare_emodb_manifest(args.dataset_root, args.output_csv)


if __name__ == "__main__":
    main()
