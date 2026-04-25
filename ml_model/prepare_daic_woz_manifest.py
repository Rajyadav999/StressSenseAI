from __future__ import annotations

import argparse
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

LABEL_COLUMNS = [
    "sample_id",
    "audio_path",
    "stress_label",
    "stress_level",
    "speaker_id",
    "split",
    "language",
    "assignment_id",
    "recording_device",
    "environment",
    "transcript",
    "notes",
]


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    speaker: str
    text: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _infer_split_from_path(path: Path) -> str:
    text = str(path).lower()
    if "train" in text:
        return "train"
    if "dev" in text or "validation" in text or "val" in text:
        return "validation"
    if "test" in text:
        return "test"
    return ""


def _extract_session_id(value: str | Path) -> str:
    text = Path(value).stem if not isinstance(value, str) else value
    match = re.search(r"(\d+)", str(text))
    if not match:
        raise ValueError(f"Could not infer DAIC session id from: {value}")
    return match.group(1)


def _detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _map_phq_to_stress(score: float, moderate_min: int, high_min: int, severe_min: int) -> tuple[str, int]:
    rounded = int(round(float(score)))
    if rounded >= severe_min:
        return "severe", 3
    if rounded >= high_min:
        return "high", 2
    if rounded >= moderate_min:
        return "moderate", 1
    return "low", 0


def _load_label_table(label_csvs: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for csv_path in label_csvs:
        table = pd.read_csv(csv_path)
        participant_column = _detect_column(
            table.columns,
            ["participantid", "participant", "id", "subjectid", "subject"],
        )
        score_column = _detect_column(
            table.columns,
            ["phq8score", "phqscore", "phq9score", "score"],
        )
        binary_column = _detect_column(
            table.columns,
            ["phq8binary", "phqbinary", "depressionlabel", "label", "binary"],
        )
        split_column = _detect_column(
            table.columns,
            ["split", "partition", "subset"],
        )

        if participant_column is None:
            raise ValueError(f"Could not find participant id column in {csv_path}")
        if score_column is None and binary_column is None:
            raise ValueError(f"Could not find PHQ score or label column in {csv_path}")

        inferred_split = _infer_split_from_path(csv_path)
        for _, record in table.iterrows():
            session_id = _extract_session_id(str(record[participant_column]))
            score_value = record[score_column] if score_column else None
            binary_value = record[binary_column] if binary_column else None
            split_value = str(record[split_column]).strip().lower() if split_column else inferred_split

            if pd.isna(score_value) and pd.isna(binary_value):
                continue

            rows.append(
                {
                    "session_id": session_id,
                    "phq_score": None if pd.isna(score_value) else float(score_value),
                    "phq_binary": None if pd.isna(binary_value) else int(binary_value),
                    "split": split_value or inferred_split,
                    "label_source_csv": str(csv_path),
                }
            )

    if not rows:
        raise ValueError("No labeled DAIC rows were loaded from the provided CSV files.")

    labels = pd.DataFrame(rows).drop_duplicates(subset=["session_id"], keep="first")
    labels["session_id"] = labels["session_id"].astype(str)
    return labels


def _auto_discover_label_csvs(dataset_root: Path) -> List[Path]:
    candidates = sorted(path for path in dataset_root.rglob("*.csv") if "split" in path.name.lower())
    if not candidates:
        raise ValueError(
            f"Could not auto-discover DAIC split CSVs under {dataset_root}. "
            "Pass them explicitly with --labels-csv."
        )
    return candidates


def _read_transcript(transcript_path: Path) -> List[TranscriptSegment]:
    table = pd.read_csv(transcript_path, sep=None, engine="python")
    start_column = _detect_column(table.columns, ["starttime", "start", "begin"])
    end_column = _detect_column(table.columns, ["stoptime", "endtime", "end", "stop"])
    speaker_column = _detect_column(table.columns, ["speaker", "participanttype", "role"])
    text_column = _detect_column(table.columns, ["value", "transcript", "text", "utterance"])

    if not all([start_column, end_column, speaker_column, text_column]):
        raise ValueError(f"Transcript file {transcript_path} is missing required columns.")

    segments: List[TranscriptSegment] = []
    for _, row in table.iterrows():
        start_value = row[start_column]
        end_value = row[end_column]
        if pd.isna(start_value) or pd.isna(end_value):
            continue

        speaker = str(row[speaker_column]).strip().lower()
        text = str(row[text_column]).strip()
        segments.append(
            TranscriptSegment(
                start_sec=float(start_value),
                end_sec=float(end_value),
                speaker=speaker,
                text=text,
            )
        )

    return segments


def _merge_segments(segments: List[TranscriptSegment], max_gap_sec: float) -> List[TranscriptSegment]:
    if not segments:
        return []

    merged: List[TranscriptSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.start_sec - previous.end_sec <= max_gap_sec:
            merged[-1] = TranscriptSegment(
                start_sec=previous.start_sec,
                end_sec=max(previous.end_sec, segment.end_sec),
                speaker=previous.speaker,
                text=f"{previous.text} {segment.text}".strip(),
            )
        else:
            merged.append(segment)
    return merged


def _write_wav_segment(audio_path: Path, output_path: Path, start_sec: float, end_sec: float) -> None:
    with wave.open(str(audio_path), "rb") as source:
        frame_rate = source.getframerate()
        sample_width = source.getsampwidth()
        channels = source.getnchannels()
        params = source.getparams()

        start_frame = max(0, int(start_sec * frame_rate))
        end_frame = max(start_frame + 1, int(end_sec * frame_rate))
        source.setpos(start_frame)
        frame_count = end_frame - start_frame
        frames = source.readframes(frame_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as target:
        target.setparams((channels, sample_width, frame_rate, 0, params.comptype, params.compname))
        target.writeframes(frames)


def prepare_daic_woz_manifest(
    dataset_root: str | Path,
    output_csv: str | Path,
    labels_csv: Optional[List[str | Path]] = None,
    segments_root: str | Path | None = None,
    min_segment_sec: float = 1.5,
    merge_gap_sec: float = 0.35,
    moderate_min: int = 5,
    high_min: int = 10,
    severe_min: int = 15,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    label_paths = [Path(path) for path in labels_csv] if labels_csv else _auto_discover_label_csvs(dataset_root)
    labels = _load_label_table(label_paths)
    label_lookup = {row["session_id"]: row for _, row in labels.iterrows()}

    segments_root_path = Path(segments_root) if segments_root else Path(output_csv).resolve().parent / "daic_woz_segments"
    rows: List[Dict[str, object]] = []

    for audio_path in sorted(dataset_root.rglob("*_AUDIO.wav")):
        session_id = _extract_session_id(audio_path.stem)
        if session_id not in label_lookup:
            continue

        label_row = label_lookup[session_id]
        phq_score = label_row.get("phq_score", None)
        if phq_score is None or pd.isna(phq_score):
            continue

        stress_label, stress_level = _map_phq_to_stress(
            score=float(phq_score),
            moderate_min=moderate_min,
            high_min=high_min,
            severe_min=severe_min,
        )
        split = str(label_row.get("split", "") or _infer_split_from_path(audio_path)).strip().lower()
        transcript_path = audio_path.with_name(audio_path.name.replace("_AUDIO.wav", "_TRANSCRIPT.csv"))

        participant_segments: List[TranscriptSegment] = []
        if transcript_path.exists():
            all_segments = _read_transcript(transcript_path)
            participant_segments = [
                segment
                for segment in all_segments
                if "participant" in segment.speaker and segment.duration_sec >= min_segment_sec
            ]
            participant_segments = _merge_segments(participant_segments, max_gap_sec=merge_gap_sec)

        if participant_segments:
            for segment_index, segment in enumerate(participant_segments, start=1):
                segment_output = (
                    segments_root_path
                    / (split or "unassigned")
                    / f"participant_{session_id}"
                    / f"{session_id}_segment_{segment_index:03d}.wav"
                )
                _write_wav_segment(
                    audio_path=audio_path,
                    output_path=segment_output,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                )
                rows.append(
                    {
                        "sample_id": f"daic_{session_id}_{segment_index:03d}",
                        "audio_path": segment_output.resolve().as_posix(),
                        "stress_label": stress_label,
                        "stress_level": stress_level,
                        "speaker_id": f"participant_{session_id}",
                        "split": split,
                        "language": "en",
                        "assignment_id": f"daic_woz_{session_id}",
                        "recording_device": "clinical_interview_mic",
                        "environment": "clinical_interview",
                        "transcript": segment.text,
                        "notes": (
                            f"label_source=DAIC-WOZ; phq_score={int(round(float(phq_score)))}; "
                            f"label_mapping=phq8_thresholds; source_audio={audio_path.name}; "
                            f"segment={segment.start_sec:.2f}-{segment.end_sec:.2f}s"
                        ),
                    }
                )
        else:
            rows.append(
                {
                    "sample_id": f"daic_{session_id}_full",
                    "audio_path": audio_path.resolve().as_posix(),
                    "stress_label": stress_label,
                    "stress_level": stress_level,
                    "speaker_id": f"participant_{session_id}",
                    "split": split,
                    "language": "en",
                    "assignment_id": f"daic_woz_{session_id}",
                    "recording_device": "clinical_interview_mic",
                    "environment": "clinical_interview",
                    "transcript": "",
                    "notes": (
                        f"label_source=DAIC-WOZ; phq_score={int(round(float(phq_score)))}; "
                        "label_mapping=phq8_thresholds; segmentation=full_audio_fallback"
                    ),
                }
            )

    if not rows:
        raise ValueError(
            "No DAIC-WOZ manifest rows were created. "
            "Check that labeled CSVs and *_AUDIO.wav files are present under the dataset root."
        )

    manifest = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    label_counts = manifest["stress_label"].value_counts().sort_index().to_dict()
    speaker_count = manifest["speaker_id"].nunique()
    print(f"Saved {len(manifest)} rows to {output_path}")
    print(f"Speakers: {speaker_count}")
    print(f"Class distribution: {label_counts}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a DAIC-WOZ manifest and participant-only WAV segments for stress-model training."
    )
    parser.add_argument("--dataset-root", required=True, help="Root directory of the licensed DAIC-WOZ dataset.")
    parser.add_argument("--output-csv", required=True, help="Path to write the manifest CSV.")
    parser.add_argument(
        "--labels-csv",
        nargs="+",
        help="Optional one or more split CSVs with participant ids and PHQ scores. If omitted, the script auto-discovers them.",
    )
    parser.add_argument(
        "--segments-root",
        help="Optional folder where participant-only WAV segments will be written. Defaults next to the manifest output.",
    )
    parser.add_argument("--min-segment-sec", type=float, default=1.5, help="Minimum participant segment duration to keep.")
    parser.add_argument("--merge-gap-sec", type=float, default=0.35, help="Merge adjacent participant turns separated by up to this gap.")
    parser.add_argument("--moderate-min", type=int, default=5, help="Minimum PHQ score mapped to Moderate.")
    parser.add_argument("--high-min", type=int, default=10, help="Minimum PHQ score mapped to High.")
    parser.add_argument("--severe-min", type=int, default=15, help="Minimum PHQ score mapped to Severe.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    prepare_daic_woz_manifest(
        dataset_root=args.dataset_root,
        output_csv=args.output_csv,
        labels_csv=args.labels_csv,
        segments_root=args.segments_root,
        min_segment_sec=args.min_segment_sec,
        merge_gap_sec=args.merge_gap_sec,
        moderate_min=args.moderate_min,
        high_min=args.high_min,
        severe_min=args.severe_min,
    )


if __name__ == "__main__":
    main()
