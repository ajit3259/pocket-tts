"""Build a deterministic paired-speaker packet for the first Hindi overfit run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from experiments.indic.build_normalization_review import (
    ReviewCandidate,
    download_audio,
    fetch_viewer_row,
    load_token,
)

DEFAULT_MANIFEST = (
    Path(__file__).with_name("outputs") / "e6_tokenizer_corpus" / "model_input_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e13_tiny_overfit_packet"
DEFAULT_SPEAKERS = ("rasa:hindi:female", "rasa:hindi:male")
SELECTION_SEED = "e13-rasa-paired-v1"
SOURCE_DATASET = "rasa"
SOURCE_SPLIT = "train"
TARGET_PAIRS = 8
TARGET_DURATION_RANGE = (2.0, 8.0)
PROMPT_DURATION_RANGE = (3.0, 6.0)


@dataclass(frozen=True)
class PacketCandidate:
    record: dict[str, Any]
    split_row_index: int


@dataclass(frozen=True)
class PacketSelection:
    role: str
    pair_index: int
    speaker_id: str
    candidate: PacketCandidate
    prompt_example_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-file", type=Path, default=Path("HF_TOKEN"))
    parser.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    parser.add_argument("--timeout", type=float, default=60)
    return parser.parse_args()


def _has_digit(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _is_eligible(record: dict[str, Any]) -> bool:
    duration = record.get("duration_seconds")
    text = record.get("text_model_input", "")
    return (
        record.get("source_dataset") == SOURCE_DATASET
        and record.get("source_split") == SOURCE_SPLIT
        and record.get("speaker_id") in DEFAULT_SPEAKERS
        and record.get("script_mode") == "devanagari"
        and isinstance(duration, int | float)
        and math.isfinite(duration)
        and TARGET_DURATION_RANGE[0] <= duration <= TARGET_DURATION_RANGE[1]
        and bool(text)
        and not _has_digit(text)
        and not record.get("normalization_changes")
        and record.get("normalization_override") is None
        and record.get("text_source_normalized") == text
        and bool(record.get("source_utterance_id"))
    )


def load_candidates(path: Path) -> list[PacketCandidate]:
    """Load strict candidates while retaining their offset in the HF viewer split."""

    split_offsets: Counter[tuple[str, str]] = Counter()
    seen_ids: set[str] = set()
    candidates: list[PacketCandidate] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = record["example_id"]
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example ID on line {line_number}: {example_id}")
            seen_ids.add(example_id)

            split_key = (record["source_dataset"], record["source_split"])
            split_row_index = split_offsets[split_key]
            split_offsets[split_key] += 1
            if _is_eligible(record):
                candidates.append(PacketCandidate(record=record, split_row_index=split_row_index))
    return candidates


def _rank(seed: str, *values: str) -> str:
    material = "\0".join((seed, *values)).encode()
    return hashlib.sha256(material).hexdigest()


def _choose_one(
    candidates: list[PacketCandidate], *, seed: str, text: str, speaker_id: str
) -> PacketCandidate:
    return min(
        candidates,
        key=lambda candidate: _rank(
            seed, "record", text, speaker_id, candidate.record["example_id"]
        ),
    )


def select_packet(
    candidates: list[PacketCandidate],
    *,
    target_pairs: int = TARGET_PAIRS,
    seed: str = SELECTION_SEED,
) -> list[PacketSelection]:
    """Select one shared prompt and paired target texts for both Rasa speakers."""

    if target_pairs < 1:
        raise ValueError("target_pairs must be positive")

    grouped: dict[str, dict[str, list[PacketCandidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        record = candidate.record
        grouped[record["text_model_input"]][record["speaker_id"]].append(candidate)

    shared: dict[str, dict[str, PacketCandidate]] = {}
    for text, by_speaker in grouped.items():
        if not all(speaker in by_speaker for speaker in DEFAULT_SPEAKERS):
            continue
        shared[text] = {
            speaker: _choose_one(by_speaker[speaker], seed=seed, text=text, speaker_id=speaker)
            for speaker in DEFAULT_SPEAKERS
        }

    prompt_texts = [
        text
        for text, by_speaker in shared.items()
        if all(
            PROMPT_DURATION_RANGE[0]
            <= by_speaker[speaker].record["duration_seconds"]
            <= PROMPT_DURATION_RANGE[1]
            for speaker in DEFAULT_SPEAKERS
        )
    ]
    if not prompt_texts:
        raise RuntimeError("No shared prompt text satisfies the prompt duration range")
    prompt_text = min(prompt_texts, key=lambda text: _rank(seed, "prompt", text))

    target_texts = sorted(
        (text for text in shared if text != prompt_text),
        key=lambda text: _rank(seed, "target", text),
    )[:target_pairs]
    if len(target_texts) != target_pairs:
        raise RuntimeError(
            f"Need {target_pairs} paired target texts, found only {len(target_texts)}"
        )

    selections: list[PacketSelection] = []
    prompt_ids: dict[str, str] = {}
    for speaker in DEFAULT_SPEAKERS:
        candidate = shared[prompt_text][speaker]
        prompt_ids[speaker] = candidate.record["example_id"]
        selections.append(
            PacketSelection(
                role="prompt",
                pair_index=0,
                speaker_id=speaker,
                candidate=candidate,
                prompt_example_id=None,
            )
        )

    for pair_index, text in enumerate(target_texts, start=1):
        for speaker in DEFAULT_SPEAKERS:
            selections.append(
                PacketSelection(
                    role="target",
                    pair_index=pair_index,
                    speaker_id=speaker,
                    candidate=shared[text][speaker],
                    prompt_example_id=prompt_ids[speaker],
                )
            )
    return selections


def validate_audio_metadata(record: dict[str, Any], audio: dict[str, Any]) -> None:
    if audio["channels"] != 1:
        raise RuntimeError(f"{record['example_id']} is not mono")
    if audio["sample_rate"] < 24_000:
        raise RuntimeError(f"{record['example_id']} cannot be downsampled to 24 kHz")
    if audio["sample_width_bytes"] not in {2, 3, 4}:
        raise RuntimeError(f"{record['example_id']} has unsupported PCM sample width")
    duration_error = abs(audio["duration_seconds"] - record["duration_seconds"])
    if duration_error > 0.1:
        raise RuntimeError(
            f"{record['example_id']} duration differs from metadata by {duration_error:.3f}s"
        )


def inspect_audio_file(path: Path) -> dict[str, Any]:
    audio_bytes = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        return {
            "sample_rate": sample_rate,
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frames": frames,
            "duration_seconds": round(frames / sample_rate, 3),
            "sha256": hashlib.sha256(audio_bytes).hexdigest(),
        }


def _speaker_slug(speaker_id: str) -> str:
    return speaker_id.rsplit(":", maxsplit=1)[-1]


def materialize_packet(
    manifest_path: Path,
    output_dir: Path,
    *,
    token: str,
    target_pairs: int = TARGET_PAIRS,
    timeout: float,
) -> list[dict[str, Any]]:
    candidates = load_candidates(manifest_path)
    selections = select_packet(candidates, target_pairs=target_pairs)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    entries: list[dict[str, Any]] = []
    for selection in selections:
        record = selection.candidate.record
        filename = (
            f"{selection.role}_{selection.pair_index:02d}_{_speaker_slug(selection.speaker_id)}_"
            f"{record['example_id']}.wav"
        )
        audio_path = output_dir / filename
        if audio_path.is_file():
            audio = inspect_audio_file(audio_path)
        else:
            review_candidate = ReviewCandidate(
                manifest_record={**record, "text_normalized": record["text_source_normalized"]},
                split_row_index=selection.candidate.split_row_index,
                review_items=(),
            )
            _, audio_url = fetch_viewer_row(session, review_candidate, timeout=timeout)
            audio = download_audio(session, audio_url, audio_path, timeout=timeout)
        validate_audio_metadata(record, audio)
        entries.append(
            {
                "schema_version": 1,
                "selection_seed": SELECTION_SEED,
                "role": selection.role,
                "pair_index": selection.pair_index,
                "speaker_id": selection.speaker_id,
                "example_id": record["example_id"],
                "prompt_example_id": selection.prompt_example_id,
                "audio_file": filename,
                "audio": audio,
                "duration_seconds_source": record["duration_seconds"],
                "text_model_input": record["text_model_input"],
                "source_dataset": record["source_dataset"],
                "source_split": record["source_split"],
                "source_license": record["source_license"],
                "source_utterance_id": record["source_utterance_id"],
                "source_locator": record["source_locator"],
                "dataset_viewer_split_row_index": selection.candidate.split_row_index,
            }
        )

    _write_jsonl(output_dir / "manifest.jsonl", entries)
    review_path = output_dir / "review.json"
    reviews = build_review_records(entries, load_review_records(review_path))
    review_path.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stats = _packet_stats(
        entries, reviews=reviews, candidates=len(candidates), manifest_path=manifest_path
    )
    (output_dir / "packet_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REVIEW.md").write_text(_review_markdown(entries, reviews), encoding="utf-8")
    return entries


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _packet_stats(
    entries: list[dict[str, Any]],
    *,
    reviews: list[dict[str, Any]],
    candidates: int,
    manifest_path: Path,
) -> dict[str, Any]:
    audio_seconds = sum(entry["audio"]["duration_seconds"] for entry in entries)
    target_entries = [entry for entry in entries if entry["role"] == "target"]
    prompt_entries = [entry for entry in entries if entry["role"] == "prompt"]
    try:
        source_manifest = manifest_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_manifest = manifest_path.as_posix()
    return {
        "schema_version": 1,
        "selection_seed": SELECTION_SEED,
        "source_manifest": source_manifest,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "eligible_recordings": candidates,
        "prompt_recordings": len(prompt_entries),
        "target_recordings": len(target_entries),
        "target_text_pairs": len({entry["pair_index"] for entry in target_entries}),
        "speakers": sorted({entry["speaker_id"] for entry in entries}),
        "audio_seconds": round(audio_seconds, 3),
        "audio_sha256": hashlib.sha256(
            "".join(entry["audio"]["sha256"] for entry in entries).encode()
        ).hexdigest(),
        "human_review": {
            "complete": all(review["decision"] != "pending" for review in reviews),
            "decisions": dict(sorted(Counter(review["decision"] for review in reviews).items())),
        },
        "selection_policy": {
            "source_dataset": SOURCE_DATASET,
            "source_split": SOURCE_SPLIT,
            "target_duration_seconds": list(TARGET_DURATION_RANGE),
            "prompt_duration_seconds": list(PROMPT_DURATION_RANGE),
            "script_mode": "devanagari",
            "digits_allowed": False,
            "normalization_changes_allowed": False,
            "normalization_overrides_allowed": False,
            "shared_text_across_speakers": True,
            "prompt_and_target_disjoint": True,
        },
    }


def load_review_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    reviews = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reviews, list):
        raise ValueError("Review file must contain a list")
    by_example: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if review.get("schema_version") != 1:
            raise ValueError("Unsupported review schema")
        example_id = review["example_id"]
        if example_id in by_example:
            raise ValueError(f"Duplicate review for {example_id}")
        by_example[example_id] = review
    return by_example


def build_review_records(
    entries: list[dict[str, Any]], existing: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for entry in entries:
        previous = existing.get(entry["example_id"])
        audio_sha256 = entry["audio"]["sha256"]
        preserve = previous is not None and previous.get("audio_sha256") == audio_sha256
        reviews.append(
            {
                "schema_version": 1,
                "example_id": entry["example_id"],
                "audio_file": entry["audio_file"],
                "audio_sha256": audio_sha256,
                "role": entry["role"],
                "pair_index": entry["pair_index"],
                "speaker_id": entry["speaker_id"],
                "decision": previous.get("decision", "pending") if preserve else "pending",
                "audio_clean": previous.get("audio_clean") if preserve else None,
                "transcript_matches": (previous.get("transcript_matches") if preserve else None),
                "voice_stable": previous.get("voice_stable") if preserve else None,
                "notes": previous.get("notes", "") if preserve else "",
            }
        )
    return reviews


def _review_markdown(entries: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> str:
    reviews_by_id = {review["example_id"]: review for review in reviews}
    lines = [
        "# E13 Tiny Overfit Packet",
        "",
        "Listen for clean audio, exact transcript agreement, and a stable voice within each",
        "speaker. Mark any bad clip before this packet is used on a GPU.",
        "",
    ]
    for entry in entries:
        review = reviews_by_id[entry["example_id"]]
        lines.extend(
            [
                f"## {entry['role'].title()} {entry['pair_index']:02d} - "
                f"{_speaker_slug(entry['speaker_id'])}",
                "",
                f"- Audio: [{entry['audio_file']}]({entry['audio_file']})",
                f"- Example: `{entry['example_id']}`",
                f"- Duration: `{entry['audio']['duration_seconds']:.3f}s`",
                f"- Text: {entry['text_model_input']}",
                f"- Decision: `{review['decision']}`",
                f"- Notes: {review['notes']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = materialize_packet(
        args.manifest,
        args.output_dir,
        token=load_token(args.token_file),
        target_pairs=args.target_pairs,
        timeout=args.timeout,
    )
    targets = sum(entry["role"] == "target" for entry in entries)
    prompts = len(entries) - targets
    print(f"Wrote {targets} targets and {prompts} prompts to {args.output_dir}")
    print(args.output_dir / "REVIEW.md")


if __name__ == "__main__":
    main()
