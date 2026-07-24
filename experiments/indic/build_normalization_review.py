"""Download only normalization-flagged audio rows into a human review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from experiments.indic.text_normalization import NormalizationReviewItem, normalize_hindi_text

DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_MANIFEST = Path(__file__).with_name("outputs") / "e4_data_audit" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e5_normalization_review"
DATASET_CONFIGS = {"rasa": "Hindi", "indicvoices_r": "Hindi"}


@dataclass(frozen=True)
class ReviewCandidate:
    manifest_record: dict[str, Any]
    split_row_index: int
    review_items: tuple[NormalizationReviewItem, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-file", type=Path, default=Path("HF_TOKEN"))
    parser.add_argument("--timeout", type=float, default=60)
    return parser.parse_args()


def load_token(path: Path) -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    if not path.is_file():
        raise FileNotFoundError(
            f"No Hugging Face token found in HF_TOKEN or {path}. The source dataset is gated."
        )
    return path.read_text(encoding="utf-8").strip()


def find_review_candidates(path: Path) -> list[ReviewCandidate]:
    """Find flagged rows while deriving their offset within each viewer split."""

    split_offsets: Counter[tuple[str, str]] = Counter()
    candidates: list[ReviewCandidate] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split_key = (record["source_dataset"], record["source_split"])
            split_row_index = split_offsets[split_key]
            split_offsets[split_key] += 1

            result = normalize_hindi_text(record["text_normalized"])
            if result.needs_review:
                candidates.append(
                    ReviewCandidate(
                        manifest_record=record,
                        split_row_index=split_row_index,
                        review_items=result.review_items,
                    )
                )
    return candidates


def _audio_url(audio_value: Any) -> str:
    assets = audio_value if isinstance(audio_value, list) else [audio_value]
    for asset in assets:
        if isinstance(asset, dict) and asset.get("src"):
            return str(asset["src"])
    raise RuntimeError("Dataset viewer row has no downloadable audio asset")


def fetch_viewer_row(
    session: requests.Session, candidate: ReviewCandidate, *, timeout: float
) -> tuple[dict[str, Any], str]:
    record = candidate.manifest_record
    locator = record["source_locator"]
    source = record["source_dataset"]
    if source not in DATASET_CONFIGS:
        raise RuntimeError(f"No dataset-viewer config registered for {source}")

    response = session.get(
        DATASET_VIEWER_ROWS_URL,
        params={
            "dataset": locator["repo_id"],
            "config": DATASET_CONFIGS[source],
            "split": record["source_split"],
            "offset": candidate.split_row_index,
            "length": 1,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("rows", [])
    if len(rows) != 1:
        raise RuntimeError(f"Expected one viewer row, received {len(rows)}")

    viewer_item = rows[0]
    if viewer_item["row_idx"] != candidate.split_row_index:
        raise RuntimeError("Dataset viewer returned an unexpected row index")
    viewer_row = viewer_item["row"]
    if viewer_row.get("filename") != record["source_utterance_id"]:
        raise RuntimeError("Dataset viewer filename does not match the manifest")
    if viewer_row.get("text") != record["text_normalized"]:
        raise RuntimeError("Dataset viewer transcript does not match the manifest")

    audio_url = _audio_url(viewer_row.get(locator["audio_column"]))
    if f"/{locator['revision']}/" not in audio_url:
        raise RuntimeError("Dataset viewer audio revision does not match the pinned manifest")
    return viewer_row, audio_url


def download_audio(
    session: requests.Session, url: str, output_path: Path, *, timeout: float
) -> dict[str, Any]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    audio_bytes = response.content
    output_path.write_bytes(audio_bytes)

    with wave.open(str(output_path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        metadata = {
            "sample_rate": sample_rate,
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frames": frames,
            "duration_seconds": round(frames / sample_rate, 3),
            "sha256": hashlib.sha256(audio_bytes).hexdigest(),
        }
    return metadata


def _review_question(items: tuple[NormalizationReviewItem, ...]) -> str:
    kinds = {item.kind for item in items}
    if "unexpanded-cardinal" in kinds:
        token = next(item.token for item in items if item.kind == "unexpanded-cardinal")
        return f'Write the exact words spoken for "{token}" (Hindi, English, digits, or other).'
    if "embedded-digit" in kinds:
        return "Is the zero spoken, or is it an erroneous separator in the compound word?"
    if "possible-punctuation-I" in kinds:
        return "Confirm that the Latin I is not spoken and mark the heard sentence boundary."
    return "Write the exact words heard around the flagged token."


def build_review_packet(
    manifest_path: Path, output_dir: Path, *, token: str, timeout: float
) -> list[dict[str, Any]]:
    candidates = find_review_candidates(manifest_path)
    if not candidates:
        raise RuntimeError("The manifest contains no normalization review candidates")

    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    entries: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        record = candidate.manifest_record
        _, audio_url = fetch_viewer_row(session, candidate, timeout=timeout)
        filename = f"{index:02d}_{record['example_id']}.wav"
        audio_metadata = download_audio(session, audio_url, output_dir / filename, timeout=timeout)
        cardinal_preview = normalize_hindi_text(record["text_normalized"], number_mode="cardinal")
        entries.append(
            {
                "index": index,
                "example_id": record["example_id"],
                "audio_file": filename,
                "audio": audio_metadata,
                "speaker_id": record["speaker_id"],
                "source_split": record["source_split"],
                "source_utterance_id": record["source_utterance_id"],
                "text_source_normalized": record["text_normalized"],
                "cardinal_preview": cardinal_preview.text,
                "review_items": [item.to_dict() for item in candidate.review_items],
                "review_question": _review_question(candidate.review_items),
                "heard_text": None,
                "decision": "pending",
                "notes": "",
            }
        )

    (output_dir / "review.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REVIEW.md").write_text(_review_markdown(entries), encoding="utf-8")
    return entries


def _review_markdown(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Hindi Normalization Audio Review",
        "",
        "Listen around the flagged token. Record what is actually spoken, not what the text",
        "was probably intended to say. Update `heard_text`, `decision`, and `notes` in",
        "`review.json` after review.",
        "",
    ]
    for entry in entries:
        issues = ", ".join(f"`{item['kind']}: {item['token']}`" for item in entry["review_items"])
        lines.extend(
            [
                f"## {entry['index']:02d}. {entry['example_id']}",
                "",
                f"- Audio: [{entry['audio_file']}]({entry['audio_file']})",
                f"- Speaker: `{entry['speaker_id']}`",
                f"- Flag: {issues}",
                f"- Source text: {entry['text_source_normalized']}",
                f"- Cardinal preview: {entry['cardinal_preview']}",
                f"- Question: {entry['review_question']}",
                "- Heard text:",
                "- Decision: `pending`",
                "- Notes:",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = build_review_packet(
        args.manifest, args.output_dir, token=load_token(args.token_file), timeout=args.timeout
    )
    print(f"Wrote {len(entries)} review clips to {args.output_dir}")
    print(args.output_dir / "REVIEW.md")


if __name__ == "__main__":
    main()
