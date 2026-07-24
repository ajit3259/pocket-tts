"""Common records for external tokenizer-text sources."""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from experiments.indic.data_manifest import script_mode

TEXT_SOURCE_SCHEMA_VERSION = 1

SLR104_DATASET = "openslr_slr104_hindi_english"
SLR104_LICENSE = "CC-BY-SA-4.0"
SLR104_OFFICIAL_URL = "https://www.openslr.org/104/"
SLR104_MIRROR_REPO = "ujs/hinglish-compressed"
SLR104_MIRROR_REVISION = "5c22260f73a889c457861b1647e96e4254dc1047"
SLR104_METADATA_FILES = {"train": "data/metadata.csv", "test": "data/metadata-test.csv"}

LIBRITTS_R_DATASET = "libritts_r"
LIBRITTS_R_LICENSE = "CC-BY-4.0"
LIBRITTS_R_OFFICIAL_URL = "https://www.openslr.org/141/"
LIBRITTS_R_MIRROR_REPO = "pharaouk/libritts_r"
LIBRITTS_R_MIRROR_REVISION = "9725807a9c85b52a5aba775d9c4be780b37d82bd"
LIBRITTS_R_PARQUET_REVISION = "42c834ebdd6db1f79120dad12347e45fa34d3650"
LIBRITTS_R_SPLIT = "train.clean.100"


def normalize_tokenizer_text(text: str) -> str:
    """Apply representation-only cleanup that does not change spoken content."""

    return unicodedata.normalize("NFC", " ".join(text.split()))


def language_mode_for_text(text: str) -> str:
    mode = script_mode(text)
    if mode == "mixed-devanagari-latin":
        return "hi-en"
    if mode == "devanagari":
        return "hi"
    if mode == "latin":
        return "en"
    return "und"


def _example_id(source: str, split: str, source_utterance_id: str) -> str:
    identity = f"{source}\0{split}\0{source_utterance_id}".encode()
    return f"{source}-{hashlib.sha256(identity).hexdigest()[:16]}"


def iter_slr104_csv(handle: TextIO, *, source_name: str) -> Iterator[tuple[int, str, str]]:
    """Yield validated path/transcript pairs from the headerless mirror metadata."""

    for row_index, row in enumerate(csv.reader(handle)):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError(
                f"{source_name} row {row_index + 1} has {len(row)} columns; expected 2"
            )
        path, text = row
        if not path.strip() or not text.strip():
            raise ValueError(f"{source_name} row {row_index + 1} has an empty field")
        yield row_index, path.strip(), text


def read_slr104_csv(path: Path, *, split: str) -> list[dict[str, Any]]:
    expected_prefix = f"{split}/"
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row_index, audio_path, text in iter_slr104_csv(handle, source_name=str(path)):
            if not audio_path.startswith(expected_prefix):
                raise ValueError(
                    f"{path} row {row_index + 1} path does not start with {expected_prefix!r}"
                )
            records.append(
                adapt_slr104_record(
                    split=split,
                    metadata_file=path.name,
                    row_index=row_index,
                    audio_path=audio_path,
                    text=text,
                )
            )
    return records


def adapt_slr104_record(
    *, split: str, metadata_file: str, row_index: int, audio_path: str, text: str
) -> dict[str, Any]:
    model_input = normalize_tokenizer_text(text)
    source_utterance_id = audio_path.removeprefix(f"{split}/").removesuffix(".wav")
    return {
        "schema_version": TEXT_SOURCE_SCHEMA_VERSION,
        "example_id": _example_id(SLR104_DATASET, split, source_utterance_id),
        "source_dataset": SLR104_DATASET,
        "source_license": SLR104_LICENSE,
        "source_split": split,
        "source_utterance_id": source_utterance_id,
        "speaker_id": None,
        "language_mode": language_mode_for_text(model_input),
        "script_mode": script_mode(model_input),
        "text_raw": text,
        "text_model_input": model_input,
        "source_locator": {
            "official_url": SLR104_OFFICIAL_URL,
            "transport_repo_id": SLR104_MIRROR_REPO,
            "transport_revision": SLR104_MIRROR_REVISION,
            "path": f"data/{metadata_file}",
            "row_index": row_index,
            "audio_path": audio_path,
            "format": "hf-csv-row",
        },
    }


def adapt_libritts_r_record(row: dict[str, Any], *, shard: str, row_index: int) -> dict[str, Any]:
    source_utterance_id = str(row["id"])
    original = str(row["text_original"])
    model_input = normalize_tokenizer_text(str(row["text_normalized"]))
    return {
        "schema_version": TEXT_SOURCE_SCHEMA_VERSION,
        "example_id": _example_id(LIBRITTS_R_DATASET, LIBRITTS_R_SPLIT, source_utterance_id),
        "source_dataset": LIBRITTS_R_DATASET,
        "source_license": LIBRITTS_R_LICENSE,
        "source_split": LIBRITTS_R_SPLIT,
        "source_utterance_id": source_utterance_id,
        "speaker_id": str(row["speaker_id"]),
        "chapter_id": str(row["chapter_id"]),
        "language_mode": "en",
        "script_mode": script_mode(model_input),
        "text_raw": original,
        "text_source_normalized": str(row["text_normalized"]),
        "text_model_input": model_input,
        "source_locator": {
            "official_url": LIBRITTS_R_OFFICIAL_URL,
            "transport_repo_id": LIBRITTS_R_MIRROR_REPO,
            "transport_revision": LIBRITTS_R_MIRROR_REVISION,
            "parquet_revision": LIBRITTS_R_PARQUET_REVISION,
            "path": shard,
            "row_index": row_index,
            "format": "hf-parquet-row",
        },
    }


def nonspace_characters(text: str) -> int:
    return sum(not character.isspace() for character in text)


def audit_text_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    texts = [record["text_model_input"] for record in materialized]
    lengths = [len(text) for text in texts]
    return {
        "records": len(materialized),
        "unique_texts": len(set(texts)),
        "duplicate_records": len(texts) - len(set(texts)),
        "characters": sum(len(text) for text in texts),
        "nonspace_characters": sum(nonspace_characters(text) for text in texts),
        "minimum_characters": min(lengths, default=0),
        "maximum_characters": max(lengths, default=0),
        "normalization_changed_records": sum(
            record["text_raw"] != record["text_model_input"] for record in materialized
        ),
        "source_splits": dict(
            sorted(Counter(record["source_split"] for record in materialized).items())
        ),
        "language_modes": dict(
            sorted(Counter(record["language_mode"] for record in materialized).items())
        ),
        "script_modes": dict(
            sorted(Counter(record["script_mode"] for record in materialized).items())
        ),
    }
