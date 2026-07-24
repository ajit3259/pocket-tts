"""Common metadata contract for Indic fine-tuning datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceLocator:
    """A row containing embedded audio in a Hugging Face Parquet shard."""

    repo_id: str
    revision: str
    shard: str
    row_index: int
    audio_column: str = "audio"
    format: str = "hf-parquet-row"


@dataclass(frozen=True)
class ManifestRecord:
    """Dataset-independent information needed before audio materialization."""

    schema_version: int
    example_id: str
    source_dataset: str
    source_license: str
    source_split: str
    source_locator: SourceLocator
    speaker_id: str
    language_mode: str
    script_mode: str
    text_raw: str
    text_normalized: str
    duration_seconds: float
    gender: str | None
    style: str | None
    source_utterance_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    revision: str
    license: str
    columns: tuple[str, ...]


RASA = SourceSpec(
    name="rasa",
    repo_id="ai4bharat/Rasa",
    revision="632f55c7ac590219d41cd7adffce5b440e4604f5",
    license="CC-BY-4.0",
    columns=("filename", "text", "gender", "style", "duration"),
)

INDICVOICES_R = SourceSpec(
    name="indicvoices_r",
    repo_id="ai4bharat/indicvoices_r",
    revision="5f4495c91d500742a58d1be2ab07d77f73c0acf8",
    license="CC-BY-4.0",
    columns=(
        "text",
        "verbatim",
        "normalized",
        "speaker_id",
        "scenario",
        "task_name",
        "gender",
        "duration",
    ),
)

SOURCE_SPECS = {spec.name: spec for spec in (RASA, INDICVOICES_R)}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    result = _text(value)
    return result or None


def _duration(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def script_mode(text: str) -> str:
    """Classify writing systems without guessing the spoken language."""

    has_devanagari = any("\u0900" <= char <= "\u097f" for char in text)
    has_latin = any(("a" <= char.lower() <= "z") for char in text)
    if has_devanagari and has_latin:
        return "mixed-devanagari-latin"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    return "other"


def _example_id(source: str, split: str, shard: str, row_index: int) -> str:
    locator = f"{source}\0{split}\0{shard}\0{row_index}".encode()
    return f"{source}-{hashlib.sha256(locator).hexdigest()[:16]}"


def _record(
    *,
    spec: SourceSpec,
    split: str,
    shard: str,
    row_index: int,
    speaker_id: str,
    text_raw: str,
    text_normalized: str,
    duration_seconds: float,
    gender: str | None,
    style: str | None,
    source_utterance_id: str | None,
) -> ManifestRecord:
    return ManifestRecord(
        schema_version=SCHEMA_VERSION,
        example_id=_example_id(spec.name, split, shard, row_index),
        source_dataset=spec.name,
        source_license=spec.license,
        source_split=split,
        source_locator=SourceLocator(
            repo_id=spec.repo_id, revision=spec.revision, shard=shard, row_index=row_index
        ),
        speaker_id=speaker_id,
        language_mode="hi",
        script_mode=script_mode(text_normalized or text_raw),
        text_raw=text_raw,
        text_normalized=text_normalized,
        duration_seconds=duration_seconds,
        gender=gender,
        style=style,
        source_utterance_id=source_utterance_id,
    )


def adapt_rasa(row: Mapping[str, Any], *, split: str, shard: str, row_index: int) -> ManifestRecord:
    """Convert one Rasa Hindi row into the common contract."""

    text = _text(row.get("text"))
    gender = _optional_text(row.get("gender"))
    speaker_suffix = (gender or "unknown").lower().replace(" ", "-")
    return _record(
        spec=RASA,
        split=split,
        shard=shard,
        row_index=row_index,
        speaker_id=f"rasa:hindi:{speaker_suffix}",
        text_raw=text,
        text_normalized=text,
        duration_seconds=_duration(row.get("duration")),
        gender=gender,
        style=_optional_text(row.get("style")),
        source_utterance_id=_optional_text(row.get("filename")),
    )


def adapt_indicvoices_r(
    row: Mapping[str, Any], *, split: str, shard: str, row_index: int
) -> ManifestRecord:
    """Convert one IndicVoices-R Hindi row into the common contract."""

    fallback_text = _text(row.get("text"))
    text_raw = _text(row.get("verbatim")) or fallback_text
    text_normalized = _text(row.get("normalized")) or fallback_text or text_raw
    speaker = _text(row.get("speaker_id")) or "unknown"
    style = _optional_text(row.get("scenario")) or _optional_text(row.get("task_name"))
    return _record(
        spec=INDICVOICES_R,
        split=split,
        shard=shard,
        row_index=row_index,
        speaker_id=f"indicvoices_r:{speaker}",
        text_raw=text_raw,
        text_normalized=text_normalized,
        duration_seconds=_duration(row.get("duration")),
        gender=_optional_text(row.get("gender")),
        style=style,
        source_utterance_id=None,
    )


def adapt_row(
    source: str, row: Mapping[str, Any], *, split: str, shard: str, row_index: int
) -> ManifestRecord:
    if source == RASA.name:
        return adapt_rasa(row, split=split, shard=shard, row_index=row_index)
    if source == INDICVOICES_R.name:
        return adapt_indicvoices_r(row, split=split, shard=shard, row_index=row_index)
    raise ValueError(f"Unsupported source: {source}")


def audit_records(records: Iterable[ManifestRecord]) -> dict[str, Any]:
    """Compute cheap metadata checks before any audio is downloaded."""

    materialized = list(records)
    durations = sorted(
        record.duration_seconds
        for record in materialized
        if math.isfinite(record.duration_seconds) and record.duration_seconds >= 0
    )
    speaker_counts = Counter(record.speaker_id for record in materialized)
    by_source: dict[str, Any] = {}
    for source in sorted({record.source_dataset for record in materialized}):
        source_records = [record for record in materialized if record.source_dataset == source]
        source_durations = [
            record.duration_seconds
            for record in source_records
            if math.isfinite(record.duration_seconds) and record.duration_seconds >= 0
        ]
        by_source[source] = {
            "records": len(source_records),
            "duration_hours": round(sum(source_durations) / 3600, 3),
            "speakers": len({record.speaker_id for record in source_records}),
            "source_splits": dict(
                sorted(Counter(record.source_split for record in source_records).items())
            ),
        }

    def percentile(fraction: float) -> float | None:
        if not durations:
            return None
        index = round(fraction * (len(durations) - 1))
        return round(durations[index], 3)

    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(materialized),
        "duration_hours": round(sum(durations) / 3600, 3),
        "duration_seconds": {
            "min": round(durations[0], 3) if durations else None,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": round(durations[-1], 3) if durations else None,
        },
        "speakers": len(speaker_counts),
        "speakers_with_multiple_utterances": sum(count >= 2 for count in speaker_counts.values()),
        "by_source": by_source,
        "source_datasets": dict(sorted(Counter(r.source_dataset for r in materialized).items())),
        "source_splits": dict(sorted(Counter(r.source_split for r in materialized).items())),
        "script_modes": dict(sorted(Counter(r.script_mode for r in materialized).items())),
        "genders": dict(sorted(Counter(r.gender or "missing" for r in materialized).items())),
        "styles": dict(sorted(Counter(r.style or "missing" for r in materialized).items())),
        "missing": {
            "text_raw": sum(not record.text_raw for record in materialized),
            "text_normalized": sum(not record.text_normalized for record in materialized),
            "duration": len(materialized) - len(durations),
            "speaker_id": sum(record.speaker_id.endswith(":unknown") for record in materialized),
        },
    }


def write_jsonl(path: Path, records: Iterable[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
