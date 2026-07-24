"""Build a deterministic Hindi, Hinglish, and English tokenizer corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.indic.tokenizer_sources import nonspace_characters

DEFAULT_HINDI_MANIFEST = (
    Path(__file__).with_name("outputs") / "e6_tokenizer_corpus" / "model_input_manifest.jsonl"
)
DEFAULT_SOURCE_DIR = Path(__file__).with_name("outputs") / "e7_tokenizer_sources"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e8_tokenizer_mixture"
DEFAULT_EXCLUSIONS = Path(__file__).with_name("tokenizer_exclusions.jsonl")
DEFAULT_WEIGHTS = {"hi": 0.60, "hi-en": 0.25, "en": 0.15}
DEFAULT_SEED = 3259
ALLOWED_FORMAT_CHARACTERS = {"\u200c", "\u200d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hindi-manifest", type=Path, default=DEFAULT_HINDI_MANIFEST)
    parser.add_argument(
        "--slr104-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "slr104_manifest.jsonl"
    )
    parser.add_argument(
        "--libritts-r-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "libritts_r_manifest.jsonl"
    )
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "example_id" not in record or "text_model_input" not in record:
                raise ValueError(f"{path} line {line_number} is not a text manifest record")
            records.append(record)
    return records


def load_exclusions(path: Path) -> dict[str, dict[str, Any]]:
    exclusions = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            exclusion = json.loads(line)
            if exclusion.get("schema_version") != 1:
                raise ValueError(f"Unsupported exclusion schema on {path} line {line_number}")
            example_id = exclusion["example_id"]
            if example_id in exclusions:
                raise ValueError(f"Duplicate tokenizer exclusion for {example_id}")
            exclusions[example_id] = exclusion
    return exclusions


def _unsafe_unicode(text: str) -> list[str]:
    return sorted(
        {
            f"U+{ord(character):04X} {unicodedata.name(character, 'UNNAMED')}"
            for character in text
            if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co", "Cn"}
            and character not in ALLOWED_FORMAT_CHARACTERS
        }
    )


def _validate_text_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        unsafe = _unsafe_unicode(record["text_model_input"])
        if unsafe:
            raise ValueError(f"Unsafe Unicode in {record['example_id']}: {', '.join(unsafe)}")


def _deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for record in records:
        text = record["text_model_input"]
        counts[text] += 1
        unique.setdefault(text, record)

    output = []
    for text, record in unique.items():
        item = dict(record)
        item["source_exact_text_occurrences"] = counts[text]
        output.append(item)
    return output


def _rank(record: dict[str, Any], *, seed: int, purpose: str) -> str:
    value = (
        f"{seed}\0{purpose}\0{record['source_dataset']}\0"
        f"{record['example_id']}\0{record['text_model_input']}"
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _sample_to_character_budget(
    records: list[dict[str, Any]], *, target: int, seed: int, stream: str
) -> list[dict[str, Any]]:
    ranked = sorted(
        records, key=lambda record: _rank(record, seed=seed, purpose=f"sample:{stream}")
    )
    available = sum(nonspace_characters(record["text_model_input"]) for record in ranked)
    if available < target:
        raise ValueError(f"{stream} has {available:,} nonspace characters, below target {target:,}")

    selected = []
    selected_characters = 0
    for record in ranked:
        selected.append(record)
        selected_characters += nonspace_characters(record["text_model_input"])
        if selected_characters >= target:
            break
    return selected


def _prepare_mixture_record(record: dict[str, Any], *, stream: str, seed: int) -> dict[str, Any]:
    output = dict(record)
    output["mixture_schema_version"] = 1
    output["mixture_stream"] = stream
    output["mixture_nonspace_characters"] = nonspace_characters(record["text_model_input"])
    output["mixture_rank"] = _rank(record, seed=seed, purpose=f"sample:{stream}")
    identity = f"{stream}\0{record['example_id']}".encode()
    output["mixture_id"] = f"tokenizer-{hashlib.sha256(identity).hexdigest()[:16]}"
    return output


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_mixture(
    hindi_manifest: Path,
    slr104_manifest: Path,
    libritts_r_manifest: Path,
    output_dir: Path,
    *,
    weights: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    exclusions_path: Path | None = None,
) -> dict[str, Any]:
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights) != {"hi", "hi-en", "en"}:
        raise ValueError("weights must contain exactly hi, hi-en, and en")
    if any(weight <= 0 for weight in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0
    ):
        raise ValueError("weights must be positive and sum to 1")

    source_records = {
        "hi": load_jsonl(hindi_manifest),
        "hi-en": load_jsonl(slr104_manifest),
        "en": load_jsonl(libritts_r_manifest),
    }
    exclusions = load_exclusions(exclusions_path) if exclusions_path is not None else {}
    available_ids = {
        record["example_id"] for records in source_records.values() for record in records
    }
    unused_exclusions = sorted(set(exclusions) - available_ids)
    if unused_exclusions:
        raise ValueError(f"Tokenizer exclusions do not exist in manifests: {unused_exclusions}")
    records_by_id = {
        record["example_id"]: record for records in source_records.values() for record in records
    }
    for example_id, exclusion in exclusions.items():
        text_digest = hashlib.sha256(
            records_by_id[example_id]["text_model_input"].encode()
        ).hexdigest()
        if text_digest != exclusion["text_model_input_sha256"]:
            raise ValueError(f"Tokenizer exclusion text does not match manifest for {example_id}")
    for records in source_records.values():
        records[:] = [record for record in records if record["example_id"] not in exclusions]
        _validate_text_records(records)

    hindi_pool = _deduplicate(
        [record for record in source_records["hi"] if record["source_split"] == "train"]
    )
    hinglish_pool = _deduplicate(
        [
            record
            for record in source_records["hi-en"]
            if record["source_split"] == "train"
            and record["script_mode"] == "mixed-devanagari-latin"
        ]
    )
    english_pool = _deduplicate(
        [
            record
            for record in source_records["en"]
            if record["source_split"].startswith("train") and record["language_mode"] == "en"
        ]
    )
    if not hindi_pool:
        raise ValueError("Hindi train pool is empty")

    pools = {"hi": hindi_pool, "hi-en": hinglish_pool, "en": english_pool}
    pool_characters = {
        stream: sum(nonspace_characters(record["text_model_input"]) for record in stream_records)
        for stream, stream_records in pools.items()
    }
    hindi_characters = sum(nonspace_characters(record["text_model_input"]) for record in hindi_pool)
    character_targets = {
        "hi": hindi_characters,
        "hi-en": math.ceil(hindi_characters * weights["hi-en"] / weights["hi"]),
        "en": math.ceil(hindi_characters * weights["en"] / weights["hi"]),
    }
    selected = {
        "hi": hindi_pool,
        "hi-en": _sample_to_character_budget(
            hinglish_pool, target=character_targets["hi-en"], seed=seed, stream="hi-en"
        ),
        "en": _sample_to_character_budget(
            english_pool, target=character_targets["en"], seed=seed, stream="en"
        ),
    }

    mixture = [
        _prepare_mixture_record(record, stream=stream, seed=seed)
        for stream in ("hi", "hi-en", "en")
        for record in selected[stream]
    ]
    mixture.sort(key=lambda record: _rank(record, seed=seed, purpose="output-order"))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "mixture_manifest.jsonl", mixture)
    (output_dir / "tokenizer_train.txt").write_text(
        "".join(f"{record['text_model_input']}\n" for record in mixture), encoding="utf-8"
    )

    actual_characters = {
        stream: sum(nonspace_characters(record["text_model_input"]) for record in selected[stream])
        for stream in selected
    }
    total_characters = sum(actual_characters.values())
    stats = {
        "mixture_schema_version": 1,
        "seed": seed,
        "exclusions_applied": len(exclusions),
        "exclusions_by_issue": dict(
            sorted(Counter(item["issue_kind"] for item in exclusions.values()).items())
        ),
        "weight_basis": "non-whitespace Unicode code points",
        "requested_weights": weights,
        "character_targets": character_targets,
        "pool_unique_records": {
            "hi": len(hindi_pool),
            "hi-en": len(hinglish_pool),
            "en": len(english_pool),
        },
        "pool_nonspace_characters": pool_characters,
        "selected_records": {
            stream: len(stream_records) for stream, stream_records in selected.items()
        },
        "selected_nonspace_characters": actual_characters,
        "actual_character_shares": {
            stream: round(characters / total_characters, 6)
            for stream, characters in actual_characters.items()
        },
        "total_records": len(mixture),
        "total_nonspace_characters": total_characters,
        "source_datasets": dict(
            sorted(Counter(record["source_dataset"] for record in mixture).items())
        ),
        "script_modes": dict(sorted(Counter(record["script_mode"] for record in mixture).items())),
        "stream_script_modes": {
            stream: dict(
                sorted(Counter(record["script_mode"] for record in stream_records).items())
            )
            for stream, stream_records in selected.items()
        },
    }
    (output_dir / "mixture_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    stats = build_mixture(
        args.hindi_manifest,
        args.slr104_manifest,
        args.libritts_r_manifest,
        args.output_dir,
        seed=args.seed,
        exclusions_path=args.exclusions,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote tokenizer mixture to {args.output_dir}")


if __name__ == "__main__":
    main()
