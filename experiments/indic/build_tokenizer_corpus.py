"""Resolve Hindi training text and build reproducible tokenizer corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.indic.data_manifest import script_mode
from experiments.indic.text_normalization import NORMALIZER_VERSION, normalize_hindi_text

DEFAULT_MANIFEST = Path(__file__).with_name("outputs") / "e4_data_audit" / "manifest.jsonl"
DEFAULT_OVERRIDES = Path(__file__).with_name("normalization_overrides.jsonl")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e6_tokenizer_corpus"


@dataclass(frozen=True)
class ResolvedRecord:
    record: dict[str, Any]
    override_applied: bool
    normalization_change_kinds: tuple[str, ...]
    resolved_review_kinds: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            override = json.loads(line)
            if override.get("schema_version") != 1:
                raise ValueError(f"Unsupported override schema on line {line_number}")
            example_id = override["example_id"]
            if example_id in overrides:
                raise ValueError(f"Duplicate override for {example_id}")
            overrides[example_id] = override
    return overrides


def resolve_record(record: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> ResolvedRecord:
    example_id = record["example_id"]
    source_text = record["text_normalized"]
    source_result = normalize_hindi_text(source_text)
    override = overrides.get(example_id)

    if override is None:
        if source_result.needs_review:
            kinds = ", ".join(item.kind for item in source_result.review_items)
            raise ValueError(f"Unresolved normalization review for {example_id}: {kinds}")
        model_input = source_result.text
        resolved_review_kinds: tuple[str, ...] = ()
    else:
        if override["source_text"] != source_text:
            raise ValueError(f"Override source text does not match manifest for {example_id}")
        override_result = normalize_hindi_text(override["text_model_input"])
        if override_result.text != override["text_model_input"] or override_result.changes:
            raise ValueError(f"Override model input is not canonical for {example_id}")
        if override_result.needs_review:
            kinds = ", ".join(item.kind for item in override_result.review_items)
            raise ValueError(f"Override remains unresolved for {example_id}: {kinds}")
        model_input = override_result.text
        resolved_review_kinds = tuple(item.kind for item in source_result.review_items)

    output = dict(record)
    output["text_schema_version"] = 1
    output["text_source_normalized"] = output.pop("text_normalized")
    output["text_model_input"] = model_input
    output["normalizer_version"] = NORMALIZER_VERSION
    output["normalization_changes"] = [change.to_dict() for change in source_result.changes]
    output["normalization_override"] = (
        {
            "decision": override["decision"],
            "issue_kind": override["issue_kind"],
            "review_method": override["review_method"],
            "reviewed_on": override["reviewed_on"],
        }
        if override is not None
        else None
    )
    output["script_mode"] = script_mode(model_input)

    return ResolvedRecord(
        record=output,
        override_applied=override is not None,
        normalization_change_kinds=tuple(change.kind for change in source_result.changes),
        resolved_review_kinds=resolved_review_kinds,
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text_lines(path: Path, texts: list[str]) -> None:
    path.write_text("".join(f"{text}\n" for text in texts), encoding="utf-8")


def build_corpora(manifest_path: Path, overrides_path: Path, output_dir: Path) -> dict[str, Any]:
    overrides = load_overrides(overrides_path)
    resolved: list[ResolvedRecord] = []
    manifest_ids: set[str] = set()

    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = record["example_id"]
            if example_id in manifest_ids:
                raise ValueError(f"Duplicate manifest example ID: {example_id}")
            manifest_ids.add(example_id)
            resolved.append(resolve_record(record, overrides))

    unused_overrides = sorted(set(overrides) - manifest_ids)
    if unused_overrides:
        raise ValueError(f"Overrides do not exist in manifest: {unused_overrides}")

    train_records = [item.record for item in resolved if item.record["source_split"] == "train"]
    train_texts = [record["text_model_input"] for record in train_records]
    unique_train_texts = list(dict.fromkeys(train_texts))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "model_input_manifest.jsonl", [item.record for item in resolved])
    _write_text_lines(output_dir / "hindi_train_all.txt", train_texts)
    _write_text_lines(output_dir / "hindi_train_unique.txt", unique_train_texts)

    stats = {
        "text_schema_version": 1,
        "normalizer_version": NORMALIZER_VERSION,
        "records": len(resolved),
        "train_records": len(train_records),
        "held_out_records": len(resolved) - len(train_records),
        "train_unique_texts": len(unique_train_texts),
        "train_duplicate_records": len(train_texts) - len(unique_train_texts),
        "overrides_available": len(overrides),
        "overrides_applied": sum(item.override_applied for item in resolved),
        "overrides_by_issue": dict(
            sorted(
                Counter(
                    item.record["normalization_override"]["issue_kind"]
                    for item in resolved
                    if item.record["normalization_override"] is not None
                ).items()
            )
        ),
        "normalization_changes": dict(
            sorted(
                Counter(
                    kind for item in resolved for kind in item.normalization_change_kinds
                ).items()
            )
        ),
        "resolved_review_items": dict(
            sorted(
                Counter(kind for item in resolved for kind in item.resolved_review_kinds).items()
            )
        ),
        "train_records_by_source": dict(
            sorted(Counter(record["source_dataset"] for record in train_records).items())
        ),
        "train_script_modes": dict(
            sorted(Counter(record["script_mode"] for record in train_records).items())
        ),
        "train_characters": sum(len(text) for text in train_texts),
        "train_unique_characters": sum(len(text) for text in unique_train_texts),
    }
    (output_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    stats = build_corpora(args.manifest, args.overrides, args.output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote tokenizer corpora to {args.output_dir}")


if __name__ == "__main__":
    main()
