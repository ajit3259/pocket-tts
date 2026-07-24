"""Audit normalization changes and ambiguities in an existing JSONL manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.indic.text_normalization import (
    NORMALIZER_VERSION,
    NumberMode,
    normalize_hindi_text,
)

DEFAULT_MANIFEST = Path(__file__).with_name("outputs") / "e4_data_audit" / "manifest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--number-mode", choices=("preserve", "cardinal"), default="preserve")
    parser.add_argument("--examples-per-kind", type=int, default=5)
    return parser.parse_args()


def audit_manifest(
    path: Path, *, number_mode: NumberMode, examples_per_kind: int
) -> dict[str, Any]:
    rows = 0
    changed_rows = 0
    review_rows = 0
    change_kinds: Counter[str] = Counter()
    review_kinds: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_text = row["text_normalized"]
            result = normalize_hindi_text(source_text, number_mode=number_mode)
            rows += 1
            changed_rows += bool(result.changes)
            review_rows += result.needs_review

            for change in result.changes:
                change_kinds[change.kind] += 1
            for item in result.review_items:
                review_kinds[item.kind] += 1
                if len(examples[item.kind]) < examples_per_kind:
                    examples[item.kind].append(
                        {"example_id": row["example_id"], "token": item.token, "text": source_text}
                    )

    return {
        "manifest": str(path),
        "normalizer_version": NORMALIZER_VERSION,
        "number_mode": number_mode,
        "rows": rows,
        "changed_rows": changed_rows,
        "review_rows": review_rows,
        "change_kinds": dict(sorted(change_kinds.items())),
        "review_kinds": dict(sorted(review_kinds.items())),
        "review_examples": dict(sorted(examples.items())),
    }


def main() -> None:
    args = parse_args()
    audit = audit_manifest(
        args.manifest, number_mode=args.number_mode, examples_per_kind=args.examples_per_kind
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
