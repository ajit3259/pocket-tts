"""Stream dataset metadata into a common JSONL manifest without reading audio."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, HfFileSystem

from experiments.indic.data_manifest import (
    SOURCE_SPECS,
    ManifestRecord,
    SourceSpec,
    adapt_row,
    audit_records,
    write_jsonl,
)

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e4_data_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_SPECS),
        help="Source to audit; repeat for multiple sources (default: all)",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "test"),
        help="Source split to audit; repeat for multiple splits (default: train and test)",
    )
    parser.add_argument("--max-shards", type=int, help="Limit shards per source/split")
    parser.add_argument("--max-rows", type=int, help="Limit total output rows")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--token-file", type=Path, default=Path("HF_TOKEN"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_token(path: Path) -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    if not path.is_file():
        raise FileNotFoundError(
            f"No Hugging Face token found in HF_TOKEN or {path}. The selected datasets are gated."
        )
    return path.read_text(encoding="utf-8").strip()


def list_shards(api: HfApi, spec: SourceSpec, split: str, max_shards: int | None) -> list[str]:
    prefix = f"Hindi/{split}-"
    shards = sorted(
        path
        for path in api.list_repo_files(spec.repo_id, repo_type="dataset", revision=spec.revision)
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(f"No {split} shards found for {spec.repo_id} at {spec.revision}")
    return shards[:max_shards]


def iter_parquet_rows(
    fs: HfFileSystem, spec: SourceSpec, shard: str, batch_size: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Dataset auditing needs PyArrow. Run `uv sync --extra indic-data` first."
        ) from error

    path = f"datasets/{spec.repo_id}@{spec.revision}/{shard}"
    with fs.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(spec.columns)):
            for row in batch.to_pylist():
                yield row_index, row
                row_index += 1


def collect_records(args: argparse.Namespace, token: str) -> list[ManifestRecord]:
    sources = args.source or sorted(SOURCE_SPECS)
    splits = args.split or ["train", "test"]
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    records: list[ManifestRecord] = []

    for source in sources:
        spec = SOURCE_SPECS[source]
        for split in splits:
            shards = list_shards(api, spec, split, args.max_shards)
            for shard in shards:
                print(f"reading metadata: {source} {shard}", flush=True)
                for row_index, row in iter_parquet_rows(fs, spec, shard, args.batch_size):
                    records.append(
                        adapt_row(source, row, split=split, shard=shard, row_index=row_index)
                    )
                    if args.max_rows is not None and len(records) >= args.max_rows:
                        return records
    return records


def speaker_split_overlap(records: list[ManifestRecord]) -> dict[str, list[str]]:
    speakers_by_split: dict[str, set[str]] = {}
    for record in records:
        speakers_by_split.setdefault(record.source_split, set()).add(record.speaker_id)

    overlap: dict[str, list[str]] = {}
    splits = sorted(speakers_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            shared = sorted(speakers_by_split[left] & speakers_by_split[right])
            overlap[f"{left}:{right}"] = shared
    return overlap


def main() -> None:
    args = parse_args()
    token = load_token(args.token_file)
    records = collect_records(args, token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.jsonl"
    audit_path = args.output_dir / "audit.json"
    write_jsonl(manifest_path, records)
    audit = audit_records(records)
    audit["speaker_overlap"] = speaker_split_overlap(records)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote {manifest_path}")
    print(audit_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
