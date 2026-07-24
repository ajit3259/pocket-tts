"""Acquire text-only SLR104 and LibriTTS-R manifests without downloading audio."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, HfFileSystem

from experiments.indic.tokenizer_sources import (
    LIBRITTS_R_MIRROR_REPO,
    LIBRITTS_R_PARQUET_REVISION,
    LIBRITTS_R_SPLIT,
    SLR104_METADATA_FILES,
    SLR104_MIRROR_REPO,
    SLR104_MIRROR_REVISION,
    adapt_libritts_r_record,
    audit_text_records,
    read_slr104_csv,
)

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e7_tokenizer_sources"
LIBRITTS_COLUMNS = ("text_normalized", "text_original", "speaker_id", "chapter_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-libritts-shards", type=int)
    return parser.parse_args()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def download_hf_file(
    fs: HfFileSystem, *, repo_id: str, revision: str, remote_path: str, local_path: Path
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_suffix(local_path.suffix + ".part")
    hf_path = f"datasets/{repo_id}@{revision}/{remote_path}"
    with fs.open(hf_path, "rb") as source, temporary_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    temporary_path.replace(local_path)


def list_libritts_shards(api: HfApi, max_shards: int | None = None) -> list[str]:
    prefix = f"clean/{LIBRITTS_R_SPLIT}/"
    shards = sorted(
        path
        for path in api.list_repo_files(
            LIBRITTS_R_MIRROR_REPO, repo_type="dataset", revision=LIBRITTS_R_PARQUET_REVISION
        )
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError("No pinned LibriTTS-R train.clean.100 Parquet shards found")
    return shards[:max_shards]


def read_libritts_shard(shard: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Tokenizer source acquisition needs PyArrow. Run `uv sync --extra indic-data` first."
        ) from error

    path = f"datasets/{LIBRITTS_R_MIRROR_REPO}@{LIBRITTS_R_PARQUET_REVISION}/{shard}"
    fs = HfFileSystem()
    with fs.open(path, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=list(LIBRITTS_COLUMNS))
    return [
        adapt_libritts_r_record(row, shard=shard, row_index=row_index)
        for row_index, row in enumerate(table.to_pylist())
    ]


def collect_libritts_records(shards: list[str], workers: int) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")

    records_by_shard: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(read_libritts_shard, shard): shard for shard in shards}
        for future in as_completed(futures):
            shard = futures[future]
            records_by_shard[shard] = future.result()
            print(f"read metadata: libritts_r {shard}", flush=True)
    return [record for shard in shards for record in records_by_shard[shard]]


def acquire_sources(
    output_dir: Path, *, workers: int = 4, max_libritts_shards: int | None = None
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    fs = HfFileSystem()

    slr104_records = []
    for split, remote_path in SLR104_METADATA_FILES.items():
        local_path = raw_dir / Path(remote_path).name
        download_hf_file(
            fs,
            repo_id=SLR104_MIRROR_REPO,
            revision=SLR104_MIRROR_REVISION,
            remote_path=remote_path,
            local_path=local_path,
        )
        slr104_records.extend(read_slr104_csv(local_path, split=split))
        print(f"read metadata: slr104 {split}", flush=True)

    api = HfApi()
    shards = list_libritts_shards(api, max_shards=max_libritts_shards)
    libritts_records = collect_libritts_records(shards, workers)

    _write_jsonl(output_dir / "slr104_manifest.jsonl", slr104_records)
    _write_jsonl(output_dir / "libritts_r_manifest.jsonl", libritts_records)
    stats = {
        "schema_version": 1,
        "slr104": audit_text_records(slr104_records),
        "libritts_r": audit_text_records(libritts_records),
        "libritts_r_shards": shards,
    }
    (output_dir / "source_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    stats = acquire_sources(
        args.output_dir, workers=args.workers, max_libritts_shards=args.max_libritts_shards
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote tokenizer source manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
