"""Compare an ID-preserving tokenizer extension with fresh and released models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
from huggingface_hub import hf_hub_download

from experiments.indic.evaluate_tokenizer_candidates import (
    DEFAULT_HINDI_MANIFEST,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROBES,
    DEFAULT_SOURCE_DIR,
    evaluate_texts,
    load_eval_sets,
    vocabulary_summary,
)
from experiments.indic.extend_tokenizer import (
    DEFAULT_DONOR,
    DEFAULT_OUTPUT_DIR,
    load_model_proto,
    validate_preserved_prefix,
)
from experiments.indic.probe_tokenizer import DEFAULT_FILENAME, DEFAULT_REPO, DEFAULT_REVISION
from experiments.indic.train_tokenizer_candidates import sha256_file

DEFAULT_EXTENDED = DEFAULT_OUTPUT_DIR / "tokenizer_extended_8000.model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--fresh", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--extended", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--hindi-manifest", type=Path, default=DEFAULT_HINDI_MANIFEST)
    parser.add_argument(
        "--slr104-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "slr104_manifest.jsonl"
    )
    parser.add_argument(
        "--libritts-r-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "libritts_r_manifest.jsonl"
    )
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def segmentation_comparison(
    baseline: spm.SentencePieceProcessor,
    tokenizer: spm.SentencePieceProcessor,
    texts: list[str],
    *,
    id_boundary: int,
) -> dict[str, Any]:
    identical_ids = 0
    identical_pieces = 0
    records_using_high_ids = 0
    high_id_tokens = 0
    total_tokens = 0
    for text in texts:
        baseline_ids = baseline.encode(text, out_type=int)
        token_ids = tokenizer.encode(text, out_type=int)
        baseline_pieces = [baseline.id_to_piece(piece_id) for piece_id in baseline_ids]
        pieces = [tokenizer.id_to_piece(piece_id) for piece_id in token_ids]
        identical_ids += baseline_ids == token_ids
        identical_pieces += baseline_pieces == pieces
        record_high_id_tokens = sum(piece_id >= id_boundary for piece_id in token_ids)
        records_using_high_ids += record_high_id_tokens > 0
        high_id_tokens += record_high_id_tokens
        total_tokens += len(token_ids)
    return {
        "identical_id_sequences": identical_ids,
        "identical_id_sequences_pct": round(100 * identical_ids / len(texts), 3),
        "identical_piece_sequences": identical_pieces,
        "identical_piece_sequences_pct": round(100 * identical_pieces / len(texts), 3),
        "records_using_ids_at_or_above_boundary": records_using_high_ids,
        "records_using_ids_at_or_above_boundary_pct": round(
            100 * records_using_high_ids / len(texts), 3
        ),
        "tokens_at_or_above_boundary": high_id_tokens,
        "tokens_at_or_above_boundary_pct": round(100 * high_id_tokens / total_tokens, 3),
    }


def _markdown_report(result: dict[str, Any]) -> str:
    id_boundary = result["preserved_vocab_size"]
    lines = [
        "# E10 ID-Preserving Tokenizer Evaluation",
        "",
        "## Compression",
        "",
        "| Tokenizer | Set | Tok/char | P95 tokens | >50 | Byte % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for tokenizer_name, tokenizer_result in result["tokenizers"].items():
        for set_name, metrics in tokenizer_result["evaluation"].items():
            lines.append(
                f"| {tokenizer_name} | {set_name} | "
                f"{metrics['tokens_per_nonspace_character']:.3f} | "
                f"{metrics['tokens_per_record_p95']} | "
                f"{metrics['records_over_token_limit_pct']:.1f}% | "
                f"{metrics['byte_token_pct']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Transfer and segmentation",
            "",
            f"| Tokenizer | Set | Identical IDs | Identical pieces | Uses ID >={id_boundary} |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for tokenizer_name, tokenizer_result in result["tokenizers"].items():
        for set_name, metrics in tokenizer_result["segmentation"].items():
            lines.append(
                f"| {tokenizer_name} | {set_name} | "
                f"{metrics['identical_id_sequences_pct']:.1f}% | "
                f"{metrics['identical_piece_sequences_pct']:.1f}% | "
                f"{metrics['records_using_ids_at_or_above_boundary_pct']:.1f}% |"
            )
    return "\n".join(lines) + "\n"


def evaluate_extension(
    baseline_path: Path,
    fresh_path: Path,
    extended_path: Path,
    evaluation_sets: dict[str, list[str]],
    output_dir: Path,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    baseline_proto = load_model_proto(baseline_path)
    extended_proto = load_model_proto(extended_path)
    validate_preserved_prefix(baseline_proto, extended_proto)
    preserved_vocab_size = len(baseline_proto.pieces)

    baseline = spm.SentencePieceProcessor(model_file=str(baseline_path))
    models = {
        "released_4k": baseline,
        "fresh_8k": spm.SentencePieceProcessor(model_file=str(fresh_path)),
        "extended_8k": spm.SentencePieceProcessor(model_file=str(extended_path)),
    }
    paths = {"released_4k": baseline_path, "fresh_8k": fresh_path, "extended_8k": extended_path}
    tokenizers = {}
    for name, tokenizer in models.items():
        tokenizers[name] = {
            "model_path": str(paths[name]),
            "model_sha256": sha256_file(paths[name]),
            "vocabulary": vocabulary_summary(
                tokenizer, None if name == "released_4k" else baseline
            ),
            "evaluation": {
                set_name: evaluate_texts(tokenizer, texts, max_tokens=max_tokens)
                for set_name, texts in evaluation_sets.items()
            },
            "segmentation": {
                set_name: segmentation_comparison(
                    baseline, tokenizer, texts, id_boundary=preserved_vocab_size
                )
                for set_name, texts in evaluation_sets.items()
            },
        }

    result = {
        "schema_version": 1,
        "max_tokens_per_chunk": max_tokens,
        "preserved_vocab_size": preserved_vocab_size,
        "evaluation_set_records": {name: len(texts) for name, texts in evaluation_sets.items()},
        "tokenizers": tokenizers,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "EVALUATION.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline or Path(
        hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, revision=DEFAULT_REVISION)
    )
    evaluation_sets = load_eval_sets(
        args.hindi_manifest, args.slr104_manifest, args.libritts_r_manifest, args.probes
    )
    result = evaluate_extension(
        baseline_path,
        args.fresh,
        args.extended,
        evaluation_sets,
        args.output_dir,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote extension evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
