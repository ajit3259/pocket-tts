"""Evaluate released and candidate tokenizers on held-out multilingual text."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import sentencepiece as spm
from huggingface_hub import hf_hub_download

from experiments.indic.probe_tokenizer import DEFAULT_FILENAME, DEFAULT_REPO, DEFAULT_REVISION
from experiments.indic.train_tokenizer_candidates import sha256_file

DEFAULT_HINDI_MANIFEST = (
    Path(__file__).with_name("outputs") / "e6_tokenizer_corpus" / "model_input_manifest.jsonl"
)
DEFAULT_SOURCE_DIR = Path(__file__).with_name("outputs") / "e7_tokenizer_sources"
DEFAULT_CANDIDATE_DIR = Path(__file__).with_name("outputs") / "e9_tokenizer_candidates"
DEFAULT_PROBES = Path(__file__).with_name("tokenizer_eval_probes.jsonl")
DEFAULT_MAX_TOKENS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--hindi-manifest", type=Path, default=DEFAULT_HINDI_MANIFEST)
    parser.add_argument(
        "--slr104-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "slr104_manifest.jsonl"
    )
    parser.add_argument(
        "--libritts-r-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "libritts_r_manifest.jsonl"
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_texts(records: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(record["text_model_input"] for record in records))


def load_eval_sets(
    hindi_manifest: Path, slr104_manifest: Path, libritts_r_manifest: Path, probes_path: Path
) -> dict[str, list[str]]:
    hindi = [record for record in _load_jsonl(hindi_manifest) if record["source_split"] == "test"]
    hinglish = [
        record
        for record in _load_jsonl(slr104_manifest)
        if record["source_split"] == "test" and record["script_mode"] == "mixed-devanagari-latin"
    ]
    english = [
        record
        for record in _load_jsonl(libritts_r_manifest)
        if record["source_split"] == "dev.clean"
    ]
    probes = _load_jsonl(probes_path)
    evaluation_sets = {
        "hindi_test": _unique_texts(hindi),
        "hinglish_test": _unique_texts(hinglish),
        "english_dev": _unique_texts(english),
        "romanized_hindi": list(
            dict.fromkeys(probe["text"] for probe in probes if probe["group"] == "romanized_hindi")
        ),
        "numbers": list(
            dict.fromkeys(probe["text"] for probe in probes if probe["group"] == "numbers")
        ),
    }
    empty = [name for name, texts in evaluation_sets.items() if not texts]
    if empty:
        raise ValueError(f"Empty tokenizer evaluation sets: {empty}")
    return evaluation_sets


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return sorted(values)[index]


def evaluate_texts(
    tokenizer: spm.SentencePieceProcessor, texts: list[str], *, max_tokens: int = DEFAULT_MAX_TOKENS
) -> dict[str, Any]:
    if not texts:
        raise ValueError("Cannot evaluate an empty text set")
    token_counts = []
    total_tokens = 0
    total_words = 0
    total_nonspace_characters = 0
    byte_tokens = 0
    unknown_tokens = 0
    records_with_bytes = 0

    for text in texts:
        token_ids = tokenizer.encode(text, out_type=int)
        record_byte_tokens = sum(tokenizer.is_byte(token_id) for token_id in token_ids)
        token_counts.append(len(token_ids))
        total_tokens += len(token_ids)
        total_words += len(text.split())
        total_nonspace_characters += sum(not character.isspace() for character in text)
        byte_tokens += record_byte_tokens
        unknown_tokens += sum(token_id == tokenizer.unk_id() for token_id in token_ids)
        records_with_bytes += record_byte_tokens > 0

    records_over_limit = sum(count > max_tokens for count in token_counts)
    return {
        "records": len(texts),
        "nonspace_characters": total_nonspace_characters,
        "words": total_words,
        "tokens": total_tokens,
        "tokens_per_nonspace_character": round(total_tokens / total_nonspace_characters, 6),
        "tokens_per_word": round(total_tokens / total_words, 6),
        "tokens_per_record_p50": _nearest_rank(token_counts, 0.50),
        "tokens_per_record_p95": _nearest_rank(token_counts, 0.95),
        "tokens_per_record_max": max(token_counts, default=0),
        "records_over_token_limit": records_over_limit,
        "records_over_token_limit_pct": round(100 * records_over_limit / len(texts), 3),
        "byte_tokens": byte_tokens,
        "byte_token_pct": round(100 * byte_tokens / total_tokens, 3),
        "records_with_byte_tokens": records_with_bytes,
        "unknown_tokens": unknown_tokens,
    }


def _learned_pieces(tokenizer: spm.SentencePieceProcessor) -> list[str]:
    return [
        tokenizer.id_to_piece(index)
        for index in range(tokenizer.vocab_size())
        if not tokenizer.is_unknown(index)
        and not tokenizer.is_control(index)
        and not tokenizer.is_unused(index)
        and not tokenizer.is_byte(index)
    ]


def vocabulary_summary(
    tokenizer: spm.SentencePieceProcessor, baseline: spm.SentencePieceProcessor | None = None
) -> dict[str, Any]:
    learned = _learned_pieces(tokenizer)

    def has_devanagari(piece: str) -> bool:
        return any("\u0900" <= char <= "\u097f" for char in piece)

    def has_latin(piece: str) -> bool:
        return any("a" <= char.lower() <= "z" for char in piece)

    summary = {
        "vocab_size": tokenizer.vocab_size(),
        "learned_pieces": len(learned),
        "byte_pieces": sum(tokenizer.is_byte(index) for index in range(tokenizer.vocab_size())),
        "pieces_with_devanagari": sum(has_devanagari(piece) for piece in learned),
        "pieces_with_latin": sum(has_latin(piece) for piece in learned),
        "pieces_with_both_scripts": sum(
            has_devanagari(piece) and has_latin(piece) for piece in learned
        ),
        "pieces_with_digits": sum(any(char.isdigit() for char in piece) for piece in learned),
    }
    if baseline is not None:
        baseline_learned = _learned_pieces(baseline)
        overlap = set(learned) & set(baseline_learned)
        same_id = sum(
            tokenizer.id_to_piece(index) == baseline.id_to_piece(index)
            for index in range(min(tokenizer.vocab_size(), baseline.vocab_size()))
        )
        summary["released_learned_piece_overlap"] = len(overlap)
        summary["released_learned_piece_retention_pct"] = round(
            100 * len(overlap) / len(baseline_learned), 3
        )
        summary["same_id_piece_count_in_shared_range"] = same_id
    return summary


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# E9 Tokenizer Candidate Evaluation",
        "",
        "## Held-out and curated compression",
        "",
        "| Tokenizer | Set | Records | Tok/char | Tok/word | P95 tokens | "
        f">{result['max_tokens_per_chunk']} | Byte % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tokenizer_name, tokenizer_result in result["tokenizers"].items():
        for set_name, metrics in tokenizer_result["evaluation"].items():
            lines.append(
                f"| {tokenizer_name} | {set_name} | {metrics['records']:,} | "
                f"{metrics['tokens_per_nonspace_character']:.3f} | "
                f"{metrics['tokens_per_word']:.2f} | "
                f"{metrics['tokens_per_record_p95']} | "
                f"{metrics['records_over_token_limit_pct']:.1f}% | "
                f"{metrics['byte_token_pct']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Vocabulary allocation",
            "",
            "| Tokenizer | Learned | Devanagari | Latin | Digits | Released overlap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tokenizer_name, tokenizer_result in result["tokenizers"].items():
        vocab = tokenizer_result["vocabulary"]
        overlap = vocab.get("released_learned_piece_overlap", vocab["learned_pieces"])
        lines.append(
            f"| {tokenizer_name} | {vocab['learned_pieces']:,} | "
            f"{vocab['pieces_with_devanagari']:,} | {vocab['pieces_with_latin']:,} | "
            f"{vocab['pieces_with_digits']:,} | {overlap:,} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_candidates(
    baseline_path: Path,
    candidate_dir: Path,
    evaluation_sets: dict[str, list[str]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    baseline = spm.SentencePieceProcessor(model_file=str(baseline_path))
    model_paths = [baseline_path, *sorted(candidate_dir.glob("tokenizer_*.model"))]
    if len(model_paths) == 1:
        raise ValueError(f"No tokenizer candidates found in {candidate_dir}")

    tokenizers = {}
    for model_path in model_paths:
        tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
        name = (
            "released_4k"
            if model_path == baseline_path
            else f"candidate_{tokenizer.vocab_size() // 1000}k"
        )
        tokenizers[name] = {
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "vocabulary": vocabulary_summary(
                tokenizer, None if model_path == baseline_path else baseline
            ),
            "evaluation": {
                set_name: evaluate_texts(tokenizer, texts, max_tokens=max_tokens)
                for set_name, texts in evaluation_sets.items()
            },
        }

    result = {
        "schema_version": 1,
        "max_tokens_per_chunk": max_tokens,
        "evaluation_set_records": {name: len(texts) for name, texts in evaluation_sets.items()},
        "tokenizers": tokenizers,
    }
    (candidate_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (candidate_dir / "EVALUATION.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline or Path(
        hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, revision=DEFAULT_REVISION)
    )
    evaluation_sets = load_eval_sets(
        args.hindi_manifest, args.slr104_manifest, args.libritts_r_manifest, args.probes
    )
    result = evaluate_candidates(
        baseline_path, args.candidate_dir, evaluation_sets, max_tokens=args.max_tokens
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote evaluation to {args.candidate_dir}")


if __name__ == "__main__":
    main()
