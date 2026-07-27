"""Train reproducible SentencePiece candidates on the E8 multilingual corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm

DEFAULT_CORPUS = (
    Path(__file__).with_name("outputs") / "e8_tokenizer_mixture" / "tokenizer_train.txt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e9_tokenizer_candidates"
DEFAULT_VOCAB_SIZES = (4000, 6000, 8000)

TRAINER_SPEC: dict[str, Any] = {
    "model_type": "unigram",
    "character_coverage": 1.0,
    "byte_fallback": True,
    "normalization_rule_name": "identity",
    "remove_extra_whitespaces": False,
    "unk_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "pad_id": 3,
    "unk_piece": "<unk>",
    "bos_piece": "<s>",
    "eos_piece": "</s>",
    "pad_piece": "<pad>",
    "input_sentence_size": 0,
    "shuffle_input_sentence": False,
    "num_threads": 1,
    "hard_vocab_limit": True,
    "max_sentence_length": 2048,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, action="append", dest="vocab_sizes")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_corpus(path: Path) -> dict[str, int]:
    lines = 0
    characters = 0
    nonspace_characters = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.rstrip("\n")
            if not text:
                raise ValueError(f"Empty tokenizer sentence on {path} line {line_number}")
            lines += 1
            characters += len(text)
            nonspace_characters += sum(not character.isspace() for character in text)
    if not lines:
        raise ValueError(f"Tokenizer corpus is empty: {path}")
    return {"lines": lines, "characters": characters, "nonspace_characters": nonspace_characters}


def train_candidate(corpus: Path, output_dir: Path, vocab_size: int) -> dict[str, Any]:
    if vocab_size <= 260:
        raise ValueError("vocab_size must leave room beyond 4 meta and 256 byte pieces")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"tokenizer_{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        minloglevel=1,
        **TRAINER_SPEC,
    )

    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.vocab_size() != vocab_size:
        raise RuntimeError(f"Candidate has {tokenizer.vocab_size()} pieces; expected {vocab_size}")
    expected_ids = {
        "unk_id": tokenizer.unk_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
        "pad_id": tokenizer.pad_id(),
    }
    if expected_ids != {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": 3}:
        raise RuntimeError(f"Unexpected special token IDs: {expected_ids}")
    byte_pieces = sum(tokenizer.is_byte(index) for index in range(vocab_size))
    if byte_pieces != 256:
        raise RuntimeError(f"Candidate has {byte_pieces} byte pieces; expected 256")

    return {
        "vocab_size": vocab_size,
        "model_path": model_path.name,
        "vocab_path": vocab_path.name,
        "model_sha256": sha256_file(model_path),
        "vocab_sha256": sha256_file(vocab_path),
        "special_token_ids": expected_ids,
        "byte_pieces": byte_pieces,
    }


def train_candidates(
    corpus: Path, output_dir: Path, vocab_sizes: list[int] | tuple[int, ...]
) -> dict[str, Any]:
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocab sizes must be unique")
    corpus_stats = inspect_corpus(corpus)
    candidates = [
        train_candidate(corpus, output_dir, vocab_size) for vocab_size in sorted(vocab_sizes)
    ]
    metadata = {
        "schema_version": 1,
        "sentencepiece_version": spm.__version__,
        "corpus_path": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "corpus": corpus_stats,
        "trainer_spec": TRAINER_SPEC,
        "candidates": candidates,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = train_candidates(
        args.corpus, args.output_dir, args.vocab_sizes or DEFAULT_VOCAB_SIZES
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote tokenizer candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
