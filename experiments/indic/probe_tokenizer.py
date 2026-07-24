"""Measure how a SentencePiece tokenizer represents Hindi and Hinglish text."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "kyutai/pocket-tts-without-voice-cloning"
DEFAULT_FILENAME = "languages/english/tokenizer.model"
DEFAULT_REVISION = "d29db7978e464fb90cb3359ee0c69a273b9142cc"
DEFAULT_PROBES = Path(__file__).with_name("probe_sentences.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, help="Local SentencePiece model")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--show-pieces", action="store_true")
    return parser.parse_args()


def load_probes(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer is not None:
        return args.tokenizer
    return Path(
        hf_hub_download(repo_id=args.repo_id, filename=args.filename, revision=args.revision)
    )


def main() -> None:
    args = parse_args()
    model_path = tokenizer_path(args)
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    probes = load_probes(args.probes)
    group_fertility: dict[str, list[float]] = defaultdict(list)

    print(f"tokenizer={model_path}")
    print(f"vocab_size={tokenizer.vocab_size()} unk_id={tokenizer.unk_id()}")
    print()
    print("id     group     chars bytes tokens tok/char unk byte")

    for probe in probes:
        text = probe["text"]
        token_ids = tokenizer.encode(text, out_type=int)
        pieces = [tokenizer.id_to_piece(token_id) for token_id in token_ids]
        nonspace_chars = sum(not char.isspace() for char in text)
        fertility = len(token_ids) / max(nonspace_chars, 1)
        unknowns = sum(token_id == tokenizer.unk_id() for token_id in token_ids)
        byte_tokens = sum(tokenizer.is_byte(token_id) for token_id in token_ids)
        group_fertility[probe["group"]].append(fertility)

        print(
            f"{probe['id']:<6} {probe['group']:<9} {nonspace_chars:>5} "
            f"{len(text.encode('utf-8')):>5} {len(token_ids):>6} "
            f"{fertility:>8.2f} {unknowns:>3} {byte_tokens:>4}"
        )
        if args.show_pieces:
            print(f"       {pieces}")

    print()
    print("Mean tokens per non-space character:")
    for group, values in group_fertility.items():
        print(f"  {group:<9} {statistics.mean(values):.3f}")


if __name__ == "__main__":
    main()
