"""Extend the released tokenizer while preserving every existing piece ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
from huggingface_hub import hf_hub_download

from experiments.indic.probe_tokenizer import DEFAULT_FILENAME, DEFAULT_REPO, DEFAULT_REVISION
from experiments.indic.train_tokenizer_candidates import sha256_file

DEFAULT_DONOR = (
    Path(__file__).with_name("outputs") / "e9_tokenizer_candidates" / "tokenizer_8000.model"
)
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e10_extended_tokenizer"
DEFAULT_TARGET_VOCAB_SIZE = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-vocab-size", type=int, default=DEFAULT_TARGET_VOCAB_SIZE)
    return parser.parse_args()


def _model_proto_type() -> Any:
    try:
        from sentencepiece import sentencepiece_model_pb2
    except ModuleNotFoundError as error:
        if error.name == "google":
            raise RuntimeError(
                "Tokenizer extension needs protobuf. Run `uv sync --extra indic-data` first."
            ) from error
        raise
    return sentencepiece_model_pb2


def load_model_proto(path: Path) -> Any:
    protobuf = _model_proto_type()
    model = protobuf.ModelProto()
    model.ParseFromString(path.read_bytes())
    return model


def has_devanagari(piece: str) -> bool:
    return any("\u0900" <= character <= "\u097f" for character in piece)


def _piece_type_name(piece_type: int) -> str:
    protobuf = _model_proto_type()
    return protobuf.ModelProto.SentencePiece.Type.Name(piece_type)


def _piece_record(piece: Any, donor_id: int) -> dict[str, Any]:
    return {
        "donor_id": donor_id,
        "piece": piece.piece,
        "score": piece.score,
        "type": _piece_type_name(piece.type),
    }


def _effective_message_values(message: Any) -> tuple[Any, ...]:
    return tuple(getattr(message, field.name) for field in message.DESCRIPTOR.fields)


def validate_preserved_prefix(baseline: Any, extended: Any) -> None:
    if len(extended.pieces) < len(baseline.pieces):
        raise ValueError("Extended model is smaller than its baseline")
    for piece_id, baseline_piece in enumerate(baseline.pieces):
        if baseline_piece.SerializeToString() != extended.pieces[piece_id].SerializeToString():
            raise ValueError(f"Baseline piece {piece_id} was not preserved exactly")
    if baseline.normalizer_spec.SerializeToString() != extended.normalizer_spec.SerializeToString():
        raise ValueError("Baseline normalizer was not preserved exactly")
    if (
        baseline.denormalizer_spec.SerializeToString()
        != extended.denormalizer_spec.SerializeToString()
    ):
        raise ValueError("Baseline denormalizer was not preserved exactly")


def extend_tokenizer(
    baseline_path: Path,
    donor_path: Path,
    output_dir: Path,
    *,
    target_vocab_size: int = DEFAULT_TARGET_VOCAB_SIZE,
) -> dict[str, Any]:
    protobuf = _model_proto_type()
    baseline = load_model_proto(baseline_path)
    donor = load_model_proto(donor_path)
    unigram = protobuf.TrainerSpec.UNIGRAM
    if baseline.trainer_spec.model_type != unigram or donor.trainer_spec.model_type != unigram:
        raise ValueError("Baseline and donor must both be unigram SentencePiece models")
    if _effective_message_values(baseline.normalizer_spec) != _effective_message_values(
        donor.normalizer_spec
    ):
        raise ValueError("Baseline and donor must use the same normalization")

    additions_needed = target_vocab_size - len(baseline.pieces)
    if additions_needed <= 0:
        raise ValueError("target_vocab_size must be larger than the baseline vocabulary")

    existing_pieces = {piece.piece for piece in baseline.pieces}
    candidates = [
        (donor_id, piece)
        for donor_id, piece in enumerate(donor.pieces)
        if piece.type == protobuf.ModelProto.SentencePiece.NORMAL
        and piece.piece not in existing_pieces
        and has_devanagari(piece.piece)
    ]
    candidates.sort(key=lambda item: (-item[1].score, item[0]))
    if len(candidates) < additions_needed:
        raise ValueError(
            f"Donor has {len(candidates)} eligible pieces; {additions_needed} are required"
        )

    selected = candidates[:additions_needed]
    extended = protobuf.ModelProto()
    extended.CopyFrom(baseline)
    for _, donor_piece in selected:
        extended_piece = extended.pieces.add()
        extended_piece.CopyFrom(donor_piece)
    extended.trainer_spec.vocab_size = target_vocab_size

    validate_preserved_prefix(baseline, extended)
    if len(extended.pieces) != target_vocab_size:
        raise RuntimeError(
            f"Extended model has {len(extended.pieces)} pieces; expected {target_vocab_size}"
        )
    piece_strings = [piece.piece for piece in extended.pieces]
    if len(piece_strings) != len(set(piece_strings)):
        raise RuntimeError("Extended tokenizer contains duplicate pieces")
    if not all(has_devanagari(piece.piece) for piece in extended.pieces[len(baseline.pieces) :]):
        raise RuntimeError("Extended tokenizer contains a non-Devanagari addition")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"tokenizer_extended_{target_vocab_size}.model"
    vocab_path = model_path.with_suffix(".vocab")
    model_path.write_bytes(extended.SerializeToString(deterministic=True))
    vocab_path.write_text(
        "".join(f"{piece.piece}\t{piece.score}\n" for piece in extended.pieces), encoding="utf-8"
    )

    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.vocab_size() != target_vocab_size:
        raise RuntimeError(
            f"Serialized tokenizer has {tokenizer.vocab_size()} pieces; expected {target_vocab_size}"
        )
    byte_pieces = sum(tokenizer.is_byte(piece_id) for piece_id in range(target_vocab_size))
    if byte_pieces != 256:
        raise RuntimeError(f"Extended tokenizer has {byte_pieces} byte pieces; expected 256")

    selected_records = [
        {"new_id": len(baseline.pieces) + index, **_piece_record(piece, donor_id)}
        for index, (donor_id, piece) in enumerate(selected)
    ]
    metadata = {
        "schema_version": 1,
        "selection_policy": "highest-scored unseen NORMAL donor pieces containing Devanagari",
        "target_vocab_size": target_vocab_size,
        "preserved_piece_count": len(baseline.pieces),
        "added_piece_count": len(selected),
        "eligible_donor_piece_count": len(candidates),
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "donor_path": str(donor_path),
        "donor_sha256": sha256_file(donor_path),
        "model_path": model_path.name,
        "model_sha256": sha256_file(model_path),
        "vocab_path": vocab_path.name,
        "vocab_sha256": sha256_file(vocab_path),
        "selected_score_max": max(piece.score for _, piece in selected),
        "selected_score_min": min(piece.score for _, piece in selected),
        "selected_pieces": selected_records,
    }
    (output_dir / "extension_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline or Path(
        hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, revision=DEFAULT_REVISION)
    )
    metadata = extend_tokenizer(
        baseline_path, args.donor, args.output_dir, target_vocab_size=args.target_vocab_size
    )
    summary = {key: value for key, value in metadata.items() if key != "selected_pieces"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote ID-preserving tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
