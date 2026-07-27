"""Expand a released Pocket TTS checkpoint for the ID-preserving 8K tokenizer."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from experiments.indic.extend_tokenizer import DEFAULT_OUTPUT_DIR as DEFAULT_TOKENIZER_DIR
from experiments.indic.extend_tokenizer import load_model_proto
from experiments.indic.probe_tokenizer import DEFAULT_FILENAME as DEFAULT_BASE_TOKENIZER_FILENAME
from experiments.indic.probe_tokenizer import DEFAULT_REPO as DEFAULT_BASE_TOKENIZER_REPO
from experiments.indic.probe_tokenizer import DEFAULT_REVISION as DEFAULT_BASE_TOKENIZER_REVISION
from experiments.indic.train_tokenizer_candidates import sha256_file

DEFAULT_SOURCE_REPO = "kyutai/pocket-tts"
DEFAULT_SOURCE_FILENAME = "languages/english/model.safetensors"
DEFAULT_SOURCE_REVISION = "39592ff23c9ef80098bb74895d104c26275fe2c9"
DEFAULT_EXTENDED_TOKENIZER = DEFAULT_TOKENIZER_DIR / "tokenizer_extended_8000.model"
DEFAULT_BASE_CONFIG = Path(__file__).parents[2] / "pocket_tts" / "config" / "english.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "e11_extended_checkpoint"
DEFAULT_SEED = 20260727
DEFAULT_MIN_TRAINED_NORM = 0.01
EMBEDDING_KEY = "flow_lm.conditioner.embed.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--base-tokenizer", type=Path)
    parser.add_argument("--extended-tokenizer", type=Path, default=DEFAULT_EXTENDED_TOKENIZER)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-trained-norm", type=float, default=DEFAULT_MIN_TRAINED_NORM)
    parser.add_argument("--skip-strict-load", action="store_true")
    return parser.parse_args()


def _without_dummy_prefix(tokenizer_path: Path) -> spm.SentencePieceProcessor:
    model = load_model_proto(tokenizer_path)
    model.normalizer_spec.add_dummy_prefix = False
    return spm.SentencePieceProcessor(model_proto=model.SerializeToString(deterministic=True))


def decompose_added_pieces(
    base_tokenizer_path: Path, extended_tokenizer_path: Path, *, preserved_vocab_size: int
) -> list[list[int]]:
    base = _without_dummy_prefix(base_tokenizer_path)
    extended = spm.SentencePieceProcessor(model_file=str(extended_tokenizer_path))
    if base.vocab_size() != preserved_vocab_size:
        raise ValueError(
            f"Base tokenizer has {base.vocab_size()} pieces; expected {preserved_vocab_size}"
        )
    if extended.vocab_size() <= preserved_vocab_size:
        raise ValueError("Extended tokenizer must be larger than the base tokenizer")

    decompositions = []
    for piece_id in range(preserved_vocab_size, extended.vocab_size()):
        piece = extended.id_to_piece(piece_id)
        token_ids = base.encode(piece.replace("▁", " "), out_type=int)
        if not token_ids:
            raise RuntimeError(f"New piece {piece_id} has an empty base decomposition")
        if any(token_id >= preserved_vocab_size for token_id in token_ids):
            raise RuntimeError(f"New piece {piece_id} decomposes outside the base vocabulary")
        decompositions.append(token_ids)
    return decompositions


def _trained_piece_ids(
    tokenizer_path: Path, embedding: torch.Tensor, *, min_norm: float
) -> list[int]:
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() + 1 != embedding.shape[0]:
        raise ValueError("Source embedding shape does not match the base tokenizer plus padding")
    norms = embedding[: tokenizer.vocab_size()].to(torch.float32).norm(dim=1)
    ids = [
        piece_id
        for piece_id in range(tokenizer.vocab_size())
        if not tokenizer.is_unknown(piece_id)
        and not tokenizer.is_control(piece_id)
        and not tokenizer.is_unused(piece_id)
        and not tokenizer.is_byte(piece_id)
        and norms[piece_id] >= min_norm
    ]
    if not ids:
        raise ValueError("No trained source embedding rows passed the norm threshold")
    return ids


def initialize_matched_random(
    source_embedding: torch.Tensor,
    trained_piece_ids: list[int],
    added_piece_count: int,
    *,
    seed: int,
) -> torch.Tensor:
    if added_piece_count < 1:
        raise ValueError("added_piece_count must be positive")
    trained = source_embedding[trained_piece_ids].to(torch.float32)
    feature_mean = trained.mean(dim=0)
    feature_std = trained.std(dim=0, unbiased=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initialized = (
        torch.randn(
            (added_piece_count, source_embedding.shape[1]), generator=generator, dtype=torch.float32
        )
        * feature_std
        + feature_mean
    )

    reference_indices = torch.randint(
        len(trained), (added_piece_count,), generator=generator, dtype=torch.int64
    )
    target_norms = trained[reference_indices].norm(dim=1)
    initialized *= (
        target_norms / initialized.norm(dim=1).clamp_min(torch.finfo(torch.float32).eps)
    ).unsqueeze(1)
    return initialized.to(source_embedding.dtype)


def decomposition_means(
    source_embedding: torch.Tensor, decompositions: list[list[int]]
) -> torch.Tensor:
    return torch.stack(
        [source_embedding[token_ids].to(torch.float32).mean(dim=0) for token_ids in decompositions]
    )


def expand_embedding(
    source_embedding: torch.Tensor, initialized_rows: torch.Tensor, *, preserved_vocab_size: int
) -> torch.Tensor:
    expected_source_rows = preserved_vocab_size + 1
    if source_embedding.shape[0] != expected_source_rows:
        raise ValueError(
            f"Source embedding has {source_embedding.shape[0]} rows; expected "
            f"{expected_source_rows}"
        )
    target_vocab_size = preserved_vocab_size + initialized_rows.shape[0]
    if initialized_rows.shape[1:] != source_embedding.shape[1:]:
        raise ValueError("Initialized rows have the wrong embedding dimension")

    expanded = torch.empty(
        (target_vocab_size + 1, *source_embedding.shape[1:]), dtype=source_embedding.dtype
    )
    expanded[:preserved_vocab_size] = source_embedding[:preserved_vocab_size]
    expanded[preserved_vocab_size:target_vocab_size] = initialized_rows.to(source_embedding.dtype)
    expanded[target_vocab_size] = source_embedding[preserved_vocab_size]
    return expanded


def _tensor_norm_summary(tensor: torch.Tensor) -> dict[str, float]:
    norms = tensor.to(torch.float32).norm(dim=1)
    return {
        "mean": norms.mean().item(),
        "std": norms.std(unbiased=False).item(),
        "p05": torch.quantile(norms, 0.05).item(),
        "p50": torch.quantile(norms, 0.50).item(),
        "p95": torch.quantile(norms, 0.95).item(),
    }


def _write_config(
    base_config_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    vocab_size: int,
) -> None:
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    config["weights_path"] = str(checkpoint_path.resolve())
    config["weights_path_without_voice_cloning"] = None
    config["flow_lm"]["lookup_table"]["n_bins"] = vocab_size
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_path.resolve())
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def prepare_checkpoint(
    source_checkpoint: Path,
    base_tokenizer_path: Path,
    extended_tokenizer_path: Path,
    base_config_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    min_trained_norm: float = DEFAULT_MIN_TRAINED_NORM,
    verify_model_load: bool = False,
) -> dict[str, Any]:
    if min_trained_norm < 0:
        raise ValueError("min_trained_norm must be non-negative")
    base = spm.SentencePieceProcessor(model_file=str(base_tokenizer_path))
    extended = spm.SentencePieceProcessor(model_file=str(extended_tokenizer_path))
    preserved_vocab_size = base.vocab_size()
    target_vocab_size = extended.vocab_size()
    if target_vocab_size <= preserved_vocab_size:
        raise ValueError("Extended tokenizer must be larger than the base tokenizer")

    state_dict = load_file(source_checkpoint, device="cpu")
    if EMBEDDING_KEY not in state_dict:
        raise KeyError(f"Source checkpoint is missing {EMBEDDING_KEY}")
    source_embedding = state_dict[EMBEDDING_KEY]
    trained_ids = _trained_piece_ids(
        base_tokenizer_path, source_embedding, min_norm=min_trained_norm
    )
    decompositions = decompose_added_pieces(
        base_tokenizer_path, extended_tokenizer_path, preserved_vocab_size=preserved_vocab_size
    )
    added_piece_count = target_vocab_size - preserved_vocab_size
    if len(decompositions) != added_piece_count:
        raise RuntimeError("Decomposition count does not match the added vocabulary")

    initialized = initialize_matched_random(
        source_embedding, trained_ids, added_piece_count, seed=seed
    )
    raw_decomposition = decomposition_means(source_embedding, decompositions)
    expanded = expand_embedding(
        source_embedding, initialized, preserved_vocab_size=preserved_vocab_size
    )
    if not torch.equal(expanded[:preserved_vocab_size], source_embedding[:preserved_vocab_size]):
        raise RuntimeError("A preserved token embedding row changed")
    if not torch.equal(expanded[target_vocab_size], source_embedding[preserved_vocab_size]):
        raise RuntimeError("The source padding row was not moved exactly")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"model_extended_{target_vocab_size}.safetensors"
    config_path = output_dir / f"config_extended_{target_vocab_size}.yaml"
    state_dict[EMBEDDING_KEY] = expanded.contiguous()
    source_checkpoint_sha256 = sha256_file(source_checkpoint)
    base_tokenizer_sha256 = sha256_file(base_tokenizer_path)
    extended_tokenizer_sha256 = sha256_file(extended_tokenizer_path)
    save_file(state_dict, checkpoint_path)
    _write_config(
        base_config_path, config_path, checkpoint_path, extended_tokenizer_path, target_vocab_size
    )

    strict_model_load = False
    if verify_model_load:
        del state_dict
        gc.collect()
        from pocket_tts import TTSModel

        model = TTSModel.load_model(config=config_path)
        actual_shape = tuple(model.flow_lm.conditioner.embed.weight.shape)
        expected_shape = (target_vocab_size + 1, source_embedding.shape[1])
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Strict-loaded embedding has shape {actual_shape}; expected {expected_shape}"
            )
        strict_model_load = True
        del model

    decomposition_lengths = torch.tensor([len(ids) for ids in decompositions])
    metadata = {
        "schema_version": 1,
        "initialization": "matched-random-v1",
        "seed": seed,
        "min_trained_norm": min_trained_norm,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "base_tokenizer": str(base_tokenizer_path),
        "base_tokenizer_sha256": base_tokenizer_sha256,
        "extended_tokenizer": str(extended_tokenizer_path),
        "extended_tokenizer_sha256": extended_tokenizer_sha256,
        "preserved_vocab_size": preserved_vocab_size,
        "target_vocab_size": target_vocab_size,
        "embedding_dimension": source_embedding.shape[1],
        "source_embedding_shape": list(source_embedding.shape),
        "expanded_embedding_shape": list(expanded.shape),
        "source_embedding_dtype": str(source_embedding.dtype),
        "trained_source_rows": len(trained_ids),
        "strict_model_load": strict_model_load,
        "new_embedding_norms": _tensor_norm_summary(initialized),
        "trained_source_embedding_norms": _tensor_norm_summary(source_embedding[trained_ids]),
        "raw_decomposition_mean_norms": _tensor_norm_summary(raw_decomposition),
        "decomposition_length": {
            "mean": decomposition_lengths.to(torch.float32).mean().item(),
            "p50": torch.quantile(decomposition_lengths.to(torch.float32), 0.50).item(),
            "p95": torch.quantile(decomposition_lengths.to(torch.float32), 0.95).item(),
            "maximum": decomposition_lengths.max().item(),
        },
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": config_path.name,
    }
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    source_checkpoint = args.source_checkpoint or Path(
        hf_hub_download(
            repo_id=DEFAULT_SOURCE_REPO,
            filename=DEFAULT_SOURCE_FILENAME,
            revision=DEFAULT_SOURCE_REVISION,
        )
    )
    base_tokenizer_path = args.base_tokenizer or Path(
        hf_hub_download(
            repo_id=DEFAULT_BASE_TOKENIZER_REPO,
            filename=DEFAULT_BASE_TOKENIZER_FILENAME,
            revision=DEFAULT_BASE_TOKENIZER_REVISION,
        )
    )
    metadata = prepare_checkpoint(
        source_checkpoint,
        base_tokenizer_path,
        args.extended_tokenizer,
        args.base_config,
        args.output_dir,
        seed=args.seed,
        min_trained_norm=args.min_trained_norm,
        verify_model_load=not args.skip_strict_load,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote expanded checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
