"""GPU-ready, CPU-testable harness for the first Hindi tiny overfit run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from experiments.indic.training_objective import (
    assemble_teacher_forced_batch,
    compute_training_objective,
    normalize_latents,
)
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import DEFAULT_EOS_THRESHOLD
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.utils.utils import download_if_necessary

DEFAULT_PACKET_DIR = Path(__file__).with_name("outputs") / "e13_tiny_overfit_packet"
DEFAULT_CACHE_DIR = Path(__file__).with_name("outputs") / "e14_latent_cache"
DEFAULT_MODEL_CONFIG = (
    Path(__file__).with_name("outputs") / "e11_extended_checkpoint" / "config_extended_8000.yaml"
)
PRESERVED_VOCAB_SIZE = 4000


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    pair_index: int
    speaker_id: str
    text_tokens: torch.Tensor
    prompt_latents: torch.Tensor
    target_latents: torch.Tensor


@dataclass(frozen=True)
class TinyOverfitConfig:
    loss_combination: Literal["branch_sum", "sample_mean"]
    seed: int = 20260727
    learning_rate: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    preserved_vocab_size: int = PRESERVED_VOCAB_SIZE
    preserved_embedding_lr_scale: float = 0.1
    head_batch_multiplier: int = 8
    flow_matching_fraction: float = 0.75
    eos_loss_weight: float = 1.0
    eos_positive_weight: float = 1.0
    eos_threshold: float = DEFAULT_EOS_THRESHOLD

    def validate(self) -> None:
        if self.loss_combination not in {"branch_sum", "sample_mean"}:
            raise ValueError("loss_combination must be branch_sum or sample_mean")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not all(0 < beta < 1 for beta in self.betas):
            raise ValueError("AdamW betas must be in (0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.preserved_vocab_size < 1:
            raise ValueError("preserved_vocab_size must be positive")
        if not 0 <= self.preserved_embedding_lr_scale <= 1:
            raise ValueError("preserved_embedding_lr_scale must be in [0, 1]")


class EmbeddingRowDeltaPolicy:
    """Apply a lower effective learning rate to preserved rows after AdamW."""

    def __init__(
        self, weight: torch.nn.Parameter, *, preserved_rows: int, preserved_lr_scale: float
    ):
        if weight.ndim != 2:
            raise ValueError("Embedding weight must be a matrix")
        if not 0 < preserved_rows < weight.shape[0] - 1:
            raise ValueError("Preserved rows must leave new rows plus one padding row")
        if not 0 <= preserved_lr_scale <= 1:
            raise ValueError("preserved_lr_scale must be in [0, 1]")
        self.weight = weight
        self.preserved_rows = preserved_rows
        self.preserved_lr_scale = preserved_lr_scale
        self.padding_row = weight.shape[0] - 1

    def capture(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.weight[: self.preserved_rows].detach().clone(),
            self.weight[self.padding_row].detach().clone(),
        )

    @torch.no_grad()
    def apply(self, snapshot: tuple[torch.Tensor, torch.Tensor]) -> None:
        preserved_before, padding_before = snapshot
        preserved_after = self.weight[: self.preserved_rows]
        preserved_after.copy_(
            preserved_before + self.preserved_lr_scale * (preserved_after - preserved_before)
        )
        self.weight[self.padding_row].copy_(padding_before)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            records.append(record)
    return records


def load_reviewed_packet(packet_dir: Path) -> list[dict[str, Any]]:
    """Validate every hash and prompt relationship before training can see a row."""

    manifest_path = packet_dir / "manifest.jsonl"
    review_path = packet_dir / "review.json"
    records = _read_jsonl(manifest_path)
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    if not records or not isinstance(reviews, list):
        raise ValueError("Packet manifest and review must be non-empty lists")

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        example_id = record["example_id"]
        if example_id in records_by_id:
            raise ValueError(f"Duplicate packet example: {example_id}")
        records_by_id[example_id] = record

    reviews_by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        example_id = review["example_id"]
        if example_id in reviews_by_id:
            raise ValueError(f"Duplicate packet review: {example_id}")
        reviews_by_id[example_id] = review
    if set(reviews_by_id) != set(records_by_id):
        raise ValueError("Packet manifest and review example IDs differ")

    prompts_by_speaker: dict[str, dict[str, Any]] = {}
    targets_by_pair: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        example_id = record["example_id"]
        review = reviews_by_id[example_id]
        if review.get("decision") != "accepted":
            raise ValueError(f"Packet example is not accepted: {example_id}")
        audio_path = packet_dir / record["audio_file"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Packet audio is missing: {audio_path}")
        actual_sha256 = _sha256_file(audio_path)
        expected_hashes = {record["audio"]["sha256"], review["audio_sha256"]}
        if expected_hashes != {actual_sha256}:
            raise ValueError(f"Packet audio hash mismatch: {example_id}")

        role = record["role"]
        if role == "prompt":
            speaker = record["speaker_id"]
            if speaker in prompts_by_speaker:
                raise ValueError(f"Multiple prompts for speaker: {speaker}")
            if record.get("prompt_example_id") is not None:
                raise ValueError("Prompt rows cannot point to another prompt")
            prompts_by_speaker[speaker] = record
        elif role == "target":
            targets_by_pair.setdefault(record["pair_index"], []).append(record)
        else:
            raise ValueError(f"Unknown packet role: {role}")

    if not prompts_by_speaker or not targets_by_pair:
        raise ValueError("Packet requires both prompts and targets")
    for pair_index, pair in targets_by_pair.items():
        if len(pair) != len(prompts_by_speaker):
            raise ValueError(f"Target pair {pair_index} does not cover every speaker")
        if len({record["speaker_id"] for record in pair}) != len(pair):
            raise ValueError(f"Target pair {pair_index} repeats a speaker")
        if len({record["text_model_input"] for record in pair}) != 1:
            raise ValueError(f"Target pair {pair_index} does not share one text")
        for record in pair:
            prompt = prompts_by_speaker.get(record["speaker_id"])
            if prompt is None or record["prompt_example_id"] != prompt["example_id"]:
                raise ValueError(
                    f"Target has the wrong same-speaker prompt: {record['example_id']}"
                )
            if record["example_id"] == record["prompt_example_id"]:
                raise ValueError("Prompt and target must be different recordings")
    return records


def _asset_path(value: str) -> Path:
    path = Path(download_if_necessary(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def model_asset_fingerprints(model: TTSModel) -> dict[str, str]:
    checkpoint = model.config.weights_path
    tokenizer = model.config.flow_lm.lookup_table.tokenizer_path
    if checkpoint is None:
        raise ValueError("Training requires a source checkpoint")
    checkpoint_path = _asset_path(checkpoint)
    tokenizer_path = _asset_path(tokenizer)
    return {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.to(torch.float32)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


@torch.no_grad()
def build_latent_cache(model: TTSModel, packet_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Encode accepted audio once with frozen Mimi and pin tokenization."""

    records = load_reviewed_packet(packet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.mimi.requires_grad_(False)
    model.mimi.eval()

    tensors: dict[str, torch.Tensor] = {}
    examples: list[dict[str, Any]] = []
    for record in records:
        example_id = record["example_id"]
        audio_path = packet_dir / record["audio_file"]
        waveform, source_rate = audio_read(audio_path)
        waveform = convert_audio(
            waveform, source_rate, model.config.mimi.sample_rate, model.config.mimi.channels
        )
        raw_latents = model.mimi.encode_to_latent(
            waveform.unsqueeze(0).to(model.device, dtype=torch.float32)
        )
        raw_latents = raw_latents[0].transpose(0, 1).to(device="cpu", dtype=torch.float32)
        if raw_latents.ndim != 2 or raw_latents.shape[1] != model.flow_lm.ldim:
            raise RuntimeError(f"Unexpected Mimi latent shape for {example_id}")
        if not torch.isfinite(raw_latents).all():
            raise RuntimeError(f"Non-finite Mimi latent for {example_id}")

        latent_key = f"latent__{example_id}"
        tensors[latent_key] = raw_latents.contiguous()
        token_key: str | None = None
        token_count = 0
        if record["role"] == "target":
            tokens = (
                model.flow_lm.conditioner.tokenizer(record["text_model_input"])
                .tokens[0]
                .to(device="cpu", dtype=torch.long)
            )
            if tokens.numel() < 1:
                raise RuntimeError(f"Empty tokenization for {example_id}")
            token_key = f"tokens__{example_id}"
            tensors[token_key] = tokens.contiguous()
            token_count = tokens.numel()
        examples.append(
            {
                "example_id": example_id,
                "role": record["role"],
                "pair_index": record["pair_index"],
                "speaker_id": record["speaker_id"],
                "prompt_example_id": record["prompt_example_id"],
                "audio_sha256": record["audio"]["sha256"],
                "latent_key": latent_key,
                "latent_frames": raw_latents.shape[0],
                "token_key": token_key,
                "token_count": token_count,
            }
        )

    cache_path = output_dir / "latents.safetensors"
    temporary_cache = output_dir / "latents.safetensors.tmp"
    save_file(tensors, temporary_cache)
    temporary_cache.replace(cache_path)
    all_tokens = torch.cat(
        [tensors[example["token_key"]] for example in examples if example["token_key"] is not None]
    )
    raw_targets = torch.cat(
        [tensors[example["latent_key"]] for example in examples if example["role"] == "target"]
    )
    normalized_targets = normalize_latents(
        raw_targets, model.flow_lm.emb_mean.detach().cpu(), model.flow_lm.emb_std.detach().cpu()
    )
    tokenizer_vocab_size = model.flow_lm.conditioner.tokenizer.sp.vocab_size()
    unknown_id = model.flow_lm.conditioner.tokenizer.sp.unk_id()
    metadata = {
        "schema_version": 1,
        "packet_manifest_sha256": _sha256_file(packet_dir / "manifest.jsonl"),
        "packet_review_sha256": _sha256_file(packet_dir / "review.json"),
        **model_asset_fingerprints(model),
        "sample_rate": model.config.mimi.sample_rate,
        "frame_rate": model.config.mimi.frame_rate,
        "latent_dimension": model.flow_lm.ldim,
        "recordings": len(records),
        "target_recordings": sum(record["role"] == "target" for record in records),
        "target_frames": raw_targets.shape[0],
        "token_audit": {
            "tokens": all_tokens.numel(),
            "unique_token_ids": all_tokens.unique().numel(),
            "preserved_token_uses": (all_tokens < PRESERVED_VOCAB_SIZE).sum().item(),
            "added_token_uses": (
                (all_tokens >= PRESERVED_VOCAB_SIZE) & (all_tokens < tokenizer_vocab_size)
            )
            .sum()
            .item(),
            "unknown_token_uses": (all_tokens == unknown_id).sum().item(),
            "minimum_token_id": all_tokens.min().item(),
            "maximum_token_id": all_tokens.max().item(),
        },
        "raw_target_latents": _tensor_stats(raw_targets),
        "normalized_target_latents": _tensor_stats(normalized_targets),
        "latent_cache_sha256": _sha256_file(cache_path),
        "examples": examples,
    }
    (output_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def validate_cache(
    model: TTSModel, packet_dir: Path, cache_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_reviewed_packet(packet_dir)
    metadata_path = cache_dir / "cache_metadata.json"
    cache_path = cache_dir / "latents.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "packet_manifest_sha256": _sha256_file(packet_dir / "manifest.jsonl"),
        "packet_review_sha256": _sha256_file(packet_dir / "review.json"),
        "latent_cache_sha256": _sha256_file(cache_path),
        **model_asset_fingerprints(model),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Latent cache fingerprint mismatch: {key}")
    if metadata.get("latent_dimension") != model.flow_lm.ldim:
        raise ValueError("Latent cache dimension does not match FlowLM")
    return metadata, records


def load_training_batches(
    flow_lm: FlowLMModel, records: list[dict[str, Any]], cache_dir: Path, *, device: torch.device
) -> list[list[TrainingExample]]:
    tensors = load_file(cache_dir / "latents.safetensors", device=str(device))
    metadata = json.loads((cache_dir / "cache_metadata.json").read_text(encoding="utf-8"))
    cached_by_id = {item["example_id"]: item for item in metadata["examples"]}
    records_by_id = {record["example_id"]: record for record in records}
    batches: dict[int, list[TrainingExample]] = {}
    for record in records:
        if record["role"] != "target":
            continue
        cached = cached_by_id[record["example_id"]]
        prompt_id = record["prompt_example_id"]
        prompt_cached = cached_by_id[prompt_id]
        raw_target = tensors[cached["latent_key"]]
        normalized_target = normalize_latents(raw_target, flow_lm.emb_mean, flow_lm.emb_std)
        example = TrainingExample(
            example_id=record["example_id"],
            pair_index=record["pair_index"],
            speaker_id=record["speaker_id"],
            text_tokens=tensors[cached["token_key"]].to(torch.long),
            prompt_latents=tensors[prompt_cached["latent_key"]].to(torch.float32),
            target_latents=normalized_target,
        )
        prompt_record = records_by_id[prompt_id]
        if prompt_record["speaker_id"] != example.speaker_id:
            raise ValueError(f"Cached prompt speaker mismatch: {example.example_id}")
        batches.setdefault(example.pair_index, []).append(example)

    expected_batch_size = len(
        {record["speaker_id"] for record in records if record["role"] == "prompt"}
    )
    ordered_batches = []
    for pair_index in sorted(batches):
        batch = sorted(batches[pair_index], key=lambda example: example.speaker_id)
        if len(batch) != expected_batch_size:
            raise ValueError(f"Incomplete cached pair: {pair_index}")
        ordered_batches.append(batch)
    if not ordered_batches:
        raise ValueError("Latent cache contains no training batches")
    return ordered_batches


def batch_index_for_step(step: int, batch_count: int, seed: int) -> int:
    if step < 0 or batch_count < 1:
        raise ValueError("Step must be non-negative and batch_count must be positive")
    epoch, offset = divmod(step, batch_count)
    order = list(range(batch_count))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def build_optimizer(
    flow_lm: FlowLMModel, config: TinyOverfitConfig
) -> tuple[torch.optim.AdamW, EmbeddingRowDeltaPolicy]:
    config.validate()
    embedding_weight = flow_lm.conditioner.embed.weight
    other_parameters = [
        parameter
        for parameter in flow_lm.parameters()
        if parameter.requires_grad and parameter is not embedding_weight
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "weight_decay": config.weight_decay},
            {"params": [embedding_weight], "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
    )
    row_policy = EmbeddingRowDeltaPolicy(
        embedding_weight,
        preserved_rows=config.preserved_vocab_size,
        preserved_lr_scale=config.preserved_embedding_lr_scale,
    )
    return optimizer, row_policy


def run_training_step(
    flow_lm: FlowLMModel,
    examples: list[TrainingExample],
    optimizer: torch.optim.Optimizer,
    row_policy: EmbeddingRowDeltaPolicy,
    config: TinyOverfitConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    flow_lm.train()
    optimizer.zero_grad(set_to_none=True)
    batch = assemble_teacher_forced_batch(
        flow_lm,
        [example.text_tokens for example in examples],
        [example.prompt_latents for example in examples],
        [example.target_latents for example in examples],
    )
    objective = compute_training_objective(
        flow_lm,
        batch,
        loss_combination=config.loss_combination,
        head_batch_multiplier=config.head_batch_multiplier,
        flow_matching_fraction=config.flow_matching_fraction,
        eos_loss_weight=config.eos_loss_weight,
        eos_positive_weight=config.eos_positive_weight,
        generator=generator,
    )
    with torch.no_grad():
        eos_logits = flow_lm.out_eos(objective.conditioning)[..., 0]
        positive_mask = batch.frame_mask & batch.eos_targets.bool()
        nonterminal_mask = batch.frame_mask & ~batch.eos_targets.bool()
        positive_logits = eos_logits[positive_mask]
        nonterminal_logits = eos_logits[nonterminal_mask]
        eos_positive_logit_mean = positive_logits.mean()
        eos_nonterminal_logit_mean = nonterminal_logits.mean()
        eos_false_trigger_rate = (
            (nonterminal_logits > config.eos_threshold).to(torch.float32).mean()
        )
        eos_missed_end_rate = (positive_logits <= config.eos_threshold).to(torch.float32).mean()
    objective.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in flow_lm.parameters() if parameter.requires_grad],
        config.max_grad_norm,
    )
    embedding_snapshot = row_policy.capture()
    optimizer.step()
    row_policy.apply(embedding_snapshot)
    return {
        "loss": objective.loss.detach().item(),
        "head_loss": objective.head_loss.detach().item(),
        "flow_matching_loss": objective.flow_matching_loss.detach().item(),
        "lsd_loss": objective.lsd_loss.detach().item(),
        "eos_loss": objective.eos_loss.detach().item(),
        "eos_positive_logit_mean": eos_positive_logit_mean.item(),
        "eos_nonterminal_logit_mean": eos_nonterminal_logit_mean.item(),
        "eos_false_trigger_rate": eos_false_trigger_rate.item(),
        "eos_missed_end_rate": eos_missed_end_rate.item(),
        "grad_norm_before_clip": torch.as_tensor(grad_norm).item(),
        "valid_frames": float(batch.frame_mask.sum().item()),
    }


def _jsonable_config(config: TinyOverfitConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config)))


def save_training_checkpoint(
    checkpoint_dir: Path,
    flow_lm: FlowLMModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    completed_steps: int,
    config: TinyOverfitConfig,
    cache_metadata_sha256: str,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "flow_lm.safetensors"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    temporary_model = checkpoint_dir / "flow_lm.safetensors.tmp"
    temporary_optimizer = checkpoint_dir / "optimizer.pt.tmp"
    state_dict = {
        key: value.detach().cpu().contiguous() for key, value in flow_lm.state_dict().items()
    }
    save_file(state_dict, temporary_model)
    torch.save(
        {"optimizer": optimizer.state_dict(), "generator_state": generator.get_state().cpu()},
        temporary_optimizer,
    )
    temporary_model.replace(model_path)
    temporary_optimizer.replace(optimizer_path)
    metadata = {
        "schema_version": 1,
        "completed_steps": completed_steps,
        "training_config": _jsonable_config(config),
        "cache_metadata_sha256": cache_metadata_sha256,
        "model_sha256": _sha256_file(model_path),
        "optimizer_sha256": _sha256_file(optimizer_path),
    }
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_training_checkpoint(
    checkpoint_dir: Path,
    flow_lm: FlowLMModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    config: TinyOverfitConfig,
    cache_metadata_sha256: str,
) -> int:
    metadata = json.loads((checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("training_config") != _jsonable_config(config):
        raise ValueError("Resume checkpoint training configuration differs")
    if metadata.get("cache_metadata_sha256") != cache_metadata_sha256:
        raise ValueError("Resume checkpoint latent cache differs")
    model_path = checkpoint_dir / "flow_lm.safetensors"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if metadata.get("model_sha256") != _sha256_file(model_path):
        raise ValueError("Resume checkpoint model hash mismatch")
    if metadata.get("optimizer_sha256") != _sha256_file(optimizer_path):
        raise ValueError("Resume checkpoint optimizer hash mismatch")

    flow_lm.load_state_dict(load_file(model_path, device=str(next(flow_lm.parameters()).device)))
    training_state = torch.load(
        optimizer_path, map_location=next(flow_lm.parameters()).device, weights_only=True
    )
    optimizer.load_state_dict(training_state["optimizer"])
    generator.set_state(training_state["generator_state"].cpu())
    return int(metadata["completed_steps"])


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _last_logged_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    last_record: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_record = json.loads(line)
    return int(last_record["completed_steps"]) if last_record is not None else None


def _cache_command(args: argparse.Namespace) -> None:
    model = TTSModel.load_model(config=args.model_config)
    metadata = build_latent_cache(model, args.packet_dir, args.cache_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def _train_command(args: argparse.Namespace) -> None:
    config = TinyOverfitConfig(loss_combination=args.loss_combination)
    config.validate()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and args.resume is None:
        raise FileExistsError("A new run requires an empty --run-dir")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model(config=args.model_config)
    cache_metadata, records = validate_cache(model, args.packet_dir, args.cache_dir)
    flow_lm = model.flow_lm.to(device)
    del model.mimi
    batches = load_training_batches(flow_lm, records, args.cache_dir, device=device)
    optimizer, row_policy = build_optimizer(flow_lm, config)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    cache_metadata_sha256 = _sha256_file(args.cache_dir / "cache_metadata.json")

    completed_steps = 0
    if args.resume is not None:
        completed_steps = load_training_checkpoint(
            args.resume,
            flow_lm,
            optimizer,
            generator,
            config=config,
            cache_metadata_sha256=cache_metadata_sha256,
        )
        last_logged_step = _last_logged_step(args.run_dir / "metrics.jsonl")
        if last_logged_step is not None and last_logged_step != completed_steps:
            raise ValueError(
                f"Run log ends at step {last_logged_step}, but resume checkpoint is "
                f"at step {completed_steps}"
            )
    while completed_steps < args.max_steps:
        batch_index = batch_index_for_step(completed_steps, len(batches), config.seed)
        metrics = run_training_step(
            flow_lm, batches[batch_index], optimizer, row_policy, config, generator
        )
        completed_steps += 1
        _append_metrics(
            args.run_dir / "metrics.jsonl",
            {
                "completed_steps": completed_steps,
                "pair_index": batches[batch_index][0].pair_index,
                **metrics,
            },
        )
        if completed_steps % args.checkpoint_every == 0:
            save_training_checkpoint(
                args.run_dir / f"step_{completed_steps:06d}",
                flow_lm,
                optimizer,
                generator,
                completed_steps=completed_steps,
                config=config,
                cache_metadata_sha256=cache_metadata_sha256,
            )

    final_dir = args.run_dir / f"step_{completed_steps:06d}"
    save_training_checkpoint(
        final_dir,
        flow_lm,
        optimizer,
        generator,
        completed_steps=completed_steps,
        config=config,
        cache_metadata_sha256=cache_metadata_sha256,
    )
    print(final_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    cache.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    cache.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    cache.set_defaults(handler=_cache_command)

    train = subparsers.add_parser("train")
    train.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    train.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    train.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--device", required=True)
    train.add_argument("--loss-combination", choices=("branch_sum", "sample_mean"), required=True)
    train.add_argument("--max-steps", type=int, required=True)
    train.add_argument("--checkpoint-every", type=int, default=100)
    train.add_argument("--resume", type=Path)
    train.set_defaults(handler=_train_command)
    args = parser.parse_args()
    if getattr(args, "max_steps", 1) < 1:
        parser.error("--max-steps must be positive")
    if getattr(args, "checkpoint_every", 1) < 1:
        parser.error("--checkpoint-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
