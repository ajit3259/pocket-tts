"""Paper-backed training objective primitives for Indic Pocket TTS experiments.

This module intentionally stops before data loading, optimization, and distributed
training. It makes the teacher-forcing and CALM/LSD contracts testable on CPU before
we spend GPU time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.flow_lm import FlowLMModel


@dataclass
class TeacherForcedBatch:
    """Padded inputs and targets for next-latent teacher forcing."""

    transformer_input: torch.Tensor
    prediction_positions: torch.Tensor
    target_latents: torch.Tensor
    frame_mask: torch.Tensor
    eos_targets: torch.Tensor


@dataclass
class TrainingObjective:
    """Scalar losses plus the conditioning tensor used to compute them."""

    loss: torch.Tensor
    head_loss: torch.Tensor
    flow_matching_loss: torch.Tensor
    lsd_loss: torch.Tensor
    eos_loss: torch.Tensor
    conditioning: torch.Tensor
    flow_matching_repeats: int
    lsd_repeats: int
    loss_combination: str


def normalize_latents(
    raw_latents: torch.Tensor, emb_mean: torch.Tensor, emb_std: torch.Tensor
) -> torch.Tensor:
    """Normalize raw Mimi latents using the frozen checkpoint statistics."""
    if torch.any(emb_std <= 0):
        raise ValueError("Every latent standard deviation must be positive")
    return (raw_latents.to(torch.float32) - emb_mean.to(torch.float32)) / emb_std.to(torch.float32)


def assemble_teacher_forced_batch(
    flow_lm: FlowLMModel,
    text_tokens: list[torch.Tensor],
    voice_latents: list[torch.Tensor | None],
    target_latents: list[torch.Tensor],
) -> TeacherForcedBatch:
    """Assemble the inference-compatible prefix and shifted audio history.

    The per-example layout is:

        [voice prefix | text prefix | input_linear(BOS, target[:-1])]

    Voice latents are raw Mimi latents, matching inference. Target latents are
    already normalized with ``emb_mean`` and ``emb_std``.
    """
    if not text_tokens:
        raise ValueError("At least one example is required")
    if not (len(text_tokens) == len(voice_latents) == len(target_latents)):
        raise ValueError("Text, voice, and target lists must have the same length")

    device = flow_lm.bos_emb.device
    model_dtype = flow_lm.bos_emb.dtype
    per_example: list[torch.Tensor] = []
    prediction_starts: list[int] = []

    for tokens, voice, target in zip(text_tokens, voice_latents, target_latents):
        if tokens.ndim != 1:
            raise ValueError("Each text token tensor must have shape [text_steps]")
        if target.ndim != 2 or target.shape[0] < 1 or target.shape[1] != flow_lm.ldim:
            raise ValueError(
                f"Each target must have shape [frames, {flow_lm.ldim}] with frames >= 1"
            )

        prefix_parts: list[torch.Tensor] = []
        if voice is not None and voice.shape[0] > 0:
            if voice.ndim != 2 or voice.shape[1] != flow_lm.ldim:
                raise ValueError(f"Each voice prompt must have shape [frames, {flow_lm.ldim}]")
            if not hasattr(flow_lm, "speaker_proj_weight"):
                raise ValueError("FlowLM has no speaker projection weight")
            if flow_lm.insert_bos_before_voice:
                prefix_parts.append(flow_lm.bos_before_voice[0])
            prefix_parts.append(
                F.linear(voice.to(device=device, dtype=model_dtype), flow_lm.speaker_proj_weight)
            )

        if tokens.numel() > 0:
            tokenized = TokenizedText(tokens.to(device=device, dtype=torch.long)[None])
            prefix_parts.append(flow_lm.conditioner(tokenized)[0])

        target = target.to(device=device, dtype=torch.float32)
        shifted_audio = torch.cat([flow_lm.bos_emb[None], target[:-1].to(model_dtype)], dim=0)
        audio_embeddings = flow_lm.input_linear(shifted_audio)
        prediction_start = sum(part.shape[0] for part in prefix_parts)
        per_example.append(torch.cat([*prefix_parts, audio_embeddings], dim=0))
        prediction_starts.append(prediction_start)

    batch_size = len(per_example)
    max_sequence = max(item.shape[0] for item in per_example)
    max_frames = max(target.shape[0] for target in target_latents)
    transformer_input = torch.zeros(
        batch_size, max_sequence, flow_lm.dim, device=device, dtype=model_dtype
    )
    prediction_positions = torch.zeros(batch_size, max_frames, device=device, dtype=torch.long)
    targets = torch.zeros(batch_size, max_frames, flow_lm.ldim, device=device, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_frames, device=device, dtype=torch.bool)
    eos_targets = torch.zeros(batch_size, max_frames, device=device, dtype=torch.float32)

    for index, (embeddings, target, start) in enumerate(
        zip(per_example, target_latents, prediction_starts)
    ):
        frames = target.shape[0]
        transformer_input[index, : embeddings.shape[0]] = embeddings
        prediction_positions[index, :frames] = start + torch.arange(frames, device=device)
        targets[index, :frames] = target.to(device=device, dtype=torch.float32)
        frame_mask[index, :frames] = True
        eos_targets[index, frames - 1] = 1.0

    return TeacherForcedBatch(
        transformer_input=transformer_input,
        prediction_positions=prediction_positions,
        target_latents=targets,
        frame_mask=frame_mask,
        eos_targets=eos_targets,
    )


def run_teacher_forced_backbone(flow_lm: FlowLMModel, batch: TeacherForcedBatch) -> torch.Tensor:
    """Return one contextual vector per target frame."""
    transformer_output = flow_lm.transformer(batch.transformer_input, None)
    transformer_output = flow_lm.out_norm(transformer_output)
    gather_index = batch.prediction_positions[..., None].expand(
        -1, -1, transformer_output.shape[-1]
    )
    return transformer_output.gather(1, gather_index).to(torch.float32)


def trig_path(
    data: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the noised latent and velocity in Pocket's inference time direction.

    Pocket inference uses time 0 for noise and time 1 for data. This is the reverse
    of the notation in the CALM paper, so:

        x(t) = sin(pi*t/2) * data + cos(pi*t/2) * noise
    """
    angle = (math.pi / 2.0) * time
    sin, cos = torch.sin(angle), torch.cos(angle)
    position = sin * data + cos * noise
    velocity = (math.pi / 2.0) * (cos * data - sin * noise)
    return position, velocity


def flow_matching_loss(
    flow_net: torch.nn.Module,
    conditioning: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Unweighted diagonal flow-matching loss from CALM Appendix A, Eq. 5."""
    examples = target_latents.shape[0]
    if time is None:
        time = torch.rand(
            examples,
            1,
            device=target_latents.device,
            dtype=target_latents.dtype,
            generator=generator,
        )
    if noise is None:
        noise = torch.randn(
            target_latents.shape,
            device=target_latents.device,
            dtype=target_latents.dtype,
            generator=generator,
        )
    position, target_velocity = trig_path(target_latents, noise, time)
    predicted_velocity = flow_net(conditioning, time, time, position)
    return (predicted_velocity - target_velocity).pow(2).mean()


def lsd_loss(
    flow_net: torch.nn.Module,
    conditioning: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    start_time: torch.Tensor | None = None,
    end_time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Unweighted off-diagonal Lagrangian self-distillation loss, Eq. 6."""
    examples = target_latents.shape[0]
    if start_time is None:
        start_time = torch.rand(
            examples,
            1,
            device=target_latents.device,
            dtype=target_latents.dtype,
            generator=generator,
        )
    if end_time is None:
        remaining_fraction = torch.rand(
            examples,
            1,
            device=target_latents.device,
            dtype=target_latents.dtype,
            generator=generator,
        )
        end_time = start_time + (1.0 - start_time) * remaining_fraction
    if noise is None:
        noise = torch.randn(
            target_latents.shape,
            device=target_latents.device,
            dtype=target_latents.dtype,
            generator=generator,
        )

    start_position, _ = trig_path(target_latents, noise, start_time)

    def flow_map(candidate_end: torch.Tensor) -> torch.Tensor:
        average_velocity = flow_net(conditioning, start_time, candidate_end, start_position)
        return start_position + (candidate_end - start_time) * average_velocity

    mapped_position, derivative = torch.func.jvp(
        flow_map, (end_time,), (torch.ones_like(end_time),)
    )
    with torch.no_grad():
        teacher_velocity = flow_net(conditioning, end_time, end_time, mapped_position.detach())
    return (derivative - teacher_velocity).pow(2).mean()


def masked_eos_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Binary cross-entropy on valid frames only."""
    if positive_weight <= 0:
        raise ValueError("EOS positive weight must be positive")
    per_frame = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
        reduction="none",
    )
    mask = frame_mask.to(per_frame.dtype)
    return (per_frame * mask).sum() / mask.sum().clamp_min(1)


def _split_head_repeats(multiplier: int, flow_matching_fraction: float) -> tuple[int, int]:
    if multiplier < 1:
        raise ValueError("Head batch multiplier must be at least one")
    if not 0.0 <= flow_matching_fraction <= 1.0:
        raise ValueError("Flow-matching fraction must be in [0, 1]")
    flow_matching_repeats = round(multiplier * flow_matching_fraction)
    if multiplier > 1 and 0.0 < flow_matching_fraction < 1.0:
        flow_matching_repeats = min(max(flow_matching_repeats, 1), multiplier - 1)
    return flow_matching_repeats, multiplier - flow_matching_repeats


def _combine_head_losses(
    flow_matching: torch.Tensor,
    lsd: torch.Tensor,
    flow_matching_repeats: int,
    lsd_repeats: int,
    mode: Literal["branch_sum", "sample_mean"],
) -> torch.Tensor:
    """Resolve a documented ambiguity between the LSD paper and reference code."""
    if flow_matching_repeats == 0:
        return lsd
    if lsd_repeats == 0:
        return flow_matching
    if mode == "branch_sum":
        return flow_matching + lsd
    if mode == "sample_mean":
        total = flow_matching_repeats + lsd_repeats
        return (flow_matching_repeats * flow_matching + lsd_repeats * lsd) / total
    raise ValueError(f"Unknown head loss combination: {mode}")


def compute_training_objective(
    flow_lm: FlowLMModel,
    batch: TeacherForcedBatch,
    *,
    loss_combination: Literal["branch_sum", "sample_mean"],
    head_batch_multiplier: int = 8,
    flow_matching_fraction: float = 0.75,
    eos_loss_weight: float = 1.0,
    eos_positive_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> TrainingObjective:
    """Compute the CPU-testable FlowLM objective while reusing backbone outputs."""
    if eos_loss_weight < 0:
        raise ValueError("EOS loss weight must be non-negative")
    conditioning = run_teacher_forced_backbone(flow_lm, batch)
    eos_logits = flow_lm.out_eos(conditioning)[..., 0]
    loss_eos = masked_eos_loss(
        eos_logits, batch.eos_targets, batch.frame_mask, positive_weight=eos_positive_weight
    )

    valid_conditioning = conditioning[batch.frame_mask]
    valid_targets = batch.target_latents[batch.frame_mask]
    fm_repeats, lsd_repeats = _split_head_repeats(head_batch_multiplier, flow_matching_fraction)
    zero = valid_targets.sum() * 0.0

    if fm_repeats:
        loss_fm = flow_matching_loss(
            flow_lm.flow_net,
            valid_conditioning.repeat(fm_repeats, 1),
            valid_targets.repeat(fm_repeats, 1),
            generator=generator,
        )
    else:
        loss_fm = zero
    if lsd_repeats:
        loss_lsd = lsd_loss(
            flow_lm.flow_net,
            valid_conditioning.repeat(lsd_repeats, 1),
            valid_targets.repeat(lsd_repeats, 1),
            generator=generator,
        )
    else:
        loss_lsd = zero

    head_loss = _combine_head_losses(loss_fm, loss_lsd, fm_repeats, lsd_repeats, loss_combination)
    loss = head_loss + eos_loss_weight * loss_eos
    return TrainingObjective(
        loss=loss,
        head_loss=head_loss,
        flow_matching_loss=loss_fm,
        lsd_loss=loss_lsd,
        eos_loss=loss_eos,
        conditioning=conditioning,
        flow_matching_repeats=fm_repeats,
        lsd_repeats=lsd_repeats,
        loss_combination=loss_combination,
    )
