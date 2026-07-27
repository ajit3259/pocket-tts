import math

import torch
from torch import nn

from experiments.indic.training_objective import (
    _combine_head_losses,
    assemble_teacher_forced_batch,
    compute_training_objective,
    flow_matching_loss,
    lsd_loss,
    normalize_latents,
    trig_path,
)
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN


class _TestConditioner(LUTConditioner):
    def __init__(self, vocab_size: int, dim: int):
        nn.Module.__init__(self)
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, tokenized):
        return self.embed(tokenized.tokens)


class _ExactTrigFlow(nn.Module):
    """Analytic average velocity for a path whose data endpoint is the condition."""

    def forward(self, data, start, end, position):
        start_angle = (math.pi / 2.0) * start
        start_sin = torch.sin(start_angle)
        start_cos = torch.cos(start_angle)
        noise = (position - start_sin * data) / start_cos

        end_position, end_velocity = trig_path(data, noise, end)
        delta = end - start
        safe_delta = torch.where(delta.abs() < 1e-6, torch.ones_like(delta), delta)
        average_velocity = (end_position - position) / safe_delta
        return torch.where(delta.abs() < 1e-6, end_velocity, average_velocity)


def _make_flow_lm() -> FlowLMModel:
    torch.manual_seed(0)
    latent_dim = 4
    model_dim = 16
    flow_lm = FlowLMModel(
        conditioner=_TestConditioner(vocab_size=32, dim=model_dim),
        flow_net=SimpleMLPAdaLN(
            in_channels=latent_dim,
            model_channels=16,
            out_channels=latent_dim,
            cond_channels=model_dim,
            num_res_blocks=2,
            num_time_conds=2,
        ),
        transformer=StreamingTransformer(d_model=model_dim, num_heads=4, num_layers=1),
        dim=model_dim,
        ldim=latent_dim,
        dtype=torch.float32,
        insert_bos_before_voice=True,
    )
    flow_lm.speaker_proj_weight = nn.Parameter(torch.randn(model_dim, latent_dim))
    return flow_lm


def test_normalization_uses_checkpoint_statistics() -> None:
    raw = torch.tensor([[[3.0, 6.0], [5.0, 2.0]]])
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([2.0, 4.0])

    normalized = normalize_latents(raw, mean, std)

    assert torch.equal(normalized, torch.tensor([[[1.0, 1.0], [2.0, 0.0]]]))


def test_reversed_trig_path_runs_from_noise_to_data() -> None:
    data = torch.tensor([[2.0, -1.0]])
    noise = torch.tensor([[0.25, 0.5]])

    at_zero, _ = trig_path(data, noise, torch.zeros(1, 1))
    at_one, _ = trig_path(data, noise, torch.ones(1, 1))

    assert torch.allclose(at_zero, noise)
    assert torch.allclose(at_one, data, atol=1e-6)


def test_exact_trig_flow_has_zero_fm_and_lsd_error() -> None:
    torch.manual_seed(1)
    data = torch.randn(5, 4)
    noise = torch.randn_like(data)
    diagonal_time = torch.linspace(0.1, 0.8, 5)[:, None]
    start = torch.linspace(0.05, 0.55, 5)[:, None]
    end = start + 0.3
    flow = _ExactTrigFlow()

    fm = flow_matching_loss(flow, data, data, time=diagonal_time, noise=noise)
    lsd = lsd_loss(flow, data, data, start_time=start, end_time=end, noise=noise)

    assert fm.item() < 1e-10
    assert lsd.item() < 1e-10


def test_teacher_forcing_shifts_targets_after_voice_and_text_prefixes() -> None:
    flow_lm = _make_flow_lm()
    text = [torch.tensor([4, 5, 6])]
    voice = [torch.randn(2, flow_lm.ldim)]
    target = [torch.randn(4, flow_lm.ldim)]

    batch = assemble_teacher_forced_batch(flow_lm, text, voice, target)

    prediction_start = 1 + voice[0].shape[0] + text[0].shape[0]
    assert torch.equal(
        batch.prediction_positions[0], prediction_start + torch.arange(target[0].shape[0])
    )
    expected_shifted = torch.cat([flow_lm.bos_emb[None], target[0][:-1]], dim=0)
    actual_audio_input = batch.transformer_input[
        0, prediction_start : prediction_start + target[0].shape[0]
    ]
    assert torch.allclose(actual_audio_input, flow_lm.input_linear(expected_shifted))
    assert batch.eos_targets[0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_head_loss_combination_keeps_source_ambiguity_explicit() -> None:
    fm = torch.tensor(2.0)
    lsd = torch.tensor(4.0)

    branch_sum = _combine_head_losses(fm, lsd, 6, 2, "branch_sum")
    sample_mean = _combine_head_losses(fm, lsd, 6, 2, "sample_mean")

    assert branch_sum.item() == 6.0
    assert sample_mean.item() == 2.5


def test_complete_objective_backpropagates_on_cpu() -> None:
    flow_lm = _make_flow_lm()
    target = [torch.randn(4, flow_lm.ldim), torch.randn(3, flow_lm.ldim)]
    batch = assemble_teacher_forced_batch(
        flow_lm,
        [torch.tensor([4, 5]), torch.tensor([6, 7, 8])],
        [torch.randn(2, flow_lm.ldim), None],
        target,
    )

    objective = compute_training_objective(
        flow_lm,
        batch,
        loss_combination="branch_sum",
        head_batch_multiplier=4,
        flow_matching_fraction=0.75,
        eos_loss_weight=0.1,
        generator=torch.Generator().manual_seed(2),
    )
    objective.loss.backward()

    assert objective.flow_matching_repeats == 3
    assert objective.lsd_repeats == 1
    assert objective.loss_combination == "branch_sum"
    assert torch.isfinite(objective.loss)
    assert flow_lm.conditioner.embed.weight.grad is not None
    assert flow_lm.transformer.layers[0].self_attn.in_proj.weight.grad is not None
    assert flow_lm.flow_net.input_proj.weight.grad is not None
    assert flow_lm.out_eos.weight.grad is not None
