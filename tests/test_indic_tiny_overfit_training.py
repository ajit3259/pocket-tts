import hashlib
import json
import wave

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from experiments.indic.tiny_overfit_training import (
    EmbeddingRowDeltaPolicy,
    TinyOverfitConfig,
    TrainingExample,
    _last_logged_step,
    batch_index_for_step,
    build_optimizer,
    load_reviewed_packet,
    load_training_batches,
    load_training_checkpoint,
    run_training_step,
    save_training_checkpoint,
)
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN


class _TestConditioner(LUTConditioner):
    def __init__(self, rows: int, dim: int):
        nn.Module.__init__(self)
        self.embed = nn.Embedding(rows, dim)

    def forward(self, tokenized):
        return self.embed(tokenized.tokens)


def _make_flow_lm() -> FlowLMModel:
    torch.manual_seed(100)
    latent_dim = 4
    model_dim = 16
    flow_lm = FlowLMModel(
        conditioner=_TestConditioner(rows=8, dim=model_dim),
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


def _write_wav(path, value: int) -> str:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(value.to_bytes(2, "little", signed=True) * 240)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_packet(tmp_path):
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    speakers = ("female", "male")
    records = []
    reviews = []
    prompt_ids = {}
    for index, speaker in enumerate(speakers, start=1):
        example_id = f"prompt-{speaker}"
        filename = f"{example_id}.wav"
        sha256 = _write_wav(packet_dir / filename, index)
        prompt_ids[speaker] = example_id
        records.append(
            {
                "example_id": example_id,
                "role": "prompt",
                "pair_index": 0,
                "speaker_id": speaker,
                "prompt_example_id": None,
                "audio_file": filename,
                "audio": {"sha256": sha256},
                "text_model_input": "प्रॉम्प्ट",
            }
        )
        reviews.append({"example_id": example_id, "decision": "accepted", "audio_sha256": sha256})
    for index, speaker in enumerate(speakers, start=3):
        example_id = f"target-{speaker}"
        filename = f"{example_id}.wav"
        sha256 = _write_wav(packet_dir / filename, index)
        records.append(
            {
                "example_id": example_id,
                "role": "target",
                "pair_index": 1,
                "speaker_id": speaker,
                "prompt_example_id": prompt_ids[speaker],
                "audio_file": filename,
                "audio": {"sha256": sha256},
                "text_model_input": "एक साझा वाक्य।",
            }
        )
        reviews.append({"example_id": example_id, "decision": "accepted", "audio_sha256": sha256})
    (packet_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (packet_dir / "review.json").write_text(
        json.dumps(reviews, ensure_ascii=False), encoding="utf-8"
    )
    return packet_dir, records


def _examples() -> list[TrainingExample]:
    return [
        TrainingExample(
            example_id=f"target-{speaker}",
            pair_index=1,
            speaker_id=speaker,
            text_tokens=torch.tensor([4, 5]),
            prompt_latents=torch.randn(2, 4),
            target_latents=torch.randn(3, 4),
        )
        for speaker in ("female", "male")
    ]


def _training_config() -> TinyOverfitConfig:
    return TinyOverfitConfig(
        loss_combination="sample_mean",
        learning_rate=1e-3,
        preserved_vocab_size=4,
        preserved_embedding_lr_scale=0.1,
        head_batch_multiplier=2,
        flow_matching_fraction=0.5,
    )


def test_reviewed_packet_requires_accepted_hash_bound_same_speaker_pairs(tmp_path) -> None:
    packet_dir, records = _write_packet(tmp_path)

    loaded = load_reviewed_packet(packet_dir)

    assert [record["example_id"] for record in loaded] == [
        record["example_id"] for record in records
    ]
    with (packet_dir / "target-female.wav").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        load_reviewed_packet(packet_dir)


def test_embedding_row_policy_scales_preserved_delta_and_restores_padding() -> None:
    weight = nn.Parameter(torch.arange(14, dtype=torch.float32).reshape(7, 2))
    policy = EmbeddingRowDeltaPolicy(weight, preserved_rows=4, preserved_lr_scale=0.1)
    before = weight.detach().clone()
    snapshot = policy.capture()

    with torch.no_grad():
        weight.add_(1)
    policy.apply(snapshot)

    assert torch.allclose(weight[:4], before[:4] + 0.1)
    assert torch.equal(weight[4:6], before[4:6] + 1)
    assert torch.equal(weight[6], before[6])


def test_batch_schedule_is_shuffled_per_epoch_and_resume_is_step_only() -> None:
    first_epoch = [batch_index_for_step(step, 8, 123) for step in range(8)]
    second_epoch = [batch_index_for_step(step, 8, 123) for step in range(8, 16)]

    assert sorted(first_epoch) == list(range(8))
    assert sorted(second_epoch) == list(range(8))
    assert first_epoch != second_epoch
    assert batch_index_for_step(11, 8, 123) == second_epoch[3]


def test_last_logged_step_ignores_blank_lines(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    assert _last_logged_step(path) is None
    path.write_text('{"completed_steps": 1}\n\n{"completed_steps": 2}\n', encoding="utf-8")
    assert _last_logged_step(path) == 2


def test_cached_raw_targets_are_normalized_only_when_batches_are_loaded(tmp_path) -> None:
    packet_dir, records = _write_packet(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    tensors = {
        "latent__prompt-female": torch.ones(2, 4),
        "latent__prompt-male": torch.ones(2, 4) * 2,
        "latent__target-female": torch.ones(3, 4) * 5,
        "latent__target-male": torch.ones(3, 4) * 7,
        "tokens__target-female": torch.tensor([4, 5]),
        "tokens__target-male": torch.tensor([4, 5]),
    }
    save_file(tensors, cache_dir / "latents.safetensors")
    metadata_examples = []
    for record in records:
        example_id = record["example_id"]
        metadata_examples.append(
            {
                "example_id": example_id,
                "latent_key": f"latent__{example_id}",
                "token_key": (f"tokens__{example_id}" if record["role"] == "target" else None),
            }
        )
    (cache_dir / "cache_metadata.json").write_text(
        json.dumps({"examples": metadata_examples}), encoding="utf-8"
    )
    flow_lm = _make_flow_lm()
    flow_lm.emb_mean.copy_(torch.ones(4))
    flow_lm.emb_std.copy_(torch.ones(4) * 2)

    batches = load_training_batches(flow_lm, records, cache_dir, device=torch.device("cpu"))

    assert len(batches) == 1
    by_speaker = {example.speaker_id: example for example in batches[0]}
    assert torch.equal(by_speaker["female"].prompt_latents, torch.ones(2, 4))
    assert torch.equal(by_speaker["female"].target_latents, torch.ones(3, 4) * 2)
    assert torch.equal(by_speaker["male"].target_latents, torch.ones(3, 4) * 3)


def test_checkpoint_resume_reproduces_the_next_stochastic_training_step(tmp_path) -> None:
    examples = _examples()
    config = _training_config()
    flow_lm = _make_flow_lm()
    optimizer, row_policy = build_optimizer(flow_lm, config)
    generator = torch.Generator().manual_seed(config.seed)
    unused_added_initial = flow_lm.conditioner.embed.weight[6].detach().clone()

    run_training_step(flow_lm, examples, optimizer, row_policy, config, generator)
    padding_after_first = flow_lm.conditioner.embed.weight[-1].detach().clone()
    unused_added_after_first = flow_lm.conditioner.embed.weight[6].detach().clone()
    assert torch.equal(unused_added_after_first, unused_added_initial)
    save_training_checkpoint(
        tmp_path / "checkpoint",
        flow_lm,
        optimizer,
        generator,
        completed_steps=1,
        config=config,
        cache_metadata_sha256="cache-hash",
    )
    reference_metrics = run_training_step(
        flow_lm, examples, optimizer, row_policy, config, generator
    )
    reference_state = {key: value.detach().clone() for key, value in flow_lm.state_dict().items()}

    resumed_model = _make_flow_lm()
    resumed_optimizer, resumed_policy = build_optimizer(resumed_model, config)
    resumed_generator = torch.Generator().manual_seed(0)
    completed_steps = load_training_checkpoint(
        tmp_path / "checkpoint",
        resumed_model,
        resumed_optimizer,
        resumed_generator,
        config=config,
        cache_metadata_sha256="cache-hash",
    )
    resumed_metrics = run_training_step(
        resumed_model, examples, resumed_optimizer, resumed_policy, config, resumed_generator
    )

    assert completed_steps == 1
    assert resumed_metrics == pytest.approx(reference_metrics)
    for key, value in resumed_model.state_dict().items():
        assert torch.equal(value, reference_state[key]), key
    assert torch.equal(resumed_model.conditioner.embed.weight[-1], padding_after_first)
    assert torch.equal(resumed_model.conditioner.embed.weight[6], unused_added_after_first)
