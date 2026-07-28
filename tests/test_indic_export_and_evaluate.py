import json

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.indic.export_and_evaluate import (
    _build_reviews,
    build_evaluation_items,
    deterministic_results_sha256,
    export_training_checkpoint,
    sha256_file,
    smoke_evaluation_items,
)


def _write_export_inputs(tmp_path):
    base_model = tmp_path / "base.safetensors"
    base_state = {
        "flow_lm.a": torch.tensor([1.0, 2.0]),
        "flow_lm.b": torch.tensor([[3.0]]),
        "mimi.encoder.weight": torch.tensor([4.0, 5.0]),
    }
    save_file(base_state, base_model)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "weights_path": str(base_model),
                "weights_path_without_voice_cloning": None,
                "flow_lm": {"lookup_table": {"tokenizer_path": "tokenizer.model"}},
            }
        ),
        encoding="utf-8",
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    training_model = checkpoint_dir / "flow_lm.safetensors"
    save_file(
        {
            "a": torch.tensor([10.0, 20.0]),
            "b": torch.tensor([[30.0]]),
        },
        training_model,
    )
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "completed_steps": 7,
                "model_sha256": sha256_file(training_model),
                "training_config": {"loss_combination": "sample_mean"},
            }
        ),
        encoding="utf-8",
    )
    return base_state, base_config, checkpoint_dir


def _packet_records():
    return [
        {
            "example_id": "prompt-female",
            "role": "prompt",
            "pair_index": 0,
            "speaker_id": "rasa:hindi:female",
            "audio_file": "prompt-female.wav",
        },
        {
            "example_id": "prompt-male",
            "role": "prompt",
            "pair_index": 0,
            "speaker_id": "rasa:hindi:male",
            "audio_file": "prompt-male.wav",
        },
        {
            "example_id": "target-female",
            "role": "target",
            "pair_index": 1,
            "speaker_id": "rasa:hindi:female",
            "prompt_example_id": "prompt-female",
            "text_model_input": "साझा प्रशिक्षण वाक्य।",
        },
        {
            "example_id": "target-male",
            "role": "target",
            "pair_index": 1,
            "speaker_id": "rasa:hindi:male",
            "prompt_example_id": "prompt-male",
            "text_model_input": "साझा प्रशिक्षण वाक्य।",
        },
    ]


def test_export_replaces_every_flow_tensor_and_preserves_mimi_bit_exact(tmp_path) -> None:
    base_state, base_config, checkpoint_dir = _write_export_inputs(tmp_path)

    metadata = export_training_checkpoint(
        checkpoint_dir,
        base_config,
        tmp_path / "export",
        verify_stock_load=False,
    )
    exported = load_file(tmp_path / "export" / "model.safetensors")
    exported_config = yaml.safe_load(
        (tmp_path / "export" / "config.yaml").read_text(encoding="utf-8")
    )

    assert torch.equal(exported["flow_lm.a"], torch.tensor([10.0, 20.0]))
    assert torch.equal(exported["flow_lm.b"], torch.tensor([[30.0]]))
    assert torch.equal(exported["mimi.encoder.weight"], base_state["mimi.encoder.weight"])
    assert metadata["flow_lm_tensors_replaced"] == 2
    assert metadata["non_flow_tensors_preserved"] == 1
    assert metadata["strict_stock_load"] is False
    assert exported_config["weights_path"] == str(
        (tmp_path / "export" / "model.safetensors").resolve()
    )
    assert exported_config["weights_path_without_voice_cloning"] == (
        exported_config["weights_path"]
    )


def test_export_rejects_an_incomplete_trainer_state(tmp_path) -> None:
    _, base_config, checkpoint_dir = _write_export_inputs(tmp_path)
    training_model = checkpoint_dir / "flow_lm.safetensors"
    save_file({"a": torch.tensor([10.0, 20.0])}, training_model)
    metadata = json.loads(
        (checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    metadata["model_sha256"] = sha256_file(training_model)
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Trainer FlowLM keys differ"):
        export_training_checkpoint(
            checkpoint_dir,
            base_config,
            tmp_path / "export",
            verify_stock_load=False,
        )


def test_evaluation_crosses_unseen_controls_with_both_fixed_prompts(tmp_path) -> None:
    probes = [
        {
            "probe_id": "unseen-hi",
            "language_mode": "hi",
            "text_model_input": "यह नया वाक्य है।",
        },
        {
            "probe_id": "unseen-mixed",
            "language_mode": "hinglish",
            "text_model_input": "आज weather अच्छा है।",
        },
    ]

    items = build_evaluation_items(_packet_records(), probes, tmp_path)

    assert len(items) == 6
    assert sum(item.category == "overfit" for item in items) == 2
    assert sum(item.category == "control" for item in items) == 4
    assert {item.speaker_id for item in items if item.category == "control"} == {
        "rasa:hindi:female",
        "rasa:hindi:male",
    }
    assert {item.reference_example_id for item in items if item.category == "overfit"} == {
        "target-female",
        "target-male",
    }
    smoke = smoke_evaluation_items(items)
    assert [item.category for item in smoke] == [
        "overfit",
        "overfit",
        "control",
        "control",
    ]


def test_control_text_cannot_duplicate_the_memorization_set(tmp_path) -> None:
    probes = [
        {
            "probe_id": "leaked",
            "language_mode": "hi",
            "text_model_input": "साझा प्रशिक्षण वाक्य।",
        }
    ]

    with pytest.raises(ValueError, match="duplicates a training text"):
        build_evaluation_items(_packet_records(), probes, tmp_path)


def test_generation_review_survives_only_for_identical_audio() -> None:
    result = {
        "item_id": "item",
        "generated_audio_file": "item.wav",
        "generated_audio_sha256": "same",
    }
    existing = {
        "item": {
            "item_id": "item",
            "generated_audio_sha256": "same",
            "decision": "accepted",
            "intelligible": True,
            "transcript_matches": True,
            "prompt_voice_matches": True,
            "notes": "Reviewed.",
        }
    }

    preserved = _build_reviews([result], existing)[0]
    reset = _build_reviews(
        [{**result, "generated_audio_sha256": "changed"}], existing
    )[0]

    assert preserved["decision"] == "accepted"
    assert preserved["transcript_matches"] is True
    assert reset["decision"] == "pending"
    assert reset["transcript_matches"] is None


def test_deterministic_result_hash_excludes_only_wall_clock_time() -> None:
    first = [
        {
            "item_id": "item",
            "generated_audio_sha256": "audio",
            "duration_seconds": 1.0,
            "generation_seconds": 0.1,
        }
    ]
    second = [{**first[0], "generation_seconds": 99.0}]
    changed = [{**first[0], "generated_audio_sha256": "different"}]

    assert deterministic_results_sha256(first) == deterministic_results_sha256(second)
    assert deterministic_results_sha256(first) != deterministic_results_sha256(changed)
