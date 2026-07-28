"""Export trainer checkpoints and run fixed Hindi/Hinglish generation evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.io.wavfile
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.indic.tiny_overfit_training import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PACKET_DIR,
    load_reviewed_packet,
)
from pocket_tts import TTSModel
from pocket_tts.utils.utils import download_if_necessary

DEFAULT_TRAINING_CHECKPOINT = (
    Path(__file__).with_name("outputs") / "e14_cpu_smoke" / "step_000001"
)
DEFAULT_EXPORT_DIR = Path(__file__).with_name("outputs") / "e15_export_cpu_smoke"
DEFAULT_EVALUATION_DIR = Path(__file__).with_name("outputs") / "e15_generation_cpu_smoke"
DEFAULT_PROBES = Path(__file__).with_name("generation_eval_probes.jsonl")
GENERATION_SEED = 20260727


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    category: str
    speaker_id: str
    prompt_example_id: str
    prompt_audio_path: Path
    text_model_input: str
    language_mode: str
    reference_example_id: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_results_sha256(results: list[dict[str, Any]]) -> str:
    deterministic_results = [
        {key: value for key, value in result.items() if key != "generation_seconds"}
        for result in results
    ]
    serialized = json.dumps(
        deterministic_results,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            records.append(record)
    return records


def _checkpoint_path_from_config(config: dict[str, Any]) -> Path:
    weights_path = config.get("weights_path")
    if not weights_path:
        raise ValueError("Base config has no weights_path")
    path = Path(download_if_necessary(weights_path))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def export_training_checkpoint(
    training_checkpoint_dir: Path,
    base_config_path: Path,
    output_dir: Path,
    *,
    verify_stock_load: bool = True,
) -> dict[str, Any]:
    """Merge a FlowLM-only trainer checkpoint into the complete TTS state."""

    checkpoint_metadata_path = training_checkpoint_dir / "checkpoint.json"
    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    training_model_path = training_checkpoint_dir / "flow_lm.safetensors"
    if checkpoint_metadata.get("model_sha256") != sha256_file(training_model_path):
        raise ValueError("Trainer checkpoint model hash does not match checkpoint.json")

    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    base_model_path = _checkpoint_path_from_config(base_config)
    base_state = load_file(base_model_path, device="cpu")
    training_state = load_file(training_model_path, device="cpu")

    base_flow_keys = {key.removeprefix("flow_lm.") for key in base_state if key.startswith("flow_lm.")}
    if set(training_state) != base_flow_keys:
        missing = sorted(base_flow_keys - set(training_state))
        extra = sorted(set(training_state) - base_flow_keys)
        raise ValueError(f"Trainer FlowLM keys differ; missing={missing}, extra={extra}")

    exported = dict(base_state)
    for key, tensor in training_state.items():
        target_key = f"flow_lm.{key}"
        if tensor.shape != base_state[target_key].shape:
            raise ValueError(f"Trainer tensor shape differs for {target_key}")
        exported[target_key] = tensor.contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_model_path = output_dir / "model.safetensors"
    temporary_model_path = output_dir / "model.safetensors.tmp"
    save_file(exported, temporary_model_path)
    temporary_model_path.replace(output_model_path)

    reloaded = load_file(output_model_path, device="cpu")
    for key, tensor in base_state.items():
        expected = exported[key] if key.startswith("flow_lm.") else tensor
        if not torch.equal(reloaded[key], expected):
            raise RuntimeError(f"Export verification failed for {key}")

    export_config = dict(base_config)
    export_config["weights_path"] = str(output_model_path.resolve())
    export_config["weights_path_without_voice_cloning"] = str(output_model_path.resolve())
    output_config_path = output_dir / "config.yaml"
    output_config_path.write_text(
        yaml.safe_dump(export_config, sort_keys=False), encoding="utf-8"
    )

    strict_load = False
    if verify_stock_load:
        loaded_model = TTSModel.load_model(config=output_config_path)
        if set(loaded_model.state_dict()) != set(reloaded):
            raise RuntimeError("Stock TTSModel state keys differ from the exported checkpoint")
        strict_load = True
        del loaded_model

    non_flow_keys = [key for key in base_state if not key.startswith("flow_lm.")]
    metadata = {
        "schema_version": 1,
        "completed_training_steps": checkpoint_metadata["completed_steps"],
        "training_config": checkpoint_metadata["training_config"],
        "training_checkpoint_metadata_sha256": sha256_file(checkpoint_metadata_path),
        "training_flow_lm_sha256": sha256_file(training_model_path),
        "base_config_sha256": sha256_file(base_config_path),
        "base_model_sha256": sha256_file(base_model_path),
        "exported_model_sha256": sha256_file(output_model_path),
        "exported_config_sha256": sha256_file(output_config_path),
        "flow_lm_tensors_replaced": len(training_state),
        "non_flow_tensors_preserved": len(non_flow_keys),
        "strict_stock_load": strict_load,
        "exported_model_path": output_model_path.name,
        "exported_config_path": output_config_path.name,
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_evaluation_probes(path: Path) -> list[dict[str, Any]]:
    probes = _read_jsonl(path)
    seen_ids: set[str] = set()
    for probe in probes:
        probe_id = probe["probe_id"]
        if probe_id in seen_ids:
            raise ValueError(f"Duplicate generation probe: {probe_id}")
        seen_ids.add(probe_id)
        if probe.get("language_mode") not in {"hi", "hinglish"}:
            raise ValueError(f"Unknown probe language mode: {probe_id}")
        if not probe.get("text_model_input"):
            raise ValueError(f"Empty generation probe: {probe_id}")
    return probes


def build_evaluation_items(
    records: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    packet_dir: Path,
) -> list[EvaluationItem]:
    prompts = {
        record["speaker_id"]: record for record in records if record["role"] == "prompt"
    }
    targets = [record for record in records if record["role"] == "target"]
    target_texts = {record["text_model_input"] for record in targets}
    for probe in probes:
        if probe["text_model_input"] in target_texts:
            raise ValueError(f"Control probe duplicates a training text: {probe['probe_id']}")

    items: list[EvaluationItem] = []
    for target in sorted(
        targets, key=lambda record: (record["pair_index"], record["speaker_id"])
    ):
        prompt = prompts[target["speaker_id"]]
        speaker_slug = target["speaker_id"].rsplit(":", maxsplit=1)[-1]
        items.append(
            EvaluationItem(
                item_id=f"overfit_{target['pair_index']:02d}_{speaker_slug}",
                category="overfit",
                speaker_id=target["speaker_id"],
                prompt_example_id=prompt["example_id"],
                prompt_audio_path=packet_dir / prompt["audio_file"],
                text_model_input=target["text_model_input"],
                language_mode="hi",
                reference_example_id=target["example_id"],
            )
        )

    for probe in probes:
        for speaker_id, prompt in sorted(prompts.items()):
            speaker_slug = speaker_id.rsplit(":", maxsplit=1)[-1]
            items.append(
                EvaluationItem(
                    item_id=f"{probe['probe_id']}_{speaker_slug}",
                    category="control",
                    speaker_id=speaker_id,
                    prompt_example_id=prompt["example_id"],
                    prompt_audio_path=packet_dir / prompt["audio_file"],
                    text_model_input=probe["text_model_input"],
                    language_mode=probe["language_mode"],
                    reference_example_id=None,
                )
            )
    return items


def smoke_evaluation_items(items: list[EvaluationItem]) -> list[EvaluationItem]:
    overfit = [item for item in items if item.category == "overfit"][:2]
    controls = [item for item in items if item.category == "control"][:2]
    if len(overfit) != 2 or len(controls) != 2:
        raise ValueError("Smoke evaluation requires two overfit and two control items")
    return [*overfit, *controls]


def _generation_frames(model: TTSModel, samples: int) -> int:
    samples_per_frame = model.config.mimi.sample_rate / model.config.mimi.frame_rate
    return round(samples / samples_per_frame)


def _load_existing_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    reviews = json.loads(path.read_text(encoding="utf-8"))
    return {review["item_id"]: review for review in reviews}


def _build_reviews(
    results: list[dict[str, Any]], existing: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    reviews = []
    for result in results:
        previous = existing.get(result["item_id"])
        preserve = (
            previous is not None
            and previous.get("generated_audio_sha256")
            == result["generated_audio_sha256"]
        )
        reviews.append(
            {
                "schema_version": 1,
                "item_id": result["item_id"],
                "generated_audio_file": result["generated_audio_file"],
                "generated_audio_sha256": result["generated_audio_sha256"],
                "decision": previous.get("decision", "pending") if preserve else "pending",
                "intelligible": previous.get("intelligible") if preserve else None,
                "transcript_matches": (
                    previous.get("transcript_matches") if preserve else None
                ),
                "prompt_voice_matches": (
                    previous.get("prompt_voice_matches") if preserve else None
                ),
                "notes": previous.get("notes", "") if preserve else "",
            }
        )
    return reviews


def _review_markdown(
    results: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> str:
    reviews_by_id = {review["item_id"]: review for review in reviews}
    lines = [
        "# E15 Generation Review",
        "",
        "For each clip, judge intelligibility, exact text coverage, and whether the voice",
        "matches the corresponding fixed E13 prompt.",
        "",
    ]
    for result in results:
        review = reviews_by_id[result["item_id"]]
        lines.extend(
            [
                f"## {result['item_id']}",
                "",
                f"- Category: `{result['category']}`",
                f"- Speaker: `{result['speaker_id']}`",
                f"- Audio: [{result['generated_audio_file']}]"
                f"({result['generated_audio_file']})",
                f"- Text: {result['text_model_input']}",
                f"- Generated duration: `{result['duration_seconds']:.3f}s`",
                f"- Likely hit maximum frames: `{result['likely_hit_max_generation']}`",
                f"- Decision: `{review['decision']}`",
                f"- Notes: {review['notes']}",
                "",
            ]
        )
    return "\n".join(lines)


@torch.no_grad()
def run_generation_evaluation(
    model_config_path: Path,
    packet_dir: Path,
    probes_path: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    records = load_reviewed_packet(packet_dir)
    probes = load_evaluation_probes(probes_path)
    items = build_evaluation_items(records, probes, packet_dir)
    if smoke:
        items = smoke_evaluation_items(items)

    model = TTSModel.load_model(config=model_config_path)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_states: dict[str, dict] = {}
    results = []
    for index, item in enumerate(items):
        if item.prompt_example_id not in prompt_states:
            prompt_states[item.prompt_example_id] = model.get_state_for_audio_prompt(
                item.prompt_audio_path
            )
        token_count = model.flow_lm.conditioner.tokenizer(
            item.text_model_input
        ).tokens.shape[1]
        if token_count > 50:
            raise ValueError(f"Evaluation item exceeds one generation chunk: {item.item_id}")
        estimated_max_frames = math.ceil(
            (token_count / model._TOKENS_PER_SECOND_ESTIMATE + model._GEN_SECONDS_PADDING)
            * model.config.mimi.frame_rate
        )
        seed = GENERATION_SEED + index
        torch.manual_seed(seed)
        started = time.monotonic()
        audio = model.generate_audio(
            prompt_states[item.prompt_example_id],
            item.text_model_input,
            copy_state=True,
        )
        elapsed = time.monotonic() - started
        if audio.numel() < 1 or not torch.isfinite(audio).all():
            raise RuntimeError(f"Generation produced invalid audio: {item.item_id}")
        filename = f"{item.item_id}.wav"
        output_path = output_dir / filename
        scipy.io.wavfile.write(
            output_path,
            model.sample_rate,
            audio.detach().cpu().to(torch.float32).numpy(),
        )
        generated_frames = _generation_frames(model, audio.numel())
        results.append(
            {
                "schema_version": 1,
                "item_id": item.item_id,
                "category": item.category,
                "speaker_id": item.speaker_id,
                "prompt_example_id": item.prompt_example_id,
                "reference_example_id": item.reference_example_id,
                "language_mode": item.language_mode,
                "text_model_input": item.text_model_input,
                "generation_seed": seed,
                "token_count": token_count,
                "generated_audio_file": filename,
                "generated_audio_sha256": sha256_file(output_path),
                "samples": audio.numel(),
                "duration_seconds": round(audio.numel() / model.sample_rate, 3),
                "generated_frames": generated_frames,
                "estimated_max_frames": estimated_max_frames,
                "likely_hit_max_generation": generated_frames >= estimated_max_frames,
                "generation_seconds": round(elapsed, 3),
            }
        )

    result_path = output_dir / "evaluation.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_path = output_dir / "review.json"
    reviews = _build_reviews(results, _load_existing_reviews(review_path))
    review_path.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REVIEW.md").write_text(
        _review_markdown(results, reviews), encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "model_config_sha256": sha256_file(model_config_path),
        "model_checkpoint_sha256": sha256_file(
            Path(download_if_necessary(model.config.weights_path))
        ),
        "packet_manifest_sha256": sha256_file(packet_dir / "manifest.jsonl"),
        "packet_review_sha256": sha256_file(packet_dir / "review.json"),
        "probes_sha256": sha256_file(probes_path),
        "generation_seed": GENERATION_SEED,
        "temperature": model.temp,
        "lsd_decode_steps": model.lsd_decode_steps,
        "eos_threshold": model.eos_threshold,
        "smoke": smoke,
        "items": len(results),
        "overfit_items": sum(result["category"] == "overfit" for result in results),
        "control_items": sum(result["category"] == "control" for result in results),
        "likely_hit_max_generation": sum(
            result["likely_hit_max_generation"] for result in results
        ),
        "human_review": {
            "complete": all(review["decision"] != "pending" for review in reviews),
            "decisions": dict(
                sorted(Counter(review["decision"] for review in reviews).items())
            ),
        },
        "deterministic_results_sha256": deterministic_results_sha256(results),
        "evaluation_sha256": sha256_file(result_path),
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument(
        "--training-checkpoint", type=Path, default=DEFAULT_TRAINING_CHECKPOINT
    )
    export.add_argument("--base-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    export.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    export.add_argument("--skip-stock-load", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--model-config", type=Path, default=DEFAULT_EXPORT_DIR / "config.yaml"
    )
    evaluate.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    evaluate.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    evaluate.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        metadata = export_training_checkpoint(
            args.training_checkpoint,
            args.base_config,
            args.output_dir,
            verify_stock_load=not args.skip_stock_load,
        )
    else:
        metadata = run_generation_evaluation(
            args.model_config,
            args.packet_dir,
            args.probes,
            args.output_dir,
            smoke=args.smoke,
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
