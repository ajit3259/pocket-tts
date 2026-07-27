import json
import wave

import pytest

from experiments.indic.build_tiny_overfit_packet import (
    DEFAULT_SPEAKERS,
    PacketCandidate,
    build_review_records,
    inspect_audio_file,
    load_candidates,
    select_packet,
    validate_audio_metadata,
)


def _record(
    example_id: str,
    text: str,
    speaker: str,
    *,
    duration: float = 4.0,
    source: str = "rasa",
    split: str = "train",
) -> dict:
    return {
        "example_id": example_id,
        "source_dataset": source,
        "source_split": split,
        "speaker_id": speaker,
        "script_mode": "devanagari",
        "duration_seconds": duration,
        "text_model_input": text,
        "text_source_normalized": text,
        "normalization_changes": [],
        "normalization_override": None,
        "source_utterance_id": f"{example_id}.wav",
    }


def _candidate(example_id: str, text: str, speaker: str, index: int) -> PacketCandidate:
    return PacketCandidate(record=_record(example_id, text, speaker), split_row_index=index)


def test_load_candidates_retains_split_offsets_and_applies_strict_filters(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    records = [
        _record("other-source", "पहली पंक्ति", DEFAULT_SPEAKERS[0], source="other"),
        _record("female-clean", "साफ़ वाक्य", DEFAULT_SPEAKERS[0]),
        _record("male-digit", "फ़ॉर्म ६", DEFAULT_SPEAKERS[1]),
        _record("male-held-out", "दूसरा वाक्य", DEFAULT_SPEAKERS[1], split="test"),
        _record("male-clean", "तीसरा वाक्य", DEFAULT_SPEAKERS[1]),
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    candidates = load_candidates(path)

    assert [candidate.record["example_id"] for candidate in candidates] == [
        "female-clean",
        "male-clean",
    ]
    assert [candidate.split_row_index for candidate in candidates] == [0, 2]


def test_select_packet_uses_shared_text_and_distinct_same_speaker_prompts() -> None:
    candidates = []
    texts = [f"साझा वाक्य {word}" for word in ("एक", "दो", "तीन", "चार")]
    for index, text in enumerate(texts):
        for speaker_index, speaker in enumerate(DEFAULT_SPEAKERS):
            candidates.append(
                _candidate(f"{speaker_index}-{index}", text, speaker, index * 2 + speaker_index)
            )

    selections = select_packet(candidates, target_pairs=2, seed="test-seed")

    prompts = [selection for selection in selections if selection.role == "prompt"]
    targets = [selection for selection in selections if selection.role == "target"]
    assert len(prompts) == 2
    assert len(targets) == 4
    assert {selection.candidate.record["text_model_input"] for selection in prompts} == {
        prompts[0].candidate.record["text_model_input"]
    }
    assert all(selection.prompt_example_id for selection in targets)
    assert all(
        selection.candidate.record["example_id"] != selection.prompt_example_id
        for selection in targets
    )
    for pair_index in (1, 2):
        pair = [selection for selection in targets if selection.pair_index == pair_index]
        assert {selection.speaker_id for selection in pair} == set(DEFAULT_SPEAKERS)
        assert len({selection.candidate.record["text_model_input"] for selection in pair}) == 1


def test_select_packet_is_deterministic_for_reordered_input() -> None:
    candidates = [
        _candidate(f"{speaker}-{index}", f"वाक्य {index}", speaker, index)
        for index in range(5)
        for speaker in DEFAULT_SPEAKERS
    ]

    forward = select_packet(candidates, target_pairs=2, seed="stable")
    reverse = select_packet(list(reversed(candidates)), target_pairs=2, seed="stable")

    assert [selection.candidate.record["example_id"] for selection in forward] == [
        selection.candidate.record["example_id"] for selection in reverse
    ]


def test_audio_validation_checks_mono_rate_and_manifest_duration() -> None:
    record = {"example_id": "example", "duration_seconds": 4.0}
    valid = {
        "channels": 1,
        "sample_rate": 48_000,
        "sample_width_bytes": 2,
        "duration_seconds": 4.02,
    }

    validate_audio_metadata(record, valid)
    with pytest.raises(RuntimeError, match="not mono"):
        validate_audio_metadata(record, {**valid, "channels": 2})
    with pytest.raises(RuntimeError, match="duration differs"):
        validate_audio_metadata(record, {**valid, "duration_seconds": 4.2})


def test_inspect_audio_file_supports_resuming_a_partial_download(tmp_path) -> None:
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\0\0" * 12_000)

    metadata = inspect_audio_file(path)

    assert metadata["channels"] == 1
    assert metadata["sample_rate"] == 24_000
    assert metadata["duration_seconds"] == 0.5
    assert len(metadata["sha256"]) == 64


def test_review_decision_survives_rebuild_only_for_identical_audio() -> None:
    entry = {
        "example_id": "example",
        "audio_file": "example.wav",
        "audio": {"sha256": "same"},
        "role": "prompt",
        "pair_index": 0,
        "speaker_id": DEFAULT_SPEAKERS[0],
    }
    accepted = {
        "example_id": "example",
        "audio_sha256": "same",
        "decision": "accepted",
        "audio_clean": True,
        "transcript_matches": True,
        "voice_stable": True,
        "notes": "Human reviewed.",
    }

    preserved = build_review_records([entry], {"example": accepted})[0]
    reset = build_review_records(
        [{**entry, "audio": {"sha256": "changed"}}], {"example": accepted}
    )[0]

    assert preserved["decision"] == "accepted"
    assert preserved["audio_clean"] is True
    assert reset["decision"] == "pending"
    assert reset["audio_clean"] is None
