import math

from experiments.indic.data_manifest import (
    adapt_indicvoices_r,
    adapt_rasa,
    audit_records,
    script_mode,
)


def test_script_mode_does_not_guess_language() -> None:
    assert script_mode("आज मौसम अच्छा है।") == "devanagari"
    assert script_mode("Aaj weather अच्छा है।") == "mixed-devanagari-latin"
    assert script_mode("Aaj mausam accha hai.") == "latin"
    assert script_mode("१२३") == "devanagari"


def test_adapt_rasa_namespaces_speaker_and_preserves_source_location() -> None:
    record = adapt_rasa(
        {
            "filename": "hindi_male_001.wav",
            "text": "मैं तैयार हूँ।",
            "gender": "Male",
            "style": "CONV",
            "duration": 2.5,
        },
        split="train",
        shard="Hindi/train-00000-of-00025.parquet",
        row_index=7,
    )

    assert record.speaker_id == "rasa:hindi:male"
    assert record.text_raw == "मैं तैयार हूँ।"
    assert record.text_normalized == record.text_raw
    assert record.source_locator.row_index == 7
    assert record.source_locator.audio_column == "audio"
    assert record.source_license == "CC-BY-4.0"


def test_adapt_indicvoices_prefers_verbatim_and_normalized_text() -> None:
    record = adapt_indicvoices_r(
        {
            "text": "fallback",
            "verbatim": "एक सौ तेईस रुपये",
            "normalized": "१२३ रुपये",
            "speaker_id": "speaker-42",
            "scenario": "Extempore",
            "gender": "Female",
            "duration": "3.25",
        },
        split="test",
        shard="Hindi/test-00000-of-00002.parquet",
        row_index=3,
    )

    assert record.speaker_id == "indicvoices_r:speaker-42"
    assert record.text_raw == "एक सौ तेईस रुपये"
    assert record.text_normalized == "१२३ रुपये"
    assert record.duration_seconds == 3.25
    assert record.style == "Extempore"


def test_audit_reports_missing_values_and_prompt_eligible_speakers() -> None:
    first = adapt_rasa(
        {"text": "नमस्ते", "gender": "Male", "duration": 1.0},
        split="train",
        shard="one.parquet",
        row_index=0,
    )
    second = adapt_rasa(
        {"text": "", "gender": "Male", "duration": None},
        split="train",
        shard="one.parquet",
        row_index=1,
    )

    audit = audit_records([first, second])

    assert audit["records"] == 2
    assert audit["speakers"] == 1
    assert audit["speakers_with_multiple_utterances"] == 1
    assert audit["by_source"]["rasa"]["records"] == 2
    assert audit["by_source"]["rasa"]["source_splits"] == {"train": 2}
    assert audit["missing"]["text_raw"] == 1
    assert audit["missing"]["duration"] == 1
    assert math.isclose(audit["duration_hours"], 1 / 3600, abs_tol=0.001)
