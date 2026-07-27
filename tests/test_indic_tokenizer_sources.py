import io
import unicodedata

import pytest

from experiments.indic.tokenizer_sources import (
    adapt_libritts_r_record,
    adapt_slr104_record,
    audit_text_records,
    iter_slr104_csv,
    normalize_tokenizer_text,
)


def test_normalize_tokenizer_text_only_changes_representation() -> None:
    decomposed = unicodedata.normalize("NFD", "बाज़ार")

    assert normalize_tokenizer_text(f"  {decomposed}   खुला है  ") == "बाज़ार खुला है"


def test_iter_slr104_csv_preserves_valid_commaless_transcript() -> None:
    rows = list(
        iter_slr104_csv(
            io.StringIO("train/example.wav,यहाँ browser open करें\n"), source_name="fixture.csv"
        )
    )

    assert rows == [(0, "train/example.wav", "यहाँ browser open करें")]


def test_iter_slr104_csv_rejects_ambiguous_extra_column() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        list(
            iter_slr104_csv(
                io.StringIO("train/example.wav,यहाँ browser, open करें\n"), source_name="fixture.csv"
            )
        )


def test_adapt_slr104_record_labels_actual_script_mix_and_provenance() -> None:
    record = adapt_slr104_record(
        split="train",
        metadata_file="metadata.csv",
        row_index=7,
        audio_path="train/example_0007.wav",
        text="यहाँ browser open करें",
    )

    assert record["language_mode"] == "hi-en"
    assert record["script_mode"] == "mixed-devanagari-latin"
    assert record["source_license"] == "CC-BY-SA-4.0"
    assert record["source_utterance_id"] == "example_0007"
    assert record["source_locator"]["transport_revision"]
    assert record["source_locator"]["row_index"] == 7


def test_adapt_libritts_record_uses_tts_normalized_text() -> None:
    record = adapt_libritts_r_record(
        {
            "id": "100_200_000001_000001",
            "text_original": "Mr. Smith paid $2.",
            "text_normalized": "Mister Smith paid two dollars.",
            "speaker_id": 100,
            "chapter_id": 200,
        },
        split="dev.clean",
        shard="clean/dev.clean/0000.parquet",
        row_index=3,
    )

    assert record["text_raw"] == "Mr. Smith paid $2."
    assert record["text_model_input"] == "Mister Smith paid two dollars."
    assert record["language_mode"] == "en"
    assert record["source_split"] == "dev.clean"
    assert record["source_locator"]["parquet_revision"]


def test_audit_text_records_reports_duplicates_and_modes() -> None:
    first = adapt_slr104_record(
        split="train",
        metadata_file="metadata.csv",
        row_index=0,
        audio_path="train/one.wav",
        text="यहाँ browser खोलें",
    )
    second = adapt_slr104_record(
        split="train",
        metadata_file="metadata.csv",
        row_index=1,
        audio_path="train/two.wav",
        text="यहाँ browser खोलें",
    )

    audit = audit_text_records([first, second])

    assert audit["records"] == 2
    assert audit["unique_texts"] == 1
    assert audit["duplicate_records"] == 1
    assert audit["language_modes"] == {"hi-en": 2}
