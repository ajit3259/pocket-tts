import json

import pytest

from experiments.indic.build_tokenizer_corpus import build_corpora, load_overrides, resolve_record


def _record(example_id: str, text: str, *, split: str = "train") -> dict:
    return {
        "schema_version": 1,
        "example_id": example_id,
        "source_dataset": "rasa",
        "source_license": "CC-BY-4.0",
        "source_split": split,
        "source_locator": {
            "repo_id": "ai4bharat/Rasa",
            "revision": "revision",
            "shard": "Hindi/train.parquet",
            "row_index": 0,
            "audio_column": "audio",
            "format": "hf-parquet-row",
        },
        "speaker_id": "rasa:hindi:male",
        "language_mode": "hi",
        "script_mode": "devanagari",
        "text_raw": text,
        "text_normalized": text,
        "duration_seconds": 1.0,
        "gender": "Male",
        "style": "CONV",
        "source_utterance_id": example_id,
    }


def _override(example_id: str, source_text: str, model_input: str) -> dict:
    return {
        "schema_version": 1,
        "example_id": example_id,
        "issue_kind": "unexpanded-cardinal",
        "source_token": "18",
        "heard_token": "अठारह",
        "source_text": source_text,
        "text_model_input": model_input,
        "decision": "replace-token",
        "review_method": "human-listening",
        "reviewed_on": "2026-07-24",
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_overrides_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "overrides.jsonl"
    override = _override("duplicate", "18 वर्ष", "अठारह वर्ष")
    _write_jsonl(path, [override, override])

    with pytest.raises(ValueError, match="Duplicate override"):
        load_overrides(path)


def test_resolve_record_applies_safe_cleanup_without_override() -> None:
    resolved = resolve_record(_record("clean", "  बाज़ार   खुला है।  "), {})

    assert resolved.record["text_source_normalized"] == "  बाज़ार   खुला है।  "
    assert resolved.record["text_model_input"] == "बाज़ार खुला है।"
    assert resolved.record["normalization_override"] is None
    assert resolved.normalization_change_kinds == ("whitespace",)


def test_resolve_record_rejects_unreviewed_number() -> None:
    with pytest.raises(ValueError, match="Unresolved normalization review"):
        resolve_record(_record("number", "मैं 18 वर्ष का हूँ।"), {})


def test_resolve_record_applies_exact_reviewed_override() -> None:
    source_text = "मैं 18 वर्ष का हूँ।"
    override = _override("number", source_text, "मैं अठारह वर्ष का हूँ।")

    resolved = resolve_record(_record("number", source_text), {"number": override})

    assert resolved.record["text_model_input"] == "मैं अठारह वर्ष का हूँ।"
    assert resolved.record["normalization_override"]["decision"] == "replace-token"
    assert resolved.resolved_review_kinds == ("unexpanded-cardinal",)


def test_resolve_record_rejects_override_for_different_source_text() -> None:
    override = _override("number", "different text", "मैं अठारह वर्ष का हूँ।")

    with pytest.raises(ValueError, match="does not match manifest"):
        resolve_record(_record("number", "मैं 18 वर्ष का हूँ।"), {"number": override})


def test_build_corpora_uses_train_only_and_emits_both_weightings(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    overrides = tmp_path / "overrides.jsonl"
    output = tmp_path / "output"
    _write_jsonl(
        manifest,
        [
            _record("train-1", "साझा वाक्य।"),
            _record("train-2", "साझा वाक्य।"),
            _record("train-3", "अलग वाक्य।"),
            _record("test-1", "परीक्षण वाक्य।", split="test"),
        ],
    )
    _write_jsonl(overrides, [])

    stats = build_corpora(manifest, overrides, output)

    assert stats["records"] == 4
    assert stats["train_records"] == 3
    assert stats["held_out_records"] == 1
    assert stats["train_unique_texts"] == 2
    assert stats["train_duplicate_records"] == 1
    assert (output / "hindi_train_all.txt").read_text(encoding="utf-8").splitlines() == [
        "साझा वाक्य।",
        "साझा वाक्य।",
        "अलग वाक्य।",
    ]
    assert (output / "hindi_train_unique.txt").read_text(encoding="utf-8").splitlines() == [
        "साझा वाक्य।",
        "अलग वाक्य।",
    ]
