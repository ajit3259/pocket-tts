import hashlib
import json

import pytest

from experiments.indic.build_tokenizer_mixture import build_mixture
from experiments.indic.data_manifest import script_mode


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(
    example_id: str, text: str, *, source: str, split: str = "train", language_mode: str
) -> dict:
    return {
        "schema_version": 1,
        "example_id": example_id,
        "source_dataset": source,
        "source_license": "test-license",
        "source_split": split,
        "source_utterance_id": example_id,
        "language_mode": language_mode,
        "script_mode": script_mode(text),
        "text_model_input": text,
    }


def _fixture_manifests(tmp_path):
    hindi = tmp_path / "hindi.jsonl"
    hinglish = tmp_path / "hinglish.jsonl"
    english = tmp_path / "english.jsonl"
    _write_jsonl(
        hindi,
        [
            _record("hi-1", "आज मौसम अच्छा है", source="hindi", language_mode="hi"),
            _record("hi-2", "कृपया दरवाज़ा खोलें", source="hindi", language_mode="hi"),
            _record("hi-duplicate", "आज मौसम अच्छा है", source="hindi", language_mode="hi"),
            _record(
                "hi-test",
                "यह परीक्षण में नहीं आना चाहिए",
                source="hindi",
                split="test",
                language_mode="hi",
            ),
        ],
    )
    _write_jsonl(
        hinglish,
        [
            _record(
                f"mix-{index}",
                f"कृपया browser tab number {index} खोलें",
                source="slr104",
                language_mode="hi-en",
            )
            for index in range(8)
        ]
        + [
            _record("mono-hi", "सिर्फ हिंदी वाक्य", source="slr104", language_mode="hi"),
            _record(
                "mix-test", "यह test split है", source="slr104", split="test", language_mode="hi-en"
            ),
        ],
    )
    _write_jsonl(
        english,
        [
            _record(
                f"en-{index}",
                f"This is replay sentence number {index}.",
                source="libritts_r",
                language_mode="en",
            )
            for index in range(8)
        ],
    )
    return hindi, hinglish, english


def test_build_mixture_is_deterministic_and_excludes_noneligible_rows(tmp_path) -> None:
    hindi, hinglish, english = _fixture_manifests(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    weights = {"hi": 0.5, "hi-en": 0.25, "en": 0.25}

    first_stats = build_mixture(hindi, hinglish, english, first, weights=weights, seed=42)
    second_stats = build_mixture(hindi, hinglish, english, second, weights=weights, seed=42)

    assert first_stats == second_stats
    assert (first / "tokenizer_train.txt").read_bytes() == (
        second / "tokenizer_train.txt"
    ).read_bytes()
    assert (first / "mixture_manifest.jsonl").read_bytes() == (
        second / "mixture_manifest.jsonl"
    ).read_bytes()
    assert first_stats["selected_records"]["hi"] == 2
    corpus = (first / "tokenizer_train.txt").read_text(encoding="utf-8")
    assert "सिर्फ हिंदी वाक्य" not in corpus
    assert "यह test split है" not in corpus
    assert "यह परीक्षण में नहीं आना चाहिए" not in corpus


def test_build_mixture_fails_when_a_stream_cannot_meet_budget(tmp_path) -> None:
    hindi, hinglish, english = _fixture_manifests(tmp_path)

    with pytest.raises(ValueError, match="below target"):
        build_mixture(
            hindi,
            hinglish,
            english,
            tmp_path / "output",
            weights={"hi": 0.01, "hi-en": 0.49, "en": 0.5},
        )


def test_build_mixture_rejects_invalid_weights(tmp_path) -> None:
    hindi, hinglish, english = _fixture_manifests(tmp_path)

    with pytest.raises(ValueError, match="sum to 1"):
        build_mixture(
            hindi,
            hinglish,
            english,
            tmp_path / "output",
            weights={"hi": 0.6, "hi-en": 0.25, "en": 0.2},
        )


def test_build_mixture_rejects_unsafe_invisible_unicode(tmp_path) -> None:
    hindi, hinglish, english = _fixture_manifests(tmp_path)
    _write_jsonl(hindi, [_record("bad-format", "फ़े\u180eब्रुअरि", source="hindi", language_mode="hi")])

    with pytest.raises(ValueError, match="MONGOLIAN VOWEL SEPARATOR"):
        build_mixture(
            hindi,
            hinglish,
            english,
            tmp_path / "output",
            weights={"hi": 0.5, "hi-en": 0.25, "en": 0.25},
        )


def test_build_mixture_applies_exact_versioned_exclusion(tmp_path) -> None:
    hindi, hinglish, english = _fixture_manifests(tmp_path)
    unsafe_text = "फ़े\u180eब्रुअरि"
    _write_jsonl(
        hindi,
        [
            _record("bad-format", unsafe_text, source="hindi", language_mode="hi"),
            _record("hi-clean", "साफ हिंदी पाठ", source="hindi", language_mode="hi"),
        ],
    )
    exclusions = tmp_path / "exclusions.jsonl"
    _write_jsonl(
        exclusions,
        [
            {
                "schema_version": 1,
                "example_id": "bad-format",
                "issue_kind": "corrupt-unicode-sequence",
                "text_model_input_sha256": hashlib.sha256(unsafe_text.encode()).hexdigest(),
            }
        ],
    )

    stats = build_mixture(
        hindi,
        hinglish,
        english,
        tmp_path / "output",
        weights={"hi": 0.5, "hi-en": 0.25, "en": 0.25},
        exclusions_path=exclusions,
    )

    assert stats["exclusions_applied"] == 1
    assert "फ़े" not in (tmp_path / "output" / "tokenizer_train.txt").read_text(encoding="utf-8")
