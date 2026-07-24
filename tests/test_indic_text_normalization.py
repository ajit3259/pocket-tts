import json

import pytest

from experiments.indic.audit_text_normalization import audit_manifest
from experiments.indic.text_normalization import (
    HINDI_NUMBERS_0_TO_99,
    MAX_CARDINAL,
    hindi_integer_to_words,
    normalize_hindi_text,
)


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, "शून्य"),
        (6, "छह"),
        (18, "अठारह"),
        (23, "तेईस"),
        (64, "चौंसठ"),
        (99, "निन्यानवे"),
        (100, "एक सौ"),
        (123, "एक सौ तेईस"),
        (864, "आठ सौ चौंसठ"),
        (1_000, "एक हज़ार"),
        (1_23_456, "एक लाख तेईस हज़ार चार सौ छप्पन"),
        (1_00_00_000, "एक करोड़"),
        (1_23_45_67_890, "एक सौ तेईस करोड़ पैंतालीस लाख सड़सठ हज़ार आठ सौ नब्बे"),
    ],
)
def test_hindi_integer_to_words(number: int, expected: str) -> None:
    assert hindi_integer_to_words(number) == expected


def test_hindi_lexicon_has_one_unique_entry_per_number() -> None:
    assert len(HINDI_NUMBERS_0_TO_99) == 100
    assert len(set(HINDI_NUMBERS_0_TO_99)) == 100


@pytest.mark.parametrize("number", [-1, MAX_CARDINAL + 1])
def test_hindi_integer_rejects_out_of_range_values(number: int) -> None:
    with pytest.raises(ValueError):
        hindi_integer_to_words(number)


def test_hindi_integer_rejects_non_integer_and_bool() -> None:
    with pytest.raises(TypeError):
        hindi_integer_to_words(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        hindi_integer_to_words(True)


def test_preserve_mode_flags_number_without_changing_training_text() -> None:
    result = normalize_hindi_text("फ़ॉर्म 6", number_mode="preserve")

    assert result.text == "फ़ॉर्म 6"
    assert result.version == "hi-v1"
    assert [(item.kind, item.token) for item in result.review_items] == [
        ("unexpanded-cardinal", "6")
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("123", "एक सौ तेईस"),
        ("१२३", "एक सौ तेईस"),
        ("1,23,456", "एक लाख तेईस हज़ार चार सौ छप्पन"),
        ("123,456", "एक लाख तेईस हज़ार चार सौ छप्पन"),
        ("कुल 864 रुपये", "कुल आठ सौ चौंसठ रुपये"),
    ],
)
def test_explicit_cardinal_mode_expands_supported_integers(text: str, expected: str) -> None:
    result = normalize_hindi_text(text, number_mode="cardinal")

    assert result.text == expected
    assert not result.needs_review
    assert {change.kind for change in result.changes} == {"cardinal"}


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("गाली0गलौज", "embedded-digit"),
        ("₹123", "unsupported-number-context"),
        ("12.5", "unsupported-number-context"),
        ("25%", "unsupported-number-context"),
        ("-12", "unsupported-number-context"),
        ("007", "invalid-cardinal"),
        ("12,34", "invalid-cardinal"),
    ],
)
def test_ambiguous_numeric_context_is_preserved_for_review(text: str, kind: str) -> None:
    result = normalize_hindi_text(text, number_mode="cardinal")

    assert result.text == text
    assert [item.kind for item in result.review_items] == [kind]


def test_cleanup_is_auditable() -> None:
    result = normalize_hindi_text("  cafe\u0301   तैयार है  ")

    assert result.text == "café तैयार है"
    assert [change.kind for change in result.changes] == ["unicode-nfc", "whitespace"]


def test_hinglish_words_are_preserved() -> None:
    text = "Aaj meeting 6 baje hai"
    result = normalize_hindi_text(text, number_mode="preserve")

    assert result.text == text
    assert result.review_items[0].kind == "unexpanded-cardinal"


def test_possible_punctuation_i_is_flagged_but_not_changed() -> None:
    text = "काम पूरा हुआI अब घर चलें।"
    result = normalize_hindi_text(text)

    assert result.text == text
    assert [(item.kind, item.token) for item in result.review_items] == [
        ("possible-punctuation-I", "I")
    ]


def test_manifest_audit_counts_changes_and_reviews(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"example_id": "clean", "text_normalized": "यह साफ़ है।"},
        {"example_id": "number", "text_normalized": "कुल 123 रुपये"},
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    audit = audit_manifest(manifest, number_mode="preserve", examples_per_kind=2)

    assert audit["normalizer_version"] == "hi-v1"
    assert audit["rows"] == 2
    assert audit["review_rows"] == 1
    assert audit["review_kinds"] == {"unexpanded-cardinal": 1}
