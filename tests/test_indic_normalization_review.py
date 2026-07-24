import json

import pytest

from experiments.indic.build_normalization_review import (
    ReviewCandidate,
    _audio_url,
    _review_question,
    find_review_candidates,
)
from experiments.indic.text_normalization import NormalizationReviewItem


def _record(example_id: str, text: str, *, source: str = "rasa", split: str = "train") -> dict:
    return {
        "example_id": example_id,
        "source_dataset": source,
        "source_split": split,
        "text_normalized": text,
    }


def test_review_candidates_retain_offset_within_each_source_split(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        _record("clean-train", "यह साफ़ है।"),
        _record("flagged-train", "फ़ॉर्म 6"),
        _record("clean-test", "यह भी साफ़ है।", split="test"),
        _record("flagged-test", "कुल 18 वर्ष", split="test"),
    ]
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    candidates = find_review_candidates(manifest)

    assert [candidate.manifest_record["example_id"] for candidate in candidates] == [
        "flagged-train",
        "flagged-test",
    ]
    assert [candidate.split_row_index for candidate in candidates] == [1, 1]


def test_audio_url_supports_viewer_list_and_object_shapes() -> None:
    assert _audio_url([{"src": "https://example.test/audio.wav", "type": "audio/wav"}]) == (
        "https://example.test/audio.wav"
    )
    assert _audio_url({"src": "https://example.test/audio.wav"}) == (
        "https://example.test/audio.wav"
    )


def test_audio_url_rejects_missing_asset() -> None:
    with pytest.raises(RuntimeError, match="no downloadable audio"):
        _audio_url([])


def test_number_review_question_requests_exact_spoken_form() -> None:
    candidate = ReviewCandidate(
        manifest_record={},
        split_row_index=0,
        review_items=(NormalizationReviewItem("unexpanded-cardinal", "864"),),
    )

    assert '"864"' in _review_question(candidate.review_items)
    assert "exact words spoken" in _review_question(candidate.review_items)
