import json

import sentencepiece as spm

from experiments.indic.evaluate_tokenizer_candidates import (
    evaluate_texts,
    load_eval_sets,
    vocabulary_summary,
)
from experiments.indic.train_tokenizer_candidates import train_candidate


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(example_id: str, text: str, *, split: str, script_mode: str = "devanagari") -> dict:
    return {
        "example_id": example_id,
        "source_split": split,
        "script_mode": script_mode,
        "text_model_input": text,
    }


def _train_tiny_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    sentences = [
        "आज मौसम बहुत अच्छा है और हम बाहर जाएंगे।",
        "कृपया browser में नया tab खोलें।",
        "This is a clear English replay sentence.",
        "Mera naam Ajit hai aur main Delhi mein rehta hoon.",
        "आपका OTP 1234 है।",
    ]
    corpus.write_text("".join(f"{sentence}\n" for sentence in sentences * 50), encoding="utf-8")
    result = train_candidate(corpus, tmp_path / "model", 340)
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(tmp_path / "model" / result["model_path"])
    )
    return tokenizer, result


def test_train_candidate_preserves_required_special_ids_and_byte_fallback(tmp_path) -> None:
    tokenizer, result = _train_tiny_tokenizer(tmp_path)

    assert tokenizer.vocab_size() == 340
    assert (tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()) == (
        0,
        1,
        2,
        3,
    )
    assert result["byte_pieces"] == 256


def test_evaluate_texts_reports_chunk_pressure_and_byte_usage(tmp_path) -> None:
    tokenizer, _ = _train_tiny_tokenizer(tmp_path)

    metrics = evaluate_texts(
        tokenizer, ["आज मौसम अच्छा है।", "यह एक थोड़ा लंबा परीक्षण वाक्य है।"], max_tokens=5
    )

    assert metrics["records"] == 2
    assert metrics["tokens"] > 0
    assert metrics["tokens_per_nonspace_character"] > 0
    assert metrics["records_over_token_limit"] > 0
    assert metrics["unknown_tokens"] == 0


def test_vocabulary_summary_separates_script_allocation(tmp_path) -> None:
    tokenizer, _ = _train_tiny_tokenizer(tmp_path)

    summary = vocabulary_summary(tokenizer)

    assert summary["learned_pieces"] == 80
    assert summary["byte_pieces"] == 256
    assert summary["pieces_with_devanagari"] > 0
    assert summary["pieces_with_latin"] > 0
    assert summary["pieces_with_digits"] > 0


def test_load_eval_sets_uses_only_held_out_and_deduplicates(tmp_path) -> None:
    hindi = tmp_path / "hindi.jsonl"
    slr104 = tmp_path / "slr104.jsonl"
    libritts = tmp_path / "libritts.jsonl"
    probes = tmp_path / "probes.jsonl"
    _write_jsonl(
        hindi,
        [
            _record("hi-train", "प्रशिक्षण", split="train"),
            _record("hi-test-1", "परीक्षण", split="test"),
            _record("hi-test-2", "परीक्षण", split="test"),
        ],
    )
    _write_jsonl(
        slr104,
        [
            _record("mix-test", "यह browser है", split="test", script_mode="mixed-devanagari-latin"),
            _record("mono-test", "केवल हिंदी", split="test"),
            _record(
                "mix-train", "यह training है", split="train", script_mode="mixed-devanagari-latin"
            ),
        ],
    )
    _write_jsonl(
        libritts,
        [
            _record("en-dev", "Held out English.", split="dev.clean", script_mode="latin"),
            _record("en-train", "Training English.", split="train.clean.100", script_mode="latin"),
        ],
    )
    _write_jsonl(
        probes,
        [
            {"group": "romanized_hindi", "text": "Mera naam Ajit hai."},
            {"group": "numbers", "text": "OTP 1234"},
        ],
    )

    sets = load_eval_sets(hindi, slr104, libritts, probes)

    assert sets["hindi_test"] == ["परीक्षण"]
    assert sets["hinglish_test"] == ["यह browser है"]
    assert sets["english_dev"] == ["Held out English."]
    assert sets["romanized_hindi"] == ["Mera naam Ajit hai."]
    assert sets["numbers"] == ["OTP 1234"]
