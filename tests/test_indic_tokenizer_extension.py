from pathlib import Path

import sentencepiece as spm

from experiments.indic.evaluate_tokenizer_extension import segmentation_comparison
from experiments.indic.extend_tokenizer import extend_tokenizer, load_model_proto
from experiments.indic.train_tokenizer_candidates import train_candidate


def _train_tokenizer(tmp_path: Path, name: str, sentences: list[str], vocab_size: int) -> Path:
    corpus = tmp_path / f"{name}.txt"
    corpus.write_text("".join(f"{sentence}\n" for sentence in sentences * 100), encoding="utf-8")
    output_dir = tmp_path / name
    result = train_candidate(corpus, output_dir, vocab_size)
    return output_dir / result["model_path"]


def _build_extension(tmp_path: Path) -> tuple[Path, Path]:
    baseline_path = _train_tokenizer(
        tmp_path,
        "baseline",
        [
            "This is an English sentence with enough recurring words.",
            "Please open the browser and read the report.",
        ],
        290,
    )
    donor_path = _train_tokenizer(
        tmp_path,
        "donor",
        [
            "आज मौसम बहुत अच्छा है और हम बाहर जाएंगे।",
            "कृपया browser में नया tab खोलें।",
            "यह हिंदी वाक्य परीक्षण के लिए है।",
        ],
        310,
    )
    output_dir = tmp_path / "extended"
    extend_tokenizer(baseline_path, donor_path, output_dir, target_vocab_size=305)
    return baseline_path, output_dir / "tokenizer_extended_305.model"


def test_extension_preserves_every_baseline_piece_and_id(tmp_path) -> None:
    baseline_path, extended_path = _build_extension(tmp_path)
    baseline = load_model_proto(baseline_path)
    extended = load_model_proto(extended_path)

    assert len(baseline.pieces) == 290
    assert len(extended.pieces) == 305
    assert all(
        baseline_piece.SerializeToString() == extended.pieces[piece_id].SerializeToString()
        for piece_id, baseline_piece in enumerate(baseline.pieces)
    )
    assert all(
        any("\u0900" <= character <= "\u097f" for character in piece.piece)
        for piece in extended.pieces[290:]
    )


def test_extension_keeps_english_segmentation_and_uses_new_hindi_pieces(tmp_path) -> None:
    baseline_path, extended_path = _build_extension(tmp_path)
    baseline = spm.SentencePieceProcessor(model_file=str(baseline_path))
    extended = spm.SentencePieceProcessor(model_file=str(extended_path))

    english = segmentation_comparison(
        baseline, extended, ["Please open the browser and read the report."], id_boundary=290
    )
    hindi = segmentation_comparison(baseline, extended, ["आज मौसम बहुत अच्छा है।"], id_boundary=290)

    assert english["identical_id_sequences_pct"] == 100
    assert english["records_using_ids_at_or_above_boundary"] == 0
    assert hindi["records_using_ids_at_or_above_boundary"] == 1
    assert hindi["tokens_at_or_above_boundary"] > 0
