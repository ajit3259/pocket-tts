# Hindi and Hinglish Data Plan

## Objective

Adapt the released Pocket TTS model to:

1. Hindi written in Devanagari.
2. Spoken Hindi represented with common Roman spellings.
3. Hindi-English code-switched speech.
4. English without catastrophic forgetting.
5. Zero-shot voice cloning across speakers not seen during adaptation.

The first training experiments will keep Mimi frozen. Audio is still passed through
Mimi to produce target continuous latents, but Mimi receives no gradient updates.

## Dataset Manifest Contract

The first manifest is metadata-only. It identifies an audio-containing Parquet row
without copying or decoding its embedded audio:

```text
example_id
source_dataset
source_license
source_split
source_locator      # repository, pinned revision, shard, row, audio column
speaker_id
language_mode
script_mode
text_raw
text_normalized
duration_seconds
gender
style
```

Keeping `text_raw` and `text_normalized` separate is important. We can always audit
what the speaker said while independently improving the text representation that
the model receives.

`language_mode` records the linguistic training interface; `script_mode` is only
an objective Unicode classification. Latin text in a Hindi dataset is not
automatically labeled as Romanized Hindi because script alone cannot distinguish
Romanized Hindi from English.

## Materialized Training Sample Contract

Each training example should contain:

```text
target_audio
target_text           # selected from or derived from the metadata manifest
speaker_id
language_mode       # hi, hi-roman, hi-en, or en
source_dataset
source_license
prompt_audio
```

`prompt_audio` and `target_audio` should be different utterances from the same
speaker. Using the target itself as its voice prompt leaks target content and can
teach the model to copy instead of learning text-to-speech.

Audio materialization and prompt pairing happen after metadata validation. This
keeps the initial audit small and prevents an accidental full dataset download on
a development laptop.

Speaker IDs must be stable within a source dataset and namespaced across datasets.
Training, validation, and test splits must be speaker-disjoint when enough speakers
are available.

## Candidate Sources

### AI4Bharat IndicVoices-R

- Role: multi-speaker Hindi and voice-cloning generalization.
- Strengths: explicit speaker IDs, natural speech, verbatim and normalized text,
  demographics, duration, SNR, C50, speaking rate, and pitch metadata.
- Quality: restored using demixing, dereverberation, enhancement, and filtering.
- Hindi scale reported by the paper: about 70 hours.
- License: CC BY 4.0.
- Access: gated; access confirmed on 2026-07-24.
- Source: https://huggingface.co/datasets/ai4bharat/indicvoices_r

### AI4Bharat Rasa

- Role: high-quality Hindi acoustics and expressive styles.
- Hindi data: 27.05 hours female plus 23.78 hours male.
- Styles include neutral, conversation, books, news, commands, and emotions.
- Limitation: only two Hindi speakers, represented by gender rather than an
  explicit speaker ID.
- Format: 48 kHz mono.
- License: CC BY 4.0.
- Access: gated; access confirmed on 2026-07-24.
- Source: https://huggingface.co/datasets/ai4bharat/Rasa

### OpenSLR SLR104

- Role: genuine Hindi-English code-switched speech.
- Hindi-English training data: 89.86 hours; test data: 5.18 hours.
- Format: 16 kHz, 16-bit speech extracted from technical spoken tutorials.
- Limitation: domain is biased toward technical and instructional language.
- License: CC BY-SA 4.0. The share-alike implications for distributed model
  weights require review before production use.
- Source: https://www.openslr.org/104/

### LibriTTS-R

- Role: English replay data to reduce catastrophic forgetting.
- Scale: 585 hours, 2,456 speakers, 24 kHz.
- We need only a sampled replay subset, not the entire corpus.
- License: CC BY 4.0.
- Source: https://google.github.io/df-conformer/librittsr/

## Sources Not Selected as Primary Training Data

- `agarwalayushi/hinglish`: useful as an index of public corpora, but it combines
  sources with different upstream terms. We should ingest selected primary sources
  directly and retain their provenance.
- Synthetic Hindi/Hinglish TTS corpora: useful for later augmentation, but not for
  the initial experiment because they can transfer another model's accent,
  pronunciation errors, and acoustic artifacts.
- Common Voice Hindi: useful for evaluation and additional speaker diversity, but
  lower and less consistent recording quality makes it secondary to IndicVoices-R.

## Proposed Pilot Sampling Mixture

The pilot will use weighted sampling rather than simply concatenating datasets:

| Stream | Sampling weight | Purpose |
|---|---:|---|
| IndicVoices-R Hindi | 35% | speaker diversity and natural Hindi |
| Rasa Hindi | 25% | clean and expressive Hindi |
| SLR104 Hindi-English | 25% | genuine code-switching |
| LibriTTS-R English | 15% | English retention |

These are starting values, not final hyperparameters. We will adjust them based on:

- Hindi and Hinglish intelligibility.
- English regression.
- Speaker similarity.
- Validation loss per data stream.
- Generated speech failure modes.

## Text Representations

The same spoken Hindi may need more than one valid textual interface:

```text
Devanagari: आज मौसम बहुत अच्छा है।
Romanized:  Aaj mausam bahut accha hai.
Code-mixed: Aaj weather bahut accha hai.
```

For some Hindi samples, we can create an additional Romanized transcript while
keeping the original Devanagari transcript. Deterministic transliteration alone is
not sufficient because Romanized Hindi has spelling variation such as `achha`,
`accha`, and `acha`. The tokenizer corpus and later augmentation should include
common variants.

English words in genuinely code-switched transcripts should remain in Latin script.
We must not normalize all borrowed English words into Devanagari because that would
remove the code-switch signal the model needs to learn.

The distinction between conservative audio-aligned training normalization and
policy-driven inference normalization is specified in
[`TEXT_NORMALIZATION.md`](TEXT_NORMALIZATION.md).

## Data Validation Gates

Before a sample can enter training:

1. Duration is within the selected training window.
2. Audio is mono and can be resampled to 24 kHz.
3. Transcript is non-empty and language/script checks pass.
4. Audio and transcript agree according to a Hindi-capable ASR audit.
5. Speaker ID has at least two usable utterances.
6. Prompt and target are different recordings.
7. Source and license are recorded.
8. Validation/test speakers do not appear in training.

## First Data Milestones

1. Download metadata and a small audio sample from each primary source.
2. Implement a common manifest schema without rewriting source datasets.
3. Run Mimi reconstruction over a diverse Hindi/Hinglish sample.
4. Measure the released tokenizer and candidate tokenizer on real transcripts.
5. Build a tiny overfit dataset before launching distributed training.
