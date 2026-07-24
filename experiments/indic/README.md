# Indic Language Experiments

This directory tracks controlled experiments for extending Pocket TTS to Hindi and
Hindi-English code-mixed speech.

The candidate corpus mixture and sample requirements are tracked in
[`DATA_PLAN.md`](DATA_PLAN.md).

## Research Question

Can the pretrained Pocket TTS architecture be adapted to Hindi and Hinglish while
retaining its small, streaming, voice-cloning design?

## Initial Hypotheses

1. The English SentencePiece tokenizer represents Devanagari inefficiently or as
   unknown/byte-fallback tokens.
2. Romanized Hinglish is easier to tokenize, but the English acoustic model will
   still produce English-biased pronunciations.
3. The pretrained Mimi codec is more language-independent than the text-conditioned
   FlowLM, so the first fine-tuning attempt should freeze Mimi.

These are hypotheses, not conclusions. Each one needs an isolated experiment.

## Experiment Sequence

### E0: Environment and baseline

- Verify dependency and gated-model access.
- Generate English audio with the released model.
- Record runtime and output duration.

### E1: Tokenizer coverage

- Run `probe_tokenizer.py` over `probe_sentences.jsonl`.
- Compare tokens per non-space character for English, Hindi, and Hinglish.
- Count unknown and byte-fallback tokens.
- Inspect whether Devanagari characters are represented as meaningful pieces.

Run:

```bash
HF_TOKEN="$(tr -d '\n' < HF_TOKEN)" \
  uv run --no-project --with huggingface-hub --with sentencepiece \
  python experiments/indic/probe_tokenizer.py
```

### E2: Unsupported-language inference baseline

- Use one fixed voice prompt for all inputs.
- Generate Hindi, Hinglish, and English controls with the released English model.
- Measure intelligibility, pronunciation, speaker similarity, and failure modes.

Run:

```bash
HF_TOKEN="$(tr -d '\n' < HF_TOKEN)" \
  uv run python experiments/indic/run_inference_baseline.py
```

### E3: Codec reconstruction

- Encode and decode real Hindi audio through Mimi without FlowLM generation.
- Compare original and reconstructed speech.
- If Hindi survives reconstruction, keep Mimi frozen for the first training run.

Run:

```bash
HF_TOKEN="$(tr -d '\n' < HF_TOKEN)" \
  uv run python experiments/indic/run_codec_reconstruction.py
```

## Interpretation Rule

A bad Hindi generation does not by itself identify the failing component:

```text
text -> tokenizer/embedding -> FlowLM -> continuous latent -> Mimi decoder
```

E1 isolates the text representation. E3 isolates the audio codec. Only after both
are measured should we choose the parameters and data needed for fine-tuning.

## Results

### 2026-07-24: E1 English-tokenizer coverage

| Group | Mean tokens/non-space character | Relative to English |
|---|---:|---:|
| English | 0.399 | 1.00x |
| Hindi | 1.754 | 4.40x |
| Hinglish | 0.539 | 1.35x |

The tokenizer reported no unknown tokens because it has byte fallback. That does
not mean Hindi has good coverage. Many Devanagari characters were represented by
raw UTF-8 byte pieces such as `<0xE0>`, `<0xA4>`, and `<0x86>`.

Two short Hindi probes required more than the default 50-token chunk limit. This
can make Hindi input fragment much earlier than equivalent English input.

### 2026-07-24: E2 English-model inference

Model: released English model. Voice: `alba`. Seed: `20260724`.

| Input | Text tokens | Audio duration | Generation time | RTF |
|---|---:|---:|---:|---:|
| English | 9 | 1.60 s | 0.244 s | 0.152 |
| Hindi | 37 | 0.40 s | 0.081 s | 0.202 |
| Hinglish | 13 | 2.08 s | 0.271 s | 0.130 |

All three WAV files contain non-silent audio. The Hindi input was below the
50-token chunk limit but generated only five 80 ms frames, so its early termination
cannot be explained by sentence splitting. The current evidence points to an
unsupported text representation and FlowLM conditioning. Human listening and ASR
evaluation are still required before judging Hinglish pronunciation or intelligibility.

Human listening assessment:

- English was clear and understandable.
- Hindi produced only a short noise lasting substantially less than one second.
- Hinglish resembled a British English speaker forcing Hindi pronunciation.

The Hinglish result indicates that Latin-script input avoids the catastrophic
Devanagari failure, but the pretrained model still applies English grapheme-to-speech
and acoustic priors to Hindi words.

### 2026-07-24: E3 Hindi codec reconstruction

Source: `dhruvkys/hi-asr-1k`, `test/audio/cv_000803.mp3` (CC0-1.0).

Transcript: `मैं मुसीबत में पड़ गया।`

| Measurement | Result |
|---|---:|
| Source sample rate | 32,000 Hz |
| Codec sample rate | 24,000 Hz |
| Source duration | 3.06 s |
| Continuous latent shape | 32 x 39 |
| Reconstruction duration | 3.12 s |
| Encode and decode time | 0.286 s |

At 12.5 Hz, 39 frames represent 3.12 seconds, so the duration increase is expected
codec-frame padding. Both source and reconstructed files contain non-silent audio.

Human A/B listening found that:

- The reconstructed sentence remained clearly understandable.
- The reconstructed speaker sounded almost the same as the source speaker.
- Hindi sounds and words were not noticeably corrupted or muffled.

**Decision:** E3 passes for this sample. Freeze Mimi for the first fine-tuning
experiment and train the text-conditioned FlowLM against latents extracted by the
pretrained codec. Repeat this evaluation over a broader Hindi/Hinglish validation
set before treating codec suitability as a general conclusion.

## E4: Metadata audit and unified manifest

`audit_datasets.py` reads selected Parquet metadata columns from the gated Rasa and
IndicVoices-R repositories. It deliberately excludes the embedded `audio` column.
Each JSONL record contains a pinned repository revision, shard, and row index so
audio can be materialized later on the training machine.

Install the optional data dependency:

```bash
uv sync --extra indic-data
```

Run a small smoke audit:

```bash
uv run --extra indic-data python experiments/indic/audit_datasets.py \
  --split test \
  --max-shards 1 \
  --max-rows 100
```

Run the complete Rasa and IndicVoices-R metadata audit by omitting
`--max-shards` and `--max-rows`. Results are written under
`experiments/indic/outputs/e4_data_audit/` and are ignored by Git.

The audit reports total records, hours, speakers, duration percentiles, metadata
distributions, missing fields, prompt-eligible speakers, and speaker overlap
between source-provided splits.

The audit preserves source-provided splits. It does not yet declare them suitable
for model evaluation; speaker overlap and the two-speaker limitation of Rasa must
be reviewed before defining our final train/validation/test policy.

### 2026-07-24: E4 complete metadata audit

| Source | Records | Duration | Speakers |
|---|---:|---:|---:|
| IndicVoices-R Hindi | 26,694 | 72.752 h | 384 |
| Rasa Hindi | 28,571 | 50.829 h | 2 |
| **Combined** | **55,265** | **123.581 h** | **386** |

All records have non-empty raw text, normalized text, duration, and speaker IDs.
All 386 speakers have at least two utterances, which makes distinct prompt/target
pairing possible.

IndicVoices-R has no speaker overlap between its source train and test splits.
Rasa's male and female speakers both occur in train and test, so Rasa's source
test split cannot measure unseen-speaker voice-cloning generalization.

The audit classified 55,262 transcripts as Devanagari and three as mixed
Devanagari/Latin. Manual inspection showed that the latter contain a stray Latin
`I` used like punctuation, not genuine Hindi-English code-switching. Therefore,
these sources provide effectively no Hinglish supervision; a separate
code-switched source remains necessary.

## E5: Hindi/Hinglish text normalization

The versioned `hi-v1` normalizer implements Unicode NFC and whitespace cleanup,
conservative review flags, and explicitly requested Hindi cardinal expansion.
Its policy and golden examples are documented in
[`TEXT_NORMALIZATION.md`](TEXT_NORMALIZATION.md).

Audit the local manifest without changing it:

```bash
uv run python experiments/indic/audit_text_normalization.py \
  --number-mode preserve
```

Preview explicit cardinal expansion:

```bash
uv run python experiments/indic/audit_text_normalization.py \
  --number-mode cardinal
```

### 2026-07-24: E5 normalization audit

| Finding | Rows or occurrences |
|---|---:|
| Unicode NFC changes | 12,581 rows |
| Whitespace changes | 845 rows |
| Standalone integers requiring review | 6 occurrences |
| Embedded digit corruption | 1 occurrence |
| Possible punctuation-like Latin `I` | 3 occurrences |

Default `preserve` mode flags 10 rows and does not expand their numbers. Explicit
`cardinal` mode proposes six integer expansions, leaving the embedded digit and
three punctuation-like `I` rows unresolved.

No training transcript has been overwritten. The six numeric utterances require
audio verification before we decide whether their aligned label should use Hindi
cardinals, English number words, or another spoken form.

Build the targeted human-review packet:

```bash
uv run python experiments/indic/build_normalization_review.py
```

The extractor uses the dataset-viewer row API so it downloads only flagged WAV
assets rather than their complete Parquet shards. It verifies the returned row
index, filename, transcript, and pinned dataset revision against the manifest
before accepting each clip.

The generated `outputs/e5_normalization_review/REVIEW.md` links all ten clips.
Reviewers should record the exact words heard around the flagged token. The
resulting decisions will become a small versioned override file; they will not be
hidden inside the general normalizer.

Human review decisions are stored in `normalization_overrides.jsonl` using the
stable manifest example ID. On 2026-07-24, both the male and female recordings of
`18 वर्ष` were confirmed to say `अठारह वर्ष`; their model-input text therefore
uses the spoken cardinal form.

The two `फ़ॉर्म 6` recordings demonstrate why overrides are per example. The male
speaker says English `six`, while the female speaker says Hindi `छह`. Their
aligned model-input texts preserve that difference, including Latin script for
the genuine English code-switch.

Both recordings containing the total `864` were confirmed to say the Hindi
cardinal `आठ सौ चौंसठ`. Together with the two `18` decisions, this resolves all
six standalone-integer occurrences in the audited corpus.

The embedded `0` in `गाली0गलौज` was not spoken. Human review confirmed the
compound `गाली-गलौज`, so this example receives an explicit corrupted-separator
override rather than a general rule that removes embedded zeros.

All three suspicious Latin `I` characters were confirmed to be unspoken
sentence-boundary corruption. They are replaced with `।`; the legitimate spoken
`ई` in `ई संजीवनी` remains unchanged. All ten E5 review candidates now have
human-audited model-input text in the override file.

## E6: Resolved Hindi tokenizer corpus

Build the exact model-input manifest and train-only Hindi text corpora:

```bash
uv run python experiments/indic/build_tokenizer_corpus.py
```

The builder applies `hi-v1` cleanup and the versioned review overrides. It fails
if an item still needs review, an override refers to different source text, an
override remains noncanonical, or an override ID is absent from the manifest.

Generated files under `outputs/e6_tokenizer_corpus/`:

- `model_input_manifest.jsonl`: all source rows with source and resolved text.
- `hindi_train_all.txt`: one line per source-train utterance.
- `hindi_train_unique.txt`: exact-text-deduplicated source-train text.
- `corpus_stats.json`: reproducibility and composition statistics.

### 2026-07-24: E6 corpus build

| Measurement | Result |
|---|---:|
| All resolved records | 55,265 |
| Source-train records | 52,031 |
| Held-out records excluded from tokenizer text | 3,234 |
| Unique source-train texts | 47,217 |
| Duplicate source-train records | 4,814 |
| Full train characters | 5,127,860 |
| Unique train characters | 4,874,789 |
| Applied human-reviewed overrides | 10 |

The full corpus preserves utterance frequency. The unique corpus reduces repeated
scripts, including prompts recorded by both Rasa speakers. We retain both so this
choice can be tested rather than embedded in preprocessing.

After review, 52,030 train rows are Devanagari and one is genuinely mixed script:
the male `फ़ॉर्म six` example. This confirms that E6 is the Hindi component of the
future tokenizer mixture, not sufficient Hinglish or English coverage by itself.

## E7: External tokenizer text sources

Acquire SLR104 Hinglish and LibriTTS-R English metadata without downloading their
audio:

```bash
uv run python experiments/indic/acquire_tokenizer_text.py
```

The SLR104 official archive bundles audio and transcripts in a 7.3 GB file. For
text acquisition, the script reads the separately exposed CSV metadata from
`ujs/hinglish-compressed` at pinned revision
`5c22260f73a889c457861b1647e96e4254dc1047`. Its loader identifies the content as
OpenSLR-104. The official OpenSLR page remains the authority for dataset identity
and its CC BY-SA 4.0 license; the Hugging Face repository is only a transport
mirror.

LibriTTS-R text is read column-selectively from `pharaouk/libritts_r` at pinned
Parquet revision `42c834ebdd6db1f79120dad12347e45fa34d3650`. PyArrow requests
only normalized/original text, utterance ID, speaker ID, and chapter ID. It does
not materialize the embedded audio column.

| Source metadata | Records | Unique text | Split detail |
|---|---:|---:|---|
| SLR104 Hindi-English | 55,961 | 45,964 | 52,825 train; 3,136 test |
| LibriTTS-R clean-100 | 33,232 | 32,888 | train only |

Of the SLR104 train rows, 44,412 are genuinely mixed Devanagari-Latin, 7,456 are
Devanagari-only, 955 are Latin-only, and two contain neither script. Dataset name
alone is therefore not used as the code-switch label.

## E8: Multilingual tokenizer mixture

Build the train-only mixture:

```bash
uv run python experiments/indic/build_tokenizer_mixture.py
```

The builder uses all unique E6 Hindi text as the anchor. It samples only
mixed-script SLR104 train rows and LibriTTS-R train rows using stable SHA-256
ranks. Budgets are measured in non-whitespace Unicode code points because
SentencePiece learns from character/subword exposure, not hours of audio.

One IndicVoices-R row is excluded in `tokenizer_exclusions.jsonl`: its normalized
February token contains an unrelated invisible Mongolian separator and a stray
combining mark, while its verbatim transcript differs. The exact text hash is
checked before applying the exclusion. Correction is deferred until the audio is
reviewed. Other unsafe control/format characters fail the build; Devanagari ZWJ
and ZWNJ remain allowed.

### 2026-07-24: E8 corpus build

| Stream | Selected lines | Non-space characters | Actual share |
|---|---:|---:|---:|
| Hindi | 47,216 | 3,892,968 | 59.9993% |
| Hinglish | 30,947 | 1,622,081 | 24.9999% |
| English | 12,391 | 973,304 | 15.0008% |
| **Total** | **90,554** | **6,488,353** | **100%** |

The generated `tokenizer_train.txt` contains no held-out rows or exact duplicate
texts. A second build was byte-identical. Its SHA-256 is
`d45e0f914e127deeef2748b01f0b0687072e402bbf467c8cfab13ac604f0cc5e`.

This English stream is a public read-speech replay proxy. It does not reproduce
the much broader private training distribution of the released Pocket TTS model,
so English regression must still be measured explicitly.
