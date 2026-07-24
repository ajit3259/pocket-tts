# Hindi and Hinglish Text Normalization

## Purpose

A TTS model should receive text that describes what the recording actually says.
Text normalization converts written forms such as `123`, `₹50`, or `10:30` into
the words that should be spoken.

This is not ordinary text cleanup. The same written token can have multiple valid
pronunciations:

| Written form | Possible spoken form | Context |
|---|---|---|
| `123` | `एक सौ तेईस` | cardinal quantity |
| `123` | `एक दो तीन` | PIN, extension, or identifier |
| `2026` | `दो हज़ार छब्बीस` | year |
| `6` | `छह` or `सिक्स` | Hindi versus Hinglish product usage |
| `10:30` | `साढ़े दस` or `दस बजकर तीस मिनट` | conversational versus formal time |

A normalizer that always chooses one form will create incorrect training labels.

## Two Different Operating Modes

### Training normalization

Training normalization must be conservative. Audio is the ground truth, so an
ambiguous written token is flagged for review rather than silently expanded.

For example, `फ़ॉर्म 6` must remain unchanged until listening or a reliable ASR
alignment determines whether the speaker said `फ़ॉर्म छह` or `फ़ॉर्म सिक्स`.

### Inference normalization

At inference time there is no reference audio. The application must select a
documented pronunciation policy. A caller can explicitly request cardinal
expansion when `123` is known to mean `एक सौ तेईस`.

Future product-level normalization will need semantic classes such as cardinal,
ordinal, currency, date, time, phone number, identifier, and year.

## Order of Operations

```text
raw text
  -> Unicode and whitespace cleanup
  -> detect semiotic classes and ambiguous tokens
  -> apply explicit pronunciation policies
  -> normalized spoken-form text
  -> SentencePiece tokenizer
```

Normalization happens before tokenizer training and inference tokenization. Mimi
is not involved because it processes audio, not text.

## Corpus Evidence

The E4 manifest contains 55,265 Hindi transcripts:

- 7 contain ASCII digits.
- 0 contain Devanagari digits.
- 0 contain currency symbols, percentages, decimals, or clock-like numeric forms.
- 3 contain a Latin `I` used like punctuation rather than genuine code-switching.

The seven digit-containing rows reduce to:

- Two copies of `18 वर्ष`.
- Two copies of `फ़ॉर्म 6` or `फॉर्म 6`.
- Two copies of a total written as `864`.
- One corrupted compound, `गाली0गलौज`.

This shows that Rasa and IndicVoices-R already contain mostly spoken-form text. It
does not eliminate the need for an inference normalizer, because users will still
enter arbitrary numbers and symbols.

## Version 1 Scope

The first deterministic implementation will:

1. Apply Unicode NFC normalization.
2. Collapse repeated whitespace.
3. Preserve ambiguous numbers by default and return review reasons.
4. Detect digits embedded inside words, such as `गाली0गलौज`.
5. Detect the observed suspicious punctuation-like Latin `I`.
6. Expand non-negative cardinal integers only when explicitly requested.
7. Accept ASCII or Devanagari digits and valid Western or Indian comma grouping.
8. Use the Indian scales `सौ`, `हज़ार`, `लाख`, and `करोड़`.

The initial integer range is `0` through `9,999,999,999`.

## Explicitly Deferred

These classes remain unchanged and are flagged until they receive their own
context-aware policy and tests:

- Signed values and decimals.
- Percentages and currencies.
- Dates, times, and years.
- Phone numbers, PINs, and identifiers.
- Measurements and units.
- Ordinals.
- Abbreviations and initialisms.
- Romanized Hindi spelling standardization.
- Transliteration between Latin and Devanagari.

English words in Hinglish must remain English. Normalization must not transliterate
or translate them merely because they occur inside a Hindi sentence.

## Golden Examples

| Input | Mode | Expected output |
|---|---|---|
| `123` | preserve | `123`, flagged for review |
| `123` | cardinal | `एक सौ तेईस` |
| `१२३` | cardinal | `एक सौ तेईस` |
| `1,23,456` | cardinal | `एक लाख तेईस हज़ार चार सौ छप्पन` |
| `₹123` | cardinal | unchanged and flagged |
| `12.5` | cardinal | unchanged and flagged |
| `फ़ॉर्म 6` | preserve | unchanged and flagged |
| `गाली0गलौज` | preserve | unchanged and flagged as embedded digit |
| `Aaj meeting 6 baje hai` | preserve | English and Roman Hindi preserved |

Hindi forms from 1 through 100 were checked against the
[Hindi Cell at NIT Hamirpur](https://nith.ac.in/library/hindi/hindi_count.php).
Where valid spelling variants exist, the canonical output should favor the form
most frequent in our audited Hindi transcripts while preserving the same
pronunciation.

## Manifest Requirements

The materialized training manifest should eventually retain:

```text
text_raw
text_source_normalized
text_model_input
normalizer_version
normalization_changes
normalization_review_reasons
```

No automated transformation should destroy the source text. Review decisions must
be reproducible and attributable to a normalizer version.
