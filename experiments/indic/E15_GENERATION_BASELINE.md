# E15 Export and Generation Baseline

## Purpose

E15 proves that a FlowLM-only trainer checkpoint can be converted back into a
complete Pocket TTS checkpoint and evaluated through the stock inference API.
It also fixes the generation set that will be used to judge the H100 overfit
runs.

The checkpoint used for this smoke had completed one CPU optimization step. It
is a pipeline baseline, not evidence of Hindi learning.

## Export audit

The exporter replaced every FlowLM tensor and retained every frozen tensor from
the E11 checkpoint.

| Item | Result |
|---|---:|
| Completed training steps | 1 |
| FlowLM tensors replaced | 127 |
| Non-FlowLM tensors preserved bit-for-bit | 87 |
| Strict stock `TTSModel` load | Passed |
| Base E11 model SHA-256 | `b6661073efe6dd9b94a1a5fc94d4cb19c506d84c62395a95d07aaca15109e148` |
| Trainer FlowLM SHA-256 | `7b5330bcbfe53d3467967f97407a8bcc638fc37bfa7225a7bdf3562df9453da7` |
| Exported full model SHA-256 | `f671b319a966322d03efc4555c667d1d4defdc080da0f59eddbb897bfc85a62c` |

The exported config points both normal and voice-cloning weight fields to the
same complete checkpoint. The tokenizer remains the E10 ID-preserving 8K model.

## Fixed evaluation set

The full evaluation contains 24 items:

- 16 memorization items: eight E13 target texts in both fixed prompt voices;
- two unseen ordinary Hindi sentences in both voices;
- one unseen Hinglish sentence in both voices; and
- one unseen OTP sentence in both voices, with `1234` represented as the spoken
  sequence `एक दो तीन चार`.

“Unseen” means absent from the 16 E13 target recordings. These controls measure
transfer beyond the tiny overfit packet; they do not claim corpus-level novelty.

Every item has a stable ordering and generation seed. Generated WAV, model,
packet, probes, and evaluation JSON hashes are recorded. A human decision is
preserved only while the generated WAV hash remains unchanged.

Two consecutive CPU reruns reproduced all four WAV hashes exactly. The
deterministic result hash, which excludes only wall-clock timing, was
`30ee7a7e50a5cd7c266e8b2b7b5094409354abfde209368d471ff220c283b100`.

## Four-item smoke result

The smoke evaluated training pair 1 and the first unseen Hindi control in both
voices.

| Item | Duration | EOS behavior | Human review |
|---|---:|---|---|
| Overfit female | 0.32 s | Early stop | Effectively silent |
| Overfit male | 0.32 s | Early stop | Brief initial noise |
| Unseen Hindi female | 9.04 s | Hit maximum frames | Sustained e-like sound |
| Unseen Hindi male | 1.28 s | Early stop | Noise-like output |

All four were rejected: no understandable words, transcript match, or
recognizable prompt voice. This is consistent with E2 and with E14's unstable
Hindi EOS baseline. One optimizer step was not expected to improve synthesis.

The final male observation is mapped from the fourth listening response by item
order because that response repeated the female file path. The ambiguity is
retained in the review note.

## H100 comparison contract

The two first overfit runs must differ only in the documented loss reduction:

| Run | Head reduction |
|---|---|
| A | `sample_mean` |
| B | `branch_sum` |

Both runs use the same E13 packet, E14 cache, model initialization, seed,
optimizer, batch order, gradient clipping, checkpoint schedule, and generation
items. Step 0 must be exported and evaluated before training so future changes
are measured against an exact common starting point.

The memorization gate requires all 16 overfit items to become intelligible,
text-complete, and prompt-voice consistent without maximum-length failures.
Control items are reported separately: they measure early transfer but are not
required to pass the narrow memorization gate.

Loss reduction is selected using the combined evidence from:

- raw FM and LSD trajectories;
- gradient norm and numerical stability;
- EOS false-trigger and missed-end rates;
- human review of all 16 overfit generations; and
- unseen-control behavior.

A lower scalar training loss alone cannot select the winner because
`sample_mean` and `branch_sum` use different scalar scales.
