# E14 CPU Training Smoke

## Purpose

This smoke test checks the complete data-to-gradient path on the released
six-layer FlowLM before spending H100 time. It is not an overfit result and does
not measure Hindi generation quality.

## Frozen latent cache

The 18 accepted E13 WAV files were resampled from 48 kHz to 24 kHz and encoded
once with frozen Mimi. Raw prompt and target latents are stored in the cache;
target normalization is applied only when a training batch is loaded.

| Item | Result |
|---|---:|
| E11 checkpoint SHA-256 | `b6661073efe6dd9b94a1a5fc94d4cb19c506d84c62395a95d07aaca15109e148` |
| E10 tokenizer SHA-256 | `c980ff465a6e8b1ee472f897560b898c9eb69beeedf803e4e6a603c3638725e1` |
| Latent cache SHA-256 | `584053630ca60cd335696664e5ceee5bc4a7c8c610e89e990bbfb1448aa0405f` |
| Cache metadata SHA-256 | `4d91e10b4a77cb90f5e77664887e34d7de890f035c6a1020f337e711ddaf1bed` |
| Latent shape contract | 32 dimensions at 12.5 Hz |
| Target frames | 1,060 |
| Token uses | 304 |
| Unique token IDs | 113 |
| Added Hindi token uses | 290 |
| Preserved token uses | 14 |
| Unknown token uses | 0 |

Raw target latents have mean `-0.124` and standard deviation `0.994`. Applying
the frozen checkpoint statistics gives mean `-0.062` and standard deviation
`0.998`. This supports retaining the released `emb_mean` and `emb_std`.

## One real optimization step

The smoke used the `sample_mean` reduction, pair 2, and 165 valid target frames.
It ran in float32 on CPU with the intended AdamW, FM/LSD split, EOS loss,
gradient clipping, and embedding-row policy.

| Metric before the update | Value |
|---|---:|
| Total loss | 1.2311 |
| Head loss | 1.1848 |
| Raw FM loss | 1.5752 |
| Raw LSD loss | 0.0136 |
| EOS loss | 0.0463 |
| Gradient norm before clipping | 8.4136 |
| EOS false-trigger rate at `-4.0` | 15.95% |
| EOS missed-end rate at `-4.0` | 0% |
| Mean nonterminal EOS logit | -5.2112 |
| Mean final-frame EOS logit | -2.5306 |

The low EOS BCE does not imply inference-safe stopping. The false-trigger rate
shows why raw logits and deployed-threshold decisions must be monitored.

## Parameter and resume audit

Pair 2 uses 25 unique added token IDs and no preserved token IDs. After one
AdamW step:

- exactly 25 added embedding rows changed;
- all 3,975 unused added rows remained bit-exact;
- all preserved rows remained unchanged for this batch;
- the padding row remained bit-exact; and
- production-sized FlowLM, AdamW, and flow-noise generator state loaded and
  re-saved at step 1 without advancing the metrics log.

Embedding weight decay is disabled so unused vocabulary rows do not drift.
Non-embedding parameters retain weight decay. Preserved English rows receive
one tenth of the actual post-AdamW update when they are used.

## Remaining GPU gate

E14 proves data integrity, tokenization, Mimi encoding, teacher forcing,
backpropagation, row-wise updates, metrics, checkpointing, and resume. The next
gate must still show:

1. all eight text pairs can be memorized;
2. FM, LSD, EOS, and gradient metrics remain finite and improve;
3. early EOS false triggers fall rather than rise;
4. generated Hindi matches the target text and prompt voice; and
5. `branch_sum` and `sample_mean` are compared under otherwise identical runs.
