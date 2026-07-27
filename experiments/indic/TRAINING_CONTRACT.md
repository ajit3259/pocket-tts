# Pocket TTS Fine-Tuning Contract

This note reconstructs the part of Pocket TTS training that is absent from the
released inference repository. It separates evidence from assumptions so that a
decreasing loss is not mistaken for a correct training pipeline.

## Evidence order

When sources disagree, use this order:

1. Released Pocket TTS model structure and inference behavior.
2. The [CALM paper](https://arxiv.org/abs/2509.06926) and
   [LSD paper](https://arxiv.org/abs/2505.18825).
3. Kyutai maintainer statements in
   [issue 30](https://github.com/kyutai-labs/pocket-tts/issues/30).
4. Community implementations as hypotheses to audit, not specifications.

Kyutai has not released its trainer. Therefore, even a paper-backed reconstruction
must be validated with a tiny overfit experiment before a long run.

## Frozen acoustic model

Mimi is independently trained. It is not part of the first Hindi/Hinglish
optimization loop.

```text
target waveform --frozen Mimi encoder--> raw latent frames
raw latent frames --checkpoint normalization--> FlowLM training targets
generated normalized frames --checkpoint denormalization--> frozen Mimi decoder
```

The target latent at each 80 ms frame has 32 dimensions. The checkpoint buffers
`emb_mean` and `emb_std` center and scale these target latents. They should remain
fixed during initial fine-tuning; recomputing them would change the coordinate
system already learned by the flow head.

Voice-prompt latents are different: inference projects the raw, unnormalized Mimi
latents through `speaker_proj_weight`. Training must preserve that behavior.

## Teacher forcing

For target normalized latent frames
`z = [z0, z1, ..., z(S-1)]`, the per-example transformer input is:

```text
[optional voice prefix | text prefix | input_linear(BOS, z0, ..., z(S-2))]
```

| Block | Input shape | Projected shape |
|---|---:|---:|
| Voice prompt | `Pv x 32` raw latents | `Pv x 1024` |
| Text | `Pt` token IDs | `Pt x 1024` |
| Shifted target history | `S x 32` normalized latents | `S x 1024` |

The contextual output at the `BOS` position predicts `z0`. The output after `z0`
predicts `z1`, and so on. All frames are trained in parallel with causal attention;
this is the continuous-latent equivalent of shifting language-model labels by one
position.

A voice prompt must come from a different utterance of the same speaker. Cutting a
prompt from the target utterance while retaining the full target transcript creates
false alignment: some target words have already been spoken in the prompt.

## Time direction

The CALM paper uses time 0 for data and time 1 for noise. Released Pocket TTS
inference starts at noise and advances from 0 to 1, so use the reversed coordinate
`u = 1 - paper_time`:

```text
x(u)  = sin(pi*u/2) * data + cos(pi*u/2) * noise
dx/du = (pi/2) * [cos(pi*u/2) * data - sin(pi*u/2) * noise]
```

Therefore:

- `u=0` is Gaussian noise.
- `u=1` is the target Mimi latent.
- `flow_net(condition, u, u, x(u))` learns `dx/du`.
- `lsd_decode` integrates in the same 0-to-1 direction.

Using the linear path `(1-u)*noise + u*data` defines a valid different flow model,
but it is not the trigonometric coordinate described for CALM. Continuing a
pretrained trigonometric flow head with the linear target changes the learned
function.

## Flow-matching loss

For a target frame `z`, sample `u` uniformly and `epsilon` from a standard Gaussian.
Construct `x(u)` and its analytic velocity using the equations above:

```text
prediction = flow_net(condition, u, u, x(u))
FM = mean_squared_error(prediction, dx/du)
```

This diagonal loss teaches the instantaneous velocity field.

## LSD loss

Lagrangian self-distillation teaches an average flow between two times. Sample a
start `a` uniformly, then an end `b` uniformly between `a` and 1:

```text
f(x(a), a, b) = x(a) + (b-a) * flow_net(condition, a, b, x(a))
```

Forward-mode automatic differentiation computes `df/db`. The self-distillation
target is the stop-gradient diagonal velocity evaluated at the mapped point:

```text
teacher = stop_gradient(flow_net(condition, b, b, f(x(a), a, b)))
LSD = mean_squared_error(df/db, teacher)
```

This is why `SimpleMLPAdaLN` contains a custom JVP-compatible layer norm. LSD is
what makes one-step or few-step generation possible; flow matching alone primarily
teaches an ODE that would normally need more integration steps.

## Head batch multiplier

The transformer is the expensive part. CALM reuses each contextual output for
eight independent time/noise draws:

```text
one transformer pass -> 8 flow-head training examples
```

The paper assigns 75% of this head batch to FM and 25% to LSD. At multiplier 8,
that is six FM draws and two LSD draws.

There is an unresolved reduction ambiguity in the primary sources:

| Source | Reduction after the 75/25 split |
|---|---|
| LSD paper Algorithm 1 | `mean_FM + mean_LSD` |
| Authors' `nmboffi/flow-maps` code | `(6*mean_FM + 2*mean_LSD) / 8` |
| CALM paper Appendix A | States `FM + LSD`; production code unavailable |

The first form gives the two objectives equal aggregate weight; the second gives
every sampled head example equal weight. `training_objective.py` requires the
caller to choose `branch_sum` or `sample_mean` explicitly. The tiny overfit should
compare both while logging the two raw errors separately.

## EOS loss

The same contextual frame feeds `out_eos`. A masked binary cross-entropy marks the
last valid target frame as positive and ignores right padding.

EOS needs separate monitoring because inference compares its raw logit with a
fixed threshold. A low BCE does not by itself prove that early false triggers or
failure to stop have been avoided.

## What trains first

For the one-GPU tiny overfit:

| Component | Initial policy |
|---|---|
| Mimi encoder and decoder | Frozen, evaluation mode |
| `emb_mean`, `emb_std` | Frozen checkpoint buffers |
| New Hindi token rows | Train |
| Existing text rows | Train with a low learning rate or separate parameter group |
| FlowLM transformer | Train |
| Flow MLP | Train |
| EOS head | Train and monitor |
| Speaker projection | Train only when valid distinct prompt/target pairs are present |

The tiny run is a diagnostic, not the final recipe. Parameter groups and freezing
ablations come after the end-to-end contract is proven.

## Community-fork audit

The `freds0/pocket-tts` training fork at commit `1d77e3f` is useful because it
identifies many required pieces, including teacher forcing, distinct same-speaker
prompts, JVP-based LSD, EOS, and export. It is not adopted directly.

Material differences found during review:

1. It uses a linear noise/data path rather than the trigonometric path described
   for CALM's consistency objective. The CALM LSD appendix leaves the interpolant
   symbolic, so checkpoint compatibility still needs an empirical audit.
2. It introduces two adaptive-weight networks and clamps their outputs; those are
   implementation choices not specified by the released model.
3. It applies text-conditioning dropout while the released six-layer Pocket model
   is already a single-pass student distilled from a CFG-guided teacher. Released
   inference has no matching conditional/unconditional CFG branch.

Its reported successful audio is evidence worth reproducing, but it does not
resolve these mathematical compatibility questions.

The fork's `mean_FM + mean_LSD` reduction agrees with the LSD paper algorithm.
The official LSD reference repository instead uses a sample-weighted mean. This
is a source ambiguity, not a demonstrated bug in the community fork.

## E12 scope and remaining unknowns

`training_objective.py` implements an unweighted, CPU-testable version of:

- checkpoint latent normalization;
- inference-compatible prefix assembly and teacher forcing;
- the reversed trigonometric FM target;
- off-diagonal LSD with JVP and a stop-gradient teacher;
- the 75/25 head split with both documented reduction choices; and
- masked EOS loss.

The adaptive weighting function is deliberately deferred. Its weights are not in
the inference checkpoint, and its exact production implementation is unavailable.
We should first prove that raw FM and LSD errors decrease on a tiny aligned sample,
then test adaptive weighting as an explicit ablation while always logging raw MSE.

Before distributed training, the remaining gates are:

1. Materialize a tiny speaker-disjoint, transcript-aligned audio set.
2. Verify raw Mimi latent extraction and fixed checkpoint normalization.
3. Overfit 8-32 utterances on one H100.
4. Confirm loss decrease, gradient flow, checkpoint resume, EOS, intelligibility,
   pronunciation, and speaker similarity.
5. Only then scale data and DDP to eight H100s.
