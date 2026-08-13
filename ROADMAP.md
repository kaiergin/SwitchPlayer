# Roadmap

Working plan for SwitchPlayer. Read `README.md` first for the project purpose;
this document assumes it.

**Guiding filter:** the agent acts with a single forward pass (~1.9 ms) and all
search runs offline as re-analysis. Work is therefore prioritised by whether it
improves the agent **independently of the search budget**. Anything whose payoff
is "search gets cheaper" is deferred until search moves onto the acting path,
which is not currently planned.

Status is verified against the code at the commit this file was added, not
assumed. Formulas are quoted from the papers; implementation details marked
*(verify)* should be checked against a reference implementation before building.

---

## Priority 1 — End-to-end value prefix

**Status: BUILT, MEASURED, AND TURNED OFF.** `ValuePrefixNetwork` is
implemented, matches LightZero's architecture, is trained alongside the per-step
head and can drive the tree via `USE_VALUE_PREFIX_IN_TREE`.

It does not learn. Across four independent measurements — culminating in a
5-arm, 2-seed, 60-iteration ablation with search in the loop — the head sits at
exactly **1.000x a predict-zero control** in every arm, i.e. collapsed to
predicting zero, while the per-step head reaches 0.24x. The flag therefore
defaults to `False`: the tree reads the signal that demonstrably learns.

This says nothing about EfficientZero or LightZero, neither of which was run.
It says *this* implementation fails *here*. Leading untested suspect: the prefix
target is a 5-step undiscounted sum, mostly zero with rare spikes, over 183
bins — the centre bin is a strong local optimum. Priority 4 (now done) may
change this; it has not been re-measured since.

The description below is kept as the design record.

### What it is

MuZero predicts a reward at every unrolled state and sums them. EfficientZero
predicts the *sum* directly from the sequence of unrolled states:

```
value_prefix = f(s_t, ŝ_t+1, ..., ŝ_t+k-1)          f = LSTM
Q(s_t, a)    = Σ_{i=0}^{k-1} γ^i r_{t+i} + γ^k v_{t+k}
                └──────── the value prefix ────────┘
```

The motivation is **state aliasing**. From the paper: predicting the exact
timestep a Pong player loses the ball is near-impossible from an early frame,
but predicting that they *will* lose it over a longer horizon is easy. Per-step
reward prediction forces the hard question at every depth and compounds the
error down the tree; predicting the prefix lets the model be vague about *when*
while staying accurate about *whether*. The LSTM is supervised at every step, so
the extra structure does not cost supervision density.

### Why it matters here

The re-analysis tree is where target quality is decided, and its Q-values are
built from exactly this sum. Reward error at depth is currently unmeasured and
compounds through `backup_simulation`. This is also the axis the project is most
interested in — how future reward is predicted through the dynamics network.

### What changes

| File | Change |
|---|---|
| `networks.py` | Drop `reward_conv`/`reward_fc` from `DynamicsNetwork`; add a `ValuePrefixNetwork` — pool each latent to a vector, feed an LSTM, emit categorical prefix logits through the existing two-hot machinery |
| `muzero.py` `train_step` | Replace the per-step `reward_loss` with a prefix loss: carry LSTM state across the unroll, supervise at every step against `Σ_{j<i} γ^j r_{t+j}` |
| `muzero.py` `MCTSNode` | Carry `value_prefix` and the LSTM hidden state `(h, c)` per node |
| `muzero.py` `batch_expand_nodes` | Step the LSTM from the parent's hidden state instead of reading a per-child reward |
| `muzero.py` `backup_simulation` | Edge reward becomes the prefix *difference* between parent and child |
| `muzero.py` `run_full_mcts` | Reset the LSTM hidden state at each search root |

### Watch for

- **Support sizing.** The prefix is a *sum* of rewards, so it needs a larger
  support than the reward head's 400. Five unroll steps with rewards clipped at
  400 reaches 2000. Getting this wrong reintroduces the all-zero-target bug
  fixed in `0ad73607` — the tests in `tests/test_scalar_categorical.py` will
  catch it if the new support is registered there.
- **Per-node memory.** Hidden state is ~4 KB/node at hidden=512, against ~110 KB
  for a latent, so this is minor next to the OOM work already done — but check
  it rather than assume.
- **Discounting.** The paper writes the prefix as `Σ γ^i r`, discounted.
  *(verify)* whether the reference implementation discounts inside the LSTM
  target or accumulates raw and discounts at use.
- **LSTM reset frequency.** *(verify)* — reset at the search root is the
  intent; the reference may also reset periodically within deep unrolls.

### How we will know it worked

1. Unit tests: prefix target equals the discounted cumulative sum; prefix at
   step 0 is zero; two-hot projection of the largest reachable prefix still sums
   to 1.
2. **The direct test of the claim:** measure predicted-vs-actual value prefix as
   a function of unroll depth, before and after. State aliasing predicts the
   per-step baseline degrades with depth faster than the LSTM version. Add this
   as a `benchmarks/bench.py` suite.
3. `train_step` latency (currently 85 ms at batch 64) must not regress badly.

---

## Priority 2 — SimSiam consistency loss

**Status: DONE.** `networks.LatentProjector` carries EfficientZero's projection
and prediction heads (fc-BN-ReLU shapes from the reference implementation);
`train_step`'s `latent_loss` is negative cosine similarity through them, online
branch through projection + prediction, target branch projection-only under
`no_grad`. `CONSISTENCY_LOSS_WEIGHT` moved to EfficientZero's 2.0. The
projector is training-only — inference consumers never project — but its
weights and optimizer ride the checkpoint/state-transfer path so training
resumes cleanly (pre-projector checkpoints still load, with a fresh projector).
Observation augmentations (the paper reports they help) remain unexplored.

### What it is

EfficientZero pulls the dynamics-predicted `ŝ_t+1` toward the representation of
the real next observation `s_t+1`, through a SimSiam arrangement: both go
through a shared **projector**, the predicted branch gets an extra **predictor**
head, the target branch is **stop-gradient**, and the loss is **negative cosine
similarity** rather than MSE. Unrolled 5 steps, pulling `ŝ_t+k` toward `s_t+k`.

### Why it matters here

Plain MSE toward a moving target is the setup most prone to representation
collapse — the network can trivially shrink all latents. Cosine similarity is
scale-invariant, and the predictor/stop-gradient asymmetry is precisely what
SimSiam introduced to prevent collapse. The dynamics model is what re-analysis
quality rests on, and this is the densest training signal available for it:
a few hundred dimensions per step against a scalar reward.

### What changed

`networks.py` gained `LatentProjector` (projector + predictor MLPs);
`train_step`'s `latent_loss` is negative cosine similarity through them, with
stop-gradient on the representation branch. Widths follow the EfficientZero
reference (1024/1024/512).

*(open)* observation augmentations — the paper reports they help, and there
are none here.

---

## Priority 3 — Signed support and an invertible value transform

**Status: DONE, and the one change with clean positive evidence.**

Note this was never an *addition* to MuZero — h() and a symmetric support are
both specified in the MuZero paper. The repo was deviating from baseline; this
brought it back.

Measured: the reward head reaches **0.24x** a predict-zero control with h() and
**1.12x** without, i.e. it only learns at all with the transform. The value
head's error opens near 650 without it and near zero with it, and cross-seed
spread in final return drops from 105 to 4.5.

Bin counts are derived per head from the transformed range (`bins_for`), because
h() shrinks the range but not the bin *count* — 51 bins still left raw 0-7 in a
single bin. The original description follows.

### Why it matters

Negative values are **unrepresentable**. Any negative reward or value silently
clamps to zero. That is survivable on MsPacman, where rewards are non-negative,
but the Tetris design in this repo is built around negative feedback
(`gamefeed.py` collects a `negative` class), so this blocks the actual target
game. It also means the value head cannot express "this state is bad", only
"this state is worth little".

Separately, MuZero applies an invertible transform before the categorical
projection:

```
h(x) = sign(x)(√(|x| + 1) − 1) + εx
```

which compresses large returns so a fixed bin count covers the range at useful
resolution. Without it, a 51-bin support over [0, 2000] has 40-point buckets —
coarse enough that small value differences vanish.

Do these together: the transform is what makes a signed support cheap.

---

## Priority 4 — Gradient scaling and loss weights on the unroll

**Status: DONE.** `scale_gradient` applies 1/K per unroll step and halves the
gradient entering each recurrent dynamics step; the value loss is weighted 0.25
against policy and reward. Verified by `tests/test_training_scale.py`, including
that a 5-step unroll now attenuates by 0.5^5 rather than amplifying by 5, and
that total gradient no longer scales with unroll depth.

MuZero scales the gradient entering each unroll step by `1/K` so that deeper
steps do not dominate, and weights the value loss (typically 0.25) against
policy and reward. Currently all five unroll steps contribute at full weight and
all loss terms are summed raw. This affects training stability at any search
budget and is cheap to add — but it changes effective learning rates, so it
wants a before/after loss-curve comparison rather than a blind commit.

---

## Priority 5 — Rate-limit self-play to the target frame rate

**Status: DONE.** `RateLimiter` paces `gameplay_process`, and
`real_time_step_period` derives the target from the env's own frameskip rather
than assuming one. That detail mattered: MsPacman-v4 samples its frameskip from
(2, 5), so real time is **17.1 decisions/second, not 60** — pacing at GAME_FPS
would have run 3.5x too fast and produced a limiter that did not limit.

Measured: unpaced self-play runs at 225 decisions/s (13x real time on this
hardware); paced it holds 17.1, with zero overruns. **The acting path uses 2.2 ms
of a 58.3 ms budget, so the agent has 96% of real time spare** — the strongest
evidence yet that the searchless-acting design has room, and the first number
this project has produced that speaks directly to its own thesis.

Set `SELF_PLAY_REAL_TIME = False` to let experiments run unpaced; the ablation
harness does not use the limiter, deliberately.

ALE runs MsPacman at ~4,900 steps/s, roughly **80x real time**. The testbed
therefore does not reproduce the constraint it exists to model, and every
throughput measurement answers a question the real system never asks — see the
banner in `benchmarks/RESULTS.md`. Capping `gameplay_process` at the target
frame rate makes producer/consumer ratios meaningful and would let us measure
whether learning actually keeps up in real time, which is the project's whole
question.

Cheap to build, and it changes how every subsequent benchmark should be read.

---

## Smaller known issues

| Issue | Where | Note |
|---|---|---|
| `get_state_dict` ignores nested optimiser state | `muzero.py` | Works only because SGD is stateless; silently ships GPU tensors if Adam returns |
| `take_random_gameplay` (né `get_max_gameplay`) samples at random | `muzero.py` | Renamed to match the behaviour; priority selection remains an option if reanalysis ever falls behind again |
| Dirichlet `alpha=0.03` over 5 actions | `muzero.py` | ~92% of noise mass on one action, documented in `tests/test_search_math.py`. Moot if Gumbel ever lands, since it removes Dirichlet |
| `playpacman.py` frame stack is broken | `playpacman.py` | `np.repeat(obs, 12, axis=0)` on `(84,84,3)` gives `(1008,84,3)`; stores single frames, not stacks. Needs a decision on the human-demo data format |

---

## Backburner

### Gumbel MuZero

**Deferred, not rejected.** Its headline benefit — policy improvement at
single-digit simulation counts — pays off when search must fit a frame budget.
With search offline at 32 simulations over 5 actions, visit counts differentiate
fine and the failure mode does not arise.

**Revisit if:** search moves onto the acting path (measured: n≤8 fits ~40% of a
60 fps frame, n=32 needs 94%), **or** if re-analysis throughput becomes binding
again and trading simulations for timestep coverage looks attractive — n=8 at
stride 2 costs 0.7x of n=32 at stride 4 while producing 2x the policy targets.
That trade assumes n=8 Gumbel targets match n=32 visit-count targets, which the
paper does not establish for Ms. Pac-Man; it would need measuring.

### EfficientZero off-policy correction

**Deferred — design tension, not just priority.** The correction reshapes a
*bootstrapped* multi-step target:

```
z_t = Σ_{i=0}^{l-1} γ^i u_{t+i} + γ^l ν_MCTS(s_{t+l}),    l ≤ k, smaller for older data
```

This repo deliberately uses **Monte Carlo returns with no bootstrapping** (see
"Custom ideas" in the README, and `create_tree_from_game`'s backup of 0). There
is no `γ^k v_{t+k}` term to correct. The off-policy problem still exists in a
different form — MC returns are unbiased for the behaviour policy, not the
current one — but the fix does not port directly. Resolve the value-target
design question first.

---

## References

- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a
  Learned Model* (MuZero), 2020
- Ye et al., [*Mastering Atari Games with Limited Data*](https://arxiv.org/abs/2111.00210)
  (EfficientZero), NeurIPS 2021 — §4.1 consistency loss, §4.2 value prefix,
  §4.3 off-policy correction
- Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022
- Wang et al., [*EfficientZero V2*](https://arxiv.org/abs/2403.00564), 2024
- [LightZero](https://github.com/opendilab/LightZero) — reference
  implementations of EfficientZero and Gumbel variants, useful for the details
  marked *(verify)* above
