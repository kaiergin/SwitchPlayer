# Baseline measurements

RTX 4070 (12 GB, sm_89) · 16 logical cores · Python 3.12.10 · torch 2.13.0+cu130 · ray 2.57.0

Reproduce with `python benchmarks/bench.py all`. Numbers are medians.

> **Read these against the frame budget, not as throughput.** This is a
> real-time RL project (see README): the number that decides whether the system
> works is acting latency per frame — `initial_inference` at **2.24 ms**, about
> 13% of a 60 fps budget.
>
> ALE runs MsPacman at ~4,900 steps/s, roughly **80x real time**. Throughput
> ratios measured here therefore overstate the pressure on the real system,
> where self-play is capped at the console's frame rate. The producer/consumer
> section below is honest about the testbed but should not be read as a
> statement about live play.

## Components

| Stage | Median | Throughput |
|---|---:|---|
| `preprocess_frame` | 0.02 ms | 52k frames/s |
| `push_frame` | 0.01 ms | — |
| `env.step` (ALE) | 0.20 ms | 4.9k steps/s |
| `initial_inference` (1 frame) | 2.24 ms | 446 acts/s |
| `batch_inference` (batch 16) | 2.37 ms | 6.8k frames/s |
| `batch_inference` (batch 32) | 3.09 ms | **10.4k frames/s** |
| `batch_inference` (batch 64) | 7.06 ms | 9.1k frames/s |
| `batch_expand_nodes` (8 nodes / 40 evals) | 3.18 ms | — |
| `batch_expand_nodes` (40 nodes / 200 evals) | 13.05 ms | — |
| `run_full_mcts` (32 sims) | 12.84 ms | 78 searches/s |
| `eval_obs` (200-step trajectory) | 24.69 ms | — |
| `train_step` (batch 64, unroll 5) | 85.37 ms | 11.7 steps/s |

The `run_full_mcts` figure is slightly pessimistic: the harness calls
`asyncio.run` per search, creating a fresh event loop each time. Inside an
actor, where searches share one loop, the effective rate is ~106/s.

## Startup

| Phase | Time |
|---|---:|
| `ray.init` | 3.41 s |
| network + storage actors | 2.42 s |
| MCTS actor construct + setup | 1.50 s |
| **total to first search** | **7.32 s** |

Not a problem. No action needed.

## MCTS actor scaling

6 trajectories × 60 steps, reanalysing every timestep.

| Actors | Wall | traj/min | Speedup | GPU MB | GPU/actor |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.4 s | 107.3 | 1.00x | 1934 | 0.60 |
| 2 | 2.2 s | 162.4 | **1.51x** | 1933 | 0.30 |
| 3 | 2.1 s | 169.5 | 1.58x | 2036 | 0.20 |

**Two actors are worth having; three are not.** Going 2 → 3 buys 4% for another
process. `NUM_MCTS_ACTORS = 2` is the right default on this machine.

**VRAM is not the constraint.** The whole fleet sits at ~2 GB of 12 GB
regardless of actor count, so the fractional `num_gpus` accounting is nowhere
near the real limit. The ceiling is GPU *contention*: MCTS issues many small
kernel launches (8 leaves × 5 actions = 40 evals per expansion), so the device
is launch-latency bound rather than compute bound, and extra processes mostly
queue behind each other.

## Reanalysis stride

Searching every timestep was the whole problem. Cost of reanalysing one
200-step trajectory as `REANALYSE_STRIDE` varies:

| Stride | Searched | Wall | Per search | traj/min | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 2.09 s | 10.5 ms | 28.6 | 1.00x |
| 2 | 100 | 1.07 s | 10.7 ms | 56.3 | 1.97x |
| **4** | **50** | **0.55 s** | **11.0 ms** | **109.1** | **3.81x** |
| 8 | 25 | 0.32 s | 12.8 ms | 188.0 | 6.56x |
| 16 | 13 | 0.16 s | 12.0 ms | 383.5 | 13.39x |

The saving is close to linear. Skipping indices does cost some tree reuse —
lazy expansion lets neighbouring searches share subtrees — but only 1.15x on
per-search cost even at stride 16, so it barely eats into the win.

## The producer/consumer gap, before and after

End-to-end through real Ray actors, 100-step trajectories:

| | 1 actor | 2 actors | 3 actors |
|---|---:|---:|---:|
| Search every step | 69.4 | 113.5 | — |
| **Stride 4** | **277.5** | **439.0** | 477.7 |

At two actors that is a **3.87x** improvement in trajectories reanalysed per
minute. Scaled to a 200-step episode against self-play's ~122 trajectories/min:

| | Rate | vs self-play |
|---|---:|---|
| Self-play generates | ~122/min | — |
| Reanalysis, every step, 2 actors | ~57/min | **2.1x deficit** |
| Reanalysis, stride 4, 1 actor | ~139/min | 1.1x surplus |
| Reanalysis, stride 4, 2 actors | ~220/min | **1.8x surplus** |

The gameplay buffer no longer overflows, so `take_random_gameplay` (then named `get_max_gameplay`) stops silently
discarding self-play at random.

Note that **one actor is now enough** to keep up, with ~14% margin.
`NUM_MCTS_ACTORS = 2` is kept for headroom — episode lengths vary and the
margin is cheap — but dropping to 1 to reclaim 2 CPUs and a GPU share is a
reasonable call.

Stride 2 would leave only ~112/min against 122/min needed, i.e. still short.
Stride 4 is the first setting with real margin, which is why it is the default.

Remaining levers, if more is ever needed:

- **Lower `TOTAL_MCTS_SIMS`** (32 today). Cost is close to linear in sims.
- **Raise `MCTS_BATCHES`** (8 today) so each expansion is a bigger kernel: 40
  nodes cost 4.1x what 8 cost for 5x the work. Costs search quality, since
  more simulations run without seeing each other's results.
- **Throttle self-play**, which is cheap anyway — inference-latency bound at
  2.24 ms/step, not env bound at 0.20 ms/step.

Scaling efficiency across actors is *worse* at stride 4 (57%) than at stride 1
(82%), which is expected: less GPU work per trajectory means coordination
overhead is a larger share. Absolute throughput is far higher regardless.

## Incidental findings

- `batch_inference` peaks at **batch 32**, which is what `eval_obs` already
  uses. Batch 64 is slower per frame; do not raise it.
- Single-frame `initial_inference` gets 446 frames/s against 10.4k frames/s
  batched — a 23x penalty, entirely launch overhead. Self-play acts one frame
  at a time by nature, but it means the acting path is latency bound and would
  not benefit from a bigger network nearly as much as it looks.
- `train_step` at 11.7 steps/s means the 100-step checkpoint interval fires
  roughly every 8.5 s, which is frequent for a `torch.save` of the full model
  plus optimiser state.

## Value encoding and the value prefix

`python benchmarks/bench.py prefix` trains a per-step reward head and an LSTM
value-prefix head on the same trunk, same batches, same steps, then compares
prefix prediction error at each unroll depth on held-out trajectories.

### Read this section with the control column

The harness now reports the MAE of the trivial **always-predict-zero**
predictor. It was added after two successive runs produced confident-looking
results that turned out to be artifacts, and it immediately invalidated both.

20 episodes, action repeat 8 (0.53 reward/step), Adam at 1e-3, 4000 steps:

| depth | predict-0 | per-step | LSTM | per-step vs 0 | LSTM vs 0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.79 | 1.01 | 0.79 | 1.27 | 1.00 |
| 3 | 1.99 | 2.38 | 1.99 | 1.20 | 1.00 |
| 5 | 3.33 | 3.70 | 3.33 | 1.11 | 1.00 |

**Neither head beats the trivial predictor at any depth.** The LSTM sits at
exactly 1.00 because it collapsed to predicting zero — its decoded output reads
`-0.00` throughout training. The per-step head appears to have converged to the
mean reward (~0.53/step), which loses to the mode (zero) on MAE because the
reward distribution is sparse. Both are constants. Neither has learned
state-dependent structure.

Any ratio between the two heads is therefore noise, and a head that collapses to
a constant will look like it "won".

### Two corrections to earlier readings of this benchmark

1. **The value prefix has never been shown to beat per-step here.** An early run
   reported 3.30x in its favour; that run gave the prefix head a hand-scaled
   support while leaving the per-step head on a broken linear one, so it
   measured the encoding, not the architecture.
2. **The "h() gives a 5x win" figure was overstated.** Per-step MAE at depth 5
   did go from 8.08 to 1.63, but against a predict-zero baseline of ~1.26 that
   is a move from *6.4x worse than a constant* to *roughly equal to a constant*.
   h() removed a pathology. It did not make the head useful.

h() is still worth keeping on its own merits — signed support (negatives were
previously unrepresentable), correct round-trip, and resolution that no longer
collapses realistic values into one bin. Those are verified by
`tests/test_scalar_categorical.py`, not by this benchmark.

### Why nothing learns here

Untested, in the order worth checking:

- **Dataset scale.** ~4,000 transitions from a random policy. EfficientZero's
  results are on 100k environment steps with a training agent.
- **Task difficulty from these observations.** Predicting whether a pellet is
  eaten 1-5 steps ahead, from an 84x84 downsample, may simply not be learnable
  at this data scale.
- **Loss competition.** The trunk is pulled by policy, value, reward, prefix and
  latent-consistency losses at once, unweighted and unscaled — see ROADMAP
  priority 4.

Until a head beats the control, this benchmark cannot discriminate between the
two architectures. Do not draw conclusions from it about the value prefix.

## Ablation: 5 arms x 2 seeds x 60 iterations (4.9 h)

`python experiments/ablation.py`. Each iteration is 2 self-play episodes,
re-analysis with MCTS, and 250 training steps — about 15,000 training steps per
run, on real self-play data with search in the loop.

### h() works. This is the first properly controlled evidence for anything.

Reward-prediction error over the last five evaluations, as a ratio to the
predict-zero control (below 1 means the head learned something):

| Arm | per-step / control | verdict |
|---|---:|---|
| per-step (with h) | **0.24** | learned |
| horizon-0.9 (with h) | **0.13** | learned |
| value-prefix (with h) | **0.29** | learned |
| **no-transform (linear)** | **1.12** | **never beat the control** |

The same encoding change also fixes the value head's start: without h() its MAE
opens at ~650 and takes 20 iterations to converge; with h() it is near zero
immediately. And it stabilises runs — final-return spread across seeds is 4.5
with h() against 105.0 without.

### The value prefix does not learn. Anywhere.

`prefix / control` is **1.000 in every single arm**, including the arm where the
prefix is driving the tree. The head is collapsed to predicting zero, so in that
arm MCTS was running with edge rewards of ~0 throughout.

That is now four independent measurements agreeing. `USE_VALUE_PREFIX_IN_TREE`
therefore defaults to False.

Leading hypothesis, untested: the prefix target is a 5-step undiscounted sum,
mostly zero with occasional spikes, encoded over 183 bins. Predicting the centre
bin is a strong local optimum, and the prefix loss competes unweighted against
four others (ROADMAP priority 4). The per-step head faces a milder version of
the same skew over 79 bins and escapes it.

### Return differences are noise at this scale

| Arm | final return | seed spread | peak | change over run |
|---|---:|---:|---:|---:|
| per-step | 85.2 ± 39.6 | 4.5 | 470 | −71.0 |
| value-prefix | 136.2 ± 81.7 | 52.5 | 350 | +15.2 |
| no-transform | 91.5 ± 58.5 | 105.0 | 222 | +24.5 |
| **horizon-0.9** | 131.0 ± 45.5 | **8.0** | **500** | **+50.8** |
| horizon-0.5 | 108.0 ± 35.2 | 10.0 | 345 | +37.8 |

value-prefix leads per-step by +51 (Cohen's d 0.75), but its seed spread is 52.5
against per-step's 4.5, on n=2. That is not a result. Note also that the
value-prefix arm was effectively planning with no reward model at all, since its
head outputs zero — worth a hypothesis that a broken reward model in the tree is
worse than none, but not a claim.

**gamma=0.9 is the most consistent arm**: smallest seed spread (8.0), largest
improvement over the run (+50.8), highest single evaluation (500). Kai's
fast-convergence intuition holds up on consistency even though the final means
are within noise of each other.
