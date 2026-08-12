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

The gameplay buffer no longer overflows, so `get_max_gameplay` stops silently
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

## Value encoding: the h transform

`python benchmarks/bench.py prefix` trains a per-step reward head and an LSTM
value-prefix head on the same trunk, same batches, same 3000 steps, then
compares prefix prediction error at each unroll depth on held-out trajectories.

### h() is a large win, for both heads

Prefix MAE at unroll depth 5, per-step head:

| Encoding | MAE @ depth 5 |
|---|---:|
| Linear support, 51 bins | 8.08 |
| **h() + bins scaled to the transformed range** | **1.63** |

A 5x improvement from the encoding alone, no architectural change. With a linear
support the bins were 40 wide while realistic prefixes were 0-5, so every target
landed in one bin. Under h() with ~0.5-wide bins in transformed space, the first
bin covers raw 0-1.25 while the support still reaches the worst case.

### The value prefix is NOT yet shown to help here

With both heads properly resolved, per-step wins at every depth:

| depth | true | per-step MAE | LSTM MAE |
|---:|---:|---:|---:|
| 1 | 0.27 | **0.28** | 24.38 |
| 3 | 0.86 | **0.88** | 1.15 |
| 5 | 1.57 | **1.63** | 2.97 |

Extended to depth 10, the ratio peaks at 0.88x around depth 5 and falls again —
there is no crossover, the LSTM never overtakes.

**An earlier run appeared to show the LSTM winning 3.30x at depth 5. That was
confounded**: the prefix head had a hand-scaled support while the per-step head
was still on the broken linear one. It measured the encoding fix, not the
architecture.

What survives is the error-growth asymmetry — over depth 1 to 10, per-step error
grows 9.61x while the LSTM's *shrinks* 0.28x, which is the direction state
aliasing predicts. But the LSTM starts from a bad enough baseline that the trend
never pays off within a 10-step unroll.

Likely explanations, none yet tested:

- **Undertrained.** The LSTM has far more parameters and a harder optimisation
  than a conv head, and 3000 steps of SGD at lr 1e-4 is very little. Its loss is
  still noisy at the end of training while the per-step head has settled.
- **Almost no signal.** Under a random policy the true prefix is 0.17-1.79 over
  ten steps; there is very little reward to predict. EfficientZero's result is on
  a training agent that actually scores.
- **Unverified architecture.** Hidden size, pooling and BatchNorm placement were
  chosen here, not taken from the reference implementation.

Do not treat the value prefix as validated on this evidence.
