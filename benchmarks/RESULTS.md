# Baseline measurements

RTX 4070 (12 GB, sm_89) · 16 logical cores · Python 3.12.10 · torch 2.13.0+cu130 · ray 2.57.0

Reproduce with `python benchmarks/bench.py all`. Numbers are medians.

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

## The producer/consumer gap

Scaled to a realistic 200-step episode:

| | Rate |
|---|---:|
| Self-play generates | ~122 trajectories/min |
| Reanalysis consumes, 1 actor | ~32 trajectories/min |
| Reanalysis consumes, 2 actors | ~49 trajectories/min |
| Reanalysis consumes, 3 actors | ~51 trajectories/min |

Self-play outruns reanalysis by **~2.5x even with two MCTS actors**, and adding
actors cannot close the gap. `MAX_GAMEPLAY = 20` therefore overflows in steady
state and `get_max_gameplay` discards the excess at random.

Adding MCTS actors is not the lever. The options that are:

- **Reanalyse a subset of timesteps** rather than all of them. Nothing requires
  a search at every index; sampling would cut the cost proportionally.
- **Lower `TOTAL_MCTS_SIMS`** (32 today). Cost is close to linear in sims.
- **Raise `MCTS_BATCHES`** (8 today) so each expansion is a bigger kernel. 40
  nodes cost 4.1x what 8 nodes cost for 5x the work, so there is real headroom.
- **Throttle self-play**, which is cheap anyway — it is inference-latency bound
  at 2.24 ms/step, not env bound at 0.20 ms/step.

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
