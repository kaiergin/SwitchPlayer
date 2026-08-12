## Reinforcement Learning on the Nintendo Switch

Games to learn:
* Tetris 99 (current)
* Mario Maker 2

## Purpose

**This is a real-time reinforcement learning research project.** The goal is an
agent that learns *while* playing a live console — video in over a capture card,
controller commands out over serial — with no pausing, no rewinding, and no
faster-than-real-time simulation.

MuZero is the **starting point, not the destination**. It was chosen because it
is the strongest model-based algorithm that reliably converges to a good policy,
but it is not a real-time algorithm: stock MuZero runs hundreds of tree-search
simulations to select each action, which cannot fit inside a 16.7 ms frame at
60 fps. The implementation here is expected to diverge from the paper, and
already does.

### What "real-time" changes

The binding constraint is **latency per action**, not sample efficiency and not
throughput. A technique that halves the sample requirement but doubles acting
latency is a regression here.

Deviations from stock MuZero that are deliberate, and should not be "corrected"
back toward the paper without discussion:

| Stock MuZero | Here | Why |
|---|---|---|
| MCTS selects every action | Agent samples the policy head directly (`MuZeroAgent.gameplay_process`) | Search does not fit the frame budget |
| Search produces actions | Search runs off the critical path as offline re-analysis over stored trajectories (`MCTSHelper`) | Keeps acting latency independent of search cost |
| Bootstrapped n-step value targets | Monte Carlo returns, no bootstrapping, during warmup | See "Custom ideas" below |

### Measuring against the goal

Benchmarks must report **acting latency against the frame budget**, not only
throughput. `benchmarks/RESULTS.md` has the current numbers; the one that
matters most is `initial_inference` at 2.24 ms/frame — about 13% of a 60 fps
budget, which is the real headroom figure.

Be careful reading testbed throughput results: ALE runs MsPacman at ~4,900
steps/s, roughly 80x real time, so producer/consumer ratios measured there do
not reflect the real system, where self-play is capped at the console's frame
rate. Rate-limiting self-play to the target frame rate would make the testbed
reproduce the actual constraint — not yet implemented.

`MsPacman-v4` is only a testbed. `muzero.py` trains against it today; wiring it
to the Switch environment is still to do.

### Not yet implemented: MuZero successor work

Several papers improve on MuZero by replacing specific components, particularly
**how future reward is predicted through the dynamics network**. None are
implemented here yet. Verify details against the papers before building.

* **EfficientZero** — replaces per-step reward prediction with an LSTM *value
  prefix* over the unrolled dynamics; adds a self-supervised latent consistency
  loss (cosine similarity through a projector) and off-policy correction. The
  plain MSE latent consistency loss in `train_step` is a weaker form of the
  same idea.
* **Gumbel MuZero** — preserves the policy improvement guarantee at very small
  simulation budgets. The most directly relevant to the real-time constraint,
  since it attacks the reason search is unaffordable rather than working around it.
* **EfficientZero V2** — the above extended to continuous control and sampled actions.
* **Stochastic MuZero** — chance nodes, for environments that are not deterministic.
* **Sampled MuZero** — large or continuous action spaces.
* **MuZero Unplugged / Reanalyse** — the offline re-analysis framing `MCTSHelper`
  already resembles.

### To do list

See **[ROADMAP.md](ROADMAP.md)** for the prioritised plan and the reasoning
behind it. Current priority is EfficientZero's end-to-end value prefix.

Unsequenced items:
* Fill replay buffer with random data
* Mult-step reward?
* Hyperparameter Tuning -- c_puct and the search budget both need re-tuning now
  that MCTS values are min-max normalised

### Custom ideas
* Heavy reward discounting and small trajectory with no bootstrapped value as initial agent
* Human gameplay as training for dynamics network

### Dependencies

The active pipeline targets **Python 3.12** with PyTorch and Ray. Not 3.13:
ray publishes no `win_amd64` wheel for cp313.

```
winget install --id Python.Python.3.12 --exact
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

Verified on an RTX 4070 with torch 2.13.0+cu130, ray 2.57.0, gymnasium 1.3.0.
`requirements.lock.txt` is a full freeze of that environment.

The legacy Switch/Tetris environment (`controller.py`, `core.py`, `tetris.py`,
`envTest.py`) targets Python 3.7 with TensorFlow and needs a separate
environment -- see the notes at the bottom of `requirements.txt`.

Currently only support for Windows machines  

### References

[Repo for sending commands to switch](https://github.com/wchill/SwitchInputEmulator)  

### Hardware Used
* [Arduino Uno R3](https://www.amazon.com/Sintron-UNO-R3-ATMEGA328P-Arduino/dp/B073DYD97C/ref=sxin_2_ac_d_pm?ac_md=1-0-VW5kZXIgJDEw-ac_d_pm&cv_ct_cx=arduino+uno+r3&keywords=arduino+uno+r3&pd_rd_i=B073DYD97C&pd_rd_r=9305d6a8-c762-46e8-b3f8-356745bc8a6d&pd_rd_w=z4ysu&pd_rd_wg=MwvP1&pf_rd_p=0e223c60-bcf8-4663-98f3-da892fbd4372&pf_rd_r=RAANRF34YJKT950W9MVQ&psc=1&qid=1584519878&s=electronics)
* [USB to serial](https://www.amazon.com/gp/product/B01CYBHM26/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
* [Camlink 4k](https://www.amazon.com/gp/product/B07K3FN5MR/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)