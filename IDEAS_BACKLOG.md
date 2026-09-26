# Ideas backlog (low priority)

Raw ideas parked for later. Nothing here is planned; ROADMAP.md and
CURRICULUM.md are the actual plans. Each entry keeps the objections raised at
the time so the idea can be picked up without re-deriving them.

## Two dynamics networks: grounded one-step + imagination (Kai, 2026-09-23)

**Idea.** Network A takes the real latent h(obs_t) and predicts only the
immediate reward and next latent (one step, no unroll). Network B does the
multi-step rollouts and is trained only to agree with A. A stays grounded in
reality; B doesn't need to predict rewards.

**Objections found (why it's parked):**
- Doesn't avoid the projector: A's one-step latent loss still has both sides
  learned, so it needs the same anti-collapse machinery. The projector collapse
  comes from the loss form, not the horizon.
- B's target: copying A's own rollout makes B a copy of A; using A applied to
  the real state at t+k-1 is a noisier version of h(obs_t+k), which is
  already available.
- Search must run multi-step through some network. A was never trained on its
  own outputs (teacher-forcing / exposure bias); the value/policy heads were
  never trained on B's latents. The value and policy losses have no place in
  the design.
- Compounding is already damped: per-step loss x 1/UNROLL_STEPS and the
  latent gradient is scaled by LATENT_GRADIENT_SCALE every step
  (muzero.py:2463, 2472).
- Pong: immediate reward is ~always 0 (hence EfficientZero's value prefix),
  and action effects are weak one step ahead. Counterfactual disc@1 is
  0.38-0.43 for every model (chance 0.33) while disc@3 separates archfix
  0.70 from ~0.50 (benchmarks/probes/probe_counterfactual_edge.log).
- Prior art: PlaNet's latent overshooting (Hafner et al. 2019) trains
  multi-step predictions toward one-step estimates with stop-grad; small gain,
  dropped in Dreamer (from memory, so check before relying on it).

**Cheaper testable core, if ever revisited:** one dynamics net, plus a
teacher-forced one-step loss at every unroll step (apply g to the real
h(obs_t+k-1), score against h(obs_t+k) and the reward), and/or stop the
gradients from steps >= 2 into the representation network.
