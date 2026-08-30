# Iterative curriculum learning (branch design record)

Goal: a generalized learning algorithm that learns fast enough for real
time, by iterating disposable networks over a shrinking abstraction ladder
— early networks overfit cheap simplified views of the game and are
discarded; later networks search finer resolutions using their
predecessors as value guides. Overfitting early is expected and accepted.
Target games: Pong first (worst case for temporal abstraction), Tetris
later (best case, and the watch-other-players motivation).

Status legend: [x] built · [ ] designed, not built · (?) open question.

## P1 — Observation and action simplification  [x]

- [x] Grayscale, two channels (`curriculum_env.py`):
  ch0 = max of last two RAW frames (flicker defence, nothing else);
  ch1 = age-decayed |frame-diff| motion trail over ~2 decision intervals
  (motion-history images): moving objects appear as a fading streak whose
  bright end marks the present — position, direction, and speed explicit
  in the input. Kai's diff-composite idea; separate channel rather than
  overlay so busy scenes stay unambiguous.
- [x] Decision contract: the chosen action is held for ALL skip raw
  frames; step reward = SUM of raw-frame rewards in the window.
- [x] Minimal actions via explicit map (`PONG_MINIMAL_ACTIONS = (0,2,3)`;
  the harness's first-N restriction would drop LEFT).
- [x] Skip feasibility scan (`experiments/skip_scan.py`). Pong result:
  scripted tracker −8.0 at skip 4, collapses to random by skip 8 —
  control bandwidth, not perception. Fixed high skip is unlearnable on
  Pong; motivated the duration menu below.
- [ ] One-hot action planes + residual skip in dynamics (P2, with the
  2-channel trunk).

## P2 — Duration menu (replaces the fixed-skip ladder)

The policy chooses (action, duration) with durations {4, 8, 16, 32}
frames. Early stages allow the full menu; later stages constrain via a
LEGALITY MASK over the duration head (never head surgery — masks keep
master compatibility and reversibility). Kai's intent: later networks are
forced toward smaller steps for generalization, using earlier networks'
value estimates as guides; long-duration bias early is a chosen
exploration pressure (explicit duration-head prior, annealed).

- Factored policy: action head × duration head (not a flat product) so
  "UP is good here" generalizes across durations and effective branching
  collapses as either head learns.
- Discounting: γ defined PER RAW FRAME; every edge discounted
  γ_frame^τ for its own duration τ (`decision_discount`). A fixed
  per-decision γ silently favors long steps (more undiscounted reward,
  same bootstrap discount) — the audit-grade hazard of this design.
- n-step targets in FRAME TIME: bootstrap at the state ≥ H raw frames
  ahead (H = n × max duration), rewards discounted by frame offset,
  bootstrap by γ_frame^H. A single 32-frame step and eight 4-frame steps
  covering the same window produce identical targets (Kai's equivalence,
  formalized). The 4.3 age-shrink also becomes frame-denominated.
- Dynamics conditioned on duration (one net, duration input) — replaces
  per-stage dynamics retraining entirely.
- Composition consistency loss: holding action a for 32 frames must land
  where composing 8× (a, 4) lands. Cosine in the SimSiam projection
  space, stop-grad on the composed short chain (plain latent MSE
  collapses; see LatentProjector docstring). Reward-sum consistency
  rider: long edge reward prediction = Σ short edge predictions. Valid
  exactly only in deterministic envs (fine on Pong); soften under
  stochasticity. Apply to a batch subsample (composition costs ~8 extra
  dynamics forwards per sample).
- (?) Duration-aware scripted probe before training: short steps when the
  ball approaches, long otherwise — validates "same play, far fewer
  decisions" for zero training cost.

## P3 — Student / master

Master = best network so far, frozen; retains only its VALUE heads
(multi-discount). Student plays; the master guides exploration through a
value deficit, never a direction (a value-only master cannot know which
action):

    δ(s) = max(0, V_m(s) − V_s(s)) / (|V_m(s)| + ε)

evaluated at the SAME frame-aligned discount (master multi-discount heads
read at the student's operating γ_frame).

- δ controls, in priority order: (a) per-state simulation budget,
  sims(s) = base × (1 + k1·δ) — search harder where the student provably
  leaves value on the table (composes with real-time budgets: high-skip
  states have the most think time); (b) root Dirichlet weight,
  ε(s) = ε0 + k2·δ, clamped; (c) optionally acting temperature.
  NEVER a per-action PUCT bonus — with no action knowledge any
  directional push is fabricated.
- Value distillation (pulling student heads toward V_m at deficit
  states) deferred: max-biased; start exploration-only.
- Promotion: student evaluated argmax over N episodes; promote on mean
  return > master's matched eval. Retired masters keep value heads only —
  which is also what makes watch-only masters identical in interface.
- Multi-discount value heads ({0.997, 0.99, 0.98, 0.95} per raw frame)
  are cheap auxiliaries on the shared trunk (hyperbolic/GVF lineage) and
  the master-comparison substrate.

## P4 — Watch-only masters and the action guesser

- Watch-only master: TD over observed frames+rewards of anyone's play
  learns the demonstrator's on-policy V at multiple discounts — no action
  labels needed. This is the Tetris-from-video path.
- Action-guessing network (NOTED 2026-08-27, deferred): inverse dynamics
  â = f(s_t, s_{t+1}) trained on the agent's OWN play (actions known),
  then run over observed master gameplay to impute actions — the BCO
  recipe (Torabi et al. 2018). MCTS integration: imputed master policy
  blends into search priors, π = (1−β)·π_student + β·π̂_m with β ∝ δ(s),
  so directional advice scales with the student's measured deficit and
  decays to zero as it catches up. High skip makes the inverse problem
  EASIER (bigger frame deltas per transition): coarse guessing first,
  refined down the ladder, matching the curriculum's shape.

## Infrastructure: concurrent harness (Ray) — P3-adjacent

Training and gameplay overlap (added 2026-08-29, Kai). The ablation
harness is deliberately single-process/sequential (paired seeds, atomic
whole-world checkpoints, clean attribution); the PRODUCT pipeline already
runs Ray actors concurrently, and EfficientZero's own system is fully
concurrent, so overlap is more faithful, not less. Worth up to ~30-40%
wall clock (iteration = max(acting, training) instead of the sum). Build
it WITH the student/master phase, which wants separate processes anyway
(student plays continuously, learner trains, master evaluates async —
the real-time architecture in miniature). Requirements: Ray actors for
self-play + learner, GPU sharing via mcts_gpu_share, pause-the-world
checkpointing (the atomic resume story must survive), and an explicit
statement that paired-seed reproducibility is given up in this mode.
Sequenced AFTER batch-16 acting and the architecture rungs, which bought
their speedups without touching concurrency semantics.

Refresh offload (designed 2026-08-29, Kai's swap-on-ready idea): a
refresh actor re-searches trajectories with the target model in the
background and returns fresh targets; the DRIVER applies the swap at a
safe point in its own loop -- single-threaded, so the loop is the mutex
and no lock is needed. Real design constraints: (1) eviction races --
swap matches trajectories by object identity and drops results for
evicted ones (wasted work, never corruption); (2) observation shipping
(~800 MB/request) argues for birthing trajectories in the Ray object
store. The payoff is NOT the current ~1-2 min/iter (communal already
made refresh cheap) but making RAISED refresh coverage free: 5-10x the
revisit rate hidden entirely under training -- pair the offload with the
coverage increase, never ship it alone.

Regime note (Kai, 2026-08-29): the transitions cap makes buffer size in
TRAJECTORIES a moving target -- ~25-35 early (short losing episodes),
~10-12 late (long rallies) at the same 31.25k cap. The eviction race is
a late-regime concern only; the identity guard is cheap insurance, not a
dominant cost. And the pieces form ONE system with the curriculum's
entropy-prioritized revisiting: the tiny buffer creates ~50x sample
reuse, which AMPLIFIES every flat stored target (it drags the policy
toward flatness dozens of times per iteration), so re-searching the
flattest targets first repairs the most-reused weakest signals -- the
measured touch-high-then-give-back oscillation is the failure mode this
aims at. Entropy-priority's value scales with reuse, reuse scales
inversely with buffer size, and the offload is what makes the volume
free. The no-sharpening guard separates repairable flatness from
genuine ambiguity, which dominates in a fresh tiny buffer.

### Phase 3 spec (2026-08-30, approved direction; build gated per-step)

Ordering principle: within-tree search stays sequential everywhere -- the
2026-08-30 re-audit proved wave-batching flattens policy targets
dose-dependently and that no budget buys it back (probe_budget_sweep:
even at matched wave count the batched distribution never converges to
sequential's, and on weak nets big budgets overshoot into noise-driven
overcommitment). All parallelism is ACROSS trees/processes.

Step 3a -- refresh actor + raised coverage (one package, never split):
- Third Ray actor holding the lagged target nets; driver ships
  trajectory refs (trajectories birthed in the object store to kill the
  ~800 MB/request copy) + born_step; actor runs the communal windowed
  re-search (width 16, probe-validated) and returns fresh targets.
- Driver applies swap-on-ready at one fixed point in its loop
  (single-threaded loop = the mutex); swaps are identity-checked
  against eviction (a result for an evicted trajectory is dropped --
  wasted work, never corruption).
- Ships WITH 5-10x revisit coverage; the offload's payoff is making
  that volume free, not saving the current ~1-2 min/iter.
- Refresh candidate ordering becomes ENTROPY-PRIORITIZED (flattest
  stored targets first) with the no-sharpening guard (stop re-searching
  steps whose refreshed entropy does not drop). If the archfix verdict
  is prior-echo stall (sharp + parked returns), the guard is mandatory
  in the same commit; if archfix is healthy, priority still ships --
  its value scales with the tiny buffer's ~50x reuse either way.
- Target-model sync cadence: actor refreshes its nets at the same
  iteration boundary the driver's lagged copy updates; a stamp on each
  result ties targets to the model version that produced them.

Step 3b -- N self-play workers + shared inference service (EffZero's
own shape; the acting twin of communal's cross-root batching):
- Lightweight env+tree workers (CPU: env stepping, tree ops, PUCT
  bookkeeping) send leaf-evaluation requests to ONE GPU inference actor
  that batches ACROSS workers. Per-tree schedule stays strictly
  sequential; the GPU batch is formed from different trees' single
  outstanding requests, so target quality is untouched by construction.
- This is a RECIPE change (N episodes/iter alters data cadence and the
  train-ratio); it gets its own design pass on cadence + 4.3 age
  bookkeeping before build, its own audit loop, and a validation A/B
  like phases 1-2. Paired-seed reproducibility is explicitly given up.
- Cancellation story unchanged in kind: pause-the-world checkpointing,
  drain in-flight episodes or drop them (same cost as today's kills).
- Batch-size note: the service naturally reproduces acting-probe
  batch-16-ish GPU efficiency WITHOUT wave semantics -- the batch is
  across independent trees.

Step 3c -- driver-side latency hiding (no semantics changes): PER
root-list caching + batch prefetch (assemble batch k+1 while k trains),
then AMP as its own measured change; CUDA graphs only if profiling
still shows launch overhead after AMP.

Zero-code lever available any time: parallel seeds of the same arm on
one GPU (evidence throughput, no semantic change to any single run).

## Carried decisions from the repro line

Tiny buffer + live refresh direction; entropy-based refresh priority is
Kai's chosen decisiveness-annealing mechanism in this regime (guard: stop
re-searching steps whose refreshed entropy does not drop). Reward-clip
growth curriculum (inert on Pong, real elsewhere) sized against the
categorical supports. Evaluation of this algorithm vs the repro line is
END-TO-END ONLY: return vs wall-clock/game-time, not per env step.

## Related work anchors

FiGAR / dynamic frame skip (duration-in-action-space), options & SMDPs
(γ^τ math), Sampled MuZero + progressive widening (action-space growth,
the continuous future), motion history images (ch1), BCO (action
guessing), hyperbolic/multi-γ heads (auxiliary value ladder).
