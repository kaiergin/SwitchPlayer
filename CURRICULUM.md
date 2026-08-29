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
