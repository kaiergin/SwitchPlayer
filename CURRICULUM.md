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

## P5 — Watching others: IDM training, advantage-banded curriculum, chunks

Builds directly on P4. P4 established WHAT to learn from observation
(watch-only value, BCO-style action guessing) and how advice enters
search (π blend weighted by δ). P5 answers three questions P4 left open:
how to train the guesser when most inputs leave no trace, how to CHOOSE
what to watch, and how to commit multi-action plans. Kai's framing: the
agent searches to find where its own inverse model is wrong, and uses
that model only to judge whether watched play is worth imitating —
never to train the policy directly.

Order of work: P5a is built; P5g validates the whole mechanism against
self-play checkpoints BEFORE any video enters. VODs are on hold until
the comparator holds up on Kai's own recorded gameplay.

### P5a — Self-recorded corpus  [x]

- [x] `session_recorder.py` (sharded npz + provenance), `record_play.py`
  (capture/calibrate/play/log), `switch_control.py`, `switch_keymap.py`.
  Round-trip verified exact over 3200 steps / 3 shards.
- [x] Stores the REGISTERED 90x160 reference frame — not the raw
  capture (60× larger), not the 84x84 observation (discards everything
  outside PLAY_AREA). Every ROI in `tetris_masks` is defined against
  that frame, so observations, board occupancy and the crop are all
  recomputable. The reward function changed four times in one week;
  storing derived observations would have meant re-recording each time.
- [x] Stores the RAW 64-bit command, not a `tetris_enum` index. Human
  play produces chords (verified: `DPAD_L + BTN_A` -> index -1) that an
  8-way space cannot represent. This also makes P5f free: an
  action-space change does not invalidate recorded data.
- [x] Native numeral strips ONLY on glyph-candidate frames. Reward is
  read at native resolution (`clear_numeral_hires`), so the reference
  frame alone cannot reproduce it — but every-step strips cost ~26 GB/h.
- [x] Provenance in `meta.json`: `source` (human/agent) and `timing`.
  An IDM is valid ONLY for the timing it was recorded under; changing
  STEP_PERIOD or baud moves the action->observation offset and silently
  invalidates every label.
- [x] Storage measured: 0.02 GB/h (flat frames) to 3.11 GB/h (pure
  noise). Real footage between; the harness prints the true figure per
  session. Records only pregame/playing by default.
- [x] Timing: STEP_PERIOD 20 -> 30 Hz. T99 runs at 60 fps and DAS is
  specified in FRAMES; 20 Hz could neither observe nor reproduce charge
  timing. Deferred ACK (`ACK_NONBLOCKING`) reads packet N's reply at the
  start of N+1: 8.00 -> 4.12 ms median, 0/200 failures, serial 24% ->
  12% of a 33 ms step. `sync()` still blocks — the handshake is
  sequential, a deferred ACK would score against the wrong packet.
- Short-tap latch: `GetAsyncKeyState` bit 0x0001 ("pressed since last
  call") as well as 0x8000. Without it any press shorter than the step
  period falls between samples and is never logged OR sent.

### P5b — IDM objective under unobservable inputs

The central difficulty, and Kai's: most inputs leave no trace. Inputs
1-7 identical with only 8 acting five frames later; or all eight inert
for the whole window. Forcing cross-entropy on an unidentifiable target
makes the net guess, and a guess is pure gradient noise competing with
real signal.

- Labels are HELD STATE, not edges (recorder already does this). With
  edge labels a DAS charge looks like seven inert frames; with held
  state every frame is correctly labelled and the delayed movement is
  explained by a rule the model can learn exactly.
- Console semantics (verified `tetris.py:142`): `step()` sends and does
  NOT clear, so consecutive identical actions are ONE continuous hold.
  `NO_INPUT` (index 7) is the only release. The agent must discover that
  release is what makes a second press exist. Keep it that way.
- [ ] CTC as the INITIAL objective (Kai's reframe). Target = the
  sequence of EFFECTIVE inputs; inert ones never appear in the label and
  blanks absorb them BY CONSTRUCTION. Identifiability dissolves into the
  loss rather than being patched with a hand-written mask. Goal of the
  coarse phase is dynamics — what happened, in what order — not timing;
  frame-exact alignment is what n-step returns resolve later.
- HAZARD: CTC merges adjacent duplicates, so `L L L` collapses to `L`.
  Held-L (DAS pause then fast run) vs three tapped Ls are the two most
  common movement idioms in the game and are visually distinguishable —
  and merging destroys exactly that distinction. Fix: MULTIPLICITY IN
  THE TOKEN (`left ×3` as its own symbol), not learned separating
  blanks, which is the case CTC handles worst. RNN-T is the upgrade path
  if run structure dominates the residual error (CTC's conditional
  independence cannot model "DAS charged, therefore a run follows").
- [ ] Coarse -> fine: CTC first, then per-frame supervision on own
  recordings, where exact timestamps DO exist. The fine stage must be
  specified, not assumed — the comparator replays inferred actions
  through the dynamics model and coarse timing may not replay faithfully.
- [ ] Cycle-consistency as the game-agnostic alternative to masking:
  train a forward model, require predicted actions to reproduce observed
  frames. Where an action had a consequence only the right one
  reproduces the future; where it had none EVERY action does equally
  well, so the loss is flat and contributes no gradient. The weighting
  falls out of the problem instead of a per-game threshold.
- Evaluation is EDIT DISTANCE over effect sequences, with inert
  detection scored separately. Per-frame accuracy is dominated by frames
  whose correct answer is unknowable and will reward confident guessing.
- MEASURE FIRST: fraction of logged commands producing no change in
  `board_occupancy` is the accuracy ceiling. Computable from the first
  recorded session; tells you whether this is a correction or the
  dominant constraint.
- Babbling data curriculum (pairs with the loss curriculum): isolated
  inputs with long gaps -> shorter gaps -> runs -> chords. The gap length
  at which sequence accuracy degrades IS the model's effective temporal
  resolution — a more useful number than accuracy.
- Bootstrap needs no human: the agent holds the controller, so motor
  babbling yields labels and prunes dead buttons. Human recording is
  convenience and arguably contaminating — an IDM trained on human
  actions encodes human statistics and is the INSTRUMENT used to judge
  others. Enforce the barrier at training-time selection by `source`,
  not at collection time; the hardware measurements (latency, ceiling)
  need commands regardless of who produced them.

### P5c — Advantage-banded watch curriculum  (Kai's core mechanism)

Separate the two jobs. RANKING (is this player better) is cheapest from
observable outcome — garbage sent, survival, efficiency — and needs no
IDM. ADVICE (what did they do) is the only part that requires one. The
IDM's confidence then gates advice rather than underwriting the ranking.

- Selection quantity is an ADVANTAGE, not a return comparison:
  `A(s) = G_video(s) − V_s(s)`, both evaluated at the SAME state. Return
  vs return compares positions, not players — a segment starting on a
  clean board against a value at a topped-out one has a meaningless sign.
- Band `0 < A < θ`: slightly better than the student (θ is the band
  ceiling; τ stays P2's duration). Experts far above
  are deferred until the basics are learned (Kai's intent).
- This band ALSO solves P5b's coverage problem, and that is the strongest
  property of the design: the IDM only ever runs on states near the
  student's own distribution, which is where it is trained. As the
  student improves the band slides upward and IDM coverage follows,
  because the student now generates its own labels there.
- HAZARD (measured): at DISCOUNT 0.997 with TD_STEPS 10, γ^10 =
  0.970 — a 10-step window is 97% bootstrap, so the comparison measures
  the student's own value net, not the video. Weights: 231 steps = 50%
  measured, 766 steps = 90%. At 30 Hz that is 7.7 s and 25.5 s. Short
  segments make the curriculum SELF-CONFIRMING — a value error selects
  footage that appears to confirm it. Anchor on measured return, use V
  only as the baseline, prefer long segments.
- Multi-γ: P3's multi-discount heads are exactly what makes a gamma
  sweep valid — V estimates return only at its training γ. A player
  better at low γ but worse at high γ is trading survival for
  immediate garbage: a signature of HOW they are better, not just
  whether. Without per-γ heads the sweep is descriptive only.
- Cold start: V ≈ 0 early makes everything look better and the band
  degenerates to "everything". Use a percentile band over the candidate
  pool (self-normalising) or gate until V is calibrated.
- Reward semantics are T99-specific (garbage sent, chosen because a
  single clear is net neutral). Variant filter REQUIRED for reward and
  advice; variant diversity is an ASSET for representation pretraining.
  Two different admission rules, decided separately.

### P5d — Search directed at IDM uncertainty

- Not intrinsic reward on self-play IDM error. Measure IDM uncertainty
  ON the watched footage, find the weakest state-regions, and direct
  search toward reaching those states in the student's OWN play, where
  actions are known and labels are free — closing the loop where the
  comparator actually needs it rather than hoping exploration wanders
  somewhere useful.
- Confound to separate: high IDM error means either the domain is
  undertrained OR the player is acting outside the student's action
  distribution. The second is the interesting case and the two produce
  the same number. Needs epistemic/aleatoric separation.
- Feedback hazard, shared with any confidence gating: commit longer or
  search less where confident -> collect less data where uncertain ->
  stay uncertain. Cap the horizon or gate on epistemic uncertainty only.

### P5e — Chunked decisions (sequence, not repeat)

Distinct from P2's duration menu: P2 holds ONE action for τ frames;
P5e commits a SEQUENCE of different actions in one decision — Kai's
"R, none, R, none, R, up" to place a piece right. Open-loop within the
chunk, no intermediate states consulted.

- Motivation is search horizon, and it cuts WITH the real-time
  constraint instead of against it. At 30 Hz a placement is ~10-20
  inputs, so depth-5 over placements plans five pieces where depth-5
  over buttons plans a sixth of a second. Same node count, ~30× horizon.
  This is the trade every classical Tetris AI makes: plan in placement
  space, not button space.
- Tetris is unusually open-loop-safe within a placement — little worth
  reacting to arrives mid-piece, and the next queue is already visible.
- Dynamics: encode the chunk as ONE latent action and make ONE
  transition, rather than unrolling k times. Reward predicted as a sum
  over the chunk, discount γ_frame^k. Reuses P2's frame-time n-step
  targets and composition-consistency machinery directly.
- Vocabulary: see below.
- Costs: credit assignment coarsens within a chunk. Tension with P5b —
  CTC teaches that timing does not matter while open-loop execution
  depends on it; resolution is that chunks are defined by intended
  EFFECT (a placement), not literal button timing.

#### Chunk vocabulary: mined by the IDM, entered as actions

Kai's mechanism. A chunk is not a separate execution path — it enters
the student's action space as ONE entry alongside the primitives, so
search, policy head and dynamics treat it identically and MCTS needs no
special casing. Each student generation therefore has its OWN action
space, which the disposable-network premise already permits.

- Mining is TOKENIZATION. Byte-pair encoding is the exact algorithm:
  count adjacent pairs over IDM-inferred sequences, merge the most
  frequent into a new symbol, iterate. P5b's `left ×3` multiplicity
  token IS a BPE merge — the CTC token design and the chunk vocabulary
  are the same construction applied at two points in the pipeline.
- Mine over BAND-SELECTED footage only (P5c), never the whole corpus.
  Counting everything just recovers the student's own habits. This is
  the channel through which better players influence the agent:
  VOCABULARY, not policy. The student still has to learn WHEN to use
  each chunk entirely on its own.
- Frequency alone is the wrong criterion — `none,none` is the most
  common subsequence in any corpus and is worthless. Score candidates on
  (a) compression gain, (b) EFFECT CONSISTENCY: low variance in the
  resulting state delta. A chunk whose outcome is unpredictable is a bad
  abstraction and will poison the dynamics model that has to summarise
  it as one transition (P5e). (c) advantage weight of the segment it was
  mined from.
- Ladder direction matches P2: early students get LONG chunks (few
  decisions, coarse control, deliberate overfitting), later students are
  constrained toward primitives for generalization. Same mechanism too —
  a LEGALITY MASK over a fixed maximum vocabulary, never head surgery,
  preserving master compatibility and reversibility.
- Master compatibility is why P3's value-only masters pay off here: a
  value head is action-space AGNOSTIC, so a master can guide a student
  whose vocabulary it never had. P4's π blend is not — the imputed
  master policy must be expressible in the student's vocabulary, which
  needs expansion to primitives and re-tokenization into the student's
  own merges. That translation is the only new coupling this introduces.
- Execution is open-loop within the chunk plus an ABORT predicate.
  Tetris is open-loop-safe within a placement but not across one. Abort
  on (a) divergence between observed state and the chunk's predicted
  outcome beyond a threshold, (b) events that invalidate the plan
  (garbage landing, unexpected lock). Without an abort, committing on
  confidence is punished exactly when it matters most.
- RETIREMENT, not just promotion: vocabulary must shrink or branching
  climbs monotonically as merges accumulate. Retire on low search visit
  share, or on effect-consistency drift as the student's own play moves
  away from the footage the chunk was mined from.
- Cold start is forced and fine: no IDM means no chunks, so the first
  student runs primitives only and vocabulary grows as the IDM matures.
- (?) Does a mined vocabulary cross the P5c barrier? It imports
  structure rather than behaviour, and the student learns when to apply
  it unaided — but it IS influence from watched play on the policy's
  action space. Decide explicitly rather than in review.

### P5f — Action space: direction × button

- `direction {none,L,R,D,U} × button {none,rotCW,rotCCW,hold}` = 20.
  Strictly more expressive than the current 8 (move-while-rotating,
  soft-drop-while-rotating, hold-while-moving) and far smaller than 2^k.
- Contradictions (L+R) are unrepresentable BY CONSTRUCTION rather than
  penalised — the net never spends capacity learning half its action
  space is nonsense.
- Gumbel search samples rather than enumerates, so 20 is comfortable at
  ACT_SEARCH_SIMS 8. Factored heads follow the precedent P2 already set
  for action × duration; the cost is the same, visit counts assume a
  categorical so marginals lose inter-button correlation.

### P5g — Validation before any video  [ ]  ← do this first

The falsifiable core, testable today with no VODs and no IDM transfer
risk. Self-play checkpoints are a skill ladder with GROUND-TRUTH
ordering, known actions and the student's own visual domain.

- Does the comparator rank checkpoints in the correct order?
- Fed deliberately bad play, does the agent IGNORE it — measured as no
  degradation against a no-video control? This is the claim most likely
  to fail silently and the cheapest to test. Bad footage is also
  generable by Kai playing badly, with true actions logged.
- Fed strong play, does it beat the control?
- Self-recorded footage has no compression or overlays, so this isolates
  COMPARATOR LOGIC from IDM domain transfer. Separate failure modes,
  tested separately.

### Open questions

- (?) Is `V_s(s)` evaluated on the raw registered frame or on the
  representation encoder's latent? The encoder is trained on Kai's
  capture domain; a foreign frame may land anywhere in latent space and
  produce a confidently wrong value, silently corrupting `A(s)`.
- (?) Encoder sharing. If the IDM shares a trunk with the agent, IDM
  gradients reshape policy features and the "no demonstration trains the
  policy" claim is void even though no demonstration entered the policy
  loss. Separate encoder costs compute and makes the claim clean;
  retrofitting later invalidates everything collected before.
- (?) Is human-sourced IDM data admissible under the barrier at all, or
  agent-babbled only?
- (?) REWARD DISCOVERY is the blocker for the general-game ambition.
  IDM, forward dynamics, representation and search are all
  self-supervised; reward is not, and P5c REQUIRES it to compute G. T99
  is an awkward first case — the obvious on-screen number is not the
  objective. Monotonic-numeral score priors and absorbing terminal
  screens are the tractable general heuristics. This single choice
  decides between "learns any game" and "learns any game you can
  instrument".
- (?) Generalising the hand-measured ROIs: the pixels you control are
  the pixels predictable from your actions, so an action-conditioned
  forward model localises the controllable region for free — same data
  the IDM needs, and the natural companion to it.

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

### Phase 3 spec (2026-08-30; 3a+3b BUILT & AUDITED 2026-08-31, 3c deferred)

Status: 3a and 3b implemented on branch phase3-speed and taken through a
four-round audit loop (two initial auditors, a verification round, a
termination round with runtime smoke -- NO FINDINGS to close). Inline
refresh semantics verified bit-identical to pre-refactor; every new knob
defaults OFF. The offload arm is effzero-repro-communal-entropyrefresh;
pool mode is --ray --ray-workers N (recipe change: N episodes/iter,
stamped into the checkpoint's validated recipe). Guard calibrated
against measured search noise (TOL 0.05, 2 consecutive strikes). Still
owed before first production use: a toy --ray A/B of the entropyrefresh
arm vs its control, and a pool-mode wall-clock measurement. 3c (PER
caching/prefetch, AMP) deferred to its own pass -- it touches the hot
training path and AMP needs its own measured A/B.

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

Step 3b.5 -- FULL INFERENCE UNIFICATION (queued 2026-09-01, Kai): the
server hosts {live, target} weight sets, every request carries a set
tag; the eval actor routes through with the live set; refresh becomes
CPU REFRESH WORKERS using the target set -- dissolving both the
refresh actor's private GPU slice (its keep-up-at-5x-coverage question
disappears into the pooled GPU) and the second weight-sync path (one
ship per set per boundary). Rationale from live measurement during the
first pool run (6 workers): GPU at 16-68% util (mean ~36%), 24-49W of
a 200W budget, 5.3/12.3GB VRAM -- the ceiling is REQUEST SUPPLY (the
nets are tiny; the server would take batch 512+ trivially), so the
speed levers are more request streams, in order: (1) more env/tree
workers (8-12 on this 16-thread box until Python tree work saturates
CPU -- recipe scales with N by the pool contract), (2) refresh workers
via this step (a second stream that fills server gaps and parallelizes
refresh across trajectories -- today the one refresh actor cooks its
batch sequentially), (3) AMP + training-batch growth (3c / recipe
change). Fractional Ray GPU shares are scheduler bookkeeping, not
partitions; unification removes them for refresh/eval entirely.

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

P5: CTC (unaligned sequence labelling, blanks as inert frames) and RNN-T
(same, with output-side dependency for runs); action chunking in
imitation learning (open-loop k-step emission); VPT-style
IDM-then-pseudo-label; GAE / advantage as the student-vs-demonstrator
comparison quantity.
