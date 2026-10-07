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

Step 3b.5 -- FULL INFERENCE UNIFICATION (BUILT & AUDITED 2026-09-01;
three-round loop closed NO FINDINGS incl. behavioral eval-freeze smoke;
the audit added a third frozen "eval" weight set -- eval through the
shared live set lost freeze semantics and put the final row one
iteration stale -- plus a pre-destruction CPU-budget refusal): the
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

### Phase 5 candidate — fully asynchronous loop (Kai, 2026-09-02)

Never bound one lane on another: the actor plays continuously (never
blocked -- THE invariant of the real-time framing), the GPU trains and
refreshes continuously, weights sync at episode boundaries no matter
how many gradient steps elapsed, collection happens whenever an episode
lands. "Iteration" dissolves; eval/checkpoint re-key to env-steps or
wall-clock; evaluation is end-to-end (return vs wall clock), per the
carried decision below. Implementation shape: the single driver loop
interleaves CHUNKS (poll episode future; ~100 train steps; a refresh
candidate) -- loop-as-mutex preserved, no threads. REQUIRED GUARD: a
train-ratio cap (MAX_TRAIN_PER_ENV_STEP, at/near today's ~2) so the
emergent hardware-dependent ratio cannot explode into tiny-buffer
overtraining early; the realized ratio logged in every row. Slot:
after the saturation tests 1-2 report (their per-iteration
comparability is what this retires). This is the live-Switch skeleton:
the game does not pause for the learner.

### Future experiment — tandem (batch-time) reanalysis, author-aware (Kai, 2026-09-08; designed, not scheduled)

Kai's original phase-5 ask, in full strength: everything trained on has
been re-searched at batch time, so no stored target exists. Cost of the
literal version: 256 x 2000 = 512k re-searches/unit vs the sweep
contract's ~12k (~28 min/unit of pure vtree search at the measured
307/s; ~3x wall clock). The tests 6/7 verdict REFRAMES it: staleness
was never the binding variable — TARGET AUTHORSHIP was (immediate
re-search by the weak LAGGED model poisoned strong on-policy targets;
keeping actor targets beat everything). So the design worth running is
AUTHOR-AWARE refresh, not maximal freshness: (a) model-progress gating
— refresh a position only when the current/target model is plausibly
better than the target's author (capability threshold or
training-steps-since-author gap); (b) batch-time reanalysis enabled
only PAST a capability threshold, where its 3x cost buys strictly
better targets; (c) pipelined prefetch (re-search batch k+1 during
step k) to hide latency if (b) ever runs. Zero-lag (live-model)
authorship is a separate risk: self-distillation loop with no anchor —
the papers' lag exists for a reason; any zero-lag variant needs its
own guard. SLOT: after the keepnew-90 endpoint and throttle-90
long-horizon reads — if keepnew-90 holds its endpoint without
oscillation, this shrinks; if it oscillates near the top, (a)/(b) are
the fix class.

### Future experiment — buffer eviction (Kai, 2026-09-03; recorded, not scheduled)

Replace/augment whole-trajectory age-FIFO. Motivating evidence: at high
capability the 31.25k FIFO buffer became a winning-rallies monoculture
(test-1 satstale run) -- age-FIFO's real failure is DIVERSITY loss, not
keeping stale data (refresh keeps targets fresh; 4.3 handles off-policy
value decay). Candidate designs, in rough order: (1) diversity
reservoir -- stratify by return/phase/novelty and protect a varied
slice from FIFO (the safe form; naive evict-the-mastered risks
evicting exactly what drift is about to forget); (2) segment-level
eviction -- split trajectories at low-priority autopilot runs, keep
informative segments with K+TD_STEPS window margins (prefix cuts free;
suffix cuts need tail-flag surgery) = several-fold effective capacity
at the same transition cap; (3) priority-eviction with a forgetting
safeguard. PREREQ: per-row buffer-composition logging (age spread,
return distribution, mean priority, guarded fraction) -- the satstale
monoculture should have been visible before the collapse, not inferred
after.

### Guard criterion note (2026-09-03, from Kai's collapsed-policy question)

The no-sharpen guard measures progress as ENTROPY DROP only -- but a
re-search that REDIRECTS a sharp target (collapsed on action a, better
action b found: sharp -> sharp, dH ~ 0) reads as no-progress, strikes,
and freezes the wrong action into the stored target for the
trajectory's remaining lifetime. A KL-based criterion -- retire a step
when re-searches stop CHANGING the target, KL(new||old) < tol -- counts
both sharpening and redirection as progress and cannot freeze a
correctable collapse. Candidate replacement if/when the guard ships in
anger.

### Future experiment — search-based value estimation (EZ-V2 SVE; Kai, 2026-09-21; designed, not built)

DECISION (Kai, 2026-09-24): fresh window T2 = 2e4 transitions (the paper's
value), not the released code's 5e3. Our stored search values are staler
than EZ-V2's batch-time reanalysis (our refresh is less complete), so SVE is
applied only to the oldest samples. With the 31,250-transition buffer that is
~36% of value targets once the buffer is full (11,250 / 31,250), not ~84%.
SVE is the next arm after the edge pair. T1 (start step: code 3e4 vs paper
4e4) and whether to run the stage-0 probe first are still open.
DECISIONS (Kai, 2026-09-28): SVE is the NEXT RUN, on top of the prioract
stack (policy-only acting + birth search + ReDo + vectorized re-search).
T1 = 4e4 training steps (the PAPER's value -- overrides the lineage rule
here), T2 = 2e4 as decided. Run the stage-0 probe first. Probe material:
the LIVE prioract run's resume checkpoint (copied, read on CPU) once its
buffer is full (~iter 25-30) -- its stored root_values are exactly SVE's
stage-1 source (birth search: online nets with root noise; overwritten by
refresh: target nets, communal). The arch1/keepnew90 checkpoints named
below are collapse-era and superseded; the faithful run's buffers are gone
(completion markers).

WHAT IT IS (verified against arXiv 2403.00564 and the released code, 2026-09-21):
the value target for STALE samples is the search's root value, i.e. the
empirical mean of the N bootstrapped estimates
V̂_n = Σ_{t<H(n)} γ^t r̂_t + γ^{H(n)} V̂(ŝ_{H(n)}). This is close to, not
exactly, our root q-value: EZ-V2's root mean also includes the root's own
network value (N+1 terms; ez/mcts/ctree/cnode.cpp), vtree's root_q averages
the N simulations only, and communal refresh values also carry backups
that reached the root through the chain (SVE audit r1, fidelity L5). Mixed target (released code, ez/agents/base.py):
- before `start_use_mix_training_steps` (3e4 training steps): the TD target
  everywhere;
- after it: TD for samples among the newest `mixed_value_threshold` (5e3)
  transitions, and the search value otherwise. The choice is made per
  sampled root and applied to all unroll positions.
- The paper states T1 = 4e4 and T2 = 2e4. The code wins (the lineage rule).
- Their search values come from batch-time reanalysis: 16-sim Gumbel through
  the target model (synced every 200 steps), at every unroll position. That
  costs them nothing extra, because they already re-search every position
  for policy targets.
- Their SVE ablations are DMC-only (Reacher Hard 795 vs 591 with TD;
  Cartpole Swingup Sparse 764 vs 632). There is no Atari ablation, and
  their README says some Atari results do not reproduce with the released
  code.

EXPECTATIONS HERE (honest priors):
- The paper's motivation (stale multi-step TD embeds an old policy's
  rewards) is WEAK for us. The 31.25k buffer spans only ~10-25 iterations,
  and Pong's rewards are sparse, so a 5-step TD target is mostly
  γ^5 · v_target(s_{t+5}). The old policy's influence is 5 decisions of path.
- What SVE might still buy: the value of the SEARCH (improved) policy,
  averaged over 50 imagined paths, instead of one bootstrap read. That means
  lower variance and faster propagation of policy improvement into V.
- Risk: at T2 = 5k, SVE sets ~84% of our value targets, and those lose their
  link to real rewards (the search sees only predicted rewards). Our
  2026-09-21 probes found plateau runs' dynamics weakly action-aware, with
  the projector collapsed. There, SVE ≈ the value net echoing itself; the
  paper itself warns that model error amplifies SVE error. Read the edge
  run's proj_erank before trusting any SVE result.

DESIGN (flag-gated, default off; build on a worktree branch `sve`, because
the live runs execute from the main working tree):
- `VALUE_TARGET_MODE` "td" (default = today, bit-identical) | "mixed", plus
  `SVE_START_TRAIN_STEPS` (3e4) and `SVE_FRESH_TRANSITIONS` (2e4; Kai 2026-09-24).
- Freshness = transition recency counted from the FIFO buffer's end, per
  sampled root (EZ-V2's buffer-index rule), in prepare_batch_paper.
  Terminal padding stays 0; truncated tails keep today's handling.
- SVE SOURCE, stage 1 = the STORED root_value: written by the acting search
  (online net, with root noise) and overwritten by every refresh (lagged
  target net, communal root value). It needs zero extra compute. Its gap vs
  EZ-V2: our values can be up to ~refresh-rotation-period old, not ≤200
  steps. Stamp `root_value_step` on every write so the age is measurable.
- SVE SOURCE, stage 2 (only if stage 1 shows signal, or stored-value age is
  the suspect) = batch-time fresh-root search through the target nets at 16
  sims, cached per target sync keyed (id(traj), pos). This is the
  BOOTSTRAP_CACHE machinery with a search in place of the value-head
  forward, restricted to SVE-masked positions. Estimated +0.2-0.5 s/step
  (1.3-2x wall clock); measure before committing.
- Priorities need no change: the writeback uses target_value[:, 0], so SVE
  samples get |v − SVE| automatically.
- Row diagnostics:
  - sve_frac: share of value targets taken from search;
  - sve_td_gap: mean |SVE − TD| where SVE is used. ~0 means SVE is a no-op;
  - sve_td_bias: mean SVE − TD. Positive means optimistic;
  - sve_value_age: training steps since the stored value was written.
- Tests:
  - "td" batches bitwise identical to today's, values and RNG;
  - the mask pinned (T1 gate, fresh window at a trajectory boundary, padding
    and truncated tails);
  - every test counterfactual-checked (the lineage rule).

STAGE 0 PROBE (before building; run on CPU so the live runs keep the GPU):
use arch1 @48 (archfix's arm, with buffer and target nets) and keepnew90
@88. On stale positions, compare three values:
- the trainer's TD target;
- the stored root_value;
- a FRESH search value (target nets, 50 and 16 sims, no noise).
On the newest trajectories, score all three against the realized
discounted return. Questions:
1. Is |SVE − TD| big enough to matter?
2. How much does stored staleness cost versus a fresh search (stage 1 vs
   stage 2)?
3. Which estimate is closest to realized returns where they are nearly
   on-policy?
If the gap is tiny, or SVE is further from realized returns than TD, stop
here.
PROBE BUILT 2026-09-28: benchmarks/probes/probe_sve_stage0.py (self-checks
its TD target against the trainer's own batch builder: max |diff| 0 on 64
roots). Scheduled on a copy of the prioract checkpoint at iter 26 (buffer
full, stale ~36%). PROPOSED AMENDMENT (written 2026-09-28 BEFORE any
full-buffer data; pending Kai): on the prioract stack the newest games are
played by sampling the prior at T = 1 until iter 45, so their realized
returns measure a far more random policy than the one search values
estimate -- the "closer to realized returns" test is structurally tilted
toward TD (the value head is trained on behaviour returns). The iter-8 dry
run already shows it: TD MAE 0.51 vs 0.58-0.61 for search values, with
search values 0.15-0.36 more optimistic. Proposed reading: stop only if the
gap is tiny (SVE would be a no-op); treat the realized-return comparison as
descriptive, judged by correlation rather than error, and re-probe at
~iter 70 (T = 0.25, behaviour near greedy) where the tilt is small.

SUPERSEDED 2026-09-28 by "Next experiment -- SVE on the prioract stack"
below (one seed, the prioract stack, T1 = 4e4; its stop rule carried over
there). Kept for the record:
RUN: the best recipe after the edge pair reports (archfix + edge if the fix
helps, archfix otherwise) + VALUE_TARGET_MODE mixed. Seeds 0 and 1
concurrently, sequential, --snapshot-weights. Pre-registered reads:
- mechanism: sve_td_gap clearly nonzero, and sve_frac ≈ 0.36 after iter 15 (T2 = 2e4);
- outcome: mean return over iters 40-75 versus the same-seed controls, plus
  crossings at 75-90. Both seeds crossing is strong evidence; one crossing
  is weak (~44% at the old 1-in-4 rate);
- stop rule: both seeds ≤ −17 at iter 30, or value/control more than doubling
  after T1 (a self-echo blow-up).

### DECISION + measurement, 2026-10-07: the temperature arm by override; search-acting latency

DECISION (Kai, 2026-10-07: "Sure go for it", to: measure low-sim search
latency, then launch the temperature arm): the tempscale arm runs by
Kai's explicit OVERRIDE of the DIRECTION proviso that refused it on
10-05 (prioract2 s1 WORSE). supervise_tempscale.py gains --kai-override,
which converts ONLY that refusal (exit 3) into a launch and logs it; an
unfired trigger (2) or a malformed record (4) still refuses. Its
pre-registered reads are UNCHANGED -- (H): first success at <= 65 supports
"acting randomness delayed the breakout" (policy-only breakouts so far:
@75, @80; search acting: @50, @65, @65); (1)-(3) the standard bands. The
arm keeps the communal re-search (the seqref read found no evidence the
refresh engine matters, and it is the faster engine).
SEARCH-ACTING LATENCY (measured 10-07, Searcher.act_search incl. its
asyncio.run, prioract2 s0 @85 weights, real Pong observations, idle GPU):
  sims        0 (policy)   4      8      16     32     50
  GPU median  1.7 ms       18     31     56    108    169 ms
  CPU median  1.9 ms       23     38     70    134    206 ms
~3.3 ms per simulation on the GPU: the engine is per-simulation-overhead
bound (one network call + Python per simulation), not compute bound. Even
4 sims exceed a 60 fps frame (16.7 ms); deciding every 4th frame (66.7 ms,
Pong's skip) fits ~8-16 sims. (muzero.ACT_SEARCH_SIMS's comment "8 sims
~40% of a 60fps frame" is stale -- now 186%.) So search acting is not
real-time viable at the control's 50 sims with this engine; a low-sim
(8-16) search-acting arm, or a batched single-root engine, would be the
route if the temperature arm does not close the gap.
LAUNCHED 2026-10-07 09:31 (main d81b3794): supervise_tempscale.py ray 0
--kai-override; the log shows the exit-3 decision, then OVERRIDE_NOTE, then
preflight (7 suites), then a fresh launch into pong-tempscale-s0-2026-10-05.
Override audit r1: nothing above LOW. LOW-1: EVERY RELAUNCH (reboot, pause,
GPU kill) MUST pass --kai-override again, or the supervisor re-decides and
exits 3 (fails closed). Deferred until after the run (changing audited files
would block a relaunch): LOW-2 (the override should also require the
trigger to have fired -- moot with the real records: no first success) and
LOW-3 test hardening; plus the deferred seqref L1/L2.

### RESULT -- seqref seed 1 + THE TWO-SEED READ, 2026-10-07

pong-seqref-s1-2026-10-05 ran 10-06 09:10 -> 10-07 ~04:50. EXACT / recal:
  iter    45     50     55     60     65     70     75     80     85     90
  raw   -14.8  -15.3  -16.2  -16.0  -16.2  -14.3  -13.6   -8.9  -10.4   -9.5
  won    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4
  recal                      -14.9  -13.3  -13.0   -9.6   -9.6  -13.5  -14.8
READ (1): NO first success (never won a phase, raw or recalibrated).
READ (2): mean 60-90 = -12.7141 -> WORSE; recalibrated -12.6667.
TWO-SEED READ (pre-registered): seed 0 broke out (@80), seed 1 did not ->
1/2 -> "the vtree stack's rate: no evidence the refresh engine matters".
(By the letter the clean outcome was already ruled out by seed 0's raw
WORSE.) Policy-only acting now stands at 2 breakouts in 4 seeds on EITHER
refresh engine -- prioract2 (communal) @75 / none, seqref (sequential) @80
/ none -- both breakouts late, at T = 0.25. Search acting broke out in 3
of 4 (control @50 and @65, noredo @65; seqvtree none), always earlier.
CONCLUSION (single-digit seeds, hedged): the communal re-search is NOT
what makes policy-only acting unreliable; the acting side is the suspect
-- consistent with, though not proof of, the acting-randomness hypothesis
(the tempscale arm, built and merged, inert until launched). Caveat: at a
true 50% rate the control's 2/2 itself happens 1 time in 4.
The audited-file freeze for seqref is lifted (both seeds done); the
deferred audit r1 L1/L2 test hardening can be done now. Next step: Kai's.

### RESULT -- seqref seed 0 (interim; seed 1 running), 2026-10-06

pong-seqref-s0-2026-10-05 ran 10-05 11:48 -> 10-06 09:10 (paused by Kai
10-05 23:50 at the iter-56 checkpoint, resumed 10-06 01:21). EXACT /
BN-recalibrated:
  iter    45     50     55     60     65     70     75     80     85     90
  raw   -13.1  -12.2  -13.9   -9.1  -12.7  -12.7   -2.5  +15.0  +16.6  -20.5
  won    0/4    0/4    0/4    0/4    0/4    0/4    2/4    4/4    4/4    0/4
  recal                      -11.1  -12.8   -7.4   -6.2  +13.9   +6.3  +18.5
READ (1): first success @80 (+15.0, 4/4 phases).
READ (2): mean 60-90 = -3.7000 -> WORSE (< -2.32); recalibrated +0.1620
(INCONCLUSIVE). The raw mean carries a full collapse at @90 (-20.5, all
phases lost; in-run H(pi) 1.77, erank 19.9) that recalibration shows is
the BN eval artifact (+18.5, 4/4) -- without it the run ended at its best.
READ (3): one collapse after the first success (@90, raw only).
FOR THE TWO-SEED READ: seed 0 broke out, so 0/2 is ruled out; but it reads
WORSE on the primary (raw) mean, so the pre-registered "both break out and
neither reads WORSE" outcome is ruled out too, whatever seed 1 does -- by
the letter the read can only land in "otherwise -> Kai's call". Seed 1
still decides whether the BREAKOUT RATE on the control's refresh engine is
2/2 (the control's) or 1/2 (the vtree stack's).
TIMING (post hoc): breakout @80, at T = 0.25 -- as late as prioract2 s0
(@75) and later than every search-acting breakout (control @50/@65, noredo
@65). The refresh engine did not move the breakout earlier; lateness so far
tracks policy-only acting, not the re-search engine.

### Next experiment (pre-registered) -- policy-only acting on the control's refresh engine (Kai, 2026-10-05)

DECISION (Kai, 2026-10-05, after seed 1 read WORSE): "Let's go with 1 since
that will truly test if the code is the issue ... audit is also a good
idea." Two seeds, s0 then s1 (the control broke out 2/2).
HYPOTHESIS (post hoc, from the cross-run pattern): the only configuration
that broke out on BOTH seeds is the faithful control, which used the
SEQUENTIAL refresh engine; every run on the vectorized communal re-search
(REFRESH_VTREE, VTREE_COMMUNAL_TRAJS 8) broke out about half the time
regardless of acting -- search acting noredo yes / seqvtree no, policy-only
prioract2 s0 yes / s1 no. So the communal re-search, not policy-only
acting, may be what makes runs unreliable.
ARM effzero-repro-faithful-prioract-noredo-seqref = prioract2's arm with
REFRESH_VTREE False (VTREE_COMMUNAL_TRAJS 1, the default): the refresh runs
the control's own code, run.refresh_targets, inline at the iteration
boundary (REFRESH_OFFLOAD is False in both arms, so --ray does not offload
it). Its EFFECTIVE delta from the faithful control is then exactly the
acting/pipeline side: USE_SEARCH_ACTING off, BIRTH_SEARCH, PRIOR_ACT_
STAGED_TEMP, RAY_EAGER_COLLECT, the --ray --cpu-actor launch (+ inert
REDO_NETS, read-only STEP_DIAGNOSTICS). Not identical to the 2/2 runs --
acting differs by design; it isolates the refresh engine.
The launch bundles HARNESS-level differences too (audit r1 L3), all on
the acting/pipeline side of the split: (a) IN-RUN EVAL REGIME -- the
control's in-run rows are 50-sim search evals on the training env; these
are prior-argmax on a CPU eval actor, so the supervisor's in-run returns
are NOT comparable to the control's in-run rows (the exact offline evals,
the pre-registered readout, are); (b) seeds are unpaired (the CPU actor
has its own RNG and reseeds the env per game); (c) acting runs on CPU;
(d) the actor plays game it+1 during the boundary work (timing only);
(e) birth search uses the vtree engine (a defect common to vtree code
would land in the "birth search" bucket of the 0/2 read). Deferred, not
blocking (audit r1 L1/L2, after both seeds: changing audited files would
block seed 1's preflight): a stronger end-to-end refresh test (what the
refresh sees and changes) and test_reanalyze_refresh / test_vtree_communal
in the preflight -- the auditor verified both properties hold today.
COST (measured 10-05 on the GPU, one refresh cycle = 4 production-length
trajectories from prioract2 s0 @60): sequential 329 s, vtree TRAJS 8
272 s (TRAJS 1 752 s, so not used) -> ~+0.5 min/iter, ~21 h per seed.
PRE-REGISTERED READS (exact prior-argmax evals, iters 45-90, per seed):
  (1)-(3) the standard reads and bands (first success: exact >= +10 or
  >= 3/4 phases; mean 60-90 vs the faithful control: NO WORSE >= +3.68,
  WORSE < -2.32; collapses after the first success); secondary: BN-
  recalibrated.
  THE TWO-SEED READ (the test):
    both seeds have a first success and neither reads WORSE -> consistent
      with the communal re-search having made the vtree stack unreliable
      (the control's 2/2 standard): keep the sequential refresh (or fix
      the communal engine) and continue on policy-only acting;
    neither seed breaks out (0/2) -> the acting/pipeline side (policy-only
      acting, birth search, the parallel actor) is implicated and the
      communal re-search is not the sole cause -> Kai's call;
    otherwise (1/2) -> the vtree stack's rate: no evidence the refresh
      engine matters -> Kai's call.
  Strength: at a true 50% rate, 2/2 still happens 25% of the time, so even
  the clean outcome is suggestive, not proof. Seed 1 runs regardless of
  seed 0 (the read needs both).

### RESULT -- prioract2 seed 1, 2026-10-05: WORSE -> the DIRECTION proviso fired

pong-prioract2-s1-2026-10-02 (same arm and code as seed 0; paused 10-04
20:30 at iter 48 by Kai, PC reboot 10-05 00:34, resumed 00:38) finished
10-05 10:09. EXACT / BN-recalibrated:
  iter    45     50     55     60     65     70     75     80     85     90
  raw    -8.0  -10.5  -11.2  -11.4  -11.3  -13.0   -8.8   -7.3  -11.4  -10.0
  won    0/4    0/4    0/4    0/4    0/4    0/4    0/4    1/4    0/4    0/4
  recal                      -11.0  -11.7   -7.7   -9.0   -6.1  -10.5  -11.1
READ (1): NO first success (best row -7.3, at most 1/4 phases).
READ (2): mean 60-90 = -10.4524 -> WORSE; recalibrated -9.5716 (WORSE).
TWO SEEDS of this arm: +3.6431 (s0, broke out @75, ended +16.8) and
-10.4524 (s1, never broke out); two-seed mean -3.4047 (WORSE band). Seed 1
ran at T=0.25 from ~iter 68 for 22 iterations without breaking out, so a
low temperature LATE did not unlock it -- the tempscale arm's regime (low
temperature throughout) is a different test, but the "T=0.25 gates the
breakout" pattern now has a counterexample.
ENFORCEMENT: supervise_tempscale.py read the records at ~10:25 and exited 3
("seed 1 reads WORSE: the DIRECTION proviso says revisit before building
further on this stack -- NOT launching; back to Kai"). main was
fast-forwarded to the audited tempscale code (d7979dce; flag default off,
inert) by the WMI hand-off after the probes. Decision is Kai's.

### DIRECTION + next experiment (pre-registered) -- sharper prior-acting temperature (Kai, 2026-10-04)

DIRECTION (Kai, 2026-10-04): develop from the no-search, fully parallel
model (policy-only acting on the CPU actor, birth search, eager collect,
no ReDo) -- the configuration that fits the real-time goal (3.7 ms/move on
the CPU actor vs ~10 min/game for 50-sim CPU search acting) -- provided it
converges (seed 1, running, is that check: if it reads WORSE, revisit
before building further on it). ReDo goes on the BACKBURNER, not deleted:
no current dormancy problem in the no-ReDo runs (dead frac 0.00-0.01,
healthy proj_erank), the one comparison points against it, and its real
use case -- long non-stationary training -- is the Switch's continual
learning (a cheap dormancy monitor would say when it is needed).
Recommended to Kai (his call): PARK noredo s1 and seqvtree s1 -- they only
attribute effects within search-acting stacks; the noredo/seqvtree
contradiction stays unresolved, acceptable since the chosen stack has its
own direct evidence (and uses the same vectorized re-search).
Order on the chosen stack: seed 1 -> this temperature arm (if triggered)
-> SVE on this stack (built and audited; training-side cost only) -> BN
recalibration stabilizer -> mid-game swaps/streaming for the Switch.

HYPOTHESIS (from seed 0's behaviour stats, post hoc): prior sampling at a
given temperature is much noisier than visit sampling, and the breakout is
gated by how random the acting is. Seed 0 at T=1 acted with H 1.44-1.51
nats (max 1.79), only ~1/3 of its actions matched the birth search's
argmax, and the prior's argmax agreed with the search's only ~55% of the
time. Every run that broke out did so after a temperature drop: the
search runs at T=0.5 (control s0 @50, s1 @65, noredo @65), both
policy-acting runs only at T=0.25 (prioract2 @75; prioract ~@75). Single
seeds; the search runs' acting entropy was not recorded (estimated
~1.2-1.3 at T=1 from their search targets).
ARM (to be built and audited on a worktree; merged only after seed 1
finishes): effzero-repro-faithful-prioract-noredo + PRIOR_ACT_TEMP_SCALE
0.5 -- the staged prior-acting schedule halved: 0.5 / 0.25 / 0.125 at the
same breakpoints (50% / 75% of training). New flag, default 1.0 (identity).
One knob; everything else identical, --ray --cpu-actor, eager collect.
TRIGGER (decided from seed 1's exact evals, mechanically): RUN this arm if
seed 1's first success (exact >= +10 or >= 3/4 phases) is at iteration
>= 70 or never -- i.e. seed 1 also needed T=0.25. If seed 1's first
success is at <= 65 (at T >= 0.5), the "temperature gates the breakout"
story loses its support -> skip this arm, go to SVE on this stack.
ENFORCED by supervise_tempscale.py (audit r1 M1: the DIRECTION proviso is
enforced too): it waits for seed 1's exact evals (rows 45-90) and exits 3
if seed 1 reads WORSE (mean 60-90 < -2.32: revisit with Kai), 2 if the
trigger does not fire (first success <= 65: SVE next), 4 on a malformed
record; otherwise it launches. (Timing note: written at 12:14 10-04, when
seed 1 had in-run rows @5 and @10 but no exact eval and nothing past
iter 10 -- none of the rows the trigger and reads use.)
PRE-REGISTERED READS (seed 0; exact prior-argmax evals, iters 45-90):
  (H) the hypothesis read: first success at <= 65 (the search runs' range)
      -> consistent with acting randomness having delayed the breakout;
      >= 70 or never -> not supported (search quality, not temperature,
      would be the suspect);
  (1)-(3) the standard reads and bands: first success; mean 60-90 vs the
      faithful control (NO WORSE >= +3.68, WORSE < -2.32); collapses after
      the first success; secondary: BN-recalibrated.
  Risk, interpretive only: if it never breaks out with low early acting
  entropy (beh_H_act), read it as exploration-starved, not as evidence
  for search acting.

### RESULT -- prioract2 (policy-only acting, no ReDo, eager collect), 2026-10-04

pong-prioract2-s0-2026-10-02 (--ray --cpu-actor, RAY_EAGER_COLLECT) ran
10-02 18:37 -> 10-04 01:28, paused by Kai 10-03 13:35 at the iter-86
checkpoint and resumed 10-04 00:33 (~13 min/iter; 158,883 env steps at
iter 90). EXACT / BN-recalibrated (offline; 45-70 exact were run mid-run
on finished snapshots, the rest after completion):
  iter    45     50     55     60     65     70     75     80     85     90
  raw   -13.1   -9.2  -10.7   -6.9  -19.9   -0.0  +12.1   +7.2  +16.4  +16.8
  won    0/4    0/4    0/4    1/4    0/4    2/4    4/4    3/4    4/4    4/4
  recal                      -11.5   -7.8   +0.3  +17.4   +5.2  +14.3  +17.2
READ (1): first success @75 (+12.07, 4/4 phases) -- later than control s0
(@50) and noredo (@65), like control s1 (@65).
READ (2): mean 60-90 = +3.6431 -> INCONCLUSIVE, 0.037 below the NO WORSE
edge (+3.68). Recalibrated (secondary) +5.0143 -> NO WORSE. The raw mean
carries a full collapse at @65 (all 30 games lost, ~-20; recal -7.8, so
partly the BN eval artifact) and the late breakout.
READ (2b): >= seqvtree - 6.5143 -- passes trivially (twin at the floor;
non-discriminating, as recorded).
READ (3): 0 collapses in 3 rows after the first success; the final rows
+16.4 / +16.8 (4/4 phases, min game +12 / +14) are the strongest
end-of-run state in this campaign (control s0 +14.8/+13.6 @85/90, s1
-14.9/+0.6, noredo +3.9/+5.3). By Atari-100k (~iter 55) it was still
-10.7 vs control s0 +10.0 -- slower to break out, stronger once it did.
EAGER COLLECT worked as built: 87 eager iterations recorded (86 lost with
the per-row window at the pause; 87 primed), every game collected
mid-block at offset 0-40 (0 x43, 10 x6, 20 x18, 30 x15, 40 x5), never
missed -- the one-iteration lag of the old parallel runs is gone.
PRE-REGISTERED INTERPRETATION: (2) INCONCLUSIVE -> seed 1. Not rounded up:
the band edge is the band edge, and a single seed with a collapse in the
window is exactly what seed 1 exists for. Kai (10-03) questioned whether
adding the game 0-40 steps in is "too early"; the sequential control adds
it at step 0, so this is the control's timing -- open for his view.
SEED 1 LAUNCHED 2026-10-04 10:17 (Kai: "yes, start seed 1 of this run"):
supervise_prioract2.py ray 1, same arm and code (fdcf5a55; the audited
files unchanged since b35af638), results in pong-prioract2-s1-2026-10-02
(the directory date is the supervisor's audited constant, not the launch
date). Same pre-registered reads and bands; eager collect unchanged.

### RESULT -- seqvtree (factorization half 2), 2026-10-02

pong-seqvtree-s0-2026-09-30 finished 10-02 18:28 (sequential, run alone
after its 09-30 pause; ~17-23 min/iter). EXACT / BN-recalibrated (offline
probes run only after completion):
  iter    45     50     55     60     65     70     75     80     85     90
  raw   -14.0  -16.1  -11.3  -15.6  -11.1  -14.3  -11.6   -8.4  -10.4  -14.9
  recal  -9.9  -13.3  -10.6  -12.8  -12.0  -10.3   -9.3  -10.4  -14.9  -17.2
READ (1): NO first success (best row -8.4; 0/4 phases won in every raw
row; recal won 1/4 at @70 and @75).
READ (2): mean 60-90 = -12.33 -> WORSE (< -2.32); recalibrated -12.41 --
not the BN eval artifact.
READ (3): n/a (no success).
DIAGNOSTICS do not separate it from the runs that broke out: H(pi)
1.33-1.52, fresh target H 1.20-1.32, proj_erank 36-75, step grad norm
0.26 -> 0.15 -- all within noredo's and the faithful pair's ranges. It
simply never broke out (in-run return -17 .. -10 throughout), like
searchact (-11.37).
MATRIX CELL (pre-registered): noredo NO WORSE (+5.86) x seqvtree WORSE
(-12.33) = "CONTRADICTION under identical flags: seed noise or pipelining
HELPS -> seed 1 of both, no attribution". The two runs' effective flags
are identical; only the launch mode differs. An 18.2-point gap between
identical-flag single seeds is ~2.2x the inferred SD of a run-to-run
difference (~8), and this stack's breakouts are all-or-nothing (archfix
recipe 1-in-4; non-breakout runs sit at -10 .. -15), so seed noise is a
live explanation; a pipelining benefit has no known mechanism (the
pipelined actor's only data effect is a one-iteration LAG). NO
ATTRIBUTION: neither the vectorized re-search nor the launch mode is
implicated or exonerated by this pair. The ReDo read (searchact -11.37
vs noredo +5.86) stands as recorded, but it rests on single seeds too, and
seqvtree shows a ReDo-off run on this stack can also fail to break out --
so the ReDo attribution is weaker than it looked.
PRE-REGISTERED NEXT: seed 1 of BOTH noredo and seqvtree (no SVE until
then). The GPU is busy with prioract2 (below; launched 18:37 10-02), and
host commit prevents two runs at once, so their order is Kai's call.
CONSEQUENCE FOR PRIOACT2's READ (2b): its twin seqvtree sits at the
no-breakout floor, so (2b) (mean >= seqvtree - 6.5143 = -18.84) can no
longer discriminate -- read (2) against the faithful control's bands is
the one that counts. (Recorded post hoc; the pre-registered reads are
unchanged.)

### Next experiment -- policy-only acting without ReDo (Kai, 2026-10-01)

DECISIONS (Kai, 2026-10-01): after the factorization implicated ReDo, test
policy-based acting again -- the prioract arm with ReDo OFF -- ALWAYS
PARALLEL (--ray --cpu-actor: the Switch must act in parallel with training
anyway), with each finished game added to the buffer at once. Let seqvtree
finish first (it isolates the batched re-search, which this arm also uses,
and it is this arm's no-lag search-acting twin). This WITHDRAWS the
launch-mode rule pre-registered at 9cae34d2 (ray vs seq chosen from
seqvtree's rows 60-90) before any of those rows existed (seqvtree was at
row 35): there is no sequential variant any more. Mid-game weight swaps and
chunked streaming are BACKLOGGED for the Switch (IDEAS_BACKLOG.md): a
policy-only Pong game takes ~8 s on the CPU actor (3.7 ms/step, measured)
vs ~13 min of training per iteration, so they would change almost nothing
here; the whole lag was the finished game waiting for the block to end.
ARM effzero-repro-faithful-prioract-noredo = prioract + REDO_EVERY 0,
STEP_DIAGNOSTICS, RAY_EAGER_COLLECT (tests/test_prioract2.py pins it);
launch pinned --ray --cpu-actor; supervise_prioract2.py ray 0 (waits for
seqvtree's completion marker).
RAY_EAGER_COLLECT (new, default off): train_block polls the actor every
EAGER_POLL_STEPS (10) gradient steps; when the episode is done it closes the
block, collects (birth search on the request-time weights, append, evict)
and continues the remaining steps in a fresh block (caches reopened). One
game per iteration is unchanged (the next is still requested at the
boundary), so each game joins ~0-40 of the iteration's 2000 steps in
(its remaining play time rounded up to the 10-step poll grid -- training
pauses for the birth search, so that adds no steps; a game that finished
during the boundary's refresh/eval/checkpoint is collected before step 1)
vs step 1 sequentially -- the sequential harness's freshness with acting
still parallel. Rows record it: eager_collect_steps, [iteration, offset]
per eager iteration (offset None = missed the block; primed iterations --
1 and the first after a resume -- have no entry; not checkpointed, so a
resumed row lists only post-resume iterations). NOT REPLAYABLE FROM THE
SEED: where the collect lands depends on wall-clock time. The birth
search draws torch's global generator on the training device (Dirichlet
root noise, tie-breaks), which training's augmentation noise also draws,
so a mid-block collect reorders those draws; numpy's generator (PER batch
sampling) is not touched by the collect, but every later PER draw sees a
buffer that changed at a wall-clock-dependent step (statistically
equivalent; resume was never bit-exact either).
WHAT IT ISOLATES: vs seqvtree (search acting, sequential, same flags
otherwise): WHO ACTS -- the prior sampled at the staged temperature + birth
search instead of the 50-sim acting search -- plus the parallel actor,
whose lag eager collect removes (residual: the first ~0-40 steps, ~0-15 s,
of each iteration, and the actor's own RNG regime). vs noredo (+5.86): who
acts AND the lag. vs prioract (-4.73): ReDo off AND eager collect.
PRE-REGISTERED READS (exact prior-argmax evals, iters 45-90; secondary:
the same on BN-recalibrated evals):
  (1) first success row (exact >= +10 or >= 3/4 phases won);
  (2) mean exact over rows 60-90 vs the faithful control: NO WORSE >= +3.68,
      WORSE < -2.32, else INCONCLUSIVE;
  (2b) vs seqvtree (the twin): policy-only acting costs no more than the
      seed spread if mean >= seqvtree's mean - 6.5143 (the faithful control
      seeds' spread);
  (3) collapses after the first success; (4) behaviour (beh_*).
INTERPRETATION (single seed, hedged): (2) NO WORSE and (2b) within the
spread -> consistent with policy-only acting being viable (the Switch
design) -> SVE on this stack next. (2) WORSE or (2b) below -> policy-only
acting costs even without ReDo and lag; first suspect the exploration
change (prior sampling: more random than visit sampling at T=1, less at
T=0.25 -- the dropped Dirichlet floor). Otherwise INCONCLUSIVE -> seed 1.

### RESULT -- noredo (factorization half 1), 2026-10-01

pong-noredo-s0-2026-09-30 finished 10-01 ~13:50 (run alone after the
seqvtree pause; ~14-15 min/iter). EXACT / BN-recalibrated:
  iter    45     50     55     60     65     70     75     80     85     90
  raw   -12.5   -9.9   -4.9   +0.6   +3.8   +3.4  +12.8  +11.3   +3.9   +5.3
  recal                       -5.2   +1.2   -1.8   +7.9  +10.1   +2.7  +11.7
READ (1): first success @65 (+3.8, 3/4 phases) -- control s1's row.
READ (2): mean 60-90 = +5.86 -> NO WORSE (>= +3.68); recalibrated +3.81.
READ (3): 0 collapses in 5 rows after the first success.
REDO COMPARISON (the only flag difference): searchact -11.37 (WORSE, never
won a phase) vs noredo +5.86 (NO WORSE) -- consistent with ReDo having
hurt searchact, and plausibly prioract (ReDo on too). noredo matches the
control's s1 on every read. Single seeds; the matrix cell is completed
by seqvtree (resumed 10-01 ~14:05 from its iter-8 checkpoint).

### Next experiment -- the factorization (bisect steps 2+3 at once; Kai, 2026-09-30)

DECISION (Kai, 2026-09-30): after searchact read WORSE, run the two
remaining bisect steps CONCURRENTLY, one seed each (0):
  noredo   = effzero-repro-faithful-searchact-noredo, --ray --cpu-actor:
             searchact with ReDo OFF;
  seqvtree = effzero-repro-faithful-seqvtree, SEQUENTIAL: the faithful
             control + the vectorized communal re-search + prior-argmax
             in-run eval, no ReDo.
Both keep STEP_DIAGNOSTICS on (row loss / grad-norm series without ReDo)
and PIN their launch mode (`_launch`, launch_gate: a relaunch in the wrong
mode fails at startup). Their effective flags are IDENTICAL apart from the
inert REDO_NETS (tests/test_factor.py); the difference is the LAUNCH MODE,
which bundles: the one-iteration data lag (and eviction lag) of the
pipelined actor, CPU-actor search numerics, and the actor's own RNG regime
(seeded per run, not checkpointed; sequential restores its training streams -- the env NoopReset stream restarts from the seed in both). With
the two finished runs this splits the suspects:
  ReDo                  searchact (-11.37)  vs  noredo
  launch mode           noredo              vs  seqvtree
  vectorized re-search  seqvtree            vs  faithful control (+12.20/+5.68)
(The vectorized re-search groups the POLICY_REFRESH_COUNT = 4 refreshed
trajectories per forest, so VTREE_COMMUNAL_TRAJS 8 acts as 4 -- in prioract
and searchact too; the "~4% of roots" figure was measured at 8 vs 1.)
EVAL_PRIOR_ONLY on seqvtree: the control's in-run SEARCH eval sat on the
sequential critical path; the readout is the offline exact eval either
way, and eval does not feed training (a reseed of the RNG streams: the
control's search eval drew np.random every step).
CONCURRENCY (factor audit r1 M2; the faithful pair did NOT use the vtree
engine): measured peak torch-reserved GPU memory per driver process at
production shape (iter-26 buffer, batch 256 training + the vectorized
refresh): 4.64 GiB each (4.04 training, +0.6 refresh); + ~0.4 GiB CUDA
context each + ~1.2 GiB desktop -> ~11.3 of 11.99 GiB (nvidia-smi's
12,282 MiB), tight; late-run games (2.5-3k steps) grow the forest by
~0.13 GiB per process and the eval concatenation adds ~0.05-0.1 ->
~11.5-11.7 GiB projected over rows 60-90 (factor audit r2). PAUSE RULE
(checked by a background monitor sampling every 10 min): pause seqvtree
(its resume restores the training RNG streams (not the env's NoopReset stream, which restarts from the seed -- statistically equivalent, not bit-exact); resume it after noredo) if
nvidia-smi memory.used >= 12,000 MiB on 3 consecutive readings, OR host
commit free < 2 GB on 3 consecutive readings, OR either run's iteration
time exceeds 2.5x its early pace for 2 consecutive rows (a Windows
driver spill; past runs grew at most 1.73x). Iteration times are
compared only between these two runs.
PRE-REGISTERED READS (exact prior-argmax evals, iters 45-90, the SAME bands
as prioract / searchact, fixed from the faithful control):
  per run: (1) first success row (exact >= +10 or >= 3/4 phases won);
  (2) mean exact over rows 60-90: NO WORSE >= +3.68, WORSE < -2.32, else
  INCONCLUSIVE; (3) collapses after the first success; secondary: the same
  on BN-recalibrated exact evals (common calibration set).
INTERPRETATION -- the full matrix (rows noredo, columns seqvtree; single
seeds, labels hedged "consistent with"):
                 seqvtree NO WORSE      seqvtree INCONCL.      seqvtree WORSE
  noredo NO W.   ReDo hurt searchact;   ReDo hurt; seqvtree    CONTRADICTION under
                 stack fine without it  seed 1 to confirm      identical flags: seed
                 -> SVE on the parallel the stack              noise or pipelining
                 stack (no ReDo)                               HELPS -> seed 1 of
                                                               both, no attribution
  noredo INCONC. launch mode suspect;   both unclear ->        re-search or ReDo
                 noredo seed 1 first    seed 1 of both         suspect; seqvtree
                                                               seed 1 first
  noredo WORSE   LAUNCH MODE hurts      launch mode suspect,   VECTORIZED RE-SEARCH
                 (lag / CPU actor);     re-search partly       hurts (both WORSE):
                 re-search exonerated;  (seqvtree below the    next seqvtree with
                 ReDo UNIDENTIFIED      control); seqvtree     VTREE_COMMUNAL_TRAJS
                 (both of its runs at   seed 1 first           1 (bit-identical to
                 the WORSE floor) ->                           the sequential engine)
                 SVE on the sequential                         BEFORE any SVE run
                 stack (+vtree)
  When a cell names two actions, the seed-1 run comes first; in the
  (WORSE, WORSE) cell the TRAJS-1 run precedes SVE.
LAUNCHED 2026-09-30 15:17 (commit 2b46acad): pong-noredo-s0-2026-09-30 and
pong-seqvtree-s0-2026-09-30. PAUSE RULE FIRED 18:39: host commit free
1.70 / 1.12 / 0.97 GB on three readings (GPU fine at ~9.97 GB used) ->
seqvtree paused at its iter-8 checkpoint (18:09); commit free recovered
to 9.08 GB, i.e. the seqvtree tree held ~8 GB of commit. HOST COMMIT (the
fixed 4.5 GB page file, 36.2 GB limit, desktop apps included) -- not the
GPU -- is what stops these two running concurrently. seqvtree resumes
after noredo finishes (relaunch supervise_factor.py seqvtree 0 via WMI;
its resume restores the training RNG streams (not the env's NoopReset stream, which restarts from the seed -- statistically equivalent, not bit-exact)), unless the page file is
enlarged first.
SVE waits for this: it goes on whichever stack reads NO WORSE (or on the
sequential faithful stack if none does).

### RESULT -- search acting on the parallel stack (bisect step 1), 2026-09-30

pong-searchact-s0-2026-09-29 finished 09-30 13:49 (paused 22:10-23:44 at
the iter-40 checkpoint; ~15-20 min/iter late -- CPU search acting).
EXACT evals (prior-argmax, all 30 noops) / BN-recalibrated:
  iter    45     50     55     60     65     70     75     80     85     90
  raw   -15.3  -13.2  -12.1  -13.2  -11.7   -9.9  -12.2  -10.6  -10.5  -11.5
  recal                      -10.4  -12.5   -8.7   -9.8  -14.1   -9.5  -10.9
No noop phase won in any row; 7-9/30 capped at 65-85 (long rallies, no
breakout).
READ (1): no success row (control 50 / 65, prioract 75). READ (2): mean
60-90 = -11.37 -> WORSE (< -2.32); secondary (recalibrated) -10.84, far
below the controls' +13.94 / +6.76. READ (3): no success, so no collapses.
READ (4): 11-20 min/iter, rising with game length.
READING (per the pre-registered matrix): WORSE -> consistent with the
parallel stack or ReDo, not policy-only acting: search acting did not
rescue the stack -- it did worse than prioract (single seeds; seed luck
is large in this campaign, but BOTH parallel-stack runs fall below BOTH
sequential controls). Shared by the two parallel-stack runs and absent
from the controls: the pipelined CPU actor (data one iteration late),
the vectorized communal re-search (8 trajectories), ReDo, and prior-
argmax in-run eval (inert for learning). Pre-registered next: step 2,
ReDo off; if still WORSE, step 3 splits the pipelined data lag
(sequential acting) from the vectorized re-search.

### BN recalibration probe (option B), 2026-09-29 -- post hoc, exploratory

probe_bn_recal_eval.py: representation + prediction BN statistics re-
estimated from root-only forwards on a COMMON calibration set (prioract
@85's greedy frames at noops 1-4, 9,226 frames), then the exact eval.
  iter    prioract raw->recal   ctl s0 raw->recal   ctl s1 raw->recal
   60       -8.4 ->  -8.0         +6.8 ->  +7.5       -1.7 ->  -0.2
   65       -3.4 ->  -5.9        +15.4 -> +18.8      +14.5 ->  +7.3
   70      -20.8 ->  -1.9        +19.8 -> +14.7      +14.2 ->  +4.1
   75       +8.1 ->  +4.2         +3.8 -> +13.8       +9.6 -> +14.5
   80       -3.0 ->  +9.3        +11.2 -> +10.1      +17.5 -> +16.3
   85      +11.2 -> +11.5        +14.8 -> +13.3      -14.9 ->  -0.2
   90      -16.8 ->  +5.9        +13.6 -> +19.4       +0.6 ->  +5.5
  mean60-90 -4.73 -> +2.16       +12.20 -> +13.94     +5.68 ->  +6.76
READING: every collapse/dip is rescued (+10 to +23: prioract @70/@80/@90,
s1 @85, s0 @75) -- the late-run collapses are mostly the eval-mode BN
artefact (running stats averaged over unroll depths 0..K; acting sees
depth 0). On recalibrated evals prioract would read INCONCLUSIVE, not
WORSE, but it still trails both controls and broke out later. The
common calibration set is NOT neutral for other agents: some good
snapshots drop (s1 @65 -7.2, @70 -10.1; s0 @70 -5.1) -- a stabilizer
should recalibrate each agent on its OWN recent root states. The
pre-registered prioract verdict (WORSE, un-recalibrated) stands.
CANDIDATE STABILIZER (next arm after the bisect, needs its own build +
audit): before each acting/eval weight sync, re-estimate depth-0 BN
statistics for the acting nets on a batch of recent replay root states.

LAUNCHED 2026-09-29 12:47: pong-searchact-s0-2026-09-29 (commit
bd5bb0fa, WMI supervisor; preflight passed in the main tree).

### Next experiment -- search acting on the parallel stack (the prioract bisect, step 1; Kai, 2026-09-29)

DECISION (Kai, 2026-09-29, after prioract read WORSE): run the
pre-registered bisect -- the prioract stack with SEARCH acting restored,
ReDo kept on -- and, in parallel, the BN-recalibration probe (option B).
ARM effzero-repro-faithful-searchact = the prioract arm with exactly
USE_SEARCH_ACTING True, BIRTH_SEARCH False, PRIOR_ACT_STAGED_TEMP False,
EVAL_PRIOR_ONLY True (tests/test_searchact.py pins the delta). One seed
(0), --ray --cpu-actor, 90 x 2000 steps, snapshots every 5 iterations.
WHAT IT ISOLATES vs prioract: who picks the actions and how they explore --
the CPU actor's 50-sim search with root noise and the staged VISIT
temperature records the targets while playing (the records birth search
reproduced after the fact), instead of sampling the prior at the staged
temperature. Everything else is prioract's: vectorized communal
re-search, the CPU actor one iteration stale, ReDo (projector excluded),
the faithful flags, the 31.25k buffer.
vs the faithful control: the parallel stack (pipelined CPU actor),
vectorized communal re-search, ReDo, and prior-argmax in-run eval
(EVAL_PRIOR_ONLY: the control evaluated with search in-run; the exact
readout below is prior-argmax for every run).
EVAL_PRIOR_ONLY (new flag, default off): greedy episodes act on the
prior's argmax and record plain 3-field steps, tail included, even in a
search-acting paper arm; self-play still searches. Needed because 8
CPU-searched eval games of up to 3000 steps take > 1 h per row, and
snapshots are written only on eval rows.
RISK: CPU search acting is ~0.25 s/step (check D) -> ~10 min per
2000-3000-step episode, about one iteration's training time: the actor may
set the pace (iteration time is logged; prioract ran 12.9 min/iter).
PRE-REGISTERED READS (exact prior-argmax evals, iters 45-90, same bands as
prioract's, fixed from the faithful control before any row):
  (1) first SUCCESS row: exact >= +10 or >= 3 of the 4 noop phases won --
      prioract's own read (1) definition (control 50 / 65; prioract 75:
      +8.1 with 3/4 phases; first exact >= +10 at 85);
  (2) mean exact over rows 60-90: NO WORSE >= +3.68, WORSE < -2.32, else
      INCONCLUSIVE (control +12.20 / +5.68, prioract -4.73);
  (3) collapses: rows AFTER the run's first success row (read 1) with
      exact < 0 -- pre-breakout rows are not collapses (control s0 0 of 8
      rows after 50; s1 1 of 5 after 65 (@85); prioract 2 of 3 after 75
      (@80 -3.0, @90 -16.8)), and the BN-variance ratio at each;
  (4) iteration time (acting-bound?) -- launched only after the
      CPU-heavy recalibration probes finish, so no other load overlaps;
      the supervisor prints elapsed_s and min/iter per row.
INTERPRETATION (actions, with hedged labels -- prioract's WORSE is one
seed, the two control seeds differ by 6.5 points and one collapse row
moves a 7-row mean by ~4-5): NO WORSE -> consistent with the stack and
ReDo being fine and policy-only acting (prior-sampled exploration / birth
search) having hurt prioract; SVE then runs on THIS stack (a searchact-sve
arm, re-derived). WORSE -> consistent with the stack or ReDo: step 2 is
ReDo off; if that is still WORSE, step 3 splits the pipelined data lag
(sequential acting) from the vectorized re-search. INCONCLUSIVE -> seed 1.
The BN-recalibration probe (option B) is read alongside: if recalibrated
BN statistics rescue the collapsed snapshots, recalibration joins every
later arm as a stabilizer, whatever this run says.
SECONDARY READOUT (pre-registered 2026-09-29, before any row of this run,
after the probe showed recalibration rescues collapses): the same reads (1)
and (2) on BN-RECALIBRATED exact evals (probe_bn_recal_eval.py, the common
calibration set = prioract @85's greedy frames at noops 1-4), computed
identically for this run, prioract and both control seeds (rows 60-90).
The PRIMARY verdict stays the un-recalibrated read (2); the secondary one
says whether a difference survives once the eval-mode BN artefact is
removed from every run alike.

### RESULT -- prioract (policy-only acting on the parallel stack), 2026-09-29

pong-prioract-s0-2026-09-27 finished 09-29 10:59 (19.3 h, ~12.9 min/iter).
EXACT evals (all 30 noop counts), vs the faithful control:
  iter      45    50    55    60    65    70    75    80    85    90
  prioract -11.3  -9.7 -12.4  -8.4  -3.4 -20.8  +8.1  -3.0 +11.2 -16.8
  ctl s0   -19.6 +12.3 +10.0  +6.8 +15.4 +19.8  +3.8 +11.2 +14.8 +13.6
  ctl s1    -9.1  -8.9  -1.2  -1.7 +14.5 +14.2  +9.6 +17.5 -14.9  +0.6
READ (2): mean over 60-90 = -4.73 (controls +12.20 / +5.68) -> WORSE
(pre-registered band < -2.32). READ (1), as pre-registered (exact >= +10
or >= 3 of 4 phases won): iter 75 (+8.1, 3/4) vs controls 50 / 65; the
first exact >= +10 is iter 85 (corrected 09-29, bisect audit r1). Under
read (3)'s definition (rows after the first success) prioract's collapses
are @80 (-3.0) and @90 (-16.8); @70 (-20.8) precedes its first success. The
BN-variance ratios below were measured at @65/@70/@75/@85/@90 (not @80). 22/30 games capped at 60-65 (long rallies, the
pre-breakout phase the controls also passed through).
The agent LEARNS to win (+8.1 @75, +11.2 @85, 4/4 phases) but collapses to
-20.8 @70 and -16.8 @90. probe_prior_bn_mismatch on the snapshots: at both
collapses the prediction head's BN running variance is ~1.5x the root
batch variance (1.06-1.12 at the good snapshots) and the eval-mode prior is
flat (H 1.68 vs 1.51-1.60); @70 the batch-stat prior is much sharper (1.51,
an eval-BN artefact), @90 it is flat too (1.62, partly real). Same
signature as the control's s1 @85 collapse (-14.9).
CONSEQUENCES (pre-registered): SVE is NOT released on this stack (launch
gate: "if prioract is WORSE, SVE moves to the search-acting stack"); the
bisect starts with search acting on the parallel stack, then ReDo off.
One seed: suggestive, not conclusive.
SVE STAGE-2 COST (measured on the GPU, iter-26 buffer, one 200-step sync
window through the real batch builder): 10,715 searches in 44 s -> ~400 s
per 2000-step iteration (~+52% wall clock, not the estimated +14-21%): the
misses trickle in ~54 per step, so every search call is small. Nearly every
stale position is searched each window anyway, so an eager per-sync search
of all stale positions in full 256-batches should cut it ~3x (needs a
short audit round).

### Next experiment -- SVE on the prioract stack (Kai, 2026-09-28; built and audited, stage 2 since 09-29)

AUDIT LOOP (2026-09-28): round 1 -- EZ-V2 fidelity (clean) + integration (1
MEDIUM: diagnostics scored against SVE targets -> td_value_targets; LOWs:
EZ-V2-exact stale boundary in transitions, fixed TD horizon under mixed,
sve_buffer_frac, mode gate, full manifest re-checked before every launch);
rounds 2 and 3 closed clean (LOWs fixed: truncated-tail test, superseded
old plan, stop rule recalibrated against the control, manifest inputs).
STAGE 2 (fresh search, 2026-09-29): round 1 -- EZ-V2 fidelity + correctness
(clean; LOWs fixed: EZ-V2's N+1 root mean, USE_TARGET_MODEL gate, non-
vacuous real-engine test, deviations documented); round 2 closed clean
(20 independent mutations caught; root visits == sims verified inside the
engine). Parity vs 78c29b05 held in every round (td arms bit-identical).

DECISION (Kai, 2026-09-28): the run after prioract is EfficientZero V2's
search-based value estimation on top of the prioract stack, with the
paper's T1 = 4e4 training steps and T2 = 2e4 fresh transitions, after the
stage-0 probe (benchmarks/probes/probe_sve_stage0.py, on a copy of the live
prioract checkpoint at iter 26).

ARM effzero-repro-faithful-prioract-sve = the prioract arm + exactly
VALUE_TARGET_MODE "mixed", SVE_START_TRAIN_STEPS 40000, SVE_FRESH_TRANSITIONS
20000, SVE_SOURCE "search", SVE_SEARCH_SIMS 16 (tests/test_sve.py pins the
delta). Same seed, same launch line
(--ray --cpu-actor); control: pong-prioract-s0-2026-09-27.

WHAT CHANGES (stage 2 since 2026-09-29 -- SVE_SOURCE "search"; the stage-1
text below describes the stale rule, which is unchanged):
- From training step 40k (iter 20) on, a sampled window ROOT whose global
  buffer index lies before the newest 20k transitions (EZ-V2's buffer-index
  rule, FIFO by trajectory) takes the stored root_value as the value target
  at EVERY real position of its window; fresh roots and all padding keep
  the TD target unchanged. Per ROOT, as in the released code.
- STAGE 2 value (the arm): at batch time, each stale real position is
  searched fresh -- fresh-root PUCT (muzero_vtree.VectorizedPUCT) through
  the lagged TARGET nets, 16 sims, no root noise, deterministic "first"
  tie-breaks (the global RNG is never drawn, so training streams cannot
  depend on cache state; a resume starts the cache empty). Cached per
  (trajectory, position) until the next target sync, and dropped at every
  block start at the same call as the bootstrap cache (it does NOT follow
  the BOOTSTRAP_CACHE toggle, which the arm leaves off: the SVE cache is
  always on). The terminal record of a terminated
  game is absorbing (0, never searched); a truncated tail is searched.
  The value is EZ-V2's root mean over N+1 estimates (the root's own network
  value + the N backups). sve_value_age = steps since the target sync
  (0-199; the double-buffered weights are one interval older). Cost rows:
  sve_searches, sve_search_s. Measured cost estimate (stage-2 audit, from
  the birth search's GPU rate): +14-21% wall clock (~11k positions per
  sync window); measured directly in the final smoke.
  EZ-V2 fidelity (stage-2 audit): matches its model (target/reanalyze nets,
  200-400 steps old), 16 sims, every position of a stale root, discount,
  value units and the N+1 root mean. Deliberate deviations: no root noise
  (EZ-V2's reanalysis search is noisy; noise-free is deterministic and
  cacheable -- measured 0.004 above the noisy value, sd 0.009 between
  draws), a truncated tail is searched (EZ-V2: 0), and PUCT instead of
  Gumbel (repo-wide). Not used: EZ-V2 also takes stale roots' POLICY
  targets from this search -- a possible follow-up, since it is paid for.
- STAGE 1 value (SVE_SOURCE "stored", kept as an option): the birth search
  (online nets at the actor's request-time step, 50 sims, root noise) until
  the refresh rotation re-searches the position (lagged target nets,
  communal). Each write is stamped (root_value_step, SVE arms only) for the
  sve_value_age diagnostic.
- Effective coverage: 0 until the buffer exceeds 20k transitions (~iter 18
  on the faithful pace), then up to ~36% of roots once it is full at 31.25k
  (~iter 25).
- Fidelity details (SVE audit r1): staleness counts TRANSITIONS, and a
  root at global transition index g is stale iff g <= N - T2 (EZ-V2's
  exact comparison: the fresh window holds T2 - 1); under "mixed" the TD
  horizon is the fixed TD_STEPS for every sample, before T1 too (EZ-V2
  zeroes delta_td for the mixed target). Inert for TRAINING: the decay
  needs a record >= 54k steps old, and such records always lie outside the
  newest 20k transitions (stale -> SVE replaces the TD value). It does
  touch diagnostics on a slow pace (~1k transitions/iter: the buffer spans
  ~62k steps): sve_td_gap / bias there compare against horizon-5 TD, while
  the control, the TD-referenced panel and the refresh priorities use the
  age-shrunk horizon. Known
  deviation: the in-progress game is appended only at the block boundary,
  so the fresh window reaches ~one game (5-12% of T2) further back than
  EZ-V2's 400-step block pushes.
- The TD target is still assembled for every position, then swapped, so
  the row diagnostics measure the exact target replaced (stage 2 adds only
  the searches). The diagnostics panel (value, value_control) is built with TD
  targets (td_value_targets), so it stays comparable with the prioract run
  and never enters the SVE window; step_loss_mean is against the TRAINING
  (mixed) target. Priorities: the batch writeback gives SVE roots |v - SVE| (as
  EZ-V2); the refresh's priority recompute stays TD-based (an SVE-based one
  would zero stale positions' priorities and starve them under PER).
- Other arms: bit-identical (parity vs 78c29b05 on the prioract, faithful
  and act-temp-1.5 arms: weights, RNG states, rows -- except the wall-clock
  birth_search_s_last).

ROW DIAGNOSTICS: sve_frac (share of real value targets from search, as
PER sampled them; 0.0 before T1), sve_buffer_frac (the buffer's stale
share, uniform over transitions -- the reference for sve_frac), sve_td_gap
(mean |SVE - TD| where used), sve_td_bias (mean SVE - TD), sve_value_age
(stage 2, the arm: training steps since the target sync, 0-199; stage 1:
steps since the stored value was computed -- a birth value is stamped with
the actor's weights' step, a refresh value with its application step,
although its target nets lag a further 200-400 steps, small against ages
of ~1e4), sve_searches / sve_search_s (stage 2: searches run, seconds). The SVE window is per row and not
checkpointed, like the other row windows: a resume's first row covers only
the iterations since the checkpoint (means unbiased).

STAGE-0 PROBE RESULT (2026-09-28 21:05; prioract checkpoint @26, training
step 52k, buffer 30,788 transitions, 35% stale; TD self-check exact):
  STALE (n 2048, data age p50 40k steps; value scale: TD mean -1.66, sd 0.43)
                    vs TD: bias   MAE   corr
    stored (stage 1)      +0.36  0.38  0.70
    fresh50 (target nets) +0.16  0.18  0.85
    fresh16 (target nets) +0.12  0.15  0.88
    stored vs fresh50:    +0.21  0.24  0.79   <- staleness
    fresh16 vs fresh50:   -0.04  0.05  0.97   <- 16 sims suffice
  NEWEST (n 3788, age p50 4k) vs REALIZED discounted return (mean -0.61):
    TD bias -0.87 MAE 0.91 corr 0.47 | stored -0.57 0.75 0.27 |
    fresh50 -0.77 0.86 0.36 | fresh16 -0.79 0.88 0.37
READING: (1) the gap is NOT tiny -- stage-1 SVE would move 35% of value
targets by ~0.9 sd, upward. (2) Staleness is most of it: stored values
(mostly birth searches ~40k steps old; refresh reaches ~2 of ~15 games per
iteration) sit +0.21 above a fresh search through today's target nets, so
stage 1 would add a systematic optimism EZ-V2's batch-time values do not
have (theirs are <= 200-400 steps old). (3) Realized returns on the newest
games are far ABOVE every estimate (the value function lags an improving
agent) -- the tilt runs opposite to the one the proposed amendment feared.
By error the stored value is closest (0.75 vs TD 0.91), by correlation TD
is best (0.47 vs 0.27): the pre-registered rule does not say stop.
DECIDED (Kai, 2026-09-29): stage 2 -- a fresh-root 16-sim search through
the target nets, the EZ-V2-faithful source; cached per target sync (every
stale position searched at most once per 200 steps: ~11k positions ->
~12 s per sync window on the vectorized engine, est. +10-20% wall clock;
measured in the final smoke once the GPU is free). BUILT on the sve branch
as SVE_SOURCE "search" (SVE_SEARCH_SIMS 16); stage 1 ("stored") kept as a
tested option.

LAUNCH GATE: benchmarks/supervisors/supervise_sve.py waits for the prioract
run to finish AND for the release file benchmarks/supervisors/sve_release,
written by hand once (a) the prioract readout says policy-only acting held
up (read (2) NO WORSE or INCONCLUSIVE-leaning-positive) and (b) the stage-0
probe does not say stop. If prioract is WORSE, SVE moves to the search-acting
stack instead (redesign), not this arm.

PRE-REGISTERED READS (vs the prioract run, exact evals, same seed; Pong
saturates at +21, so SVE can only show as SPEED or STABILITY):
  (1) speed: first row with exact >= +10, and first with >= +15;
  (2) level: mean exact over rows 60-90 (bands as prioract's read (2),
      recomputed from the prioract + control curves before launch);
  (3) late stability: number of rows 60-90 with exact < 0 (the faithful
      control had 1 of 7 in s1);
  (4) diagnostics: sve_buffer_frac is 0 before T1, then the builder's own
      max(0, (N - T2 + 1) / N) averaged over the row's batches -- by
      construction, so it only checks the buffer SIZE (expect ~0.29-0.36
      once full: the cap counts records and eviction drops whole games), not
      the stale rule, which the tests pin (brute-forced against an
      independent enumeration by the round-2 audit). sve_frac is read
      relative to it (PER shifts it either way); sve_td_gap / bias against
      the stage-0 probe's prediction (its fresh16 row at @26: bias +0.12,
      MAE 0.15 -- an N-only mean with random tie-breaks; the arm's N+1 root
      mean sits ~0.006 lower, so predict ~+0.11); the TD-referenced value
      error and the
      collapse probes.
STOP RULE -> REVIEW WITH KAI (not an automatic kill; replaces the design's
"value/control doubling", which the SVE audit r3 showed fires on the
faithful control itself): the TD-referenced `value` (mean |v(s0) - TD z0|
over the diagnostics panel) above 2x the PRIORACT run's value at the same
row for 3 consecutive rows after T1 -- a value self-echo would look like
this. Calibration: between the two faithful seeds on the same recipe the
same-row ratio reached 6x in single rows but never stayed >= 2x in the
same direction for even 2 consecutive rows. Context, not a trigger:
sve_td_bias drifting upward row over row (SVE increasingly optimistic
against the reward-anchored TD target). The design's other condition (both
seeds <= -17 at iter 30) is dropped: the faithful control itself was below
it (exact -20.8 / -18.1 at iter 30).
One seed: a null or negative result is suggestive only.

### Next experiment -- policy-only acting on the parallel stack (Kai, 2026-09-26; built, audit pending)

ARM effzero-repro-faithful-prioract, run with --ray --cpu-actor, seed 0,
90 iterations, archfix's command otherwise. Deltas vs the faithful arm:
- VECTORIZED COMMUNAL RE-SEARCH (REFRESH_VTREE, VTREE_COMMUNAL_TRAJS 8):
  same communal semantics and coverage (4 trajectories / 2 iterations) at
  ~3.4x throughput; ~4% of roots flip visit counts vs the sequential engine,
  below the search's own repeat noise. Communal kept (not fresh-root) so the
  re-search semantics match the faithful run.
- POLICY-ONLY ACTING (USE_SEARCH_ACTING False, PRIOR_ACT_STAGED_TEMP): the
  CPU actor samples prior^(1/T), T on the papers' staged schedule (1.0 /
  0.5 / 0.25). Eval becomes the prior's argmax -- the number the Switch
  would see (and no more slow CPU search eval).
- BIRTH_SEARCH: every new game is searched ONCE on arrival, on the GPU,
  through the weights the ACTOR PLAYED WITH (online, not the lagged target
  model -- re-stamping newcomers with it was the keepnew-era poison; under
  --ray the collect runs after the next training block, so the pipeline
  snapshots the synced weights at request time -- round-1 audit HIGH-1),
  fresh-root, root noise on acted steps, greedy truncated tail -- exactly
  the records the acting search would have made. Cost ~10-30 s/iteration.
  Only WHO PICKS THE ACTIONS changes vs search acting.
- ReDo every 1000 steps, tau 0.025 (redo.py), passes mid-period (1.5k,
  2.5k, ...: never on an eval row, checkpoint or the final step):
  single-consumer units only
  (block-internal convs, representation conv1/conv3, the prediction
  heads' 1x1 policy/value convs, the value-prefix input conv and hidden
  FC); the SimSiam
  PROJECTOR IS EXCLUDED (REDO_NETS; Kai 2026-09-26 -- ~94% of dormant units
  sit in its hidden layers, so including it would re-randomize most of it
  every pass); BN channel reset with gamma/beta = the survivors' mean
  |gamma| / mean beta (the identity made recycled units 7-75x louder and
  blew the loss up 5-6x for several steps); no-BN head units (policy/
  value 1x1 convs) keep an unscaled default-init row and get their BIAS
  solved so their activity equals the median survivor's (norm matching
  left them dead on arrival; row rescaling at bias 0 inflated them 4-24x
  into sparse units -- round-2/3 audits; bias-solved rows measure p50
  1-6x / p99 <= 13x the survivors' norm, held-out activity 0.97-1.03x the
  target); outgoing weights zeroed; momentum zeroed. Motivation: dormant-unit accumulation under SGD+wd in the
  faithful run (half the first-layer filters dead in s1 by iter 75).
  Literature caveat: ReDo was only validated with Adam and without
  BN/residuals; a 2024 comparison ranked it weakest of the plasticity fixes.
  Rows log redo_recycled_last/_total and redo_dormant_frac_last.
- SEQUENTIAL -> --ray --cpu-actor: the actor plays one iteration ahead of
  training (the pipeline's one semantic delta, data incorporation).
NOT changed: buffer size (31.25k; EZ keeps all 100k -- a separate test),
coverage, the optimizer, and every faithful flag.
EVAL IS NOT LIKE-FOR-LIKE with the control (round-1 audit MEDIUM-1): the
control's rows are 50-sim search eval; this arm's are the prior's argmax.
LIKE-FOR-LIKE CONTROL CURVE (computed by the round-1 fidelity audit):
pong-faithful-s0's snapshots under prior-argmax eval, eval-actor seeds
(seed*100003 + it*613 + 500000 + e), 8 games each:
  iter   5  10  15  20   25   30   35   40   45   50   55   60   65   70   75   80   85
  prior -20.9 -20.6 -19.0 -18.5 -13.8 -20.8 -12.9 -12.1 -19.0 +7.8 +7.9 +6.4 +16.4 +20.6 +9.5 +12.9 +15.8
  (search eval for reference: -20.1 ... -1.4 @50, +18.3 @55, +21.0 @65,
  +18.6 @80). Capped rallies are common under prior-argmax (4/8 @50, 5/8
  @55) and games are near-deterministic across seeds -- read eval_capped
  and treat 8 games as fewer than 8 samples.
READOUT (round-2 design audit, HIGH): prior-argmax eval is DETERMINISTIC
given the NoopReset count and depends mostly on noops mod 4, so an 8-game
row is ~4-8 distinct games and single rows were off by up to +-6 (control
iter 75: row +9.5, exact +3.8). The PRIMARY readout is therefore offline:
benchmarks/probes/probe_exhaustive_eval.py plays every noop count 1..30 per
snapshot (exact expectation, fixed thread count, ~5.5 min/snapshot CPU;
records in benchmarks/probes/probe_exhaustive_eval-<run>.jsonl),
for prioract AND both control seeds (s1 is the seed-spread yardstick; under
--ray "same seed" shares only the network init, not the episode streams).
Pre-registered reads (on EXACT means; env_steps reported alongside, since
prior acting changes episode length):
  (1) first row with exact mean >= +10 (or >= 3 of the 4 noop phases won);
  (2) mean exact over rows 60-90 vs the two control seeds:
        NO WORSE  if >= the lower control seed's mean - 2;
        WORSE     if more than 8 below BOTH control seeds;
        otherwise INCONCLUSIVE -> run seed 1 before concluding;
  (3) the collapse probes at 15-30 (batch-stat rank, shift/state, raw disc);
  (4) behaviour (per-row means of the acted steps, logged as beh_*):
      beh_act_agree (sampled action == argmax of the NOISY birth target;
      round-1 prediction from the control's weights: ~0.34-0.38 at T=1,
      ~0.46-0.51 at T=0.5, ~0.58-0.62 at T=0.25 -- the search-acting
      control's own top-1 rate was ~0.47 / 0.61 / 0.70), beh_prior_agree
      (temperature-free prior argmax vs birth argmax; control ~0.50-0.55),
      beh_kl_target_prior, beh_H_act and beh_T (step-weighted mean) /
      beh_T_values (the exploration actually applied), beh_rare_rate,
      beh_action_counts. Under --ray the birth priors are recomputed on the
      driver GPU while the actor sampled from a batch-1 CPU forward (cuDNN
      may use TF32), so beh_prior_agree can flip at near-ties -- a
      diagnostic-precision caveat, not a behaviour difference;
  (5) ReDo attribution: redo_recycled_window / redo_by_net_window,
      redo_gn_pre20 vs redo_gn_post20 (grad-norm burst after passes),
      step_loss_mean / step_grad_norm_mean. Offline: redo.dormancy_scores on
      the control snapshots gives dormancy at matched iterations.
  Per-row windows are not checkpointed: a resume loses at most the partial
  window in flight (cumulative totals ARE checkpointed).
EXPLORATION CHANGES MATERIALLY (the variable under test, audit-measured on
the control's weights): at T=1 prior sampling is MORE random than visit
sampling (H 1.52 vs 1.32); at T=0.25 it is LESS exploratory (H 0.69 vs
0.87; rare-action rate 0.5% vs 1.9%) -- the Dirichlet floor is gone. If
the run stalls late, this is the first suspect.
Offline follow-up: greedy 50-sim search eval of this arm's snapshots, so
both arms also have the search number. A failure is ambiguous by design (bundle, one seed): the
bisect starts with search acting on this stack, then ReDo off.
Order after it: SVE (redesigned for whichever acting mode survives).
AUDIT LOOP (Kai: audit before launch): six fresh-eyes rounds, 09-26..09-28.
Rounds 1-3 found and fixed HIGH/MEDIUM issues (birth-search weight timing,
BN recycling loudness, no-BN head recycling -> bias solve, readout
determinism -> exact eval); rounds 4-6 closed clean (no finding above LOW),
their LOWs fixed and re-audited. Left unfixed on purpose (cosmetic, fail
safe; fixing would reopen the loop): a JSON manifest that is not an object
raises in the preflight (nothing launches), and a test-failure reason that
embeds an object address reprints every preflight poll. The launch is gated
on benchmarks/supervisors/prioract_manifest.json (sha256 of the audited
files) plus tests/test_prioract.py passing in the main tree.
CONTROL CAVEAT for read (2): faithful s1 collapsed late (exact +17.5 @80 ->
-14.9 @85, all four noop phases lost) -- the control itself is unstable over
rows 60-90, so read (2) compares against both seeds' full curves.
READ (2) BANDS, fixed 2026-09-28 before any prioract row (control exact
means over rows 60-90, all seven rows): s0 +12.20, s1 +5.68 -> NO WORSE if the
prioract mean >= +3.68; WORSE if < -2.32 (more than 8 below both); else
INCONCLUSIVE -> seed 1. Control exact @90: s0 +13.57, s1 +0.63 (its search
eval read +11.5 -- the greedy prior only partly recovered from the @85
collapse).

### Campaign record, 2026-09-22 .. 09-26 (edge pair, fidelity audit, the faithful run)

- EDGE PAIR FINAL (iter 90): s0 −8.0, s1 −9.0. Both beat the three non-winning
  archfix-recipe runs (−11.4/−14.0/−11.4), no breakout. Both projectors
  collapsed (batch-stat rank ~2 by iter 15; 33-48% dead final-BN gammas).
- COLLAPSE MECHANISM (probe_shift_collapse): augment_observations shared ONE
  shift/intensity draw between the online root and the SimSiam targets, so the
  shift was the one feature the consistency loss could always match. Collapsed
  projectors encoded 83-85% shift, 2-5% game state. EfficientZero's released
  code draws root and targets independently (core/train.py:73-75).
- FIDELITY AUDIT (two fresh agents, then two more on the fix): verified
  deviations from EfficientZero's released code -- shared augmentation draw,
  AdamW 3e-4 instead of SGD 0.2/0.9/1e-4 with a 1000-step warmup, gradient
  placement (their whole loss x1/K incl. step 0; hook halves heads too),
  min-max latent renormalisation (their Atari config: off), no consistency
  mask past trajectory ends, clamp after intensity jitter. All ported as
  default-off flags; arm effzero-repro-refresh5x-communal-edge-faithful turns
  them all on (plus a non-finite-gradient skip, their AMP scaler's behaviour).
  tests/test_effzero_fidelity.py: every flag mutation-checked.
- THE FAITHFUL RUN (pong-faithful-s{0,1}-2026-09-24, sequential, seeds 0/1,
  rows = mean of 8 eval games): no projector collapse in either seed (rank
  41-85 throughout, 99% game state, one-step action discrimination 0.8-0.99
  vs ~0.54 in every earlier model). Scores: s0 −13.2/−4.1/−1.4/+18.25/+21.0
  at iters 25/35/50/55/65; s1 −11.4/−9.1/−1.4/+11.75/+19.0. At iter 55
  (~101k env steps, the Atari-100k budget) the 2-seed mean is +15.0 vs
  EfficientZero's published 20.1. Both seeds ~+18.5 at iter 80. Archfix
  needed ~150k steps for +17.1 and was 1 in 4; this recipe is 2 for 2.
- OPEN: transient BN-statistics dips (eval-mode running stats drift from the
  network's activations; prior flattens; s0 −20.2 at iter 30, −19.1 at 45;
  both recovered within one row). Model is fine under batch statistics
  (probe_prior_bn_mismatch). Every dip so far falls at a schedule boundary
  (horizon decay ~iter 27, visit temperature at 45 and 67.5). Unverified.

### Campaign record, 2026-09-11 .. 09-21 (archfix replication, edge visits, collapse probes)

- ARCHFIX IS 1 IN 4. The recipe's four runs at iter 90: +17.1 (archfix,
  seed 0) / -11.4 (check D, same recipe on the Ray stack, seed 0) / -14.0
  (sequential control, seed 1) / -11.4 (r12: 3x refresh coverage via the
  vectorized communal engine). All reach a -8..-15 plateau by iter 40-60;
  only archfix broke out (+3/+16/+17.1 at 80/85/90). The parallel stack is
  exonerated, and 3x refresh volume did not buy consistency. Single-seed
  verdicts cannot resolve arm effects smaller than this spread: run >= 2
  seeds per arm.
- EDGE VISITS (DESIGN.md "Edge visits", flag COMMUNAL_EDGE_VISITS): in the
  communal chain tree, root n+1's own simulations were counted on root n's
  behaviour action. The result was a ~50% behaviour-mass floor in 15/16 of
  every communal target, and an anti-behaviour cliff at the window's last
  slot that the policy memorizes. archfix trained on these targets too.
- EARLY-COLLAPSE HYPOTHESIS (Kai, 09-21): does heavy early refresh with a
  weak model collapse policy diversity, so the dynamics never learns?
  Probed (benchmarks/probes/probe_buffer_diversity.py,
  probe_counterfactual_dynamics.py, probe_counterfactual_latent.py + logs).
  NOT supported: every buffer, from iter 10 (whole history) through iter
  88 and including the saturation arms, uses NOOP/UP/DOWN near-uniformly
  (3-way entropy >= 1.085 of 1.099). The iter-10 model already tells its
  actions apart (emulator-branched counterfactuals). Rarely-chosen actions
  are modelled within 5-10 points of favoured ones. FOUND INSTEAD: the
  SimSiam projector dimensionally collapses in 5 of 7 non-winning models
  (effective rank 5-8 vs ~20 archfix, 45 at iter 10; ~1/3 of final-BN
  gammas dead), and archfix has the most action-aware dynamics in both
  projection and raw latent space. This is correlational (n=1 winner).
  Augmentation is already on in this lineage. Candidate fixes if snapshots
  show collapse leads the plateau: a variance/covariance hinge on projector
  outputs, or the lagged target nets as a BYOL-style consistency target.
- CORRECTION 09-22 (benchmarks/probes/probe_erank_artifact.py + log): the
  ranks above were EVAL-mode projections. BatchNorm running stats lag the
  batches, so eval mode understates collapse. Measured with batch
  statistics, as the consistency loss sees it:
  - FINALS: archfix 23.5; check D 4.6, control s1 4.5, r12 3.2. So control
    s1 IS collapsed, and collapse separates the one winner from all three
    non-winners.
  - EARLY COLLAPSE IS UNIVERSAL: 53 at vsat @10, then 1.9-2.8 in every
    mid-run checkpoint (vtree @26, throttle @26, satguard @34, arch1 @48,
    satstale @50) and in both edge seeds by iter 15-20.
  - The raw latent never collapses (rank 83-126, rising).
  - Reading: archfix recovered from, or escaped, a collapse that every run
    goes through by iter ~20. Recovery-then-breakout vs breakout-then-
    recovery is untestable without its snapshots. The edge run's snapshots
    can time it if either seed breaks out.
  - The rows' proj_erank (eval mode) lags. For analysis, use batch
    statistics on the snapshots.
- EDGE RUN (launched 09-21): arm effzero-repro-refresh5x-communal-edge =
  archfix's exact arm + COMMUNAL_EDGE_VISITS + BOOTSTRAP_CACHE off,
  sequential, seeds 0 and 1 concurrently (pong-edge-s{0,1}-2026-09-21,
  benchmarks/supervisors/supervise_edge.py). --snapshot-weights keeps the
  nets at every eval row, and rows now carry proj_erank / proj_dead_frac,
  so collapse onset can be timed against the plateau. Controls: the three
  archfix-recipe runs above.

## Carried decisions from the repro line

Tiny buffer + live refresh direction; entropy-based refresh priority is
Kai's chosen decisiveness-annealing mechanism in this regime (guard v2:
retire only stable-sharp targets -- sharp AND reproduced by re-search;
flat targets stay eligible). Reward-clip
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
