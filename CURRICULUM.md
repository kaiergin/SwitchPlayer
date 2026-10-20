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

DECISIONS 2026-10-17 (Kai; the build starts on branch curriculum-p2):
- TOKENS, not action head x duration head. A held action is already a
  token ("UP x16" -- P5b's multiplicity token), and P5e's mined chunks
  are tokens too, so the policy scores ONE vocabulary of tokens with a
  FIXED number of slots; P2's 12 duration tokens (Pong's minimal NOOP /
  UP / DOWN x {4, 8, 16, 32} frames) are the first vocabulary. Each
  token is one dynamics transition; its reward is the sum over its
  frames; its edge discounts gamma_frame^frames.
- The vocabulary NEVER changes while a network trains. At a generation
  swap (P3) the least-used tokens are dropped, and the data the master
  recorded with a dropped token is replayed to the next model as raw
  4-frame steps (so P3's runs record games at raw-step resolution with
  token boundaries; the inner steps get policy targets by re-search).
  This replaces the iteration-scheduled menu shrink: long tokens live
  on while they are used.
- Trial and error (Kai): simplest defaults first -- plain per-slot
  embeddings (content-built token embeddings are a later variant), all
  12 tokens legal for the whole first run, a mild early long-token
  bias, per-token usage and payoff logged.
- Pong's minimal 3 actions (the FIRE variants duplicate them; 12
  children keep search at ~2x the base's 6 instead of ~4x for 24).
- The composition-consistency loss is the SECOND step (P2b), measured
  on its own.
- Reward-clip growth (Kai, 2026-08-27; previously a one-line carried
  decision) is built on this branch too: raw rewards stored, clipped at
  target-build time under a rising ceiling. Inert on Pong (every reward
  is +-1), so it is tested as its own Ms. Pac-Man run.

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

### Next experiment (pre-registered) -- the token arm on Ms. Pac-Man, 3x longer (Kai, 2026-10-20)

DECISION (Kai, 2026-10-20): after TK's 90 iterations (deployed mean 60-90
856.62 against the twin's 649.14, NO DIFFERENT at +207; raw BETTER) -- "it
takes much longer for ms Pac-Man to truly converge compared to other
games" -- an extended run: the TK arm (tokens + clip growth), 270
iterations (3x), FRESH, every run-length schedule stretched. He keeps
developing the token line even where it does not yet win outright: tokens
are the route to large, combinatorial action spaces (robotics' combined
actions; Smash Bros / Mario Maker's many joystick directions combined with
buttons), learned partly from WATCHING others' play, where a simple
tokenization of inputs is what a third-person observer can search over.
QUESTION: does the token arm keep improving on Ms. Pac-Man when the run is
planned 3x longer?
ARM: TKX = effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow
  (TK, unchanged). Recipe = TK's (supervise_rgrow.py tk) with --iterations
  270: MsPacmanNoFrameskip-v4 --frameskip 4, 270 x 2000 train steps, 3
  episodes (one game's lives) per iteration, --episode-cap 3000, eval every
  5, --checkpoint-every 2, --snapshot-weights, --ray --cpu-actor; seed 0;
  folder mspacman-tokens-rgrow-x270-s0-2026-10-20. TK-90 = the 90-iteration
  run (mspacman-tokens-rgrow-s0-2026-10-19). Same seed as TK-90, but the
  stretched schedules diverge -- PER beta from step 1, the ceiling and the
  bias from iteration 2 -- so TKX's rows are not a replay of TK-90's.
SCHEDULES, stretched (fractions of the planned 540k steps; computed with
the arm's config under _run_length_globals(270, 2000) at each iteration's
starting step (it - 1) x 2000; pinned in tests/test_rgrowx.py):
  - reward ceiling log-linear 1 -> 160 by 50% (iteration 136): ~5.2 at
    row 45, ~12.4 at 68, ~28.4 at 90, ~154 at 135; 160 from iteration 136
    on (row 140 the first logged at 160). TK-90 reached 160 at iteration 46.
  - long-token bias 0.5 -> 0 by 25%: 0 from iteration 69 (TK-90: 24).
  - prior-acting temperature (0.5 x the staged 1 / 0.5 / 0.25): 0.5 to
    iteration 135, 0.25 from 136, 0.125 from 204 (step 405k falls inside
    iteration 203; TK-90: 46 and 69).
  - PER beta 0.4 -> 1.0 over the run; the off-policy horizon decay
    (OFFPOLICY_DECAY_STEPS = 0.3 x total) over 162k steps (TK-90: 54k).
  UNCHANGED: the replay cap, 31,250 transitions (~48 iterations of data at
  TK-90's ~650 base steps per iteration: the last ~18% of TKX's run against
  about half of TK-90's); LR 0.2 constant after the 1k-step warmup (no
  decay, as in TK-90); 2000 train steps and 3 episodes per iteration; TK's
  compute knobs (VTREE_REFRESH_BATCH 64, VTREE_COMMUNAL_TRAJS 1).
START: after RG has completed AND its 36 evals have landed -- the
supervisor (supervise_rgrowx.py tkx 0) waits for RG's completion marker
and for both RG probe jsonls holding rows 5-90, then for no ablation.py at
all, so the GPU and the CPUs are free (expected ~00:30 2026-10-21). A
logged `--no-wait` override exists (Kai's call), refused while any
supervise_rgrow.py process lives.
READS. EXACT eval as TK-90's: whole games (every life), 30 noop starts,
prior argmax, PROBE_ENV=MsPacmanNoFrameskip-v4; rows 5-270 (54 rows);
DEPLOYED (acting states) primary, raw secondary; g = (score - 307.3) /
973.9.
  NEAR RANDOM first: a last-7-row deployed mean with g < 0.2 (< 502.08) is
      diagnosed before any other read.
  (1) level: TKX's deployed mean of its last 7 rows (240-270) against
      TK-90's rows 60-90, 856.62, with the same +-232 band (two 7-row
      windows; the band is the twin's row spread, wider than TK-90's own
      108.3): BETTER >= 1088.6 (g 0.80), WORSE <= 624.6 (g 0.33), else NO
      DIFFERENT. Raw likewise against TK-90's 830.81, +-205 (BETTER >=
      1035.8, WORSE <= 625.8; amended, audit r1 A-L4: each band is 2 SE of
      a 7-row difference from the LARGER of TK-90's and the twin's row
      spreads --
      deployed the twin's 216.9 over TK-90's 108.3, raw TK-90's own 191.6
      over the twin's 129.3). Also g, and descriptively against
      EfficientZero's 1281.2 (g = 1).
  (2) still climbing: TKX's deployed mean of rows 240-270 against its rows
      150-180 (7 rows each, 90 iterations earlier): >= +232 STILL CLIMBING,
      <= -232 FALLING, else PLATEAU.
  (3) data: score against env steps (base steps) beside TK-90 and the
      twin; the Atari-100k read: the 7-row window centred on the first row
      whose env_steps >= 100k (~row 155 at TK-90's pace), descriptively
      against EfficientZero's 1281.2.
  (4) same iterations, descriptive: TKX's rows 60-90 against TK-90's rows
      60-90 (TKX still at temperature 0.5 and ceiling ~9-28 there): what
      the stretch costs early.
  (5) tokens, descriptive (as TK-90's read 4): per-duration play and
      search shares over the run, frames per decision, decisions per life,
      the deployed player's decisions per second.
  (6) stability: the largest drop between adjacent rows from row 140;
      rows below 502.08 (the near-random floor) after the first row >=
      881 (the twin's BETTER threshold, 649.14 + 232).
  (7) clip growth, descriptive: reward_ceiling, reward_kept, reward_max.
DECISION AFTER (completed before launch, audit r1 A-M2). Single seed: a
BETTER / WORSE reads "consistent with". The DEPLOYED reads decide; raw is
reported beside them, and a deployed / raw disagreement is reported, not
acted on. Read (1) x read (2):
  BETTER + STILL CLIMBING -> longer still pays (Kai's call: 4x or more, or
    a seed-1 replicate first); the long length becomes the Ms. Pac-Man
    length for P2b / P3 comparisons.
  BETTER + PLATEAU -> 270 iterations is the Ms. Pac-Man length for P2b /
    P3 (Kai confirms).
  BETTER + FALLING -> it peaked inside the run: diagnose the decline
    (read 6; the 0.125 temperature stage from iteration 204; the replay
    cap) before adopting any length; the best 7-row window is reported
    descriptively.
  NO DIFFERENT + STILL CLIMBING -> the stretch delayed learning and it is
    still rising: NOT "90 suffice" -- Kai's call (longer still, or the
    90-iteration schedules held and the run continued).
  NO DIFFERENT + PLATEAU -> 90 iterations suffice at this recipe; the limit
    is elsewhere (data per iteration, the replay cap) -> Kai's call.
  NO DIFFERENT + FALLING -> diagnose the decline as above.
  WORSE (any trend) -> diagnose (the stretched schedule: the longer high
    temperature, the slower clip growth; the replay cap) before anything
    else; with STILL CLIMBING the stretch may only have delayed learning.
  ABANDONED (stopped under the pace rule or after a crash loop, Kai's
    call) -> read descriptively from the rows it has; no verdict.
  The TOKENS verdict of the 90-iteration pair (TK against RG, scored when
  RG's evals land) is separate and unchanged.
COST: ~48 h at TK-90's 10.3-11.1 min per iteration (its pace stayed flat
while decisions per life went 40 -> 84); up to ~60-70 h if lives lengthen
(the 36-token refresh scales with decisions per life). PACE RULE: a row
past 25 min/iteration is an ALERT in the supervisor and Kai's call.
Interim evals: the hand-off (scratchpad post_rgrowx.py) evaluates each
snapshot once it is SETTLED (the resume checkpoint is newer than it -- see
AMENDMENTS), ONE probe at a time on the CPU while TKX trains (~1-4 min per
probe, two per 5 iterations ~ 53 min: a small CPU load beside the actor --
a deviation from TK-90, whose evals ran after it), the rest 2 at a time
after completion. ~3 GB of snapshots (54 x ~55
MB). Known risks: a Windows Update restart (relaunch: scratchpad
launch_rgrowx.ps1; the supervisor resumes from the last checkpoint, the
hand-off reruns only missing rows) and Kai's Switch-control sessions,
which kill GPU pythons (pause the supervisor and its trainer by command
line, relaunch after).
NO GPU SMOKE: the run executes code byte-identical to TK-90's -- `git diff
a81bcf39` is empty for every core file it runs (muzero.py, redo.py,
muzero_vtree.py, networks.py, scalar_categorical.py, memory.py,
EpisodicLifeEnv.py, atari_wrappers.py, residual_block.py, experiments/,
the probe; checked 2026-10-20 at 756f34ff), and TK-90 completed its 90
iterations with no deaths. The only change is --iterations, whose effects
(the schedules above) are pinned by CPU tests at 270 x 2000. The probe
pins _run_length_globals(90, 2000) for its own process; that has no effect
on its greedy prior-argmax play (no temperature, no bias), so TKX's evals
are exactly TK-90's.
AMENDMENTS (audit r1, before launch; A and B nothing HIGH):
  - A-M1 / B-M1 (both auditors): a resume replays the iterations after its
    checkpoint and rewrites their snapshots, so a row probed before its
    covering checkpoint could keep a dead timeline's result for good. The
    hand-off probes a snapshot only once the resume checkpoint is newer
    than it (checkpoints every 2 iterations: a resume starts at most 2
    back, so a settled snapshot is final; <= 1 iteration of delay).
  - A-M2: the decision table completed (above); the deployed reads decide.
  - B-M2 / A-L3: one supervisor and one hand-off per run -- each refuses a
    second instance of itself, and launch_rgrowx.ps1 is idempotent (it
    skips a running component, so after a pause that killed only the
    supervisor and its trainer, rerunning it starts just the supervisor).
  - A-L1: a RESUME (TKX's own checkpoint exists) skips the wait for RG.
    PROCEDURE for --no-wait: kill TKX's waiting supervisor, RG's supervisor
    and trainer, and RG's hand-off post_rgrow.py by command line, then
    launch `tkx 0 --no-wait`.
  - A-L2: RG complete for over 2 h with its evals still missing is an
    hourly ALERT (is RG's hand-off alive?).
  - A-L4: the raw band (above).
  - A-L5: the trainer guard counts python processes only, and sees only
    experiments/ablation.py trainers (a GPU probe or Kai's Switch code is
    not seen -- the Switch rule above still applies).
  - A-L6: no --ray --cpu-actor run has gone past 90 iterations (~16 h);
    each row ALERTs when host commit falls below 4 GB (memory creep).
  - B-L1-L6 (hand-off): stdout cannot die on a non-ASCII echo; an error in
    its loop is an ALERT and a retry, not a dead hand-off; a probe running
    past 30 min (one snapshot) / 3 h (a group) is an hourly ALERT; the
    final pass never runs two probes on one jsonl. ABANDON and FRESH
    RESTART: see audit r2.
AMENDMENTS (audit r2, before launch; A and B nothing HIGH; both MEDIUMs the
same hole):
  - A-M1 / B-M1: a fresh restart into the same run folder leaves the old
    snapshots, which look settled by mtime once the new run checkpoints --
    the dead timeline's evals would land as done. Now: the supervisor
    refuses a run folder holding snapshots without a checkpoint; the
    hand-off's settled also requires the run's own ablation jsonl to hold
    row i (drained before its covering checkpoint: no delay). PROCEDURE
    for a fresh restart: move the WHOLE run folder (checkpoint, ablation
    jsonl, snapshots, logs) and the run's two probe jsonls aside, or use a
    new folder.
  - ABANDON (A-L1, B-L2): FIRST stop the supervisor and its trainer by
    command line, THEN create the empty file ABANDONED in the run folder.
    The supervisor refuses every launch while it exists; the hand-off
    evaluates the snapshots that have an in-run row and logs "all done".
    Un-abandoning (removing the flag, resuming) must first drop the probe
    rows of snapshots newer than the checkpoint.
  - A-L2: a RESUME is also "this run's supervisor.log logged a LAUNCHED /
    NO-WAIT override line" (a --no-wait start that died before its first
    checkpoint does not wait for RG again).
  - A-L3 / A-L4: each attempt waits for no trainer, THEN checks the
    manifest, then launches; rows keep streaming while it waits.
  - A-L5-L7: tests never read host memory or the real run folder, fail
    instead of hanging, and cover the python-only singleton filter.
  - A-L8: raw thresholds from the rounded +-205 (above).
  - A-L9 (noted, pre-existing): the shared venv is outside the manifest;
    the hand-off ALERTs on a .venv change. No package installs into
    Y:\SwitchPlayer\.venv while TKX runs.
  - B-L1, B-L3-L7: "all done" only after a final pass ran (else "ALERT:
    final pass FAILED"); single-snapshot final jobs; exact-name singleton
    checks (the launcher's regex anchored); the health checks run while
    probes run; only PROBE_ENV records count as landed, and no manual
    probes into TKX's jsonls while the hand-off lives; the --no-wait
    procedure also stops RG's hand-off (above).
AMENDMENTS (audit r3, before launch; A and B nothing HIGH; both MEDIUMs the
fresh-restart procedure again):
  - A-M1 / B-M1: the supervisor also refuses a run folder with no
    checkpoint while TKX's probe jsonls hold records (a restart that
    forgot to move them would keep the old scores); the hand-off, if a
    probe row it saw land ever VANISHES (a jsonl moved under it), ALERTs,
    starts no further probe and exits. PROCEDURE for a FRESH RESTART
    (supersedes r2's): (1) stop the supervisor and its trainer, (2) stop
    the hand-off and its probes (python processes running post_rgrowx.py /
    probe_exhaustive_eval.py on this run) and wait for them to exit, (3)
    move the whole run folder AND the run's two probe jsonls aside (or use
    a new folder), (4) relaunch with launch_rgrowx.ps1.
  - A-L1: while RG has no marker and nothing drives it (no live
    supervise_rgrow.py, or its log silent > 4 h) the wait ALERTs hourly.
    RECOVERY after a reboot while RG is unfinished: rerun RG's launcher
    (scratchpad launch_rgrow.ps1) AND launch_rgrowx.ps1.
  - A-L2 / A-L3: completion and ABANDONED are re-checked right before
    every launch; rows stream from before the preflight.
  - A-L4: RG's eval rows count only as Ms. Pac-Man records (env).
  - A-L5 / B-L4: --no-wait is also refused while RG's hand-off
    post_rgrow.py lives. PROCEDURE for --no-wait (supersedes r1's): kill
    TKX's waiting supervisor, RG's supervisor and trainer, and RG's
    hand-off -- matching the exact script names (post_rgrow.py is NOT
    post_rgrowx.py) -- then run `launch_rgrowx.ps1 -NoWait` (it starts the
    supervisor with --no-wait and keeps TKX's hand-off). RETRACTING a
    --no-wait (once logged, every relaunch counts as a resume): stop TKX's
    supervisor and trainer, move TKX's run folder and its two probe jsonls
    aside, relaunch.
  - A-L6: the supervisor's stdout cannot die on a non-ASCII character.
  - A-L7: tests: a failing manifest blocks the launch; a flag or marker
    appearing during an attempt's waits stops it; tests never hang.
  - B-L1-L3, B-L5, B-L6: the hand-off's start position counts whole lines;
    "all done -- N rows missing" when rows are missing; the abandon pass
    logs the rows newer than the checkpoint (drop their probe rows before
    any un-abandon, then relaunch with launch_rgrowx.ps1); a supervisor
    refusal is an ALERT; interim probes need 8 GB of free commit beside
    the trainer (the final pass 4).
  - The NO GPU SMOKE core-file diff is re-run at the manifest commit.
AMENDMENTS (audit r4, before launch: A and B NOTHING ABOVE LOW -- the loop
closes; its LOWs fixed, then a confirmation round):
  - A-L1: RETRACTING a --no-wait = the FRESH RESTART procedure (r3 steps
    1-4), and a launch without a checkpoint re-checks the stray snapshots /
    probe rows right before it starts.
  - A-L2: the "nothing drives RG" ALERT says which: no live RG supervisor
    (rerun launch_rgrow.ps1) or a live but silent one (check it; do NOT
    rerun launch_rgrow.ps1 beside it -- RG's launcher is not idempotent).
  - A-L3 / A-L4: rows also stream between preflight retries; the RG
    supervisor check counts python processes only.
  - A-L5: the --no-wait procedure also stops RG's hand-off's probes; RG's
    missing evals then run after TKX completes (or beside it with Kai's
    approval).
  - B-L1-L9 (hand-off): a probe jsonl that cannot be read (anything but
    "not found") raises -- the loop ALERTs and retries -- so a transient
    lock never reads as vanished or missing rows; a vanish during the final
    pass ends "stopped", not "all done"; interim probes need 6 GB of free
    commit (the 4-GB ALERT line + a probe + margin; the final pass 4); the
    run's own jsonl is read only when some missing row could settle;
    supervisor.err echoes start at a whole line; a PREFLIGHT failure line
    is an ALERT; ABANDONED while this run's trainer lives is an ALERT and a
    wait; `launch_rgrowx.ps1 -NoWait` beside a running supervisor says it
    is ignored.
  - Confirmation round (r5): nothing above LOW; its LOW (a locked probe
    jsonl at the hand-off's start would kill it outside the retry loop)
    fixed in the hand-off. The audited repo code is the manifest commit
    below.
AUDIT + LAUNCH: five rounds (r1-r3 each one MEDIUM, fixed; r4 nothing above
LOW -- the loop closed; r5 confirmed r4's LOW fixes, nothing above LOW; a
last check of r5's one-line LOW fix, nothing above LOW). Manifest dd5a7ce2
(40 AUDITED files); the core files are byte-identical to TK-90's a81bcf39
(`git diff --stat a81bcf39 dd5a7ce2 -- <core>` empty). Run worktree
Y:\SwitchPlayer-rgrowx-run, detached at dd5a7ce2. LAUNCHED 2026-10-20 15:24
(WMI, scratchpad launch_rgrowx.ps1): the hand-off post_rgrowx.py, then
`supervise_rgrowx.py tkx 0`, WAITING for RG's marker and its 36 evals
(expected ~00:30-01:00 2026-10-21).

### Next experiment (pre-registered) -- P2a on Ms. Pac-Man, with reward-clip growth (Kai, 2026-10-19)

DECISION (Kai, 2026-10-19): after P2a's Pong NO BREAKOUT, "try the exact
same setup on ms pacman", with reward clipping added first ("reward
clipping will be needed for ms pacman"; his rule of 10-17: "clip heavily
in the beginning and over time decrease the amount clipped").
QUESTIONS: (A) does reward-clip growth lift the real-time algorithm on Ms.
Pac-Man over its sign-clipped twin? (B) on top of it, do duration tokens
hold up in a game of held directions, where Pong's aim-the-return problem
does not arise?
TWO ARMS, SEED 0, run ONE AFTER THE OTHER -- TK first (Kai's question),
then RG (amended before launch, audit r1: see AMENDMENTS):
  RG = effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow
    = the Ms. Pac-Man arm + REWARD_GROWTH.
  TK = effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow
    = RG + TOKEN_VOCAB (Ms. Pac-Man's 9 actions x {4, 8, 16, 32} frames =
    36 tokens) + P2a's long-token bias (b 0.5 -> 0 by 25% of training).
  TWIN = mspacman-s0-2026-10-16 (the same arm without either; deployed
    mean 60-90 649.14, g 0.351; raw 585.62; row 90 at 53.4k env steps).
RG vs TWIN isolates the clip growth; TK vs RG isolates the tokens (the
same 9 actions -- unlike Pong, no 6 -> 3 confound: the twin already acts
over Ms. Pac-Man's minimal set); TK vs TWIN is Kai's question. Same
recipe as the twin: MsPacmanNoFrameskip-v4 --frameskip 4, 90 x 2000
train steps, 3 episodes (one game's lives) per iteration, --episode-cap
3000 (per life, in base steps), eval every 5, --checkpoint-every 2,
--snapshot-weights, --ray --cpu-actor.
REWARD-CLIP GROWTH (code, default OFF everywhere):
  - The actor records RAW rewards, one per base step (a token keeps its
    inner steps' rewards); the learner clips them. Clipped reward per base
    step = sign(r) x min(|r| / UNIT, c), UNIT = 10 (a pellet), summed over
    a token's inner steps -- a token's reward is what the base player
    would have collected over the same frames. (P2a's Pong tokens clipped
    the token's sum; on Pong the two agree, as points never come twice
    within 32 frames. On Ms. Pac-Man clipping the sum would pay a token
    that eats 4 pellets the same as one that eats 1.)
  - Ceiling c: log-linear from 1 at step 0 to 160 at 50% of training,
    c = 160 ** min(1, t / 90,000), stepped per iteration (from the
    iteration's starting step): ~2.8 at row 10, ~12.0 at row 23, ~143 at
    row 45, 160 from row 46 on (row 50 the first logged at 160). At c = 1
    every Ms. Pac-Man reward is 1 -- exactly the
    twin's sign clip. At 160 every reward of the first boards passes in
    pellet units: pellet 1, power pellet 5, ghosts 20 / 40 / 80 / 160,
    cherry 10, strawberry 20.
  - The whole replay is re-clipped at each iteration's ceiling (every
    stored record keeps its raw rewards), so the value and value-prefix
    targets, insertion priorities, the communal refresh's chains and the
    diagnostics see one ceiling at a time; Ms. Pac-Man's buffer spans ~50
    iterations and never mixes ceilings. Stored priorities are not
    recomputed at a ceiling change (training writebacks and the refresh
    rotation update them).
  - Not normalized by c: pellets stay 1 throughout, so the value scale
    grows from the twin's (tens) to hundreds. The categorical supports
    already cover it -- built at import for raw ceilings 400 / 2000 / 5000
    (reward / value prefix / value); RG's network is the twin's (TK's has
    36 action slots). The heads the paper arms train are the value prefix
    (per-step reward head unused, REWARD_LOSS_WEIGHT 0) and the value: a
    5-record prefix past 2000 saturates -- only reachable when tokens
    stack all four ghosts and fruit; rows log the largest stored record
    reward (reward_max). The gate refuses a ceiling whose 5-step prefix
    exceeds the prefix support, and a game whose reward unit is not
    REWARD_UNIT (REWARD_UNITS: Ms. Pac-Man 10).
  - Rows also log reward_ceiling and reward_kept (clipped / raw-unit
    reward mass over the buffer).
  - REGRESSION (pinned in the tests): RG with the ceiling held at 1
    reproduces the twin's arm bit for bit (actor records and a train +
    refresh slice).
TOKENS (TK): P2a's mechanics unchanged (frame-time discount and value
targets, base-step accounting of env_steps / the episode cap / the replay
cap / the BN window, the bias, prior-argmax eval) over 36 tokens. A life
loss ends a token early (EpisodicLifeEnv), counted at its nominal length
in the replay units as in P2a.
READS. EXACT eval as the twin's: whole games (every life), 30 noop
starts, prior argmax; rows 5-90; DEPLOYED (acting states) primary, raw
secondary; g = (score - 307.3) / 973.9.
  (1) level: each arm's deployed mean 60-90 against the twin's 649.14,
      with bands from the twin's own row spread (SD 216.9 over rows
      60-90; 2 SE of a difference of two 7-row means = 232): BETTER >=
      881.0 (g 0.59), WORSE <= 417.3 (g 0.11), else NO DIFFERENT. TK is
      also read against RG with the same +-232. Raw likewise (twin
      585.62; SD 129.3, +-138) as secondary.
  (2) data: score against env steps (base steps) beside the twin.
  (3) clip growth, descriptive: reward_ceiling, reward_kept, reward_max,
      the row value scale. The rows' value / value_control / prefix /
      step-loss fields are in clipped-reward units, which grow with the
      ceiling: not comparable with the twin's rows, nor across rows.
  (4) tokens, descriptive (P2a read 4): per-duration play and search
      shares, frames per decision, decisions per game (from the probe:
      the rows' tok_decisions_per_game counts per TRAJECTORY = per life on
      Ms. Pac-Man), payoff (in clipped units at the insertion ceiling
      against a birth value from nets trained on smaller ceilings: biased
      up while the ceiling grows -- compare tokens within a row, not
      across rows), the deployed player's decisions per second against
      the twin's 15.
  (5) stability: the largest drop between adjacent rows from row 45 (the
      twin's -443).
DECISION AFTER (amended before launch, audit r1 B-M3/M4). The bands are a
WITHIN-RUN noise floor from one seed (rows are autocorrelated, lag-1 r ~
0.27, which widens the SE ~1.3x; seed spread is larger -- archfix 1-in-4,
Gumbel 1/2): a single-seed BETTER / WORSE reads "consistent with", and a
permanent change to the base needs a seed-1 replicate first.
  NEAR RANDOM, per arm, first (audit r2 B-M1): an arm whose deployed
    mean 60-90 has g < 0.2 (< 502.08) is diagnosed before any other read
    of it.
  TOKENS (TK, Kai's question): HOLD UP if TK has g >= 0.2 AND is not
    WORSE than the twin AND not WORSE than RG -> P2b (the composition
    loss) on the TK arm (if RG is WORSE, the base P2b runs on is Kai's
    call). FAIL if TK is WORSE than the twin -> tokens fail in both games
    -> the composition diagnosis first (does the model value X32 above
    X4 x 8, the same frames?). MIXED (TK not WORSE than the twin but WORSE
    than RG) -> Kai's call: tokens cost something here that growth alone
    does not. INCONCLUSIVE (TK near random -- g < 0.2 -- but neither WORSE
    than the twin nor than RG; audit r3 L1) -> the composition diagnosis,
    then Kai's call; no P2b. (Every FAIL is also near random -- 417.3 <
    502.08 -- and its diagnosis IS the composition diagnosis.) An
    ABANDONED TK (stopped under the pace rule) is read descriptively from
    the rows it has; no TOKENS verdict.
  GROWTH (RG): BETTER than the twin -> a seed-1 replicate, then it joins
    the Ms. Pac-Man base. NO DIFFERENT -> Kai's call (keep it for its
    intent, or retune the end ceiling / schedule). WORSE -> diagnose (the
    value scale) before any replicate; TK is then judged against the twin
    only.
COST: one run after the other on the one GPU: TK ~12-17 h at Pong-like
12-24 frames per decision, up to ~25-30 h if play drifts to 4-frame tokens
or lives lengthen (the 36-token refresh, ~0.21 s per position, scales with
decisions per life: 0.8-1.6 min per iteration at the twin's ~230 base
steps per life, ~5-8 min at 4 frames or ~400 steps), then RG ~17 h (the
twin's 17.2 h), + ~1 h of evals (18 rows x 2 variants x 2 runs; TK's run
at 2 CPU probes x 2 threads while RG preflights and starts -- a small CPU
load beside RG's actor). PACE RULE: a row past 25 min/iteration is an
ALERT in the supervisor and Kai's call (continue, or stop TK and start RG
with the logged `rg 0 --no-wait` override).
AMENDMENTS (audit r1, before launch):
  - Sequential, not concurrent: two concurrent Ray runs took host commit
    to ~1 GB free in the GPU smoke (each Ray instance prestarts ~16
    workers; no earlier pair of full-length concurrent --ray runs), and
    the twin trained ALONE -- the eager collect's timing (when a game joins
    the buffer mid-block) depends on the actor's pace, so solo runs match
    the twin's regime. rg's supervisor waits for tk's completion marker.
  - TK's compute knobs (memory; the algorithm unchanged): both vectorized
    engines hold A latents per expansion, so 36-slot trees at the base's
    shapes overflow the 12 GB GPU (measured: birth search at 256 roots per
    chunk peaked 11.4 GB; the communal refresh at 8 trajectories per forest
    ran out of memory, at 2 reserved 8.9 GB). TK runs birth search in
    chunks of 64 roots (VTREE_REFRESH_BATCH 64: 3.0 GB) and the refresh one
    trajectory per forest (VTREE_COMMUNAL_TRAJS 1, the sequential engine's
    exact batch shapes: 3.8 GB, ~0.21 s per position -- ~1-2 min per
    iteration at Ms. Pac-Man's life lengths). RG keeps the base's 256 / 8.
    Batch shapes change float numerics only (G = 8 vs 1: ~4% of roots
    flip visit counts, DESIGN.md), not the search.
  - Insertion priorities are not recomputed at a ceiling change (above);
    training writebacks (~512k per iteration over ~31k positions) re-score
    most positions within an iteration. A small PER tilt the twin lacks.
  - The twin ran code b9e3af68 (Ms. Pac-Man manifest f07d1d44). The branch
    code reproduces its iter-90 exact eval exactly (acting 642.33, raw
    837.0; probe from the curriculum tree on a copy of the snapshot); the
    training path at ceiling 1 is pinned by the regression tests.
  - Kai's Switch-control sessions kill GPU pythons: pause the running
    supervisor during them (supervisor v2 pattern), resume after.
AMENDMENTS (audit r2, before launch): the per-arm near-random floor above;
the COST range and the pace rule; supervise_rgrow.py ALERTs on a row whose
reward_ceiling is off the schedule (the iteration-top re-clip did not run),
on a pace past 25 min/iteration, and on a stored record reward past 400
units; `rg 0 --no-wait` (logged) starts RG without TK's completion marker
if TK is abandoned -- refused while TK's supervisor is alive. PROCEDURE:
kill TK's supervisor, TK's trainer AND the waiting `rg 0` supervisor (by
command line), then launch `rg 0 --no-wait`; the hand-off mutes TK and
drops its evals only if TK is still unfinished once RG's evals are done
(audit r4, nothing above LOW; its LOWs: this procedure, and the hand-off
keeping TK's evals after a retracted override). RECOVERY: after a
reboot, relaunch the hand-off and the supervisors the same way (they
resume from the last checkpoint; a finished run's supervisor exits at
once; rg re-waits) -- but if TK was abandoned, relaunch only rg, with
--no-wait. To pause, kill the running supervisor and its trainer by
command line (never taskkill /T) and relaunch later. (Audit r3: nothing
above LOW; its LOWs fixed -- the INCONCLUSIVE case, the abandoned-TK path
in the supervisor / hand-off / RECOVERY, a finished run's supervisor
exiting before its preflight, the hand-off echoing supervisor lines while
its probes run.)
AUDIT + LAUNCH: four rounds. r1 -- A nothing above LOW; B four MEDIUM
(host commit with two concurrent Ray runs -> sequential; TK's GPU memory
unmeasured -> measured, TK's batch knobs; the decision rules' twin floor;
the bands' seed variance) -> fcb2beed. r2 -- A nothing above LOW; B one
MEDIUM (a per-arm near-random floor) -> 5965c105. r3 and r4 -- nothing
above LOW (LOWs -> b5ad5128, 79d0edd5 and the hand-off). Manifest
a81bcf39 (38 AUDITED files of 79d0edd5). The runs execute from the run
worktree Y:\SwitchPlayer-rgrow-run, detached at a81bcf39; GPU smoke from it
(tk then rg, each killed mid-iteration + resumed, probes raw + acting) OK.
LAUNCHED 2026-10-19 15:57 (WMI, scratchpad launch_rgrow.ps1): the hand-off
post_rgrow.py, then `supervise_rgrow.py tk 0` (preflight ok, attempt 1
fresh at 16:03) and `supervise_rgrow.py rg 0` (waiting for tk's marker).
TK DONE 2026-10-20 07:35 (attempt 1, no deaths, 15.4 h, 10.3-11.1 min per
iteration; row 90 at 58.5k env steps against the twin's 53.4k); its 18 x 2
evals done 07:42 (probe jsonls committed). INTERIM, read (1) against the
twin only: deployed mean 60-90 = 856.62 (g 0.564) against 649.14, +207 --
NO DIFFERENT (BETTER would be >= 881.0); raw 830.81 against 585.62, +245
-- BETTER on the secondary read (band +-138). Not near random. The TOKENS
verdict waits for RG (TK is WORSE than RG only if RG's deployed mean is
>= 1088.6). RG started 07:35 (preflight ok, attempt 1 fresh).

### Next experiment (pre-registered) -- P2a: duration tokens on Pong (Kai, 2026-10-17)

DECISION (Kai, 2026-10-17): after Ms. Pac-Man, "go with my list" -- his
curriculum algorithm, built on branch curriculum-p2 (decisions in P2
above). Trial and error, simplest defaults first.
QUESTION: does the real-time base learn Pong when every decision is a
TOKEN -- NOOP, UP or DOWN held for 4, 8, 16 or 32 frames -- and which
tokens does it come to use?
ARM effzero-repro-faithful-prioract-noredo-temp05-bnrecal-tokens = the
real-time base (temperature + BN stabilizer, 2/2 breakouts) +
TOKEN_VOCAB (the 12 tokens) + the early long-token bias. SEED 1, so the
twin is the SAME arm at the same seed without tokens (and on Pong's
six ALE actions -- see NOTES):
pong-bnrecal-s1-2026-10-08 (deployed first success @75, raw @70; deployed
mean 60-90 +5.80, raw +5.47). Same recipe: Pong-v4, --frameskip 4,
90 x 2000 train steps, 1 game per iteration, eval every 5, --ray
--cpu-actor, --snapshot-weights; the episode cap is 3000 x 4 = 12000
frames (the base's game-length cap in time).
MECHANICS (code, default OFF everywhere):
  - A token (ALE action, F frames) plays F/4 consecutive steps of the
    paper wrapper stack (MaxAndSkip 4) with that action; its reward is
    the sum of the inner rewards, then the base's sign clip. The frame
    stack is pushed at EVERY inner step, so the observation is always
    the last 16 raw frames at 4-frame spacing, whatever tokens were
    played.
  - Discount per raw frame: an edge into a token of F frames discounts
    by DISCOUNT^(F/4) (DISCOUNT = the base's 0.997^4 per 4 frames) -- in
    the birth search and the communal refresh (per node, by its in-edge
    token) and in the value targets.
  - Value targets in FRAME time: real rewards up to the first position
    at least 4 x horizon frames ahead (horizon = TD_STEPS 5 under the
    4.3 age shrink: 20 frames at age 0), then the target-net bootstrap
    there; priorities likewise.
  - Unroll K = 5 tokens; the value-prefix target is unchanged (an
    undiscounted sum); policy and dynamics see 12 slots (plain per-slot
    embedding, as the base's 6 actions).
  - Acting: prior sampling over the 12 tokens at the base's staged
    temperature x 0.5, with a long-token bias: weights x (F/4)^b, b
    falling linearly from 0.5 at step 0 to 0 at 25% of training. Eval
    and deployment: the prior's argmax token.
  - env_steps = frames / 4 (the base's decision-equivalents, so rows
    share the base's x-axis); cumulative decisions are logged beside it
    (tok_decisions_total, checkpointed). The episode cap (3000 base
    steps) is checked between decisions, so the last token can run past
    it by up to 7 base steps.
  - The replay transition cap (31,250) counts BASE steps under tokens, so
    the buffer spans the same game time as the twin's (~12-30 games)
    whatever tokens are played -- counted in records, ~200-decision token
    games would have made MAX_REPLAY 85 the binding cap and the buffer
    ~3-7x the twin's tiny one (audit r1). Rows log replay_trajs /
    replay_records / replay_base_steps. Refresh keeps the twin's 4
    trajectories per 2 iterations, so the per-trajectory revisit rate
    matches; with fewer decisions per game each gradient step's samples
    reuse fewer distinct positions -- intrinsic to tokens, reported.
  - The BN stabilizer's calibration window (BN_RECAL_FRAMES 4096) also
    counts BASE steps under tokens: the newest trajectories totalling
    >= 4096 base steps, all of their decision states (~1k at ~4 base
    steps per decision) -- the twin's recency in game time, rather than
    4096 decisions spanning ~4x the game time (audit r2).
  - REGRESSION (pinned in the tests): TOKEN_VOCAB = the base's six ALE
    actions x 4 frames reproduces the base arm bit for bit.
READS. Exact eval as the line's: 30 noop starts, prior argmax (now over
tokens), rows 45-90; DEPLOYED (acting states) primary, raw secondary.
  (1) breakout: first success (exact >= +10 or >= 3/4 phases);
  (2) level: deployed mean 60-90 against the bands (NO WORSE >= +3.68,
      WORSE < -2.32) and the twin's +5.80;
  (3) game time: score against cumulative frames (env_steps) beside the
      twin, so a gain is not mistaken for more data;
  (4) tokens, descriptive: per row the acting share of each duration,
      mean frames per decision, decisions per game, the birth search's
      visit share per duration, and each token's payoff (mean z - V over
      the positions where it was played); for the deployed player,
      decisions per second (60 / the probe's frames per decision,
      pooled over the row's games) -- the
      real-time quantity;
  (5) stability: rows below 0 after the first success.
NOTES (audit r1, before launch): the row policy_entropy is over 12 tokens
(max ln 12 = 2.48 against the twin's ln 6 = 1.79), so H(pi) values are not
comparable across the pair; a token's payoff (read 4) uses the 20-frame
n-step horizon, which for a long token is its own duration -- descriptive,
not like-for-like across durations. The twin ran code 9a7f3388 (branch
bnrecal); the plain Pong path's equivalence since then is pinned by the
Ms. Pac-Man real-Pong actor regression and P2a's base-token regressions
(TOKEN_VOCAB = six ALE actions x 4 frames reproduces the plain arm bit for
bit: the real-Pong actor and a whole train + refresh slice). The
twin acted over Pong-v4's six ALE actions (FIRE / RIGHTFIRE / LEFTFIRE
duplicate NOOP / UP / DOWN, splitting its priors and birth-search visits
across duplicate children); this arm's tokens are built on the minimal
three, a second difference from the twin that this run does not separate
(a minimal-actions x 4-frames control would). A token cut short by a
terminal is counted at its nominal length in replay_units /
tok_frames_per_decision / replay_base_steps (at most 7 base steps per
game; env_steps counts the steps actually played).
DECISION AFTER:
  HOLDS UP -- a first success by row 75 (the twin's deployed one) AND
    deployed mean 60-90 NO WORSE: tokens hold up -> P2b (the composition
    loss) on this arm, then P3.
  PARTIAL -- a first success, but later than row 75 or a mean below
    +3.68 -> Kai's call (P2b may repair long/short disagreement; or a
    shorter menu).
  NO BREAKOUT -> diagnose before P2b: token usage (stuck on long
    tokens?), the frame-time targets, the bias.
COST: ~20 h (the base's ~13.5 min/iteration; 12 search children ~2x the
base's per expansion, on fewer decisions per game) + ~30 min evals.
AUDIT + LAUNCH: four audit rounds. r1: HIGH (records stored the
post-token frame stack) + MEDIUM (replay cap in records) -> 1f21a464;
r2: MEDIUM (BN-stabilizer window in decisions) -> 7014e1b1; r3 and r4:
nothing above LOW in the code (r3's launch-plan HIGH -- bare `python`
under WMI has no psutil -- answered by launching with the venv
interpreter; LOW test gaps -> 3494b7e7). Manifest 6a66298a (34 AUDITED
files of 3494b7e7). The run executes from the run worktree
Y:\SwitchPlayer-p2a-run, detached at 6a66298a (development continues on
curriculum-p2 elsewhere); GPU smoke from it (kill mid-iteration +
resume, probe raw + acting) OK. LAUNCHED 2026-10-18 00:18 (the folder
keeps the pre-registration date, pong-tokens-s1-2026-10-17): preflight
ok (manifest + quick probe + 13 suites), attempt 1 fresh; eval hand-off
scratchpad post_tokens.py (rows 45-90, acting + raw; logs "all done").
Deferred, test-only: pin the probe's pooled frames_per_decision exactly
(r4 NIT-1).

### RESULT -- P2a: duration tokens on Pong (seed 1), 2026-10-18

pong-tokens-s1-2026-10-17 trained the tokens arm from the run worktree
(detached at 6a66298a) 10-18 00:18 -> 15:01 in one attempt (52,950 s,
10.0 min/iteration against the estimated ~13.5 -- ~490 decisions per game
instead of the twin's ~2,000; no deaths, supervisor.err empty). The
hand-off ran the evals 15:01-15:14: code check ok (tokens manifest), all
10 rows in both variants. EXACT eval, 30 noop starts, prior argmax over the
12 tokens; DEPLOYED = acting states (primary), RAW secondary; "won" =
phases won of 4; "capped" = games of 30 that hit the 3000-step cap; dps =
the deployed player's decisions per second (60 / pooled frames per
decision); TWIN = pong-bnrecal-s1-2026-10-08 deployed:
  iter       45     50     55     60     65     70     75     80     85     90
  deployed -17.0  -10.2  -10.3  -15.4  -11.8  -16.4   +3.7   -7.3   -2.9   +4.0
    won     0/4    0/4    0/4    0/4    0/4    0/4    2/4    1/4    1/4    2/4
    capped    6     24     18     24     19      9     22     22     23     20
    dps     3.0    3.0    2.8    2.7    2.6    2.6    2.3    2.5    2.6    2.5
  raw      -16.1  -11.3  -12.2  -14.9  -14.6  -13.9   -7.3  -10.9   -6.1   -1.9
    won     0/4    0/4    0/4    0/4    0/4    0/4    1/4    0/4    1/4    1/4
    capped    1     20     21      7     18      8     22     21     26     27
  twin     -14.3  -13.6  -13.1  -13.1   -1.0   -3.8  +10.1   +9.5  +20.4  +18.4
    capped    0      0      2      7     23      2      0      0      0      0
  env k    65.4   78.5   92.2  106.0  119.8  133.8  146.7  160.7  175.4  189.9
  twin k   68.0   79.8   92.1  104.2  117.2  130.9  145.2  158.3  168.9  179.3
In-run eval (8 games): -17.0 @45, -10.5 @50, -12.4 @55, -16.4 @60, -10.6
@65, -14.8 @70, -0.6 @75, -8.8 @80, -6.9 @85, +3.9 @90.
READ (1) breakout: NONE. No row, deployed or raw, reached exact >= +10 or
3/4 phases; the best deployed rows are +3.7 @75 and +4.0 @90 (2/4 each).
The twin's deployed first success was @75 (+10.1, 3/4).
READ (2) level: deployed mean 60-90 = -6.59 -> WORSE (< -2.32; the twin
+5.80). Raw -9.95 (the twin +5.47).
READ (3) game time: matched -- 146.7k env steps @75 against the twin's
145.2k, 189.9k @90 against 179.3k. Not a data shortfall: at the same game
time the twin had broken out.
READ (4) tokens:
  - The 32-frame share of play rose from 0.20-0.33 (rows 5-25) to 0.51
    @45 and 0.64 @90, most of it AFTER the long-token bias reached 0
    (25% of training, row ~22), so the bias did not drive the late drift
    (it may have seeded it). The birth search's 32-frame visit share sat at
    0.44-0.51 from row 35 on; from row 60 the acting share ran 0.13-0.19
    above it (the staged temperature 0.25 -> 0.125 sharpens the prior's
    favourites). 8-frame tokens were nearly unused (2-5% of play from row 30, 6-13%
    of search). Window counts @90: NOOP32 553, DOWN32 550, UP32 468, NOOP4
    304, DOWN16 210, every other token <= 94.
  - Frames per decision (acting) 11.9 @10 -> 23.6 @90; deployed 20.0-25.8,
    i.e. 2.3-3.0 decisions per second against the twin's 15 (5-6x fewer --
    the real-time quantity). Decisions per game 390-520 (rows 45-90); 38,199
    decisions in all.
  - Payoff (mean z - V where played): small (|.| <= 0.21 over all rows)
    with no consistent sign by duration; each row's most negative token is
    a 4-, 8- or 16-frame one, never a 32-frame one (UP8 @75 / @80 / @90:
    -0.19 / -0.18 / -0.16). policy_entropy 1.0-1.5 over rows 45-90 (over
    12 tokens; ln 12 = 2.48).
READ (5) stability: no first success. Descriptively the deployed player
fell from +3.7 @75 to -7.3 @80.
GAME SHAPE (descriptive): the deployed games at 75-90 run 2,740-2,970 of the
3,000-step cap and 20-23 of 30 hit it. Finished games went all one way per
row -- all won @75, @85, @90 (8, 7, 10 games), all lost @80 (8) -- and
most capped games were behind (15 of 20 @90). The twin's games at 75-90
never hit the cap (1,710-2,240 steps). The token player keeps rallies
going but rarely gets the ball past the CPU paddle; whether 32-frame holds
(0.53 s) are too coarse to aim returns is a hypothesis, not tested here.
PRE-REGISTERED DECISION: NO BREAKOUT -> diagnose before P2b: token usage
(stuck on long tokens? -- the play shares say increasingly so, with the
bias gone), the frame-time targets, the bias (ended at row ~22). The 6 -> 3
action change stays unseparated. Kai picks the diagnosis.

### RESULT -- the real-time algorithm on Ms. Pac-Man (seed 0), 2026-10-17

mspacman-s0-2026-10-16 trained the temperature + stabilizer arm
(effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman) from
10-16 08:55 to 10-17 02:09 in one attempt (62,011 s, 11.9 min/iteration, no
deaths, supervisor.err empty). The hand-off ran the evals 02:10-02:25: code
check ok, all 18 rows in both variants. EXACT eval: whole games, 30 noop
starts, prior argmax. ACTING = the deployed policy (primary); RAW is
secondary; IN-RUN = the run's own 8-game eval:
  iter      5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  acting  462 1109  453  606  495  684  490  706  588  643  508  781  338  361  730  816  875  642
  raw     459  499  482  374  406  298  416  479  592  576  878  478  479  515  520  627  643  837
  in-run  534 1115  325  385  539  668  424  800  575  650  580  975  395  481  766  851  889  685
  env k   3.0  6.4  9.0 12.1 15.3 18.2 20.8 23.7 26.3 29.7 32.9 35.8 38.9 41.8 44.5 47.3 50.0 53.4
Each row is 4 distinct games, one per noops mod 4: every row has exactly 4
distinct scores, as the pre-registration expected.
READ (1) learning: acting mean 60-90 = 649.14 (2.11x random). The acting
row means by thirds are 625 (5-25), 603 (30-55) and 649 (60-90); without
the @10 outlier (1109) rows 5-25 are 504. Raw by thirds: 444 / 543 / 586.
Most of the gain over random came in the first ~10 iterations (~6k env
steps); after that the level rose little.
READ (2) level: g = (649.14 - 307.3) / 973.9 = 0.351 -> PART-WAY
(0.2 <= g < 0.5; HOLDS UP needed a mean >= 794.25). Raw 585.62 -> g =
0.286, also PART-WAY. Pong's same line: g = 0.57-0.75.
READ (3) data: no row reached 100k env steps. Each iteration collected ~600
env steps (one game), so the run ended at 53.4k: half the Atari-100k
budget, with 180k gradient steps (3.4 per env step, Pong's ratio ~1). Per
the pre-registration this read falls back to row 90: 642.3 (g 0.34) at
53.4k env steps, against EfficientZero 1281.2 / SPR 1318.4 at 100k --
descriptive; this run had half the references' env steps.
READ (4) stability: the largest acting drop from row 45 on is -443 (781 @60
-> 338 @65; raw -399, 878 @55 -> 478 @60). No capped game in any row. Game
length 450-750 decisions. In-run 60-90 mean 720 vs exact 649: the 8-game
in-run eval reads ~70 higher. Deployed acting states vs raw: +63 over
60-90 on average, but mixed row by row (4 rows higher, 3 lower).
PRE-REGISTERED DECISION: PART-WAY -> Kai's call: a second seed, or a look at
the port first. Beyond the letter: the curve flattened after ~10
iterations, and the run saw only half the data the references use at 3.4x
Pong's gradient steps per env step. The port's train/env-step ratio is the
first suspect on the pre-registered diagnose list. A second seed would
measure seed variance in a recipe whose data budget is itself in question,
so the port deserves a look first. The freeze is lifted (the evals are
done).

### RESULT -- Gumbel search acting, seed 1 + THE TWO-SEED DECISION, 2026-10-16

pong-gumbelact8-s1-2026-10-15 ran 10-15 12:23 -> 10-16 08:12 in one attempt
(71,340 s, no deaths). Evals by the hand-off 08:13-08:33 (code check ok:
audited manifest + recal probe). EXACT prior argmax (raw) / external recal
(calibration pong-prioract-s0 @85), 30 games per row; "capped" = games that
hit the 3000-step cap:
  iter       45     50     55     60     65     70     75     80     85     90
  raw      -15.4  -16.3  -16.9   -8.6   -6.3   -8.3   -7.7   -8.2   -7.0   -3.7
    won     0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    1/4    1/4
    capped    0      0      0     23     22     15     16     15      9     30
  recal                          -8.2   -4.1   -0.8  -10.8   -4.2  -10.2   -3.6
    won                           0/4    0/4    2/4    0/4    1/4    0/4    1/4
  env steps 68.3k  79.8k  90.6k 101.1k 112.9k 123.3k 135.6k 148.5k 161.9k 173.6k
In-run eval (prior argmax, 8 games): -16.0 @45, -17.9 @55, -7.8 @60, -7.5
@70, -7.8 @80, -3.2 @90 (8/8 games at the cap).
READ (1) breakout: NONE -- no row 45-90 at exact >= +10 or >= 3/4 phases
(best raw -3.73 @90, 1/4 phases; best recal -0.77 @70, 2/4).
READ (2) level: raw mean 60-90 = -7.1143 -> WORSE band (< -2.32); recal
-5.9857. Same-seed temperature reference (bnrecal s1) raw +5.47.
READ (3) late stability: UNTESTED (no first success).
READ (4) data: no first success; 173.6k env steps @90 (seed 0: 196.3k).
SHAPE (descriptive): at @60 the policy learned to defend -- most games run to
the 3000-step cap with few points either way -- and it stayed on that
plateau to @90 (30/30 games capped) without learning to score: the first
won games are @85 (8 of 30) and @90 (3 of 30). Seed 0 (20-22 of 30 games
capped @55-65) and bnrecal s1 (16-23 @60-65) passed through the same
plateau and broke out of it @70; seed 1 had not left it by @90.
TWO-SEED DECISION (pre-registered): NO BREAKOUT -- Gumbel-8 acting 1/2; the
temperature + stabilizer config (2/2) stays the base of the real-time line.
Scored by hand; it matches the Ms. Pac-Man supervisor's DECISION line
(10-16 08:34, "NO BREAKOUT -- no first success at rows 45-90"), so the
Ms. Pac-Man run trains effzero-repro-faithful-prioract-noredo-temp05-bnrecal-
mspacman.
Descriptive only, not a decision read: the two-seed raw mean 60-90 is
(+9.74 - 7.11) / 2 = +1.31, against the temperature config's +4.08. Seed 1
stays the hard seed for policy-only acting: 4 of the 5 configs run on it
never broke out (prioract2, seqref, SVE, Gumbel-8); only the temperature +
stabilizer config did.
MS. PAC-MAN LAUNCH: the merge watcher fast-forwarded main to the audited
f07d1d44 at 08:34. Seed 1 finished earlier than forecast, so the pre-launch
Ray GPU smoke ran after the merge: the supervisor was stopped at 08:44,
before its preflight finished. The smoke ran both -mspacman arms at seed 0
(2 iterations x 30 train steps, from the worktree). The temp arm was killed
mid-iteration 2 and resumed. The eval probe (PROBE_MAX_NOOP=2) then ran on
each iter-2 snapshot, raw, plus acting states for temp: whole 3-life games
of 372-540 decisions, each ending at real game over. Everything passed and
no process was left over. The supervisor was relaunched from main's copy
(same args, auto 0) at 08:51: it gave the same decision, "preflight ok
(manifest + quick probe + 15 suites)", and "LAUNCHED s0 temp attempt 1
(fresh)" at 08:55, with exactly the pre-registered command line. The
freeze in the Ms. Pac-Man pre-registration (below) applies until its evals
finish.

### Next experiment (pre-registered) -- the real-time algorithm on Ms. Pac-Man (Kai, 2026-10-16)

DECISION (Kai, 2026-10-16): "I'd like to test the current algorithm on ms
pacman when the current code is done running" (10-15: "... start moving our
working algorithm to a different problem to solve to see if it holds up on
something like pac man").
QUESTION: does the real-time line's algorithm -- Pong's recipe, with only
the env and the unit conversion below changed -- learn Ms. Pac-Man, and how
close does it get to the published Atari-100k results?
ARM = the base that the Gumbel seed-1 decision (below, 10-15) leaves as the
real-time line's training configuration:
  REPLICATES  -> effzero-repro-faithful-gumbelact8-birth-mspacman
                 (Gumbel-8 acting, 50-sim birth targets);
  NO BREAKOUT -> effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman
                 (temperature + BN stabilizer, prior sampling);
  PARTIAL     -> Kai's call between the two (nothing launches until then).
The supervisor applies this decision from seed 1's readout files ("auto":
first success, mean 60-90, collapses with the recal classification, first
success <= 80), logs it, and launches. I re-check it by hand from the same
files and the hand-off's code-check line (post_gumbel_s1.log); if they
disagree, the run is stopped and Kai decides.
SEED 0, one run, dir benchmarks/results/mspacman-s0-2026-10-16.
ENV: MsPacmanNoFrameskip-v4 (EfficientZero's env id; v4 = no sticky actions,
the 9-action minimal set), through the lineage's wrapper stack unchanged:
NoopReset(30) -> MaxAndSkip(4, flicker max-pool) -> EpisodicLife for
training. Rewards clipped to +-1 as in Pong (REWARD_CLIP 1.0 = EfficientZero's
sign clip: every Ms. Pac-Man reward -- 10 / 50 / 200-1600 / fruit -- is +1);
discount 0.997^4, 84x84 RGB, 4-frame stack, network, search, refresh engine,
buffer (31,250 transitions), schedule (90 iterations x 2000 train steps,
eval every 5, checkpoint every 2, staged temperature) all unchanged; --ray
--cpu-actor; --episode-cap 3000 (per life).
UNIT CONVERSION (the only arm change): EpisodicLife ends each training
episode at a life loss (EfficientZero's episode_life), so a Ms. Pac-Man
trajectory is one LIFE and a game is 3; in Pong a trajectory was a game.
Knobs counted in trajectories are kept per GAME (x3): --episodes-per-iter 3
(one game's lives per iteration, as Pong's one game; the actor's env carries
the match on across them, one respawn no-op per continued life),
MAX_REPLAY 85 -> 255 (keeps the 31,250-transition cap the binding one, as
it always was in Pong -- 85 random-play lives hold ~15k),
POLICY_REFRESH_COUNT 4 -> 12 (4 games' worth re-searched per 2 iterations).
Per POSITION the refresh coverage still differs from Pong's (Ms. Pac-Man
games are shorter) -- reported, not corrected. Row diagnostics that
describe the last collected trajectory (fresh_target_entropy,
birth_*_last) now describe one life, not a game: not comparable with
Pong's.
Measured random play through this stack (5 games): 3 lives, ~180 decisions
per life, ~545 per game, mean score 286 (published random 307.3). So an
iteration collects ~550 env steps early (Pong ~800-3000); expect roughly
60-120k env steps by iteration 90 (Pong 150-200k): the same 180k gradient
steps on less data.
EVAL CHANGE (code): the in-run eval (8 games per row) and the exact probe
play whole GAMES -- Run.play_game: every life to the real game over, the
3000 cap per life (a capped life ends the game and reads capped); the Ray
eval actor and the probe reseed the env per game first (a real reset; the
game then starts from play_episode's own real reset, the double reset
every eval already had). Before, each eval "game" was one play_episode,
which in a multi-life game is one life, and the next one resumed the match
mid-way. A Pong game is one life: identical results (pinned in
tests/test_mspacman.py against the old code on the real Pong-v4 env). The
probe takes PROBE_ENV (default Pong-v4; the action count from the env).
REFERENCES (Atari-100k = 100k env steps = 400k frames; the BBF table, arXiv
2305.19452): random 307.3, human 6951.6, DER 1187.4, DrQ(eps) 1218.1,
SPR 1318.4, EfficientZero 1281.2, BBF 2008.3.
PRE-REGISTERED READS. The exact eval: 30 noop starts (1..30), whole games,
prior argmax, rows 5-90 (all 18 snapshots). PRIMARY variant = the deployed
policy: raw for the Gumbel arm; the acting states (PROBE_ACTING_STATES=1)
for the temperature + stabilizer arm, raw secondary (as bnrecal s1). No
external BN recal (no Ms. Pac-Man calibration set; Pong's frames do not
transfer). Greedy play from a noop count is deterministic and the 30
counts collapse into a few distinct games (noops mod 4, as on Pong; audit
measured 2 distinct scores on random nets), so one row is a weighted mean
of a handful of games -- the reads average rows. The probe's phases_won is
Pong's win test (every Ms. Pac-Man phase mean is positive): not a read.
  (1) learning: the primary exact score per row and its mean over rows
      60-90;
  (2) level against EfficientZero: g = (score - 307.3) / (1281.2 - 307.3),
      the fraction of EfficientZero's gain over random, for the primary
      mean over rows 60-90. On Pong the same line sits at g = 0.57-0.75
      over 60-90 (random -20.7, EfficientZero 20.1: tempscale s0 raw +2.70
      -> 0.57, bnrecal s1 deployed +5.80 -> 0.65, Gumbel s0 raw +9.74 ->
      0.75). Bands: HOLDS UP g >= 0.5 (mean >= 794.25, just below that
      Pong range); PART-WAY 0.2 <= g < 0.5 (502.08 <= mean < 794.25);
      DOES NOT HOLD UP g < 0.2 (mean < 502.08, under 1.63x random).
      Read (2) is NOT matched on data: Pong's rows 60-90 sat at ~120-200k
      env steps, Ms. Pac-Man's will likely sit lower, and EfficientZero's
      1281.2 is a final score at exactly 100k -- read (3) is the
      data-matched read;
  (3) data: env steps per row; the primary exact score at the first row
      with env_steps >= 100k (the Atari-100k point; or row 90 with its env
      steps if never reached) against EfficientZero 1281.2 / SPR 1318.4 --
      descriptive (one seed against 3-seed means);
  (4) stability, descriptive: the largest drop between consecutive rows from
      row 45 on, capped games, game length (eval_len_mean), in-run vs exact.
DECISION AFTER:
  HOLDS UP -> the algorithm transfers off Pong: next is Kai's (his list of
    things to try, or onward toward the Switch); a second seed is optional.
  PART-WAY -> Kai's call: a second seed, or a look at the port first.
  DOES NOT HOLD UP -> do not carry it to new games yet; diagnose the port
    first (refresh coverage per position, the higher train/env-step ratio,
    84x84 vs pellets, 9-action exploration) before blaming the algorithm.
CODE (worktree Y:\SwitchPlayer-mspacman, branch mspacman; this section is
committed on main itself, so the branch touches no file a write-up does):
Run.play_game + the sequential / Ray eval paths + the probe's PROBE_ENV;
SelfPlayActor.play reseeds only at a new game; the two -mspacman arms;
benchmarks/supervisors/supervise_mspacman.py {auto|gumbel|temp} 0 (waits
for Gumbel seed 1's completion marker AND its full readout -- exact 45-90,
recal 60-90; ALERTs if the readout is still incomplete 2 h after seed 1
completed -- so it never overlaps seed 1 or its evals; applies the decision
in "auto", exits 2 on PARTIAL unless exactly one variant's checkpoint (live
or completion marker) is already in the folder -- Kai's pick, then resumed;
an explicit variant must agree with a definite decision unless
--kai-override (Kai's instruction only); refuses if the other variant's
checkpoint is in the folder; holds, with an ALERT, while a live ablation.py
of EITHER -mspacman arm at the seed is running; the manifest is read from
the main tree; manifest + quick probe + 15 suites once per supervisor
process, the manifest again before every relaunch; a failing preflight is
an ALERT once main holds the manifest, and so is a manifest still missing
2 h into the preflight); tests/test_mspacman.py. Merged into
main only after seed 1's evals finish (seed 1's relaunch and hand-off check
the old manifest), by the WMI watcher scratchpad mspacman_merge.py with the
audited SHA pinned (a full 40-hex SHA that carries the manifest and whose
files hash to it; it waits for seed 1's complete readout -- a failed seed-1
probe is rerun by hand on the pre-merge code first -- holds if seed 1's
hand-off logged a code difference, checks main is still
this line, waits out Kai's in-progress git operations and uncommitted edits
to the files it updates, and re-hashes main against the manifest after the
merge, ALERTing on any difference; it also ALERTs hourly if seed 1 is
complete with a whole readout but its hand-off has not logged "all done"
for 2 h).
The supervisor is launched hidden via WMI from the worktree path (stdout
-> supervisor.log, stderr -> supervisor.err in the run dir); after the
merge every relaunch uses main's identical copy, with the same arguments.
FREEZE: no edit to an AUDITED file (supervise_mspacman.py's list) in the
main tree from the manifest commit (the audited SHA) until this run's evals
finish -- the merge and every relaunch check main against the manifest; the
deferred Gumbel test-only items (they touch tests/test_gumbelact8.py) wait
too. Nor any pip install into the shared .venv (the manifest cannot see
it; post_mspacman.py ALERTs on a change). AUDITED also pins four finished
runs' probe records the suites read (prioract2 s0/s1 exact, Gumbel s0 exact
and recal): no raw/recal re-probe of those runs (it would append to them)
until the evals finish. Gumbel seed 1's run folder and its two probe jsonls
stay where they are too (every supervisor relaunch re-reads them).
EVALS: the WMI hand-off scratchpad post_mspacman.py (log post_mspacman.log;
logs "all done"; repeats the supervisor's DECISION / ALERT / HOLD lines,
the merge watcher's ALERTs and tracebacks and any supervisor.err text;
ALERTs when supervisor.log goes 4 h silent;
ALERTs on a FAILED run, a HOLD, a code difference from the manifest, a
probe exiting nonzero, or a row of the 18 missing from a jsonl) writes
benchmarks/probes/probe_exhaustive_eval-mspacman-s0-2026-10-16.jsonl (raw)
and, for the temperature arm, ...-acting.jsonl (deployed).
COST: ~20-21 h (Pong's ~13.5 min/iteration; acting runs beside training)
+ evals ~30-60 min.

### Next experiment (pre-registered) -- Gumbel search acting, seed 1 (Kai, 2026-10-15)

DECISION (Kai, 2026-10-15): "let's run gumbel one more time then decide
what comes next" -- next being his list of things to try, or moving the
working algorithm to a different game (e.g. Ms. Pac-Man) to see whether it
holds up.
QUESTION: does seed 0's result -- breakout @70, the line's highest 60-90
mean (+9.74), no collapse after it -- replicate on another seed? (The level
test is the NO WORSE band, not seed 0's +9.74; the seed-1 mean is also
reported against the same-seed temperature reference, bnrecal s1 +5.47.)
ARM unchanged: effzero-repro-faithful-gumbelact8-birth (the audited 10-12
arm), SEED 1, dir pong-gumbelact8-s1-2026-10-15. Supervisor change only:
supervise_gumbelact8.py accepts seeds 0 and 1 (each with its own run-folder
date and the run it queues behind: seed 1 behind Gumbel seed 0's completion
marker, already there).
SEED-1 REFERENCES (no PUCT-8 run exists at seed 1): pong-prioract2-s1 (the
same base arm with prior sampling): no breakout, raw mean 60-90 -10.45;
pong-bnrecal-s1 (temperature + stabilizer): raw first success @70
(deployed @75), raw mean +5.47 (deployed +5.80). Seed 1 has been the harder
seed for policy-only acting: 3 of the 4 other policy-only configs run on it
never broke out (prioract2, seqref, SVE); the 50-sim search-acting control
(pong-faithful-s1) did (recal @65).
PRE-REGISTERED READS (exact prior-argmax evals, rows 45-90; external recal
60-90 secondary -- as seed 0):
  (1) breakout: first success (exact >= +10 or >= 3/4 phases) at any row;
  (2) level: mean 60-90 vs the bands (NO WORSE >= +3.68, WORSE < -2.32);
  (3) late stability: rows below 0 after the first success. A COLLAPSE is
      such a row unless its external recal row is >= 0 (the known BN
      artifact, as tempscale s0 @90 -- reported, not counted); rows before
      60 have no recal and count as they are. Recal only classifies a raw
      row here; the raw series stays primary (seed 0 has no row below 0
      after its first success under either reading). STABLE = no collapse
      AND at least two rows after the first success (first success <= 80);
      a first success at 85 or 90 leaves stability UNTESTED;
  (4) data: env steps at the first success (seed 0: 148.9k @70; bnrecal s1
      raw: 130.9k @70), reported so a gain is not mistaken for extra data.
TWO-SEED DECISION:
  REPLICATES -- seed 1 breaks out AND its mean 60-90 is NO WORSE AND it is
    STABLE (which needs the first success by 80): the breakout and the
    stability hold on seed 1 and its level clears the NO WORSE band ->
    Gumbel-8 acting becomes the training configuration for the real-time
    line (the deployed player stays the prior argmax, read (D) 10-15).
  PARTIAL -- a first success, but the mean below +3.68, a collapse, or
    stability UNTESTED (first success at 85 / 90): 2/2 breakouts like the
    temperature config, but seed 0's level or stability is not shown on
    seed 1 -> Kai's call between the two (cost per decision during
    training: Gumbel-8 ~40 ms vs prior sampling ~2 ms; both fit a 4-frame
    decision).
  NO BREAKOUT -- Gumbel-8 1/2; the temperature config (2/2) stays the base.
Reported in every case, descriptive only: the two-seed raw mean against
the temperature config's (+2.70 s0 / +5.47 s1 = +4.08) -- seed 0's +9.74
carries it above +4.08 for any seed-1 mean above -1.58, so it is not a
decision read.
The search-agent read (D) is not repeated (answered on both engines, 14
snapshot rows, consistent).
COST: ~20 h (seed 0: 90 iterations in 71,631 s of training) + ~30 min
evals (exact 45-90, recal 60-90) by the WMI hand-off scratchpad
post_gumbel_s1.py (log post_gumbel_s1.log; logs "all done"; a FAILED run
is ALERTed and waited on, so a relaunch still gets its evals; before the
evals it re-hashes the run's audited files and both probe scripts against
the launch-time manifest and ALERTs on any difference; a probe with a
nonzero exit is ALERTed).

### RESULT -- Gumbel search acting (8 sims, staged scale), seed 0, 2026-10-15

pong-gumbelact8-s0-2026-10-12 ran 10-13 18:58 -> 10-15 03:48 (paused by Kai
10-13 20:57 -> 10-14 01:27 at the iter-10 checkpoint; a power cut 10-14
~03:26 lost iters 19-20, resumed 11:19 from the iter-18 checkpoint, which
read back whole). Evals by the hand-off 03:49-07:15. EXACT prior argmax
(raw) / external recal / the 8-sim SEARCH AGENT, and the twin
(pong-searchact8-s0, PUCT-8 acting):
  iter       45     50     55     60     65     70     75     80     85     90
  raw      -13.7   -9.6   -7.2   -7.5   -0.5   +7.3  +17.9  +17.7  +15.4  +17.9
    won     0/4    0/4    0/4    0/4    1/4    3/4    4/4    4/4    4/4    4/4
  recal                         -10.4   -0.8  +17.4  +19.1  +18.2  +17.5  +17.0
  Gumbel-8 agent                 -9.2   -1.1  -13.1  +12.4  +12.7  +12.6  +14.1
  twin raw  -8.6  -10.9  -10.4   -9.2   -4.6   +2.1   +5.9   -1.4   +3.8  +19.4
  PUCT-8 agent (twin)           -11.2   -3.5   +2.3  -15.7   -7.2   -4.7  +19.5
  env steps  81.5k  94.5k 107.5k 121.0k 134.7k 148.9k 162.1k 174.1k 185.3k 196.3k
  twin env   60.9k  69.9k  80.8k  89.7k 100.6k 113.1k 125.2k 137.2k 150.6k 162.7k
BREAKOUT @70 (+7.3, 3/4 phases), then four rows +15.4..+17.9, every game
>= +8 from @75 on. In-run eval (prior argmax) +1.9 @65, +7.5 @70, +18.8 @75,
+18.8 @90.
READ (A): first success @70 -> NOT SUPPORTED by the pre-registered letter
(<= 65): 8 Gumbel sims did not recover 50-sim search acting's @50-65.
READ (B) vs the twin (descriptive, single seed): first success @70 vs @90;
mean 60-90 +9.74 vs +2.30. CONFOUND, larger than pre-registered ("slightly
fresher data"): the faster Gumbel actor collected more env steps per
iteration in iters 10-55 (+35% by @50), so by iteration the Gumbel run had
more data. At matched env steps the gap shrinks: Gumbel first success at
148.9k env steps vs the twin's at 162.7k; the twin at ~113k (@70) was +2.1,
the Gumbel run at ~121k (@60) -7.5.
READ (1) speed: first success @70, first >= +15 @75.
READ (2) level: mean 60-90 = +9.7381 -> NO WORSE (>= +3.68); recal
+11.1334. The highest 60-90 mean of the real-time line (temperature config:
tempscale s0 raw +2.70, bnrecal s1 deployed +5.80; twin +2.30).
READ (3) late stability: 2 of 7 rows 60-90 below 0 (both before the
breakout); no collapse after it (tempscale s0 had -18.9 @90, the twin -1.4
@80).
READ (D), deployment (the 10-10 question): searching at 8 sims at
deployment HURTS on both engines. Paired game by game against the same
snapshot's prior argmax, rows 60-90: Gumbel-8 -5.67 (worse in 145 of 210
games), PUCT-8 on the twin -5.23 (worse in 131). Each has one collapsed row
(Gumbel @70 -13.1 vs the prior's +7.3; PUCT @75 -15.7 vs +5.9) and neither
beats the prior at any row by more than +1.03. The pre-launch audit
observation (-4.4) holds up. A Switch player trained with search acting
should still deploy the prior argmax.
PRE-REGISTERED INTERPRETATION: (A) no -> "8-sim acting does not move the
breakout with either engine; the temperature + stabilizer config stays the
base for policy-only acting." Queue item (3) (Gumbel + SVE) does not run
(SVE s0 did not work out). Beyond the letter, single seed: the Gumbel run
broke out no earlier than the temperature config (@70 / @75) but held the
highest, most stable late level of the line -- whether that replicates
(and how much is the extra data) needs a second seed. Gumbel-8 acting costs
~40 ms/decision during training (inside a 4-frame budget); the deployed
player stays policy-only either way. Next step is Kai's call. The
audited-file freeze is lifted; deferred test-only items: Gumbel r2 LOW-1,
NIT-1, NIT-3 (the SVE2 deferrals are moot with SVE shelved).

### RESULT -- SVE seed 0 (temperature + stabilizer config), 2026-10-13

pong-sve2-s0-2026-10-10 ran 10-12 15:31 -> 10-13 18:23 (96,751 s, one
attempt, no interruptions). Evals by the hand-off 18:24-18:53. EXACT evals,
DEPLOYED (shipped acting states) / RAW / external-calibration recal, and
the reference (pong-tempscale-s0, no stabilizer, no deployed series):
  iter      45     50     55     60     65     70     75     80     85     90
  deployed -14.2  -13.7  -13.5  -12.0  -13.6   -9.1   -8.8   -7.5  -12.4   -8.1
    won     0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4
  raw      -15.3  -13.1  -16.5   -9.6  -14.1  -14.0  -10.4   -9.1  -12.1   -6.3
    won     0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    1/4
  ext recal                     -14.2  -14.1   -7.7   -6.6   -4.5  -12.4   -7.0
  ref raw   -7.8   -2.0  -19.5   +2.6   +2.5  +11.6  +13.9   -2.1   +9.2  -18.9
  ref recal                     +1.7   +5.0   -0.9  +11.1  +17.8  +10.3  +16.4
NO BREAKOUT. Slow climb from ~-14 (45-65) to -7.5..-9 (70-90, many games
capped, deployed phase 0 at -1.0 @80); the only phases won are raw @90
(phase 0, +1.1) and recal @90 (phase 2, +0.1). In-run eval (prior argmax)
-9.2 @70, -8.6 @90.
READ (1) speed: no first success, no row >= +15 (reference raw @70).
READ (2) level: deployed mean 60-90 = -10.2143 -> WORSE (< -2.32); raw
-10.7906; external recal -9.4859 (reference raw +2.6951, recal +8.7809).
READ (3) late stability: 7 of 7 rows 60-90 below 0 (reference raw 2).
READ (4) diagnostics: sve_buffer_frac 0.29 @25, then 0.32-0.35 (in range);
sve_frac 0.27-0.31; sve_td_bias +0.155 @25 drifting DOWN to +0.077 @90;
sve_td_gap 0.21 -> 0.13. As in seed 1, SVE behaved as built.
STOP RULE (from the two jsonl files): same-row ratio above 2x at 35 (3.44)
and 45 (2.73) only -- longest streak 1; NOT triggered (live: the same).
PAIRING: env_steps already differ at row 5 (4,746 vs 4,349); late rows
trail the reference (170,671 vs 180,020 @90). Start-only pairing, as
pre-registered.
WORKS OUT -> NO (no first success; (2) WORSE). TWO-SEED SUMMARY: SVE 0/2
breakouts (deployed means -13.70 / -10.21) vs 2/2 for the temperature
config without SVE -> SVE SHELVED on the policy-only line, as
pre-registered. Queue item (3), Gumbel + SVE, does NOT run. Queue item (2),
Gumbel acting, launched as armed: gumbel_merge fast-forwarded main to
a980ed44 at 18:53, the supervisor launched at 18:58 (preflight 14 suites).
PAUSED 10-13 20:57 (Kai asked) after its iteration-10 checkpoint (in-run
rows 5/10: -21.0 / -19.2; twin searchact8 s0 -20.6 / -17.6; ~11.5
min/iter).

### Next experiment (pre-registered) -- Gumbel search acting (Kai, 2026-10-12)

DECISION (Kai, 2026-10-12): "plan gumbel to run immediately after [SVE
seed 0] ... gumbel without SVE". Queue item (2) of "SVE seed 0, then Gumbel
acting" below; the searchact8 result's pre-registered follow-up ("Gumbel
search at low sims (designed for that regime)").
HYPOTHESIS: 8-sim PUCT acting did not move the breakout (@90) because PUCT
at 8 sims is a weak improvement over the prior. Gumbel MuZero's sequential
halving is built for tiny budgets: at 8 sims over Pong's 6 actions it gives
EVERY action a one-step lookahead, then re-checks the best two -- an
improvement step the paper guarantees at any budget when the Q estimates
are right.
ARM effzero-repro-faithful-gumbelact8-birth = the searchact8 arm
(prioract-noredo + 8-sim search acting on the CPU actor, 50-sim PUCT birth
search makes every target, EVAL_PRIOR_ONLY) + two new flags (default off):
ACTOR_GUMBEL_SEARCH -- play_episode's per-step searches run the repo's
Gumbel engine (MCTSHelper.run_gumbel_mcts) instead of PUCT, while
USE_GUMBEL_SEARCH stays off, so birth search, refresh and every other search
keep PUCT -- and ACTOR_GUMBEL_STAGED_SCALE (below). The deltas vs
searchact8 are the actor's search engine and its exploration mapping (the
staged temperature on the Gumbel draws, which also cools the search's own
visit allocation -- which 3 actions survive phase 0, which 2 get the extra
sims -- where PUCT-8's temperature acts only after its search). SEED 0 (twin
pong-searchact8-s0-2026-10-09, same seed; the runs split within the first
iterations, so this pairs the start).
THE ENGINE AT THIS BUDGET (m = min(16, 6 actions, 8 sims) = 6): phase 0
visits each of the 6 root actions once (6 sims, one batched expansion),
keeps the top 3 by g + logits + sigma(q_hat); phase 1 spends the last 2 sims
on the top 2 of those; the final halving keeps 1, which acts. sigma = (50 +
max visits) x 0.1 (the paper's Atari c_visit / c_scale); q_hat is normalized
by the search's min-max range with EfficientZero's range floor (EZ-V2 style;
the floor never binds here: root-Q spread median 0.097 / 0.053). Below the
root: the paper's deterministic visit matching. No Dirichlet noise.
Deviations from mctx, measured on prioract2 s0 @50 / @85 (audit r2 re-run
of the r1 probe, scratchpad audit_gumbel_r2/engine_probe2.log): the final
pick is among the 3 survivors, not mctx's most-visited 2 (the played action
differs in 6.1% / 10.3% of steps); min-max over the tree, not per node
(10.4% / 15.3%); the greedy argmax differs in 8.9% / 11.4%.
EXPLORATION SCHEDULE (ACTOR_GUMBEL_STAGED_SCALE): the actor's self-play
Gumbel draws are scaled by the twin's staged visit temperature (1 / 0.5 /
0.25 at 50% / 75% of the run): argmax(T*g + logits + sigma(q_hat)) is a
temperature-T sample of the improved policy among the halving survivors.
Measured (below): at the paper's scale 1, Gumbel-8 matches a 50-sim search
about as often as PUCT-8 at T=1, so without the schedule the late rows would
compare different exploration levels. Deviation from the paper (scale 1
throughout). Greedy searches keep scale 1 (they act by the improved-policy
argmax).
MEASURED BEFORE LAUNCH (prioract2 s0 weights, real Pong states, CPU;
scratchpad gumbel_measure.py / gumbel_measure2.py):
  latency per decision (act_search, 1 torch thread as the production Ray
  actor, machine loaded by test suites -- ratios hold, absolutes high):
  PUCT-8 66.9 ms, Gumbel-8 47.9 ms (0.72x: phase 0 batches its 6 root-
  forced simulations), Gumbel-4 29.7, Gumbel-16 103.5, PUCT-16 128.1. With
  the unloaded PUCT-8 figure (~54 ms/step incl. env) -> Gumbel-8 ~40
  ms/step, inside a 4-frame decision (66.7 ms). Faster games also join the
  eager collect earlier than the twin's (slightly fresher data) -- noted.
  power -- how often the acted action matches a 50-sim PUCT search's
  greedy argmax (K=16 draws per state; prior rows exact):
                         @50    @85
    PUCT-8 at T 1       0.485  0.406
    PUCT-8 at T 0.5     0.523  0.452
    PUCT-8 at T 0.25    0.552  0.479
    prior at T 1        0.407  0.290
    prior at T 0.5      0.559  0.373
    prior at T 0.25     0.671  0.443
    Gumbel-8, scale 1   0.470  0.397
    greedy: Gumbel-8 0.598 / 0.488, PUCT-8 0.661 / 0.535, prior argmax
    0.748 / 0.465.
  staged scale, paired with PUCT-8 at the same T on the same states (K=8):
                         @50            @85
                     Gumbel  PUCT   Gumbel  PUCT
    T 1               0.483  0.470   0.372  0.425
    T 0.5             0.530  0.516   0.443  0.441
    T 0.25            0.600  0.535   0.450  0.464
  -> with the schedule, Gumbel-8's exploration tracks the twin's at every
  stage (within +-0.065).
  The reference is itself a (50-sim) PUCT search that leans on the prior,
  so agreement favours PUCT-8 and the prior; it is a manipulation check,
  not a quality measure. Q changes the played action relative to pure
  prior sampling in 37.2% / 39.4% of steps (engine_probe2.log) -- the
  manipulation is real.
  PRE-LAUNCH OBSERVATION (audit r1, searchact8 s0 @90, noops 1-8): the
  greedy Gumbel-8 agent scored +15.1 vs the prior argmax's +19.5 on the
  same games (paired -4.4); min-max stretches small one-step Q differences
  to ~5 logits of sigma, so at deployment the 8-sim Gumbel agent can be
  WORSE than the prior it searches from. Read (D) measures this properly.
PRE-REGISTERED READS (exact prior-argmax evals, rows 45-90 -- comparable
with every run; secondary: external recal 60-90):
  (A) the action-quality read (as searchact8's): first success (exact >=
      +10 or >= 3/4 phases) at <= 65 -> consistent with low-sim Gumbel
      acting recovering search acting's earlier breakout (50-sim search
      acting: @50-65); >= 70 or none -> not supported at 8 Gumbel sims;
  (B) vs the twin (PUCT-8, same seed): first success @90, mean 60-90
      +2.30 -- reported, single-seed, descriptive;
  (1)-(3) the standard reads and bands (NO WORSE >= +3.68, WORSE < -2.32).
  (D) the deployment read (tertiary, new): the SEARCH agent a search-
      acting Switch player would deploy -- probe_exhaustive_eval with
      PROBE_SEARCH_SIMS 8 on rows 60-90: greedy Gumbel-8 (improved-policy
      argmax; np.random reseeded per game) on this run, and greedy PUCT-8
      on searchact8 s0's snapshots (the 10-10 question: what the 8-sim
      agent itself scores). Each against its run's prior-argmax mean on
      the same rows. Known asymmetries: PUCT-8's max visit count ties in
      6.7% / 18.3% of states (@50 / @85) and its greedy rule then takes the
      lowest action index; the greedy Gumbel-8 agent keeps scale-1 draws in
      its halving (one draw of a stochastic agent per noop count, reseeded
      for reproducibility) while greedy PUCT-8 is deterministic.
INTERPRETATION (single seed, hedged): (A) yes and not WORSE -> Gumbel-8
acting is a real-time-viable improvement -> its seed 1. (A) yes but WORSE
-> mixed (an early breakout that does not hold) -> seed 1 before any
conclusion. (A) no -> 8-sim acting does not move the breakout with either
engine; the temperature + stabilizer config stays the base for policy-only
acting. Either way, queue item (3), Gumbel + SVE, stays exactly as Kai set
it: conditional only on SVE seed 0 working out (this result is reported to
him with it). (D) decides separately whether a deployed Switch player
should search at 8 sims at all.
COST: the actor's ~2000-3000-step game takes ~1.5-2 min, inside the ~13-min
training block (eager collect joins it mid-block); ~13-14 min/iter -> ~20
h. Evals after the run: exact + recal (~30 min), then the search-agent
probes (14 snapshots, ~107 s per Gumbel game at 2 threads -> ~5 h).
QUEUE MECHANICS: supervise_gumbelact8.py ray 0 (WMI, launched from the
worktree path before the merge; run folder created first) waits for SVE
seed 0's completion marker (ALERT and hold if it FAILED), then polls its
preflight (the manifest needs the gumbelact branch on main); a WMI watcher
(gumbel_merge.py) merges gumbelact into main only after SVE seed 0's eval
hand-off logs "all done" AND the run completed (up to 4 attempts); the
supervisor launches at its next 10-min poll. post_gumbel.py runs the evals.
The worktree Y:\SwitchPlayer-gumbel must stay on gumbelact until the run
completes (the supervisor reads its manifest from there on every
relaunch). A stall on a failure path (the SVE hand-off dying, a manual SVE
relaunch after a CRASH LOOP) shows only as WAITING / PREFLIGHT lines.
AUDIT: round 1 one HIGH (a preflight suite pinned the probe's old source --
the queue could never have launched) + LOWs, fixed with the staged scale
added; round 2 nothing above LOW -- the loop closes. Round-2 fixes to text
and scripts only (numbers above, queue item (3) as registered, the merge
watcher's explicit identity, this pre-registration on the gumbelact branch
so the merge fast-forwards). Deferred (test-only): r2 LOW-1 (the staged-
scale tests use patched temperatures and scales 0/1 only), NIT-1
(behavioural supervisor test), NIT-3 (test thread/sys.path residue).

### Next experiment (pre-registered) -- SVE seed 0, then Gumbel acting (Kai, 2026-10-12)

DECISION (Kai, 2026-10-12): "run a second SVE seed, then plan gumbel to run
immediately after it ... gumbel without SVE, and if the second SVE run works
out, we can do a combination of gumbel and SVE after the gumbel only run".
QUEUE: (1) SVE seed 0 (this section); (2) Gumbel search acting without SVE,
pre-registered in its own section before it is built, armed to launch as
soon as (1) and its evals finish; (3) Gumbel + SVE after (2), only if (1)
WORKS OUT (below).
(1) ARM unchanged: effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve
(the audited 10-10 arm), SEED 0, dir pong-sve2-s0-2026-10-10 (the date is
the supervisor's audited constant). Supervisor change only:
supervise_sve2.py accepts seed 0 and seed 1, each with its own stop-rule
reference, and takes the deferred r2 hardening (LOW-A: streak keyed by
iteration; LOW-B: a behavioural test; LOW-C: NaN rows, the stop rule
computed before the prints, a startup "stop rule armed: N twin rows" line).
REFERENCE: no seed-0 run of the stabilizer config exists. The reference is
pong-tempscale-s0-2026-10-05, the temperature config at seed 0 without the
stabilizer (which changes only the actor's BN statistics): raw first
success @70, raw mean 60-90 +2.70, external recal +8.78. As with seed 1,
the parallel actor splits the games within the first iterations, so this
pairs the start only.
PRE-REGISTERED READS (as seed 1's; exact evals of the DEPLOYED policy,
rows 45-90; raw and external recal secondary): (1) speed: first success
(exact >= +10 or >= 3/4 phases) and first row >= +15 -- reference @70
(raw); (2) level: mean 60-90 vs the bands (NO WORSE >= +3.68, WORSE <
-2.32); (3) late stability: rows 60-90 below 0; (4) diagnostics:
sve_buffer_frac ~0.29-0.36 once full, sve_td_bias near +0.11 and not
drifting upward.
STOP RULE: as seed 1's, against tempscale s0's same-row `value` (rows
25-90 all present, 0.17-1.06); advisory live, evaluated from the jsonl
files at the end.
WORKS OUT (gates (3)): the deployed series reaches a first success at any
row 45-90 AND its mean 60-90 is not WORSE (>= -2.32). Two-seed summary
either way: SVE 0/2 breakouts -> SVE shelved on this line (the
temperature config without SVE: 2/2); 1/2 -> no evidence SVE helps or
hurts, but (3) is run as Kai asked.
COST: ~26 h (seed 1: 93,571 s).

### RESULT -- SVE on the temperature + stabilizer config (seed 1), 2026-10-12

pong-sve2-s1-2026-10-10 ran 10-10 19:36 -> 10-11 21:35 (93,571 s, one
attempt, no interruptions; ~19-20 min/iter after T1, inside the 18-21
estimate). EXACT evals, DEPLOYED (shipped acting states) / RAW (trainer
statistics) / external-calibration recal:
  iter      45     50     55     60     65     70     75     80     85     90
  deployed -13.4  -16.5  -13.5  -15.8  -12.8   -8.0  -14.5  -13.7  -14.5  -16.7
    won     0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4
  raw      -14.2  -16.7  -12.2  -14.3  -12.6  -13.3  -12.5  -13.5  -12.7  -14.8
    won     0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4    0/4
  ext recal                     -11.4  -12.5  -14.6  -11.9  -14.8  -12.2  -14.0
  twin dep -14.3  -13.6  -13.1  -13.1   -1.0   -3.8  +10.1   +9.5  +20.4  +18.4
NO BREAKOUT. The closest it came: deployed @70 (-8.0; phases 1/3 at -2.75 /
-1.0, 14 games capped). In-run eval (prior argmax) -9.4 @70, -17.8 @90.
READ (1) speed: no first success, no row >= +15 (twin @75 / @85).
READ (2) level: deployed mean 60-90 = -13.7001 -> WORSE (< -2.32); raw
-13.3904; external recal -13.0619 (twin +5.8046 / +5.4667 / +6.7669).
READ (3) late stability: 7 of 7 rows 60-90 below 0 (twin 3) -- no breakout
to collapse from.
READ (4) diagnostics: sve_buffer_frac 0 through row 20; row 25 0.151 (below
the 0.20-0.25 predicted: this run's slower pace put 26.6k transitions in
the buffer by iteration 25 vs the twin's 30.1k), then 0.33-0.35 (in
range). sve_frac 0.146 @25, then 0.285-0.312 (~0.87 of the buffer share:
PER samples stale roots slightly less). sve_td_bias +0.143 @25 (stage-0
prediction ~+0.11), then drifting DOWN to +0.079 @90; sve_td_gap 0.19 ->
0.14. SVE did what it was built to do; nothing in the value diagnostics
looks broken.
STOP RULE (authoritative, from the two jsonl files): same-row ratio above
2x at 25 (2.08), 50 (2.46), 55 (2.21) -- longest streak 2 (@50-55); NOT
triggered. The live supervisor printed the same streaks.
BEHAVIOUR VS THE TWIN, rows 75-90: search-target entropy 1.29-1.44 (twin
1.24-1.31), act/target agreement 0.40-0.47 (0.46-0.51), prior maxp
0.36-0.38 (0.38-0.40). The twin's late rows are a winning agent's, so these
read as consequences as much as causes.
PAIRING: the twin pairs the initial weights only. env_steps already differ
at row 5 (4,327 vs 4,911), so the parallel actor's timing split the games
in the first iterations, well before T1 -- in practice an independent run
from the same start.
PRE-REGISTERED INTERPRETATION: SVE HELPS -> NO (no first success; (2)
WORSE). "One seed: a null or negative result is suggestive only." Base
rate: the policy-only line has had a non-breakout seed before (prioract2 s1,
-10.45; the unscaled config broke out 2/4), and the temperature config
without SVE broke out 2/2. This seed cannot separate SVE hurting from an
unlucky run, but it gives no sign of the speed gain SVE was run for, at
+52% training wall clock. Next step is Kai's call: a second SVE seed (~26
h), or keep the temperature + stabilizer config as the base and move on.
The audited-file freeze is lifted; the r2 LOW-A/B/C stop-rule hardening
stays deferred (moot unless SVE runs again).

### Next experiment (pre-registered) -- SVE on the temperature + stabilizer config (Kai, 2026-10-10)

DECISION (Kai, 2026-10-10: "building SVE makes sense as a next step"):
EfficientZero V2's search-based value estimation on the strongest
real-time configuration. SVE was built and audited on 09-28/29 (stage 2:
fresh 16-sim search through the target nets) but never run; its
prerequisite -- drop ReDo -- is met by this base.
ARM effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve = the BN
stabilizer arm (prior temperature x0.5, BN_RECAL_ACTING, policy-only CPU
actor, eager collect, birth search, communal refresh, no ReDo) + SVE
exactly as pre-registered on 09-28: VALUE_TARGET_MODE "mixed",
SVE_START_TRAIN_STEPS 40000 (T1), SVE_FRESH_TRANSITIONS 20000 (T2),
SVE_SOURCE "search", SVE_SEARCH_SIMS 16. SVE's searches run the raw-
statistics target nets (search runs the prediction head at depth > 0; the
stabilizer touches only the actor / eval copies). SEED 1, paired with its
twin pong-bnrecal-s1-2026-10-08 (same initial weights and env seeds --
verified bit-identical through T1 by the sve2 audit; SVE's first batch is
iteration 21's first step, its first row 25; runs are not replayable, so
this pairs the start, not the trajectory). COST: stage 2 measured +52%
wall-clock on 09-29 -> ~20 min/iter, ~30 h; eager collect's mid-block
reopen re-runs some cached SVE searches (sve2 audit r1: +6-8 s/iter,
under 1% of wall clock; estimate 18-21 min/iter after T1).
PRE-REGISTERED READS (the 09-28 SVE reads, against the twin; exact evals
of the DEPLOYED policy, rows 45-90; raw and external recal secondary;
Pong saturates at +21, so SVE can only show as SPEED or STABILITY):
  (1) speed: first success (exact >= +10 or >= 3/4 phases) and first row
      >= +15 -- twin: @75 and @85;
  (2) level: mean 60-90 vs the bands (NO WORSE >= +3.68, WORSE < -2.32)
      and vs the twin's +5.80;
  (3) late stability: rows 60-90 below 0 -- twin: 3 (@60/@65/@70, all
      before its first success);
  (4) diagnostics: sve_buffer_frac 0 before T1, then ~0.29-0.36 once the
      buffer is full (row 25 will read lower, ~0.20-0.25: on the twin's
      pace the 31.25k buffer fills only at ~iter 26); sve_frac relative to
      it; sve_td_gap / bias against the stage-0 prediction (~+0.11).
  SVE HELPS (single seed, hedged) if (1) moves the first success to <= 65
  -- search acting's range, which no policy-only run has reached -- without
  (2) reading WORSE; same or later -> no evidence of a speed gain at this
  seed.
STOP RULE -> REVIEW WITH KAI (09-28): the TD-referenced `value` (mean
|v(s0) - TD z0| over the diagnostics panel) above 2x the TWIN's value at
the same row for 3 consecutive rows after T1 (a value self-echo) --
monitored live: supervise_sve2.py prints each row's value, the twin's and
the ratio from row 25 and an "ALERT: STOP RULE" line on the third
consecutive row above 2x (sve2 audit r1 MEDIUM-1). The live alert is ADVISORY:
the rule is evaluated from the two jsonl files at the end (authoritative).
AUDIT: round 1 one MEDIUM (stop-rule quantity not printed) + three LOWs,
fixed; round 2 nothing above LOW -- the loop closes. Deferred until after
the run (an audited file): r2 LOW-A (a resume that re-runs a row can shift
the live streak by one row -- key it by iteration), LOW-B (a behavioural
test of the wiring), LOW-C (NaN value, a row skipped by an earlier print
error, a startup "stop rule armed: N twin rows" line).

### RESULT -- low-sim (8) search acting with birth targets, 2026-10-10

pong-searchact8-s0-2026-10-09 ran 10-09 12:49 -> 10-10 09:11 (paused by Kai
16:47-17:03 at the iter-18 checkpoint). EXACT / external recal:
  iter    45     50     55     60     65     70     75     80     85     90
  raw    -8.6  -10.9  -10.4   -9.2   -4.6   +2.1   +5.9   -1.4   +3.8  +19.4
  won    1/4    1/4    0/4    0/4    2/4    2/4    2/4    1/4    2/4    4/4
  recal                      -6.1   -5.2  -10.0   -4.6   -6.0  +17.1  +13.2
READ (A): first success @90 (+19.4, 4/4; recal @85) -> NOT supported (<= 65).
Competitive from @65 (2/4 phases) but under the success bar until the end.
READ (2): mean 60-90 = +2.2999 -> INCONCLUSIVE; recalibrated -0.2050.
READ (3): no collapse after the first success (it came last).
VS ITS TWIN (prioract2: identical but prior sampling): @75 / none. The
8-sim PUCT search did not move the breakout earlier -- if anything later.
PRE-REGISTERED INTERPRETATION: (A) no -> PUCT at 8 sims is too weak to
help; next candidates: Gumbel search at low sims, or accept the
temperature config as the base and move to SVE -- Kai's call. (Single seed.)
STANDING SUMMARY OF THE REAL-TIME (policy-only) LINE: the temperature
config (prior T x0.5) broke out on 2/2 seeds (@70 / @75) -- tempscale s0
raw +2.70 (recal +8.78), bnrecal s1 deployed +5.80 NO WORSE, final
+18.4/+20.4 -- the strongest real-time configuration so far; the BN
stabilizer works (weakly tested); 8-sim act-time search adds 54 ms/decision
and no earlier breakout. The audited-file freeze is lifted.

### Next experiment (pre-registered) -- low-sim search acting with birth-search targets (Kai, 2026-10-09)

DECISION (Kai, 2026-10-09: the cleanup first, "then you can do that" -- the
8-sim PUCT search-acting arm, pre-registered and audited): the stabilizer
result's pre-registered branch (S works, T not met -> acting quality).
HYPOTHESIS: policy-only breakouts come late (@70-80 across five seeds;
2/2 for the temperature config) while search acting broke out @50-65; the
quality of the ACTIONS (search-improved vs sampled from the prior) is what
times the breakout. Test it at a search budget that fits real time: 8
simulations = 31 ms (GPU) / ~54 ms on the production 1-thread CPU actor
per decision, inside a 4-frame decision (66.7 ms).
ARM effzero-repro-faithful-searchact8-birth = prioract2's arm
(effzero-repro-faithful-prioract-noredo: --ray --cpu-actor, eager collect,
no ReDo, communal refresh, 50-sim BIRTH search for the targets) with the
ACTIONS chosen by an 8-sim PUCT search on the CPU actor (act_search: root
noise, visit-count sampling at the staged visit temperature 1 / 0.5 /
0.25). New flags (default off / None): ACTOR_SEARCH_SIMS (the actor's
per-step search sims; None = ACT_SEARCH_SIMS) and SEARCH_ACT_BIRTH_TARGETS
(under USE_SEARCH_ACTING the actor records plain (obs, action, reward)
steps and skips the tail search, so the 50-sim BIRTH search still makes
every target -- only the action choice changes). EVAL_PRIOR_ONLY (in-run
eval = prior argmax, comparable with every run); PRIOR_ACT_STAGED_TEMP off
(inert: no prior sampling). The behaviour diagnostics (beh_*) are not
recorded in this mode -- they describe prior sampling. No BN stabilizer
(its gate refuses search acting: root-only statistics would distort a
search) and no temperature scale (inert under search). Seed 0.
COMPARISONS: prioract2 s0/s1 (identical except who acts: @75 / none);
noredo (50-sim search acting with its own 50-sim targets, parallel: @65,
+5.86); the control (50-sim search acting, sequential: @50 / @65); the
temperature config (@70 / @75).
COST (corrected by audit r1 LOW-1 before launch): the production Ray CPU
actor runs ONE torch thread (Ray sets OMP_NUM_THREADS=1) -> ~54 ms per step
at 8 sims, env step included (the 38 ms figure was 2 threads) -- 81% of a
4-frame decision, still inside it; ~1.8 min per 2000-step game, ~2.7 min at
the 3000 cap; with eager collect a game joins ~150-450 steps into its
2000-step block (vs 0-40 for policy-only) -- slightly staler data, noted.
POWER (audit r1, measured on prioract2 s0 weights, 127 real states): how
often the acted action matches the 50-sim search's argmax, 8-sim search vs
prior sampling: T=1 0.48 vs 0.34 (@50) / 0.43 vs 0.32 (@85); T=0.5 0.52 vs
0.45; T=0.25 0.55 vs 0.53. The manipulation is real while T >= 0.5 (to
~iter 67 -- covering read (A)'s window) and nearly nil at T=0.25. Read (A)
scores the PRIOR argmax (comparable with every run), not the 8-sim agent
that would be deployed.
AUDIT: round 1 nothing above LOW (four LOWs fixed: latency, a 50-sim birth
budget test, ACTOR_SEARCH_SIMS only with SEARCH_ACT_BIRTH_TARGETS,
docstrings/print); round 2 nothing above LOW -- the loop closes. Deferred:
a test for a stray ACTOR_SEARCH_SIMS on a policy-only arm (inert either
way).
PRE-REGISTERED READS (exact prior-argmax evals, iters 45-90; secondary:
external-calibration recal):
  (A) the action-quality read: first success at <= 65 -> consistent with
      a little act-time search recovering search acting's earlier
      breakout; >= 70 or none -> not supported at 8 PUCT sims;
  (1)-(3) the standard reads and bands.
INTERPRETATION (single seed, hedged): (A) yes and not WORSE -> 8-sim
search acting is a real-time-viable improvement -> seed 1, then Kai's call
on combining it with the stabilizer-equivalent for search (a root-only
recalibration is not valid for search). (A) no -> PUCT at 8 sims is too
weak to help; next candidates: Gumbel search at low sims (designed for
that regime), or accept the temperature config as the base and move to
SVE -- Kai's call.

### HOUSEKEEPING, 2026-10-09: the deferred audit items closed

Done (commit "Cleanup: the deferred audit items", audit round 1 nothing
above LOW): the bnrecal request->collect chain test (r2 L-1); tempscale
override LOW-1 (relaunch hint + docstring), LOW-2 (override_code: the
override never launches over an unfired trigger), LOW-3 (tests); seqref L1
(the e2e refresh test sees both games, stamps, counts); the ACT_SEARCH_SIMS
comment now carries the measured latency. Folded into the next build (its
audit covers them): a target-change assertion in the seqref e2e test, an
override_code test for a trigger fired at >= 70, and seqref L2 -- the next
supervisor's preflight runs test_reanalyze_refresh and test_vtree_communal.

### RESULT -- the BN stabilizer on the temperature config (seed 1), 2026-10-09

pong-bnrecal-s1-2026-10-08 ran 10-08 12:41 -> 10-09 08:13, no
interruptions. EXACT evals, DEPLOYED (the shipped acting states) / RAW
(trainer statistics) / external-calibration recal:
  iter      45     50     55     60     65     70     75     80     85     90
  deployed -14.3  -13.6  -13.1  -13.1   -1.0   -3.8  +10.1   +9.5  +20.4  +18.4
    won     0/4    0/4    0/4    0/4    1/4    1/4    3/4    3/4    4/4    4/4
  raw      -20.2  -12.5  -12.8   -8.4   -6.2   +7.8   +2.7   +8.7  +17.5  +16.2
    won     0/4    0/4    0/4    0/4    1/4    3/4    2/4    3/4    4/4    4/4
  ext recal                     -9.0   +2.7   -0.4   +3.4  +10.7  +19.6  +20.5
READ (S): WORKS by the letter -- no deployed row below 0 after the deployed
first success, and deployed mean 60-90 +5.8046 >= raw +5.4667. Weakly
informative: the raw series did not collapse after its own first success
either (this seed had no late BN collapse to prevent); the deployed policy
never did worse than raw after @70 and ended higher (+20.4/+18.4 vs
+17.5/+16.2).
READ (T): first success (deployed) @75 -> NOT repeated by the letter (<= 70);
the raw series reached 3/4 phases @70.
READ (2): deployed mean 60-90 = +5.8046 -> NO WORSE (>= +3.68); raw +5.4667
(NO WORSE); external recal +6.7669.
READ (3): 0 collapses after the first success.
THE TEMPERATURE CONFIG ACROSS TWO SEEDS: both broke out (s0 @70 raw;
s1 @75 deployed / @70 raw) -- the first policy-only configuration to break
out on 2/2 seeds (the unscaled config: 2/4). Breakouts still later than
search acting's (@50-65). Final rows +18.4/+20.4 -- near EfficientZero's
published 20.1.
PRE-REGISTERED INTERPRETATION: S works, T does not -> acting quality is the
suspect -> low-sim search acting (8-16 sims fit a 4-frame decision). The
audited-file freeze is lifted; the deferred items (bnrecal chain test,
tempscale LOW-2/3, seqref L1/L2) can be done now.

### Next experiment (pre-registered) -- the BN stabilizer on the temperature config (Kai, 2026-10-08)

DECISION (Kai, 2026-10-08: "Sure"): build the BN recalibration stabilizer
and run it on the temperature config, after the temperature arm's raw
mean (+2.70) sat 6.1 points under its recalibrated mean (+8.78) because of
snapshot-level BN collapses (@55, @80, @90).
MECHANISM (BN_RECAL_ACTING, new, default off; --ray, policy-only acting
only): at every weight ship to the CPU actor and the eval actor (each
iteration boundary), the BN running statistics of the REPRESENTATION and
PREDICTION nets are re-estimated from root-only forward passes
(momentum=None, BN in train mode, no grad) over the newest
BN_RECAL_FRAMES (4096) clean stacked observations in the replay buffer
(newest trajectories first, deterministic stride -- no RNG draws). Only
the copies shipped to the actor / eval actor carry them. Unchanged: the
trainer's nets, the target nets, birth search (it keeps the raw
request-time states -- search runs the prediction head on dynamics latents
at depths > 0, where root-only statistics would be wrong), refresh, and
training. Snapshots also store the shipped acting states, so the exact
probe can evaluate the DEPLOYED policy.
ARM effzero-repro-faithful-prioract-noredo-temp05-bnrecal = the temperature
arm + BN_RECAL_ACTING. SEED 1 (not 0: seed 0 would share the temperature
arm's initial weights and env seeds, so it would not be an independent
second sample of the temperature config).
PRE-REGISTERED READS (exact prior-argmax evals, iters 45-90):
  (S) the stabilizer read, on the DEPLOYED policy (exact eval with the
      snapshot's acting states): WORKS if no deployed row after the
      deployed first success falls below 0 AND the deployed mean 60-90 is
      >= the raw (trainer-statistics) mean 60-90 on the same snapshots;
      otherwise it does not. UNDETERMINED if the deployed series has no
      first success (the first clause would be vacuous) -- audit r1 L5,
      worded before any row existed. (Also reported: the external-calibration
      recal probe, for comparison with every earlier run.)
  (T) the temperature read, as a second seed of the temperature config:
      first success (deployed) at <= 70 -> the early breakout repeats;
      >= 75 or never -> it does not.
  (1)-(3) the standard reads and bands on the deployed series (primary);
      raw as secondary.
INTERPRETATION (single seed, hedged): S works and T repeats ->
temperature + stabilizer becomes the base config (then SVE on it, or a
further seed -- Kai's call); S works and T does not -> acting quality is
the suspect -> low-sim search acting (8-16 sims fit a 4-frame decision);
S does not work -> BN needs a structural fix (e.g. LayerNorm) -> Kai's
call. Caveat: the actor plays with recalibrated statistics, so the
behaviour (and thus training data) can differ from the temperature arm's
-- by design; (T) compares configurations, not identical training.
AUDIT ROUND 1 (2026-10-08) changed, before launch: the behaviour
diagnostics (beh_prior_maxp, beh_H_act, beh_prior_agree, beh_kl_target_
prior) now use the prior the actor SAMPLED (the acting nets), while birth
search still searches with the raw nets (M1); the row fields
policy_entropy / maxp stay RAW-statistics diagnostics, return_mean is the
deployed policy. The recalibration uses INTERLEAVED batches (frames[b::nb])
-- contiguous batches of one game's stretch under-estimated the variance
(x0.96, L1) -- so its estimator differs slightly from the offline
external-calibration probe (contiguous), which stays as it was for
comparability with every earlier run. Probe records gain "variant"
("raw" / "acting") (L3); the supervisor refuses any seed but 1 (L4).
AUDIT ROUND 2: nothing above LOW -- the loop closes. Verified end to end on
a real game: the logged behaviour prior equals the actor's pre-temperature
prior to 8e-7 (the raw prior would differ by TV 0.16-0.24); interleaved
recalibration matches exact full-batch statistics (variance ratio 0.999,
argmax agreement 0.999 at @80/@90; contiguous gave 0.94-0.96). Wording,
precisely: the calibration set is a 4096-frame stride sample of the newest
whole trajectories totalling >= 4096 frames; the row key is policy_maxp;
the very first ship (empty buffer) carries raw statistics. DEFERRED until
after the run (an audited test file; a mid-run change would block a
relaunch): a request->collect chain test pinning the middle of the M1
wiring (r2 L-1; the auditor's version is in scratchpad audit_bn2).

### RESULT -- the temperature arm (PRIOR_ACT_TEMP_SCALE 0.5), 2026-10-08

pong-tempscale-s0-2026-10-05 ran 10-07 09:31 -> 10-08 09:56 (paused by Kai
19:48-21:07 at the iter-48 checkpoint; PC shutdown after the 23:02 iter-54
checkpoint, resumed 01:33 -- both relaunches with --kai-override). EXACT /
BN-recalibrated:
  iter    45     50     55     60     65     70     75     80     85     90
  raw    -7.8   -2.0  -19.5   +2.6   +2.5  +11.6  +13.9   -2.1   +9.2  -18.9
  won    0/4    1/4    0/4    2/4    2/4    4/4    4/4    1/4    4/4    0/4
  recal                       +1.7   +5.0   -0.9  +11.1  +17.8  +10.3  +16.4
READ (H): first success @70 (+11.6, 4/4) -> by the letter NOT supported
(the criterion was <= 65; @60/@65 won 2/4 phases, short of 3/4). It is the
earliest policy-only breakout so far (prioract2 @75, seqref @80) and was
competitive from @50 (-2.0, a phase won), but not at search acting's
@50-65.
READ (2): mean 60-90 = +2.6951 -> INCONCLUSIVE; recalibrated +8.7809 ->
NO WORSE -- the best policy-only recalibrated mean (prioract2 s0 +5.01,
seqref s0 +0.16), between the control seeds' (+13.94 / +6.76).
READ (3): two raw collapses after the first success, @80 (-2.1) and @90
(-18.9), both the BN eval artifact (recal +17.8 and +16.4, 4/4); also a
pre-success raw collapse @55 (-19.5). The raw/recal gap (+6.1 on the mean)
is now the largest drag on the primary read. The behaviour windows did
not collapse with them (beh_prior_maxp 0.43-0.46, act_agree 0.55-0.60
throughout), so these are snapshot-level eval-statistics drops.

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
