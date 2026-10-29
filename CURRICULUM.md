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
- DECISION 2026-10-21 (Kai: "I'd like the distillation to be implemented
  and ran after the current run"): a DISTILLATION STEP at the generation
  swap, before the student plays. The master's games, recorded at raw
  4-frame resolution with its token boundaries, are re-tokenized into the
  student's vocabulary (a dropped token's frames become raw 4-frame steps
  -- the 10-17 rule) and become the student's first replay; the student
  trains on them offline with the master's SEARCH policies as its policy
  targets, then plays as usual. The student's VALUE is learned by its own
  TD on that data, never pulled toward V_m, so the deficit
  delta = max(0, V_m - V_s) stays an honest measurement. Re-tokenizing
  raw-step recordings is the same pipeline that later carries other
  players' games (P4 / P5) into a vocabulary. First test: "the
  distillation step" experiment below (no deficit exploration yet -- one
  change at a time).
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

### Next experiment (pre-registered) -- Test C: BBF-style resets in one 100k run, against a no-reset twin (Kai, 2026-10-29)

DECISION: Kai 2026-10-29, while Test B2's T0 trains:
  - Test C runs resets "alone first". BBF's n-step and discount ramps are
    left for a follow-up.
  - It gets a no-reset control twin: "a no reset control isn't a bad
    idea".
Background:
  - In Test B, after F0, every generation is flat within itself: -8.2 to
    +2.1 deployed per 1k env steps. The progress came at the swaps. One
    reading is lost plasticity: a network that stops improving on fresh
    data. A swap is a full reset that passes knowledge on by distillation.
    BBF instead resets part of the network on a schedule and keeps the
    replay buffer.
  - Our stack shows signs of it. Under SGD with coupled weight decay, the
    faithful Pong runs had lost up to half of the representation's
    first-layer filters, and about 8% of all BN channels, by iteration 85
    (redo.py's docstring). TKX, one TK run to 270 iterations, plateaued.
  - Against: ReDo, a gentler plasticity method (recycling dormant units
    every 1,000 steps), HURT on Pong under search acting (searchact -11.37
    against noredo +5.86). Every arm since runs without it.
QUESTION: in one 100k env-step run from scratch, do periodic BBF-style
resets reach a higher deployed score at 100k than the same run without
them?
BBF'S RESET (verified 10-29 against the released bbf/configs/BBF.gin and
bbf/agents/spr_agent.py):
  - every 40k gradient steps (the config's reset_every 20,000 agent steps
    at 2 updates per step; the paper: "reset every 40k gradient steps"),
    the encoder and transition model move halfway to a fresh random init
    (shrink_perturb_keys "encoder,transition_model", shrink_factor 0.5,
    perturb_factor 0.5; SR-SPR used 20%);
  - every other layer (the Q head, the SPR projection and predictor) is
    re-initialized;
  - the target network is reset with them (reset_target);
  - the optimizer state is re-initialized except for the kept layers';
  - resets continue to one interval before the end (no_resets_after =
    the run's 100k).
THE RUNS (seed 0; one generation each; every env step counted):
  C0 (control) is T0's recipe, run from scratch to 100,000 env steps:
    - TK's 36 tokens, policy-only acting, PREFIX_BN_PER_STEP, the own-play
      log;
    - the sign clip c0 = 1 throughout and DISCOUNT 0.98805 per base step;
    - the env-step stop: the final row is the iteration that passes
      100,000.
    The within-run schedules are keyed to the run's env-step progress over
    a nominal 150 iterations (Test B's 30 per 20k, scaled); the
    --iterations cap is 300. With no swap, the clip and the 36-token
    vocabulary never change.
  C (resets) is C0 plus resets. A reset falls at the end of every 20th
    iteration (every 40,000 gradient steps at 2,000 per iteration), while
    the run has played at most 80,000 env steps:
    - SHRINK AND PERTURB (BBF's encoder and transition model):
      representation and dynamics. Every parameter becomes 0.5 x itself +
      0.5 x a fresh draw of its layer's initializer (BN gamma and beta
      included: fresh 1 and 0). BN running statistics are kept; training
      re-estimates them.
    - RE-INITIALIZED (BBF's head, projection and predictor): prediction
      (policy and value), value_prefix (the reward LSTM and its head) and
      the projector, BN running statistics included.
    - THE TARGET MODEL is set to the reset online network (BBF's
      reset_target). Its copies every 200 steps then resume.
    - THE OPTIMIZER: SGD momentum is zeroed for the re-initialized
      parameters and kept for the shrink-and-perturbed ones, as BBF keeps
      the copied layers' state. The 1,000-step linear lr warmup restarts.
      This is OUR ADDITION: EfficientZero's SGD at lr 0.2 starts fresh
      weights with it; BBF's Adam has no warmup.
    - THE ACTOR HOLD, also OUR ADDITION: we act from the policy head, which
      a reset re-initializes. For the 10,000 gradient steps after a reset
      (5 iterations; BBF's post-reset annealing window), weight ships to
      the actor carry the last pre-reset weights and their recalibrated
      acting BN statistics. The env-step count still syncs. The birth
      search, the refresh and every target keep using the learner's
      online and target nets as always, so no target comes from the
      pre-reset network.
    Resets fall after iterations 20, 40, 60 and so on: about 6 in 100k env
    steps at ~700 env steps per iteration, the last at 66k-80k. In each
    cycle, row 20 is the pre-reset network and row 25 the network at the
    hold's end.
  The runs go one after the other, C first, since the host's memory holds
  one trainer.
READS. EXACT eval as before (whole games, 30 noop starts, prior argmax,
PROBE_ENV MsPacmanNoFrameskip-v4); DEPLOYED primary, trainer-BN secondary.
Rows every 5 iterations, plus the final.
  (1) PRIMARY at 100k: C's deployed rows in [90k, C's end] against C0's
      rows in [90k, C0's end]. The band is 2 x 216.9 x sqrt(1/n_C +
      1/n_C0): BETTER / NO DIFFERENT / WORSE. Trainer-BN likewise.
  (2) descriptive, beside Test B, Test B2 and the current lineage:
      - the deployed mean per 20k window of cumulative env steps (Test B's
        RESULT's definition);
      - C0's [90k, end] window against Test B's (1015.9, n = 4), T2's and
        the current lineage's (1163.8), with the same band rule. This asks
        whether one long run matches the swap lineages; C0 keeps the sign
        clip, so the clip differs.
  (3) per reset (C):
      - the dip: row 25 of the cycle against row 20;
      - the recovery: the first row at or above row 20;
      - the cycle's improvement rate: OLS deployed per 1k env steps over
        its rows 25-40, against C0's rate over the same rows.
  (4) plasticity, both runs, every row:
      - the dormant-unit fraction per net, ReDo's score at <= 0.025 and at
        <= 0.1. It is scored on a fresh replay sample drawn from a
        dedicated random generator, on copies of the nets, and nothing is
        recycled, so training is unchanged.
      - the representation's first-layer filter norms (the fraction below
        10% of their init norm);
      - the logged erank.
  (5) the prior: policy_maxp and the search targets' entropy by row.
  (6) mechanism: probe_prefix_bn FIXED, as in PB, at each run's final row
      and at C's rows 20 and 25 of its first and last cycles.
  (7) stability: the largest adjacent deployed drop. C's reset dips are
      reported separately, in (3).
DECISION AFTER:
  - (1) BETTER: resets help in our stack. Next is Kai's call among:
    BBF's n-step and discount ramps on top; resets inside the swap lineage
    (or a shrink-and-perturbed student); a higher replay ratio.
  - (1) NO DIFFERENT: resets alone do not pay at this resolution. Kai's
    call between the ramps bundle and Test D.
  - (1) WORSE: resets as specified hurt here, as ReDo did. Reads (3) and
    (4) say why: the dips, the recovery times, and whether the dormant
    fraction moved.
CAVEAT: two parts of the reset are ours, the actor hold and the warmup
restart. The hold changes the data C collects for about 25% of its steps.
C and C0 differ in nothing else. One seed; the band is the within-run row
noise.
BUILD (before launch; audited like every build):
  - the trainer: the reset schedule (every 20 iterations, up to 80k env
    steps); the per-net shrink-and-perturb and re-init; the target model;
    the optimizer state and the warmup restart; the actor hold; the reset
    state in the checkpoint, so a resume neither repeats nor skips a reset
    and resumes a hold mid-way; the dormant-fraction and filter-norm row
    diagnostics (both arms).
  - the arms C0 and C (ablation.py); a supervisor (C, then C0 once C is
    complete and evaluated); a hand-off over both runs; a launcher; tests;
    a manifest and a fresh run tree.
COST: the build and its audit loop while Test B2 trains (B2 ends about the
evening of 10-30), then about 28 h per run: results about 11-02.
AMENDMENTS (the build, before launch):
  (A1) THE STORED TARGETS. The text above says "no target comes from the
      pre-reset network". That holds for every NEW target: the birth
      search, the refresh, the value bootstraps through the target model.
      It does not hold for the policy targets already in the replay at a
      reset. Those are the pre-reset network's birth searches. They stay,
      as BBF keeps its replay, until the refresh re-searches them (12 lives
      every 2 iterations, so a full pass takes about 25 iterations) or
      they are evicted (the buffer turns over in about 50 iterations). So
      for much of each 20-iteration cycle the new heads also learn from the
      old network's searches. This makes our reset softer than BBF's,
      which stores no policy targets. Re-searching the whole buffer at a
      reset was rejected: the fresh heads' near-uniform searches would
      replace good targets with noise for about 25 iterations. PER
      priorities are kept too.
  (A2) NAMES. C is r0: arm ...-tokens-reset-r0, run
      mspacman-reset-r0-s0-2026-10-30. C0 is n0: arm ...-tokens-long-n0,
      run mspacman-long-n0-s0-2026-10-30. They are the chain supervisor's
      one-run lineages "r" and "n". n0 starts "after" r0, once R0 is
      complete and its exact evals have landed.
  (A3) READ (4)'s references:
      - The filter read compares each first-layer filter with 10% of the
        median filter norm of a fresh draw of that layer (Conv2d's default
        init). The run's own init norms are not kept.
      - The dormant fractions are over ReDo's units in all five nets
        (single-consumer layers; residual-stream producers excluded), the
        projector included.
  (A4) THE HOLD ships the actor's last pre-reset weights, with their
      BN-recalibrated acting statistics. The eval rows and the snapshots
      always take the learner's. The behaviour diagnostics' prior follows
      the hold.
BUILT 2026-10-29 (pre-audit; the audit loop follows before any launch):
  - muzero.py: the RESET_* knobs and DORMANT_DIAGNOSTIC (all off by
    default); LR_WARMUP_ORIGIN (0, so every other arm's warmup is
    unchanged).
  - experiments/ablation.py:
      - the arms ...-tokens-long-n0 (C0) and ...-tokens-reset-r0 (C);
      - reset_gate, run at startup;
      - reset_due, fresh_networks (a dedicated torch RNG keyed on the seed
        and the reset index), bbf_reset;
      - the reset lands after the row's snapshot and eval request and
        before its checkpoint; the in-flight game is birth-searched by the
        reset nets;
      - RayPipeline._actor_states (the hold): sync and the behaviour prior
        use it, the eval actor and the snapshots do not;
      - the checkpoint's "resets" (the log and the hold, for reset arms
        only), restoring the warmup origin on a resume;
      - warmup_origin around each run;
      - the rows' "resets" / "acting_held" (reset arms) and the plasticity
        read (Run.dormant_stats: dormant_025 / dormant_10 per net and
        overall, conv1_weak_frac);
      - Run.uniform_windows: ReDo's sampler extracted and shared, its draws
        unchanged.
  - benchmarks/supervisors/supervise_testb.py: the lineages "r" and "n"
    (RESET_KNOBS pinned by the quick probe); "after" with upstream(); the
    "after" run's names bound with the lineage's; run_name dated by the
    variant's own lineage; AFTER ABANDONED; tests/test_testc.py among the
    suites (25).
  - Tests: tests/test_testc.py (new); test_supervise_testb.py (the Test C
    lineages; 25 suites); the arm lists in test_pbn.py and test_harness.py.
  - Scratchpad: post_testc.py (generated from post_testb2.py by
    make_post_testc.py), check_post_testc.py (CHECK OK), launch_testc.ps1
    (it refuses until T2's marker, Test B2's hand-off's "all done" and no
    Test B2 process alive), smoke_testc.py (a tiny reset run; a kill after
    its first reset's checkpoint and a resume; a tiny control run; GPU,
    after Test B2).
AUDIT r1 (two fresh-eyes auditors on 25b41262, static: host memory stayed
below the test floor while T0 trains): A (the trainer core) and B (the
chain) found nothing above LOW but one MEDIUM:
  - B-MEDIUM: the smoke's stop (2,500 env steps) could end it at iteration
    4, before its second reset, failing a correct build.
  - A-LOW-1: fresh_networks seeded through torch.manual_seed, which also
    reseeds every CUDA generator; fork_rng(devices=[]) restores only the
    CPU one. The run would continue on a restarted CUDA stream (root
    noise) after each reset: deterministic, no bias, but not the claimed
    isolation.
  - A-LOW-2: reset_gate lacked refusals the registered arms cannot reach
    (LR_DECAY_AT, which a warmup restart would undo; search acting whose
    stored targets come from the actor; LR_WARMUP_STEPS 0).
  - A-LOW-3 / B-NIT-1: the row field acting_held was always False (the
    hold ends exactly on a row), so nothing showed the hold happened.
  - B-LOW-1: nothing watched the reset schedule while the runs train.
  - B-LOW-2: the scorer only pointed at Test B2's T2 window.
  - B-LOW-3: no documented path if Test B2 ends abnormally.
  - Test gaps: A-TEST-1 (a fake pipeline could not catch a hold that
    carries the reset weights), A-TEST-2 (a vacuous synced-at check),
    A-TEST-3 (no CUDA or Python-random check), B-NIT-2/3 (an unbounded
    sleep stub; the run date's own-lineage dating unpinned).
  - Notes, no change: A-NIT-1 (held games are stamped with their request
    step, under-aging them by 2-8k steps against a 90k-step horizon
    decay); A-NIT-2 (a reset's target stamp can equal the sync before it;
    every stamp-keyed cache is dropped explicitly); A-NIT-3 (iteration 20's
    refresh runs before the reset, so its 12 lives keep pre-reset targets
    for a full rotation -- within A1).
FIXES r1 (a fresh-eyes r2 follows):
  - the smoke: a reset every 3 iterations with a 2-iteration hold, so a
    held ship really happens; stop 5,000; its assertions follow the run's
    final iteration and read the trainer's own lines; the kill lands after
    the iteration-4 checkpoint, mid-hold;
  - fresh_networks seeds torch.default_generator only (+ test);
  - reset_gate refuses LR_DECAY_AT, LR_WARMUP_STEPS 0, and new targets not
    all from the birth search (+ tests);
  - acting_held is replaced by held_ships: the distinct steps at which the
    actor was shipped the held weights, checkpointed (a resume's re-ship
    counts once);
  - the supervisor checks each r0 row's resets and held_ships against the
    schedule (reset_alert: 4 held ships per reset) and prints both with
    the plasticity read;
  - the scorer computes T2's window and the trainer-BN columns, read (7),
    the dormant fractions per net, and lists rows lacking an eval;
  - the launcher documents the path if Test B2 ends abnormally (Kai's
    call), and that post_testb2.log must be kept;
  - the hand-off counts n0's renewed WAITING lines as live (B-NIT-5);
  - quick() pins LR_WARMUP_ORIGIN 0 and warmup_origin (B-NIT-7);
  - tests: the hold on the real pipeline (A-TEST-1), the target double
    buffer after a reset (A-TEST-2), the RNG checks (A-TEST-3), a bounded
    sleep, the run dates.
AMENDMENTS (audit r1):
  (A5) The rows carry "held_ships" (cumulative) instead of "acting_held".
  (A6) The CAVEAT's "25% of its steps" is about 20%: the hold changes
      iterations 22-25 of each cycle (4 ships, 12 lives). Iteration 21
      plays the pre-reset weights through the pipeline's normal
      one-iteration lag either way.
AUDIT r2 (fresh eyes on 97a846d3, static: memory below the test floor):
both auditors found the same MEDIUM, and LOWs:
  - MEDIUM: held_ships counted 5 for a reset after a resume from that
    reset iteration's own checkpoint (every reset iteration is a
    checkpoint iteration at --checkpoint-every 2). The pipeline's
    construction sync re-ships, at the reset's step, what the run shipped
    there before the reset. No effect on training; the counter would be
    wrong and every later row would raise a false ALERT.
  - A-LOW-1: the r1 target tests still could not detect the loss of
    bbf_reset's snapshot or stamp lines: the reset's own target sync fired
    200 steps after the helper's.
  - A-LOW-2: BBF's reset_target is not "the target becomes the reset
    online network" (see A7).
  - B-LOW-1: the smoke never covered the cutoff, the supervisor's check on
    real rows, or a resume from a reset's own checkpoint.
  - B-LOW-2: the scorer's read (7) included C's reset dips (read (3)'s).
  - NITs: the gate's docstring, the run_one order the hold relies on (sync
    and the next request before the reset) unpinned, the supervisor
    docstring, the hand-off's live lines (RESUMING, STOP), the launcher's
    KEEP note, the scorer's notes and labels, the smoke's Tee and out dirs.
FIXES r2 (a fresh-eyes r3 follows):
  - the hold records its reset's step ("from"), and a held ship counts only
    after it (+ a test of the resume re-ship);
  - the target tests: the helper syncs 100 steps before the reset, and the
    double-buffer test runs in both orders (that, and the run's own: the
    refresh's sync at the reset's step);
  - the smoke: a reset every 4 iterations (each a row and a checkpoint, as
    in the run), the cutoff at 6,000 env steps (it bites mid-run), every
    row through the supervisor's own reset_alert, the kill at reset 1's
    own checkpoint (the MEDIUM's resume point), fresh out dirs;
  - the scorer's read (7) without the reset dips (shown apart), read (6)'s
    rows, notes when a trainer-BN verdict cannot be given;
  - the NITs above.
  (A7) THE TARGET: the pre-registration credits "the target model is set
      to the reset online network" to BBF's reset_target. BBF's code
      (spr_agent.py jit_reset, read 10-29 by audit r2a) draws a SECOND
      independent init for the target instead: its encoder and transition
      model become 0.5 x the target's own weights + 0.5 x that draw, and
      its other layers that draw. We copy the reset online nets into both
      buffers of our hard-lagged target and keep that. A random target
      unrelated to the online nets would make the next 200-400 steps'
      bootstraps come from an untrained value head of another network,
      and BBF's EMA target (tau 0.005) is pulled to the online net within
      about 1,000 steps anyway.

### RESULT -- Test B: a fast lineage from scratch, five generations F0-F4 in 100k env steps (seed 0), 2026-10-29

(Pre-registration: "Next experiment (pre-registered) -- Test B", below
Generation 2's RESULT; its interim notes per generation are there.)
RUN (from the run worktree Y:\SwitchPlayer-testb-run, detached at 77d5df40,
manifest from the audited 90862862): F0 05:33 10-28 -> F4's last evals
13:57 10-29 (32.4 h). Each generation ran in one attempt with no deaths;
every supervisor.err is empty, the hand-off raised no ALERT, and no eval
game was capped. Every master gate passed. Training steps per generation
were 68k / 76k / 72k / 74k / 74k, 364k in all, including the 4 x 20k
offline steps.
EXACT eval (30 noop starts, prior argmax). Every score is the unclipped
score of a whole game. DEPLOYED is primary; trainer-BN (the same snapshot
under the trainer's BN statistics, the pre-registration's "raw") is
secondary. cum = the lineage's cumulative env steps, in thousands.
       row     0     5    10    15    20    25  final
  F0   cum     -   2.8   6.1   9.1  12.2  15.6  18.1 (30)  20.5 (34)
       dep     -   299   353   544   573   593   657        538
       tBN     -   264   392   883   713   949   757        729
  F1   cum  20.5  24.1  27.2  30.2  34.1  37.7  40.0 (28)
       dep   872   946   967   974   945   922   962
       tBN   683   788   735   846  1142   764   898
  F2   cum  40.0  43.9  47.9  51.6  55.6  59.4  60.1 (26)
       dep   746   799   923   865   846   772   834
       tBN   884   714   840   932  1162  1112   921
  F3   cum  60.1  64.7  68.0  71.4  75.3  79.0  80.6 (27)
       dep  1037  1032  1222   968   854  1072   851
       tBN   811   787  1022  1532   772   812   803
  F4   cum  80.6  83.8  87.2  91.7  95.4  99.1 100.5 (27)
       dep  1005  1174  1091   952  1045   873  1194
       tBN  1253  1103  1068  1203   944  1274  1202
READ (1) HEADLINE at 100k: F4's rows in [90k, 100.5k] are rows 15 / 20 /
  25 / 27 (n = 4): 952.3 / 1045.0 / 872.7 / 1193.7. Their mean is 1015.9
  deployed (g 0.728), against the current lineage's 1163.8 (DS2 rows 0 /
  5 / 10; g 0.879): -147.9 -> NO DIFFERENT (band +-331.3). Trainer-BN is
  1155.6 against 1259.9: -104.3 -> NO DIFFERENT (band +-331.3).
  References, not verdicts: EfficientZero 1281.2, BBF 2008.3, EZ-V2
  2251.0. Test B's window is below all three (one seed).
  Descriptive, the curve against the current lineage: Test B is at 538
  @20.5k, 962 @40.0k, 834 @60.1k, 851 @80.6k and 1194 @100.5k. G0 was at
  740 @19.8k, 1154 @39.5k and 1190 @59.2k, then the 31.8k recording, then
  DS2 at 1065 @91.0k.
  Means per 20k window of cumulative env steps. A window holds every
  in-run row whose cumulative env step falls in it, the students' row 0
  included (Test B2's read (7) uses the same definition):
      window             0-20k  20-40k  40-60k  60-80k  80-100.5k
      Test B deployed      503     881     845    1003    1023
      current deployed     540     896    1107       -    1164 (91-98k)
      Test B trainer-BN    660     812     934     951    1106
      current trainer-BN   524     790    1147       -    1260
  (The Test B2 pre-registration quoted F1 953 and F2 840 for 20-40k and
  40-60k. Those are generation means over rows 5 to final; the strict
  windows are 881 and 845.)
READ (2) per swap: transfer is row 0 against the master's final row; the
  jump is the student's rows 5 and 10 against the master's last two rows;
  the rate is the student's OLS deployed rate per 1k env steps.
      swap      transfer   jump   student rate
      F0 -> F1      +334   +360           +2.1
      F1 -> F2      -217    -82           +1.5
      F2 -> F3      +203   +324           -8.2
      F3 -> F4      +154   +171           -2.1
  F0's own rate was +16.8.
READ (3) the clip, the discount, the vocabulary and the swap data:
      gen  clip   discount  tokens  reward_kept  set records  lives by temp
      F0   1      0.97          36  0.98 -> 0.77           -  51 / 24 / 27
      F1   11.62  0.985         18  0.86-0.88         10,418  48 / 18 / 18
      F2   18.10  0.9925        16  0.93-0.94         10,102  42 / 18 / 18
      F3   24.02  0.99625       16  0.96-0.97          6,601  42 / 18 / 21
      F4   32.07  0.997         16  0.975-0.977        4,671            -
  The set records column is the distillation set built from that
  generation's master's log. "Lives by temp" counts that generation's own
  logged lives at temperature 0.5 / 0.25 / 0.125; F4's log was never
  read, since no swap followed it. Every value check passed: the largest
  discounted return was 28.8 / 46.5 / 73.6 / 71.1 units, against a
  5,000-unit support.
READ (4) the prior: policy_maxp by generation:
  - F0: 0.49 -> 0.64;
  - F1: 0.70 -> 0.55;
  - F2: 0.62 -> 0.49;
  - F3: 0.53-0.59;
  - F4: 0.57-0.65.
  The search targets' entropy is 0.82-1.45 / 1.20-1.52 / 1.23-1.58 /
  1.34-1.59 / 1.20-1.63. There is no collapse.
READ (5) mechanism: FIXED at every probed row. p1_ratio at row 0 / final:
  F0 - / 0.76, F1 0.97 / 0.56, F2 0.93 / 1.05, F3 1.04 / 0.78, F4 0.47 /
  0.67.
READ (6) stability: the largest adjacent deployed drop within each
  generation is -119 / -29 / -74 / -254 / -172. Across the swaps, row 0
  against the master's final row is +334 / -217 / +203 / +154. Every
  value is inside the one-row band (+-464).
PRE-REGISTERED DECISION: (1) NO DIFFERENT -> Kai's call: the same level
  at no recording cost. Kai's call (10-29) is already made: Test B2 (two
  swaps at a constant 0.98805, pre-registered above) runs next.
Beyond the letter (one seed, within-run bands, so "consistent with"):
  - Data cost: the current lineage's 1164 window took G0's 59.2k, a 31.8k
    recording and DS2's first 6.7k. Test B's 1016 took 100k of its own
    play and no recording. It is 148 lower, inside the band.
  - The gains come at the swaps. After F0, every generation is flat or
    falling within itself (+2.1 / +1.5 / -8.2 / -2.1). The jumps sum to
    +773 (+360 / -82 / +324 / +171). That is more than the lineage's whole
    progress from F0's final row to F4's (538 -> 1194, +656), so the drift
    within generations is net negative. Each 20k generation spends most of
    its online steps on a flat stretch. Test B2 asks whether a longer generation improves within
    itself or only adds to the flat stretch.
  - The gap to the current lineage opens at 40-60k (845 against 1107),
    where G0 was still improving within its own generation. F0 at
    discount 0.97 kept pace with G0 over 0-20k (503 against 540), so the
    low starting discount shows no visible cost.
  - The tokens lengthen each generation. Frames per decision in play are
    6.8 at F0's end, then 7.1-10.4 (F1), 11.7-13.1 (F2), 15.4-19.5 (F3)
    and 18.1-22.8 (F4). The distillation sets shrink with them (10,418 ->
    4,671 records over about 20k base steps each), while the offline
    phase stays 20,000 steps.
  - Deployed against trainer-BN: the deployed acting states are not
    consistently ahead in this lineage. Trainer-BN is higher in F2 (rows
    5-26: 947 against 840) and F4 (1132 against 1055), lower in F1 and F3.

### Next experiment (pre-registered) -- Test B2: two swaps in 100k env steps at the original constant discount (Kai, 2026-10-29)

DECISION: Kai 2026-10-29, while Test B's F4 trains: "two swaps with
constant 0.988 after Test B". Kai asked whether Test B learns more slowly
and whether swapping every 20k is too often. The interim reads behind the
question:
  - After the first generation, Test B's generations barely improve
      within themselves: rates +2.1 / +1.5 / -8.2 deployed per 1k env
      steps for F1-F3, against F0's +16.8. The gains come at the swaps
      (jumps +360 / -82 / +324).
  - Against G0, deployed means per 20k window of cumulative env steps:
      0-20k G0 540 vs F0 503; 20-40k 896 vs F1 953; 40-60k 1107 vs F2 840.
      The gap opens at the second swap.
  - The distillation sets shrink as the tokens lengthen: 10,418 / 10,102
      / 6,601 / 4,671 records.
  - The discount: the token arms' original is 0.997 per frame, i.e.
      0.997^4 = 0.98805 per base step, constant over G0 -> DS2 -> G2.
      Test B's ladder ran 0.97 -> 0.997 per base step. F0 at 0.97 kept
      pace with G0, so the low start did not visibly hurt. The constant is
      chosen because the ladder is tied to the swap count (with two swaps
      it would stop at 0.9925) and because 0.98805 is the setting behind
      every positive result so far.
QUESTION: on the same 100k env-step budget, do fewer and longer
generations (two swaps, three generations of ~33.3k env steps) at the
original constant discount reach a higher deployed score at 100k than
Test B's four swaps? And than the current lineage?
THE LINEAGE (seed 0; three generations T0-T2; every env step counted):
  T0 is from scratch: Test B's F0 recipe (TK's 36 tokens, policy-only
    acting, PREFIX_BN_PER_STEP, sign clip c0 = 1, the own-play log) at
    DISCOUNT 0.98805 per base step. It runs until 33,333 of its own env
    steps.
  T1, T2: each is the distilled student of the previous generation's final
    snapshot, through Test B's swap unchanged: the master's own play over
    its whole generation (~33k base steps), the halved-clip rule, the 1%
    vocabulary rule, the master's 50-sim birth search targets, 20,000
    offline steps, then online play. THE ONE CHANGE: the student's
    discount equals the master's (0.98805 at every swap). The value check
    (A2) still runs.
  THE STOP (Test B's A1): generation k's --stop-env-steps is
    round(100,000 x (k + 1) / 3) minus the final env steps of T0..T(k-1):
    33,333 for T0. T2 ends at the first iteration past 100k cumulative.
  WITHIN each generation, TK's schedules restart over its env steps, as in
    Test B. The nominal run is 50 iterations (Test B's 30 for 20k, scaled
    by 5/3); the --iterations cap is 100.
  MASTER GATE at each swap: Test B's (the master's last two deployed rows
    averaging NEAR RANDOM, < 502.08, stop the chain; Kai's call).
READS. EXACT eval as before (whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4); DEPLOYED primary, trainer-BN secondary.
Rows every 5 iterations of each generation, plus each generation's final
snapshot and the students' row 0 and offline curves. Cumulative env steps
are the lineage's own online env steps.
  (1a) PRIMARY at 100k: the mean deployed score of T2's rows in [90k, T2's
      end] cumulative env steps (its final included; n rows) against
      Test B's rows in the same window (F4's, its final included; n_B
      rows). The band is 2 x 216.9 x sqrt(1/n + 1/n_B): BETTER / NO
      DIFFERENT / WORSE. Trainer-BN likewise.
  (1b) against the current lineage, as in Test B's (1): DS2 rows 0 / 5 /
      10 (mean 1163.8; trainer-BN 1259.9), band 2 x 216.9 x sqrt(1/n +
      1/3). References, not verdicts: EfficientZero 1281.2, BBF 2008.3,
      EZ-V2 2251.0.
  (2)-(6): Test B's reads (2)-(6) per swap and per generation (transfer,
      jump and the OLS improvement rate; the clip, the discount, the
      vocabulary, reward_kept, the set's size and temperature mix; the
      prior; read (5) probe_prefix_bn at each student's row 0 and each
      generation's final row; stability).
  (7) descriptive: the deployed mean per 20k window of cumulative env
      steps, beside Test B's and the current lineage's (G0 540 / 896 /
      1107 at 0-20k / 20-40k / 40-60k; DS2 1163.8 at 91-98k).
  (8) descriptive: T0 against G0 over the first 33k env steps (the same
      discount; T0 adds PREFIX_BN_PER_STEP, the own-play log and schedules
      compressed to 33k).
DECISION AFTER:
  - (1a) BETTER: fewer, longer generations beat four swaps. The generation
    length becomes an input to Test D's trigger (a minimum length).
  - (1a) NO DIFFERENT: the swap count at this resolution is not the lever.
    Kai's call between Test C and Test D.
  - (1a) WORSE: more frequent swaps are better; Test D can explore
    shorter generations.
  CAVEAT: against Test B two things change, the swap count (4 -> 2) and
  the discount (the ladder -> constant 0.98805), so (1a) is on the pair.
  Against the current lineage, the swap timing (one swap after 59k plus a
  31.8k recording, against two at ~33k and ~67k with no recording) and
  the own-play data differ.
BUILD (before launch; audited like every build): the arms T0-T2; the
prep's constant-discount rule; Test B's chain supervisor, hand-off and
launcher generalized to a three-generation lineage; tests, a manifest, a
fresh run tree. No change to Test B's behaviour.
COST: the build and its audit loop while F4 trains, then ~1.3 days of
training, launched after Test B's RESULT.
AMENDMENTS (the build, before launch):
  (A1) THE PREFILL CAP. A master's whole generation (~33.4-34k base steps)
      is more than the replay holds (MAX_REPLAY_TRANSITIONS 31,250). The
      student's prefill therefore evicts the oldest master lives to the
      cap, as Test 2's DS2 did: roughly the master's first 3-4 iterations
      (its 0.5-temperature start). The prep's clip, vocabulary and value
      check still read the whole log. Read (3) adds the trainer log's
      "distillation set: ... evicted to the caps" line, the lives and base
      steps the student actually started from.
  (A2) The discount's value is 0.997^4 = 0.988053892 (0.98805; the text
      above first said 0.98807), bit-identical to G0's DISCOUNT.
BUILT 2026-10-29 (pre-audit): the arms T0-T2 (experiments/ablation.py
TWO_ARM: F0 / F1 with DISCOUNT 0.997^4, _nominal_iterations 50 and
_discount_rule "keep"); the prep's rule (distill.discount_rule /
swap_discount: the master arm's _discount_rule, Test B's arms none =
"halve"; vocab.json "discount_rule"), student_arm's check of a prep's rule
against its master arm's; supervise_testb.py generalized to a lineage
table (variant letter f = Test B, t = Test B2; configure(), target()); a
hand-off post_testb2.py, a launcher launch_testb2.ps1 (refuses until F4's
marker, Test B's hand-off's "all done" and no Test B probe, prep, trainer
or supervisor), a checker check_post_testb2.py and a smoke
smoke_testb2.py in the scratchpad;
tests in test_testb.py and test_supervise_testb.py, and the T arms added
to test_pbn.py's and test_harness.py's arm lists.
AUDIT r1 (two fresh-eyes auditors, 10-29): r1a (the core) one MEDIUM --
the T arms broke two preflight suites' arm lists (test_pbn, test_harness)
-- and LOWs: the prefill cap (A1), no student-side check of the prep's
rule; r1b (the chain) nothing above LOW -- the hand-off's echo list held
F names, the launcher only checked that Test B's hand-off exited, the
smoke lacked a trainer guard and keep checks. FIXES r1: all of them, plus
the NITs (swap_discount raises on an unknown rule; the bite test accepts
only AssertionError; a Test B2 run_prep test; T1's offline phase in the
slow test; the text's 0.98805). Not changed: comparing the manifest's
_commit to the run tree's HEAD (that HEAD is the manifest's own commit;
the preflight's hash check catches a stale manifest). The first suite run
then caught a test bug of the r1 fixes (the bite test's distill patch
leaked into later probes of the same child): fixed (its own child).
AUDIT r2 (fresh eyes, static: host memory below the test floor while F4
trains): r2a (the core) nothing above LOW -- LOW: no test of T2's rule
check against a STUDENT master (T1); NITs: an unknown master arm, a kept
discount past the cap only refused at the student, wording. r2b (the
chain) nothing above LOW -- LOW: the launcher's Test B process pattern
also matched a preflight's quick-probe children (a false refusal); the
smoke did not require Test B to be over; NITs: run folders created before
the checks, the manifest pins also matching Test B's tree, target() not
checking the lineage. FIXES r2: all of them (a T2 unit test and the
smoke's prep2 step taking prep2 as T2; student_arm refuses an unknown
master arm; swap_discount refuses a kept discount past 0.997; the
launcher matches only Test B scripts, checks the tree is Test B2's
build, creates the folders last; the smoke refuses until F4 is complete
and no supervise_testb.py is alive; target() refuses a variant of
another lineage).
AUDIT r3 (fresh eyes, the whole build, static): nothing above LOW --
LOW: the smoke's guard did not wait for Test B's hand-off to finish F4's
evals (the launcher's condition); NITs: the prep's two new refusals not
in PREP_CHECKS (retried, not stopped -- unreachable behind the quick
probe), the launcher not checking Test B's supervisors, the smoke's
trainer check missing module-form trainers. FIXES r3: all four (the
smoke now also requires post_testb.log's "all done" and no Test B
hand-off, probe or supervisor; PREP_CHECKS gains "_discount_rule " and
"past the cap", with tests).
AUDIT r4 (fresh eyes, the r3 fixes): nothing above NIT -- THE AUDIT LOOP
IS CLOSED (r2, r3, r4 nothing above LOW). ACCEPTED r4 (no edit, so no
further round): the smoke's script-name check has no python-only filter
(a non-python process naming those files makes it refuse: a refusal,
never a risk); the launcher's Test B supervisor pattern misses a
hand-started supervisor with a quoted path (Test B's launcher never
quotes); the launcher does not refuse while smoke_testb2.py runs (the
smoke runs to completion before the launch, by hand).
PRE-LAUNCH (10-29, after Test B's RESULT): the 24 suites all pass in the
build tree (13:58). check_post_testb2.py reports CHECK OK. The GPU smoke
(smoke_testb2.py, a tiny lineage to 900 env steps per generation) passes
every step: T0 -> prep1 (the discount kept at 0.988053892081) -> T1
(offline, then online to its stop) -> prep2 from T1's log (a student
master; kept again), and T2's student_arm takes prep2 at the same
discount.
LAUNCHED 14:03 10-29 from the run worktree Y:\SwitchPlayer-testb2-run,
detached at a8c83a58: testb_manifest.json rewritten from the audited build
(c57d2a16; seven AUDITED files changed against Test B's). In that tree,
before the launch: the manifest check and t0's quick probe passed. T0's
stop is 33,333. launch_testb2.ps1 passed its checks (F4's marker, Test B's
hand-off log ending "all done", no Test B process alive) and started the
hand-off post_testb2.py and the supervisors t0-t2. t0 runs its preflight
(the 24 suites in the run tree), then T0. t1 and t2 wait on their
predecessors. Expected: about 45-50 iterations per generation, about 28-30
h in all.

### RESULT -- Generation 2: DS2 as the master, one distilled student G2 (seed 0), 2026-10-28

(Pre-registration: "Next experiment (pre-registered) -- Generation 2",
below Test B's section; its interim notes there.)
RUN (from the run worktree Y:\SwitchPlayer-gen2-run, detached at ce5e1488,
manifest from the audited 48716683; one attempt; supervisor.err empty):
the preflight and prep 11:32 -> 11:34 10-27; G2 11:35 10-27 -> 04:48 10-28
(62,010 s; the prefill and the offline phase ~1 h 39 min, then ~10.4
min/iteration); the hand-off's last evals 04:54 10-28 -- every row (0 and
5-90, the offline curve 1-4), all five read (10) rows, no capped game. One
ALERT: host commit free 3.0 GB at row 10 (other applications; the trainer
held steady).
MASTER = DS2 weights-iter090 (deployed 60-90 1656.52, g 1.385). THE PREP
(above in the interim notes): 36 DS2 games (mean 2127.8 at temperature
0.125), 31,862 base steps; c1 11.258 -> c2 22.961; 18 of 21 tokens kept.
EXACT eval (30 noop starts, prior argmax); DEPLOYED primary, RAW (the
trainer's BN statistics) secondary, IN = the run's 8-game eval:
  iter       0    5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  G2 dep  1350 1925 1798 2244 1686 1824 2040 1828 1968 1997 1955 1665 2036 2315 2074 1910 2403 2259 1831
  G2 raw  1526 1268 1624 1610 1772 2001 1900 1868 1846 1961 1885 1897 1909 2028 2663 1885 1985 2615 2435
  G2 in      - 1629 1764 1878 1684 1694 2309 1882 1859 2230 1836 1555 1998 2280 2165 1966 2609 2185 1954
  G2 env     -  4.1  9.0 13.0 16.3 20.9 25.2 29.3 33.4 37.1 41.5 45.8 49.9 54.3 58.9 62.8 66.6 71.1 75.4
  G2 maxp    - 0.58 0.60 0.59 0.57 0.60 0.64 0.63 0.63 0.63 0.65 0.58 0.67 0.65 0.67 0.71 0.67 0.65 0.67
(env in thousands of G2's own base steps; every score is the unclipped game
score of a whole game.)
NEAR RANDOM: no (60-90 deployed g 1.859, raw 1.961).
READ (0) transfer: row 0 1350.3 deployed (raw 1526.3) against MASTER
  1656.52: -306.2, descriptive (+-464); NOT TRANSFER FAILED. Offline curve,
  deployed / raw: 1722 / 1398 @4k, 1498 / 1682 @8k, 1510 / 1725 @12k, 1645
  / 1674 @16k, 1350 / 1526 @20k. DS2's own recorded games: mean 2127.8.
READ (1) PRIMARY, against the master: G2 60-90 deployed 2118.24 (SD 213.9)
  against MASTER 1656.52: +461.7 -> ABOVE (band 232). Raw 2217.14 against
  1600.05: +617.1 -> ABOVE (band 205).
READ (2) head start 5-25: G2 1895.20 against DS2's 1330.27: +564.9 ->
  BETTER (band 274.4). Raw 1654.93.
READ (3) two generations: G2 g 1.859 against G0's 0.828 (+1004.4 points);
  DS2 was g 1.385.
READ (4) time to level (the first row >= 1424.52): row 5 (4,083 env
  steps). Every deployed row 5-90 is above MASTER itself (lowest 1665 @55).
  G2's lowest row in 5-25 is 1686.0 (@20) against row 0's 1350.3: no
  forgetting.
READ (5) tokens: the 3 dropped tokens held 1.5% of DS2's decisions. Play
  shares (4 / 8 / 16 / 32 frames): 0.14 / 0.03 / 0.19 / 0.64 @5 -> 0.11 /
  0.01 / 0.12 / 0.76 @90 (DS2 @90: 0.10 / 0.03 / 0.22 / 0.64); search
  shares 0.20 / 0.09 / 0.16 / 0.55 @5 -> 0.22 / 0.04 / 0.10 / 0.65 @90.
  Frames per decision, deployed, rows 60-90: 24.4-26.7 (2.2-2.5 decisions
  per second; DS2 22.6-24.5); row 0 23.2.
READ (6) stability, the largest adjacent deployed drop from row 45: -428
  (2259 @85 -> 1831 @90). Raw: -778 (2663 @70 -> 1885 @75).
READ (7) the clip: S(c1) 0.273, c2 22.961, S(c2) 0.136 on DS2's games (the
  prep). reward_kept by row: 0.87 @5 -> 0.91 @90, rising (DS2 at c1 fell
  0.83 -> 0.78).
READ (8) prior: R = maxp 75-90 / 30-55 = 0.674 / 0.627 = 1.08 -> HELD.
  Fresh-target entropy without trend (0.48-1.87; 0.48 a one-row dip @80).
READ (9) the plateau rule (Step 0 (ii)), after the fact: on deployed the
  noise-scaled rule fires at row 65 (sigma 224.4; best window then 35-65,
  1966.2). The rows after it averaged 2095.3, so this is a false plateau.
  The rank-based variant never fires. On raw neither fires (sigma 120.9;
  the last window, 60-90, is the best).
READ (10) mechanism: FIXED at every row. p1_ratio 0.88 / 0.51 / 0.64 /
  0.47 / 0.56 at rows 0 / 30 / 55 / 75 / 90. Against the pooled
  statistics the step-1 offset would be 1.52 / 10.23 / 8.51 / 8.18 / 7.66
  SDs. No caveat on reads (1)-(9).
PRE-REGISTERED DECISION: (1) ABOVE -> the loop compounds over two swaps:
  Kai's call on generation 3 and on the loop's automation. Kai's standing
  call (10-27): Test B (a from-scratch lineage with a swap every 20k env
  steps, pre-registered above) runs next. Generation 3 of this lineage is
  not scheduled.
Beyond the letter (one seed, within-run bands -- "consistent with"):
  - Both swaps passed their master by about two bands: G0 1114 -> DS2
    1657 (+543) -> G2 2118 (+462). The gain per swap is not shrinking much
    yet.
  - Data cost: G2's level came from about 270k cumulative env steps:
    - G0's 59.2k;
    - two 31.8k recordings;
    - DS2's 71.4k;
    - G2's 75.4k.
    EZ-V2 reports 2251.0 at 100k and BBF 2008.3, so G2's score is in their
    range, but on about 2.7x the data. Test B tests whether the same loop
    gets there inside 100k (own-play swap data, so no recording cost).
  - The offline phase: G2's deployed curve was best at 4k offline steps
    (1722) and lower by 20k (1350). DS2's rose to 16k (1326) and fell at
    20k (1065). On deployed, the last part of the 20,000 offline steps did
    not help either student. Test B keeps 20k: noted as a candidate lever
    for a shorter swap.
  - Token habits: G2 moves further onto the 32-frame tokens (0.64 -> 0.76)
    while keeping some 16-frame play (0.12), and acts 2.2-2.5 times a
    second.
  - The looser clip (c2 22.96, one ghost of 200 points whole) keeps more of
    a stronger player's reward: reward_kept rises through the run, where
    DS2's fell at c1.
  - The plateau rules fire on noise here (the noise-scaled rule at 65). In
    Test 2 the rank-based rule did (DS2 at 55). Neither rule yet tells a
    real slowdown from a noisy level stretch. This supports Kai's
    constant-swap choice for Test B, and gives Test D's smart trigger a
    concrete false-positive case to beat.

### Next experiment (pre-registered) -- Test B: a fast lineage from scratch (a swap every 20k env steps, own-play distillation, a per-generation discount) (Kai, 2026-10-27)

DECISION: Kai 2026-10-27, while Generation 2 trains. The comparison that
prompted it: Ms. Pac-Man at Atari 100k is EZ-V2 2251.0, BBF 2008.3 and
EfficientZero 1281.2. Our lineage G0 -> DS2 reaches 100k cumulative env
steps (G0's 59.2k + the 31.8k recording + DS2's own) at DS2 rows 5-15,
deployed 1200-1295. Kai's choices:
  - "make the swap constant for now ... after 20k environment steps each
    time" (the "large slowdown" trigger is discussed below, under Test D);
  - the discount copied from BBF, 0.97 -> 0.997, changed only at swaps;
  - the swap data is the master's own recent play;
  - a 100k env-step budget;
  - Generation 2 finishes first;
  - BBF-style resets are a later test (Test C).
Kai approved the plan's four recommendations:
  (1) the discount's gap to 1 halves at each swap, capped at 0.997;
  (2) the within-run schedules restart each generation, compressed to its
      20k env steps;
  (3) 20,000 offline steps per swap;
  (4) a build check that the value support holds the largest value.
QUESTION: with cheap swaps every 20k env steps, does a lineage from scratch
reach a higher deployed score at 100k cumulative env steps than the current
lineage (G0 -> DS2)?
THE LINEAGE (seed 0; five generations and four swaps; every env step
counted):
  F0 is from scratch. It uses TK's recipe (the 36 tokens, policy-only
    acting, PREFIX_BN_PER_STEP) at the sign clip (c0 = 1) and DISCOUNT 0.97
    per base step. A base step is 4 frames, the unit of BBF's discount; the
    token arms' current 0.997 per frame is 0.988 per base step. F0 runs
    until 20,000 of its own env steps.
  F1-F4: each is the distilled student of the previous generation's FINAL
    snapshot, the one from the iteration in which it passed 20,000 env
    steps. The swap is Test 2's / Generation 2's, with these changes:
      - THE DATA is the master's own play over its whole generation (~20k
        base steps). The master's trainer logs it at raw-step resolution,
        with the decision records, every inner frame and the raw
        per-base-step rewards, as distill.record captures them. There is no
        fresh recording, so a swap costs no env steps. The targets are the
        master's 50-sim birth search over its tokens (the masked read-out),
        with the master's spec (master_spec).
      - THE CLIP: the halved-clip rule on the logged games.
      - THE VOCABULARY: the 1% rule on the logged decisions.
      - THE DISCOUNT: the gap 1 - g halves, capped at 0.997, so 0.97,
        0.985, 0.9925, 0.99625, 0.997.
    Then 20,000 offline steps (snapshots every 4,000; the last is row 0),
    then online play until 20,000 of its own env steps.
  WITHIN each generation, TK's schedules restart and run over its 20,000 env
    steps, keyed to its env-step progress; the offline steps advance none
    of them. The schedules are the prior-acting temperature (0.5 / 0.25 /
    0.125, staged), the long-token bias (0.5 -> 0 by 25%), PER beta (0.4 ->
    1) and the off-policy horizon decay. A generation is ~25-30 iterations
    at 650-800 env steps each, ~50-60k gradient steps (BBF resets every
    40k).
  MASTER GATE at each swap: the master's last two deployed rows averaging
    NEAR RANDOM (< 502.08) stop the chain (Kai's call). For scale, G0 had
    740 at 19.8k env steps.
READS. EXACT eval as before: whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4; DEPLOYED primary, raw secondary. Rows are
taken every 5 iterations of each generation, plus each generation's final
snapshot and the students' row 0 and offline curves. Cumulative env steps
are the lineage's own online env steps (the swaps and the offline phases
add none).
  (1) HEADLINE at 100k: the mean deployed score of the lineage's rows in
      the window [90k, 100k] cumulative env steps (F4's rows there, its
      final included) against the current lineage's rows in the same window
      (DS2 rows 0 / 5 / 10 at 91.0k / 94.0k / 97.7k: 1065.0, 1226.3, 1200.0;
      mean 1163.8). The band is 2 x 216.9 x sqrt(1/n + 1/3) for the lineage's
      n rows (+-354 for n = 3): BETTER / NO DIFFERENT / WORSE. Raw likewise
      (DS2 raw 1119.3, 1022.7, 1637.7; mean 1259.9). References, not
      verdicts: EfficientZero 1281.2, BBF 2008.3, EZ-V2 2251.0 (multi-seed
      means at 100k agent steps; ours is one seed). Descriptive: the
      deployed curve against cumulative env steps beside the current
      lineage's (G0 740 @19.8k, 1154 @39.5k, 1190 @59.2k).
  (2) per swap: transfer (the student's row 0 against the master's final
      row), the jump (the student's rows in its first 10 iterations against
      the master's last two rows), and each generation's improvement rate
      (deployed score per 1k env steps, OLS over its rows), the input to
      Test D's trigger.
  (3) the clip, the discount and the vocabulary per generation;
      reward_kept by row; the logged set's size and its temperature mix.
  (4) the prior: policy_maxp and the search targets' entropy by row
      (generations are too short for Test 1's R windows).
  (5) mechanism: probe_prefix_bn at each student's row 0 and each
      generation's final snapshot, FIXED as in PB.
  (6) stability: the largest adjacent deployed drop within a generation,
      and across each swap (row 0 against the master's final).
DECISION AFTER:
  - (1) BETTER -> the fast lineage becomes the base; Test C (resets) builds
    on it.
  - (1) NO DIFFERENT -> Kai's call: the same level at no recording cost.
  - (1) WORSE -> diagnose before Test C: transfer at the early swaps (read
    (2)), the low early discount, the smaller distillation sets.
  CAVEAT: three things change at once against the current lineage (the
  swap timing, the own-play data, the discount schedule), so the verdict
  is on the package.
BUILD (before launch; audited like every build):
  - The trainer: a raw-resolution log of its own play; an env-step stop
    with a final snapshot at the stopping iteration; the schedules keyed
    to the generation's env-step progress; DISCOUNT per arm; the value
    support checked against the largest value (ceiling / (1 - g)).
  - The prep: distillation from a master's log instead of a recording;
    the discount rule (vocab.json "student_discount"); DISCOUNT among the
    student keys.
  - The arms F0-F4; a chain supervisor (each generation waits for the
    previous one's completion and evals); an eval hand-off over the five
    runs; tests, a manifest, a fresh run tree.
COST: the build and its audit loop first, then ~1.3 days of training,
launched after Generation 2 completes (~05:00 10-28).
AMENDMENTS (the build, before launch):
  (A1) THE STOP is cumulative, so overshoots do not accumulate. Generation
      k's --stop-env-steps is 20,000 x (k + 1) minus the final env steps
      of F0..F(k-1), each read from that run's final in-run row. F0's stop
      is 20,000. A generation ends at the iteration in which its own env
      steps pass its stop, and F4 ends at the first iteration past 100k
      cumulative. The headline window is therefore [90k, F4's end], which
      is within one iteration of 100k.
  (A2) THE VALUE CHECK is empirical. The worst case, ceiling / (1 - g),
      would exceed the value support (5,000 units) from F3 on, although
      Ms. Pac-Man's values stay far below it. The prep instead requires
      the value support to hold 2 x the largest discounted, clipped return
      at any base step of the master's logged games, under the student's
      ceiling and discount; it records that return in vocab.json
      ("value_max_student"). Failing the check is PREP FAILED (Kai's
      call).
  (A3) The schedules' planned run is the arm's nominal 30 iterations: it
      is the scale of the env-keyed fractions and of the 4.3 decay (0.3 x
      30 x 2,000 training steps). The --iterations cap is 60.
  (A4) The MASTER GATE uses the mean of the master's last two deployed
      rows, i.e. its two latest in-run iterations (as written above). The
      master's final snapshot is weights-iterKKK, with K its final
      iteration (its last in-run row).
  (A5) READ (5) runs on each student's row 0 and on each generation's
      final row K. K is known once the run is complete, so the hand-off
      adds that row in its final pass.
BUILT 2026-10-27 (pre-audit; the audit loop follows before any launch):
  - muzero.py: SCHEDULE_ENV_TARGET and SCHEDULE_ENV_STEPS. With a target
    and a planned run, schedule_steps() returns the env-step progress
    scaled to the planned run; otherwise it is bit for bit the old
    function. OWN_PLAY_LOG flag.
  - experiments/ray_actors.py: SelfPlayActor.set_weights takes the env-step
    count; under OWN_PLAY_LOG, play() returns each life's inner frames,
    real_done and acting temperature.
  - experiments/ablation.py:
      - the arms F0 (TK at the sign clip, PREFIX_BN_PER_STEP, DISCOUNT
        0.97, OWN_PLAY_LOG, _env_stop, _nominal_iterations 30) and F1-F4
        (the distilled student of the previous generation, with
        _discount_from_prep);
      - OwnPlayLog / read_own_play, one pickle file per iteration,
        truncated at a process's first write, so a resume replaces a dead
        timeline's lives;
      - env_schedule, env_stop_gate, own_play_dir, --stop-env-steps
        (part of the resume recipe);
      - the loop's LAST iteration is the one whose collect passes the
        stop; nothing is requested after it, and its weights, snapshot and
        drained row are written, then the loop breaks;
      - the pipeline ships SCHEDULE_ENV_TARGET to the actor, the env-step
        count at every sync (only when keyed), and logs every collected
        life before admission;
      - student_arm / student_gate / student_expect handle the prep's
        discount (student_discount, capped at 0.997); an env-stopped
        student's master_iterations is pinned by master_checks, not by its
        arm.
  - experiments/distill.py: prep(from_log=...) / --from-log, which read
    the master's own-play log in place of the recording (check_life on
    every life, games grouped as record() groups them);
    student_discount_rule; discounted_return_max; the value check (A2);
    DISCOUNT and _student_discount as student keys only for
    _discount_from_prep arms.
  - tests/test_testb.py (49 tests, two slow: an F0-like master's own play
    logged as the actor logs it, its prep from the log, then F1's offline
    phase at the prep's discount, whose row 0 a later prep reads back as
    a student master). The flag-on and augmentation arm lists in
    test_pbn / test_harness gain F0-F4.
  - benchmarks/supervisors/supervise_testb.py: five variants f0-f4 in one
    run tree. f0 has no wait and no prep, but the any-trainer guard still
    holds it while Generation 2 trains. Each f{k} waits for F{k-1}'s
    marker and evals (dynamic rows), applies the master gate (A4), preps
    from F{k-1}'s own-play log, and computes its stop (A1). The file has
    24 suites and testb_manifest.json. tests/test_supervise_testb.py has
    183 cases; eight seeded supervisor faults were each caught.
  - Scratchpad: post_testb.py (from post_gen2.py: five runs, dynamic
    rows, read (5) per (A5), the next run's supervisor waiting on the
    previous run's exact evals); make_check_testb.py -> check_post_testb.py
    (CHECK OK); launch_testb.ps1 (worktree Y:\SwitchPlayer-testb-run;
    starts the hand-off and the five supervisors).
  - NOT YET RUN, waiting for host memory while Generation 2 trains: the
    full 24 suites, and a real end-to-end smoke (Ray, the CLI): a tiny F0
    to its stop -> its own-play log -> a prep --from-log -> a tiny F1.
    Both run before launch.
AUDIT r1 (two fresh-eyes auditors on a6d026c3): A (the core) found
nothing above LOW. B (the orchestration) found one MEDIUM.
Verified in r1:
  A: a SelfPlayActor built as _make_actor builds it played 12 real lives
  at synced env counts 0 / 3k / 12k / 16k. The logged temperatures were
  0.5, 0.5, 0.25, 0.125 and the long-token bias 0.5, 0.2, 0, 0, as the
  env-keyed schedules require; check_life held on every life;
  real_done grouped the lives into four games; the round trip through
  OwnPlayLog was exact. `last` holds on every collect path (pipelined,
  primed, eager mid-block); nothing is requested after it; a death
  between row K and the marker replays consistently. Each life is logged
  once, in order, before admission. The prep --from-log, the discount rule
  and the value check are correct. Generations 1 and 2 are unchanged.
  test_testb, test_gen1, test_gen2, test_distill and test_pbn pass.
  B: names and paths agree across the supervisor, the hand-off, the
  launcher and what ablation / distill write. The next generation waits
  for exactly the rows the hand-off produces. The A1 stop is recomputed
  identically on every start. Stale snapshots and log files from a dead
  timeline are never read. The manifest covers the 24 suites' imports.
  No test reads live state, and every stubbed wait is bounded.
  test_supervise_testb passes (183); check_post_testb reports CHECK OK;
  make_check_testb regenerates byte for byte.
FIXES r1 (a fresh-eyes r2 check follows):
  - B-M1 (MEDIUM): the any-trainer guard holds F0 back only while a G2
    trainer process is alive. A pause or a reboot during G2 could let
    F0's preflight and launch race G2's resume. Fix: launch_testb.ps1
    refuses to start until G2's completion marker (its checkpoint below
    1 MB) or its ABANDONED file exists.
  - B-L1: distill.py's deterministic refusals (the value check A2, the
    halved ceiling past the reward supports, an incomplete or empty
    own-play log) were retried 3 times, 15 min apart. They now stop at
    once with PREP FAILED: PREP CHECK (Kai's call; +4 cases).
  - B-L2: set_intact checks vocab.json's master_sha256 against the master
    snapshot's bytes, so a fresh restart of the master that ends at the
    same K is caught (+2 cases).
  - B-L4: only STOP NOT POSITIVE (a spent budget) is Kai's call; STOP
    UNKNOWN clears on a rerun (+1 test).
  - A-L1: the behaviour diagnostics (beh_T, H_act) read the env count the
    actor played at. collect now advances SCHEDULE_ENV_STEPS after the
    lives' reanalysis.
  - A-L4: student_gate refuses a Test B student whose set is not an
    own-play log, so a swap is never a fresh recording (+2 cases).
  - A-L5: env_stop_gate requires --snapshot-weights, since the final
    snapshot is the next generation's master (+1 case).
  - Test gaps: the value check's failure is now pinned by a slow test, and
    log_lives's game grouping by a unit test.
  - NIT: the start-up banner names the env-keyed schedules (A-NIT9); the
    distill docstring names --from-log.
ACCEPTED r1:
  - B-L3: the master gate reads its rows once; a transient read error
    stops the generation until a relaunch.
  - A-L2: game returns across a resume can merge a partial game (header
    only; no read uses it).
  - A-L3: orphaned weights-iterK / iter-*.pkl after a resume are never
    read, since K comes from the pruned jsonl (B verified).
  - A-NIT6: a token's undiscounted inner sum can exceed the per-base-step
    return by g^-(f/4-1), at most 1.24; this is within the 2x headroom.
  - A-NIT7: the header's acting_temperature is the master's final value;
    log_temperatures holds the mix.
  - A-NIT8: read-only stacks are harmless.
  - A-NIT10: the eval actor plays greedy, so it reads no schedule.
  - B-N1 to N7: the gate runs after the preflight; a resume beside a
    finishing orphan runs the preflight; Rows assumes dict rows; the
    hand-off's ALERT throttles are shared across runs; check_post_testb
    must not be rerun beside a live Test B probe; the test helpers' child
    pythons have no timeout of their own; the hand-off's run_rows does
    not filter by arm (one arm per folder).
  - Not added: a test that runs SelfPlayActor.play() itself (A's scratch
    run and the pre-launch smoke cover it), and a prep from a STUDENT
    master's log (the smoke's F1 -> F2 swap covers it).
AUDIT r2 (a fresh-eyes check of FIXES r1 at 55a2d820 and the launcher):
nothing above LOW. THE AUDIT LOOP IS CLOSED.
Verified:
  - The launcher's G2 gate, run alone against mock G2 folders, throws on a
    full checkpoint, on exactly 1,000,000 bytes and on no checkpoint, and
    passes on a marker or on ABANDONED. It runs before anything starts,
    and nothing else starts the supervisors.
  - The PREP CHECK substrings match distill's real messages as a real
    child process logs them; progress lines, OOM and paths do not match.
    The stop reaches the next generation's wait as Kai's call.
  - master_sha256 matches what distill writes, and no legitimate resume
    breaks.
  - STOP UNKNOWN is a plain give-up.
  - During collect the count equals the last sync's on every path.
  - A-L4 and A-L5 bind only the env-stopped arms.
  - Mutants of each fix fail their tests, except A-L1 (below).
  - test_testb: 51 pass; test_supervise_testb: 190; CHECK OK.
FIXES r2 (LOW / NIT: tests and text; confirmed by a fresh-eyes r3 check):
  - LOW-1: A-L1 had no test (reverting it passed every test). A test now
    records the count inside reanalyse ([1000, 1000]; [1003, 1006] fails).
  - NIT-2: the stale docs. The supervisor's CHAIN paragraph,
    live_trainers and the launcher's header said Test B could be launched
    beside G2 and that the guard holds F0 back; the launcher now waits for
    G2, and the guard is a backstop. pred_master_stop's docstring now
    names the narrowed Kai's-call stops. The hand-off's give-up ALERT and
    docstring now name STOP NOT POSITIVE / PREP CHECK as Kai's call: a
    relaunch repeats them. check_post_testb asserts it, via
    make_check_testb.
  - NIT-4: a prep killed while reading the log (its own progress line
    last) is retried, not stopped (+1 test); set_intact fails closed on an
    unreadable master snapshot (+1 test).
  - NIT-1: the launcher notes KEEP Y:\SwitchPlayer-gen2-run until Test B
    completes (its G2 gate reads the marker there on every rerun).
ACCEPTED r2: NIT-3 -- check_life's AssertionError and a corrupt iter-NNN.pkl
are deterministic but retried (both unlikely: the marker comes after a
whole log); a transient "device not ready" read as a missing iteration
would be a false PREP CHECK stop (negligible on a local drive). One
auditor run started at 7.5 GB of free commit, below the 8 GB rule.
AUDIT r3 (a fresh-eyes check of FIXES r2 at 802ea321 and the scratchpad
edits): nothing above NIT. Verified:
  - The A-L1 test fails on the a6d026c3 placement ([1003, 1006]), on a
    placement after reanalyse and with the line removed; the control
    passes.
  - The progress-line test fails only when PREP_CHECKS loses its trailing
    space. The snapshot test fails on a fail-open set_intact and on a
    removed try/except.
  - The text edits change no behaviour. make_check_testb regenerates its
    check byte for byte and CHECK OK holds; the new assertion fails on the
    pre-r2 hand-off.
ACCEPTED r3 (NIT):
  - A killed prep writes no progress line of its own; the test pins the
    colon-vs-space distinction, which is what matters.
  - live_trainers and FIXES r2 say the launcher "waits" for G2; it
    refuses, and is rerun after G2 completes.
  - Two test docstrings keep the "launched beside G2" framing (now the
    backstop case).
  - make_check_testb.py has lines over 79, like its earlier lines that
    mirror check_post_gen2.
  - Two short rewrapped lines.
  - Pre-existing: prep_log_tail reads the whole append-mode prep.log, so
    an attempt killed before writing reports the previous attempt's last
    line (Kai's call either way; rare).
PRE-LAUNCH CHECKS (10-28, after G2 completed; nothing else running):
  - THE 24 SUITES (one at a time, CPU): 23 passed. test_mspacman failed:
    test_actor_play_continues_the_match_with_one_respawn_noop raised
    AttributeError, 'SelfPlayActor' object has no attribute 'mz'.
    - Cause: its _actor_shell builds the actor with object.__new__,
      skipping __init__ (which sets self.mz), and Test B's play() reads
      self.mz.OWN_PLAY_LOG. Production builds every actor through
      __init__.
    - FIX: the shell sets actor.mz = muzero.
    - Rounds r1-r3 ran only a subset of the files, and the supervisor's
      preflight (pytest -x over the 24) would have refused the launch.
    After the fix test_mspacman passes, as do test_inference_server,
    test_ray_harness, test_searchact and test_vtree_refresh.
  - THE SMOKE (scratchpad smoke_testb.py, the real CLI with Ray and the
    GPU; 20 train steps per iteration, env stop 900, 40 offline steps):
    f0 79 s, prep1 9 s, f1 160 s, prep2 19 s, all OK.
    - F0 stopped after iteration 2 (1,205 env steps). It wrote:
      - an own-play log of 3 lives per iteration (temperature 0.5, then
        0.25: the env-keyed schedule);
      - weights-iter002;
      - a 2.7 kB completion marker.
    - Prep 1 read F0's log (source "own-play log", 6 lives, 2 games, no
      partial game):
      - discount 0.97 -> 0.985;
      - ceiling 1 -> 13.5 (clip share 0.407 -> 0.204);
      - value check 21.8 against 5000.
    - F1: its 40 offline steps (snapshots 20 and 40), then online to its
      own stop (2 iterations, 1,249 env steps, training steps 80). Its
      schedules restarted (temperature 0.5 again).
    - Prep 2, from a STUDENT master (F1):
      - master offline steps 40;
      - vocabulary 36 -> 23 (13 dropped, all 16- or 32-frame);
      - discount 0.985 -> 0.9925, read back from F1's snapshot;
      - ceiling 13.5 -> 16.75;
      - value check 24.5.
AUDIT r4 (fresh eyes: the fix, plus a sweep of the 26 test files outside
the 24 and the five above): nothing above LOW. THE AUDIT LOOP IS CLOSED.
  Verified:
    - The fix runs exactly the pre-Test-B play() path (65d83b1d).
    - Every production SelfPlayActor is built through __init__ (the
      training actor, the eval actor, probe_evalpath). The Run shells
      inside the actors never reach a Test B attribute.
    - Only sync() passes env_steps, and the 4-tuple branch is reachable
      only on --ray --cpu-actor F arms.
    - On a resume, the first sync ships the restored count.
    - The resolved F0 / F1 specs are as pre-registered.
    - Sweep: all 26 files pass (slow tests included); none failed.
  ACCEPTED r4:
    - LOW-1: no unit test calls SelfPlayActor.play() on the own-play
      branch (the test_testb fixture copies it inline). The smoke ran
      that branch through real Ray actors, and both preps' check_life
      held on every logged life. Covered, as ACCEPTED r1 planned.
    - NIT-1: ablation.py's INNER_FRAMES comment still says only distill
      sets it (play() now does too). Comment only, left in the audited
      file.
    - NIT-2: the eval actor gets the target but never the count; greedy
      eval reads no schedule. The refresh actor and the inference server
      are unreachable for F arms.
    - NIT-3: collect sets the count for every arm; inert outside Test B.
LAUNCHED 05:33 10-28 (after Generation 2's RESULT) from the run worktree
Y:\SwitchPlayer-testb-run, detached at 77d5df40; testb_manifest.json from
the audited 90862862 (55 files). Before the launch, in that tree: the
manifest check and F0's quick probe passed. launch_testb.ps1 (G2's marker
present) started the hand-off post_testb.py and the supervisors f0-f4.
f0: STOP 20,000 (BUDGET - 0), preflight (the 24 suites), then F0. f1-f4
wait on their predecessors. Expected: about 30 iterations per generation,
the first prep after F0; five generations in about a day and a half.
F0 COMPLETE (interim, 11:15 10-28; not read against the pre-registration
until F4). Scores in these notes are whole-game unclipped scores (the env's
summed points). "Trainer-BN" is the pre-registration's "raw" column: the
same snapshot under the trainer's BN statistics instead of the deployed
acting states.
  One attempt, no deaths: 34 iterations in 5.4 h (19,530 s; 8.8 -> 11.0
      min/iter). F0 stopped at 20,508 env steps after 68,000 training
      steps (3.3 per env step).
  Deployed (exact, acting) at rows 5-30 and the final row 34: 299 / 353 /
      544 / 573 / 593 / 657 / 538, at 2.8k / 6.1k / 9.1k / 12.2k / 15.6k /
      18.1k / 20.5k env steps. Trainer-BN: 264 / 392 / 883 / 713 / 949 /
      757 / 729. For scale, the current lineage's G0 was at 740 deployed
      at 19.8k.
  READ (2): the improvement rate is +16.8 deployed per 1k env steps (OLS
      over the 7 rows).
  MASTER GATE (the swap to F1): rows 30 and 34 average 597.2, at or above
      502.08, so it PASSES. F1's master is weights-iter034.
  READ (3): clip c0 = 1, discount 0.97, 36 tokens. reward_kept is 0.976 at
      row 5 and falls to 0.773 at row 34 as larger rewards appear.
  READ (4): policy_maxp 0.50 -> 0.64 (0.62-0.67 from row 15 on). The
      search targets' entropy is 0.82-1.45 (1.11 at row 34).
  READ (5) at row 34: p1_ratio 0.76, so FIXED.
  READ (6): the largest adjacent deployed drop is -119 (row 30 -> 34;
      one-row band +-464).
  Tokens: the 4-frame share of play rises from 0.43 (row 10) to 0.82 (row
      34) as the long-token bias decays. Frames per decision 14.7 -> 6.8.
F1 began at 11:13 with STOP 19,492 (40,000 - 20,508): its preflight, then
the prep from F0's own-play log.
SWAP 1 PREP (F0 -> F1; 11:22-11:27 10-28, attempt 1; read (3)):
  THE DATA: F0's own-play log, iterations 1-34. 34 whole games, 102
      lives, 20,508 base steps, no partial games. Game returns: mean
      543.8, median 470. Lives by temperature: 0.5 51, 0.25 24, 0.125 27.
      This is the whole generation's play, from its start, rather than a
      fresh recording at the final temperature.
  THE CLIP: the master's ceiling 1 removes 0.227 of 1,849 reward units.
      The student's ceiling is 11.62, which removes 0.114. 0.9% of the
      steps are clipped, against 2.2% at the master's ceiling.
  THE DISCOUNT: 0.97 -> 0.985. VALUE CHECK: the largest discounted return
      is 28.8 units; 2x that is within the 5,000-unit support.
  THE VOCABULARY: 18 of 36 tokens. All 9 4-frame tokens are kept; dropped
      are (0,16) (0,32) (1,16) (2,8) (2,16) (2,32) (3,8) (4,16) (4,32)
      (5,8) (5,16) (5,32) (6,8) (6,32) (7,8) (7,16) (8,8) (8,16).
  TARGETS: 10,418 records, from F0's 50-sim birth search under its spec.
      470 dropped-token decisions are split into 1,855 extra records.
      There were 38 uniform fallbacks. The mean visit mass discarded on
      dropped tokens is 0.183. The prep took 268 s.
F1 LAUNCHED 11:28 10-28 (attempt 1, fresh): the prefill, the 20,000
offline steps, then online play to 19,492 env steps.
F1 ROW 0 (interim, 13:35 10-28; offline phase ~2 h):
  READ (2) transfer: row 0 (20,000 offline steps, before F1 plays) is 872.0
      deployed (SD 201.5) and 683.0 trainer-BN. The master's final row is
      537.7 (trainer-BN 729.0), so +334.3 deployed, descriptive (one-row
      band +-464). Against the master's last two rows (597.2): +274.8.
      The earlier swaps' row 0 was BELOW the master (DS2 1065 against
      1114, G2 1350 against 1657). Here it is above, at 20.5k cumulative
      env steps (G0 had 740 at 19.8k).
  The offline curve (deployed / trainer-BN): 563 / 807 at 4k, 590 / 637
      at 8k, 602 / 593 at 12k, 1002 / 664 at 16k; row 0 (20k) 872 / 683.
  READ (5) row 0: p1_ratio 0.97, so FIXED (18 tokens, ceiling 11.62).
  Frames per decision, deployed: 7.8 at row 0.
F1 COMPLETE (interim, 18:20 10-28; not read against the pre-registration
until F4):
  One attempt, no deaths: 6.8 h (24,390 s), of which the prefill and the
      offline phase took ~2 h. 28 iterations at ~11 min each; F1 stopped
      at 19,494 env steps (76,000 training steps including the 20,000
      offline). Cumulative env steps: 40,002.
  Deployed (exact, acting), rows 0-25 and the final row 28: 872 / 946 /
      967 / 974 / 945 / 922 / 962, at 20.5k / 24.1k / 27.2k / 30.2k /
      34.1k / 37.7k / 40.0k cumulative env steps. Trainer-BN: 683 / 788 /
      735 / 846 / 1142 / 764 / 898. For scale, the current lineage's G0
      was at 1154 deployed at 39.5k, so the fast lineage is ~190 behind at
      40k.
  READ (2): jump: rows 5 and 10 (mean 956.7) against the master's last two
      rows (597.2) give +359.5. The improvement rate is +2.1 deployed per
      1k env steps (OLS over the 7 rows, row 0 included), so F1 is flat
      from row 5 on: its whole gain over F0 came at the swap.
  MASTER GATE (the swap to F2): rows 25 and 28 average 942.3, at or above
      502.08, so it PASSES. F2's master is weights-iter028.
  READ (3): clip 11.62, discount 0.985, 18 tokens; reward_kept 0.86-0.88.
  READ (4): policy_maxp 0.70 at row 5, then 0.54-0.64 (0.55 at row 28).
      The search targets' entropy is 1.20-1.52.
  READ (5) at row 28: p1_ratio 0.56, so FIXED (G2's final row: 0.56).
  READ (6): the largest adjacent deployed drop is -29 (row 15 -> 20).
      Across the swap: row 0 is +334 over the master's final row.
  Tokens: the 4-frame share of play is 0.63 at row 5 and 0.78 at row 28.
      Frames per decision 10.4 -> 7.1.
F2 began at 18:19 with STOP 19,998 (60,000 - 40,002).
SWAP 2 PREP (F1 -> F2; 18:27-18:30 10-28, attempt 1; read (3)):
  THE DATA: F1's own-play log, iterations 1-28. 28 whole games, 84 lives,
      19,494 base steps. Game returns: mean 777.5, median 705, max 1510.
      Lives by temperature: 0.5 48, 0.25 18, 0.125 18.
  THE CLIP: the master's ceiling 11.62 removes 0.131 of 2,177 reward
      units. The student's ceiling 18.10 removes 0.066. The share of steps
      clipped stays at 1.4%: no step reward falls between the two
      ceilings, so only the clipped excess shrinks.
  THE DISCOUNT: 0.985 -> 0.9925. VALUE CHECK: the largest discounted
      return is 46.5 units; 2x that is within the 5,000-unit support.
  THE VOCABULARY: 16 of 18 tokens; dropped (0,8) and (6,16). The (7,4)
      token (DOWNRIGHT, 4 frames) holds 52.6% of the decisions.
  TARGETS: 10,102 records. 183 dropped-token decisions are split into 364
      extra records. There were 4 uniform fallbacks. The mean visit mass
      discarded on dropped tokens is 0.059. The prep took 191 s.
F2 LAUNCHED 18:31 10-28 (attempt 1, fresh).
F2 ROW 0 (interim, 20:38 10-28):
  READ (2) transfer: row 0 is 745.7 deployed (SD 138.0) and 884.0
      trainer-BN. The master's final row is 962.3 (trainer-BN 897.7), so
      -216.6 deployed, descriptive (one-row band +-464); trainer-BN is
      -13.7. Unlike swap 1, there is no jump at this swap; it sits below
      the master like the current lineage's swaps (DS2 -49, G2 -306).
  The offline curve (deployed / trainer-BN): 803 / 736 at 4k, 844 / 843
      at 8k, 805 / 755 at 12k, 784 / 803 at 16k; row 0 (20k) 746 / 884.
  READ (5) row 0: p1_ratio 0.93, so FIXED (16 tokens, ceiling 18.10).
  Frames per decision, deployed: 8.7 at row 0.
F2 COMPLETE (interim, 01:05 10-29; not read against the pre-registration
until F4):
  One attempt, no deaths: 6.4 h (23,070 s), 26 iterations. F2 stopped at
      20,124 env steps (72,000 training steps including the 20,000
      offline). Cumulative env steps: 60,126.
  Deployed (exact, acting), rows 0-25 and the final row 26: 746 / 799 /
      923 / 865 / 846 / 772 / 834, at 40.0k / 43.9k / 47.9k / 51.6k /
      55.6k / 59.4k / 60.1k cumulative env steps. Trainer-BN: 884 / 714 /
      840 / 932 / 1162 / 1112 / 921.
  Rows 5-26 average 839.9 deployed, against F1's rows 5-28 at 952.7
      (-112.8). Trainer-BN averages 946.8 against F1's 862.2 (+84.6). The
      deployed acting states run 300-340 BELOW trainer-BN at rows 20-25,
      against the usual deployed >= trainer-BN.
  For scale, the current lineage's G0 was at 1190 deployed at 59.2k, so
      the fast lineage is ~356 behind at 60k.
  READ (2): jump: rows 5 and 10 (mean 860.7) against the master's last two
      rows (942.3) give -81.6. The improvement rate is +1.5 deployed per
      1k env steps (OLS over the 7 rows, row 0 included). F2 is flat like
      F1.
  MASTER GATE (the swap to F3): rows 25 and 26 average 803.2, at or above
      502.08, so it PASSES. F3's master is weights-iter026.
  READ (3): clip 18.10, discount 0.9925, 16 tokens; reward_kept 0.93-0.94.
  READ (4): policy_maxp falls from 0.62 at row 5 to 0.49 at row 26. The
      search targets' entropy is 1.23-1.58.
  READ (5) at row 26: p1_ratio 1.05, so FIXED.
  READ (6): the largest adjacent deployed drop is -74 (row 20 -> 25).
      Across the swap: row 0 is -217 against the master's final row.
  Tokens: F2 plays longer tokens than F1. The 4-frame share of play is
      0.58-0.65 (F1: 0.63-0.80), the 32-frame share 0.25-0.29, and frames
      per decision 11.7-13.1 (F1: 7.1-10.4).
F3 began at 01:04 10-29 with STOP 19,874 (80,000 - 60,126).
SWAP 3 PREP (F2 -> F3; 01:13-01:15 10-29, attempt 1; read (3)):
  THE DATA: F2's own-play log, iterations 1-26. 26 whole games, 78 lives,
      20,124 base steps. Game returns: mean 980.8, median 835, max 1970,
      above F2's deployed exact evals (~840). Lives by temperature: 0.5
      42, 0.25 18, 0.125 18.
  THE CLIP: the master's ceiling 18.10 removes 0.063 of 2,550 reward
      units. The student's ceiling 24.02 removes 0.031. The share of steps
      clipped falls from 1.9% to 0.3%.
  THE DISCOUNT: 0.9925 -> 0.99625. VALUE CHECK: the largest discounted
      return is 73.6 units; 2x that is within the 5,000-unit support.
  THE VOCABULARY: 16 of 16 tokens; none dropped (every token is at or
      above 1%). (7,4) holds 45.6% of the decisions, (7,32) 10.5%.
  TARGETS: 6,601 records (F2's longer tokens: fewer decisions per base
      step), no splits, no uniform fallbacks. The prep took 130 s.
F3 LAUNCHED 01:16 10-29 (attempt 1, fresh).
F3 ROW 0 (interim, 03:16 10-29):
  READ (2) transfer: row 0 is 1037.0 deployed (SD 199.0) and 811.0
      trainer-BN. The master's final row is 834.3 (trainer-BN 921.3), so
      +202.7 deployed, descriptive (one-row band +-464); trainer-BN is
      -110.3.
  The offline curve (deployed / trainer-BN): 863 / 862 at 4k, 915 / 1093
      at 8k, 977 / 1058 at 12k, 1091 / 1086 at 16k; row 0 (20k) 1037 /
      811. Deployed rises through the offline phase, unlike F2's flat
      curve.
  READ (5) row 0: p1_ratio 1.04, so FIXED (16 tokens, ceiling 24.02).
  Frames per decision, deployed: 15.6 at row 0, the longest so far.
F3 COMPLETE (interim, 07:36 10-29; not read against the pre-registration
until F4):
  One attempt, no deaths: 6.2 h (22,439 s), 27 iterations. F3 stopped at
      20,511 env steps, 637 past its stop of 19,874 (74,000 training
      steps including the 20,000 offline). Cumulative env steps: 80,637.
  Deployed (exact, acting), rows 0-25 and the final row 27: 1037 / 1032 /
      1222 / 968 / 854 / 1072 / 851, at 60.1k / 64.7k / 68.0k / 71.4k /
      75.3k / 79.0k / 80.6k cumulative env steps. Trainer-BN: 811 / 787 /
      1022 / 1532 / 772 / 812 / 803.
  Rows 5-27 average 999.8 deployed (F2: 839.9, F1: 952.7) and 954.7
      trainer-BN (F2: 946.8, F1: 862.2).
  READ (2): jump: rows 5 and 10 (mean 1127.0) against the master's last
      two rows (803.2) give +323.8. The improvement rate is -8.2 deployed
      per 1k env steps (OLS over the 7 rows, row 0 included): the gain
      came at the swap and faded within the generation.
  MASTER GATE (the swap to F4): rows 25 and 27 average 961.5, at or above
      502.08, so it PASSES. F4's master is weights-iter027.
  READ (3): clip 24.02, discount 0.99625, 16 tokens; reward_kept
      0.96-0.97.
  READ (4): policy_maxp 0.53-0.59. The search targets' entropy is
      1.34-1.59.
  READ (5) at row 27: p1_ratio 0.78, so FIXED.
  READ (6): the largest adjacent deployed drop is -254 (row 10 -> 15).
      Across the swap: row 0 is +203 against the master's final row.
  Tokens: longer again. The 32-frame share of play is 0.38-0.52 (F2:
      0.25-0.29), the 4-frame share 0.36-0.49, and frames per decision
      15.4-19.5.
F4 began at 07:35 10-29 with STOP 19,363 (100,000 - 80,637). Its rows from
90k cumulative env steps on (F4's own 9,363), its final included, form
read (1)'s headline window.
SWAP 4 PREP (F3 -> F4; 07:42-07:44 10-29, attempt 1; read (3)):
  THE DATA: F3's own-play log, iterations 1-27. 27 whole games, 81 lives,
      20,511 base steps. Game returns: mean 1026.3, median 950, max 1960.
      Lives by temperature: 0.5 42, 0.25 18, 0.125 21.
  THE CLIP: the master's ceiling 24.02 removes 0.046 of 2,771 reward
      units. The student's ceiling 32.07 removes 0.023. The share of steps
      clipped stays at 0.4%.
  THE DISCOUNT: 0.99625 -> 0.997 (the cap; the halving rule gives
      0.998125). VALUE CHECK: the largest discounted return is 71.1 units;
      2x that is within the 5,000-unit support.
  THE VOCABULARY: 16 of 16 tokens; none dropped. (7,4) holds 29.8% of the
      decisions; the 32-frame tokens (3,32), (1,32), (7,32), (8,32) hold
      47.1% together.
  TARGETS: 4,671 records, no splits, no fallbacks. The prep took 106 s.
      The distillation set shrinks each swap as the tokens lengthen:
      10,418 / 10,102 / 6,601 / 4,671 records over ~20k base steps each.
F4 LAUNCHED 07:45 10-29 (attempt 1, fresh).
F4 ROW 0 (interim, 09:44 10-29):
  READ (2) transfer: row 0 is 1005.3 deployed (SD 280.2) and 1253.0
      trainer-BN. The master's final row is 851.3 (trainer-BN 803.3), so
      +154.0 deployed, descriptive (one-row band +-464); trainer-BN is
      +449.7.
  The offline curve (deployed / trainer-BN): 983 / 1077 at 4k, 1607 /
      1042 at 8k, 940 / 1325 at 12k, 1046 / 1180 at 16k; row 0 (20k) 1005
      / 1253.
  READ (5) row 0: p1_ratio 0.47, so FIXED (16 tokens, ceiling 32.07).
  Frames per decision, deployed: 20.2 at row 0.
F4 COMPLETE (13:57 10-29; read against the pre-registration in "RESULT --
Test B", above):
  One attempt, no deaths: 6.1 h (22,020 s), 27 iterations. F4 stopped at
      19,845 env steps, 482 past its stop of 19,363 (74,000 training steps
      including the 20,000 offline). Cumulative env steps: 100,482. The
      hand-off's last evals finished at 13:57 ("all done").
  Deployed (exact, acting), rows 0-25 and the final row 27: 1005 / 1174 /
      1091 / 952 / 1045 / 873 / 1194, at 80.6k / 83.8k / 87.2k / 91.7k /
      95.4k / 99.1k / 100.5k cumulative env steps. Trainer-BN: 1253 / 1103
      / 1068 / 1203 / 944 / 1274 / 1202.
  Rows 5-27 average 1054.8 deployed (F3: 999.8) and 1132.2 trainer-BN (F3:
      954.6).
  READ (2): jump: rows 5 and 10 (mean 1132.5) against the master's last two
      rows (961.5) give +171.0. The improvement rate is -2.1 deployed per
      1k env steps (OLS over the 7 rows, row 0 included).
  READ (3): clip 32.07, discount 0.997, 16 tokens; reward_kept 0.975-0.977.
  READ (4): policy_maxp 0.57-0.65. The search targets' entropy is
      1.20-1.63.
  READ (5) at row 27: p1_ratio 0.67, so FIXED.
  READ (6): the largest adjacent deployed drop is -172 (row 20 -> 25).
      Across the swap: row 0 is +154 against the master's final row.
  Tokens: the 32-frame share of play is 0.44-0.62 (F3: 0.38-0.52), the
      4-frame share 0.21-0.34, and frames per decision 18.1-22.8 in play
      (16.7-21.7 deployed).
LATER (Kai 2026-10-27, not pre-registered yet):
  - TEST C: BBF-style resets on the winning lineage. Candidates: a student
    started from a shrink-and-perturb copy of the master (conv layers 50%
    toward a fresh init), or resets within a generation every ~40k
    gradient steps.
  - TEST D: a smart swap trigger, rate-based and game-agnostic. Kai: a
    plateau can be "a large slowdown in improvement", not only no gain.
    Candidates: the rate falling below 1/3 of the generation's own peak
    (on past curves this fires at 40-85 iterations, noisily), or swapping
    when staying is slower than the rate a swap delivers (the marginal
    value rule). Both are calibrated on Test B's per-generation rates.

### Next experiment (pre-registered) -- Generation 2: DS2 as the master, one distilled student (Kai, 2026-10-27)

DECISION: Kai 2026-10-27, after Test 2's RESULT: "Generation 2 next" (over
the pre-registered seed-1 repeat; see the RESULT's last note).
QUESTION: does the loop compound? With DS2 (generation 1's distilled
student) as the master, the clip's removed share halved again and the
vocabulary trimmed again by the same rules, does the generation-2 student
pass DS2 as DS2 passed G0?
THE SWAP (generation 1 -> 2): every rule is Test 2's; only the master
changes.
  MASTER = DS2's final snapshot (weights-iter090: 20,000 offline + 180,000
    online steps; its acting states); MASTER = its deployed 60-90 mean
    1656.52 (raw 1600.05). Not NEAR RANDOM (g 1.385).
  RECORDING, VOCABULARY, RE-TOKENIZATION, TARGETS: Test 2's -- DS2's fresh
    games at its final temperature (0.125) through its acting states, at
    raw-step resolution, whole games until >= 31,250 base steps, seed 0;
    the 1% rule over DS2's 21 tokens, never a 4-frame token; dropped tokens
    as raw 4-frame records; the masked read-out of DS2's 50-sim birth
    search, under DS2's own spec (its per-step prefix BN statistics: its
    search is clean by Test 2's read (10)).
  THE CLIP (rule (a): each later swap halves again): the master's ceiling
    is DS2's constant c1 = 11.258 (read from its snapshot's spec); the
    student's c2 is the smallest c >= c1 with S(c) <= S(c1) / 2 on DS2's
    recorded games. c2 past the reward supports -> PREP FAILED, Kai's call.
    Logged: S(c1), c2, S(c2), the shares of base steps clipped.
ARM (seed 0, 90 iterations): G2 = DS2's arm (the TK recipe,
  PREFIX_BN_PER_STEP) over the generation-2 vocabulary at the CONSTANT
  ceiling c2. It gets the distillation step's prefill (DS2's re-tokenized
  games with master-search targets) and 20,000 offline steps, then TK's
  within-run schedules keyed to its ONLINE steps (SCHEDULE_OFFSET_STEPS
  20,000): prior-acting temperature 0.5 / 0.25 / 0.125 staged, long-token
  bias 0.5 -> 0 by 25%, PER beta 0.4 -> 1, the off-policy horizon decay.
  This is the proposal's default (b): the clip is the one setting that
  moves only at a swap. No fresh twin: Kai's choice, since the question is
  compounding, and Test 2 settled distilled against fresh on one seed.
READS. EXACT eval as before (whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4); G2's rows 0 and 5-90 plus its offline
curve (4k, 8k, 12k, 16k); DEPLOYED primary, raw secondary; MASTER = DS2's
1656.52 (raw 1600.05).
  NEAR RANDOM first: G2's 60-90 deployed mean < 502.08 is diagnosed
      before any other read.
  (0) transfer: row 0 against MASTER, descriptive (+-464); TRANSFER FAILED
      if row 0 is NEAR RANDOM. The offline curve and DS2's own scores in
      the recorded games beside it.
  (1) PRIMARY, against the master: G2's deployed mean 60-90 against
      MASTER 1656.52, +-232: ABOVE / SAME / BELOW. Raw against 1600.05,
      +-205.
  (2) head start: G2's deployed mean of rows 5-25 against DS2's (1330.27),
      +-274.4: BETTER / NO DIFFERENT / WORSE (two distilled starts, one
      generation apart).
  (3) two generations, descriptive: G2's 60-90 against G0's 1113.86 (in
      g).
  (4) time to level: the first row >= MASTER - 232 (1424.52), in
      iterations and in G2's own env steps; G2's lowest row in 5-25
      against row 0 (forgetting).
  (5) tokens: the dropped tokens and their share of DS2's recorded play;
      G2's per-duration play / search shares, frames per decision, the
      deployed decisions per second.
  (6) stability: the largest adjacent deployed drop from row 45.
  (7) the clip: S(c1), c2, S(c2) on DS2's games; reward_kept by row.
  (8) the prior: R = mean policy_maxp over rows 75-90 / over rows 30-55;
      HELD >= 0.8, FLATTENED <= 0.6, else PARTIAL; the search targets'
      entropy by row.
  (9) the general plateau rule (Step 0 (ii)) after the fact,
      descriptive.
  (10) mechanism: probe_prefix_bn.py at rows 0, 30, 55, 75 and 90, FIXED
      as in PB (p1_ratio <= 2). If G2 is not FIXED at one or more of rows
      55, 75 and 90, reads (1)-(9) carry that caveat.
DECISION AFTER:
  - (1) ABOVE -> the loop compounds over two swaps: Kai's call on
    generation 3 and on the loop's automation (a plateau-triggered swap,
    Step 0 (ii)).
  - (1) SAME -> the second swap kept the master's level without passing
    it: Kai's call (candidates: the halved clip now reaching rewards the
    score weighs most; the vocabulary trim; a seed repeat; a fresh twin).
  - (1) BELOW -> diagnose before any generation 3: transfer by read (0),
    the clip by read (7), the tokens by read (5).
  - (0) TRANSFER FAILED -> the generation-2 prep is diagnosed first.
BUILD (before launch; audited like every build):
  - experiments/distill.py: a STUDENT master (an arm with `_student`).
    Its runtime spec comes from its snapshot's spec: the vocabulary, the
    constant ceiling, the schedule offset and the per-step BN flag,
    checked against its arm's static keys. Its final training_steps is
    iterations x 2,000 + its offline steps. The master's ceiling is its
    constant one.
  - experiments/ablation.py: the -gen2-distill arm (DS2's arm with
    `_master_arm` = DS2's arm). student_arm / student_gate must accept a
    master vocabulary that is itself a student vocabulary: the prep's
    master_vocab, an ordered subset of TK's 36 that holds every 4-frame
    token.
  - A supervisor (supervise_gen2.py, derived from supervise_gen1.py):
    the master gate on DS2 (its completion marker and both probe jsonls'
    rows 60-90), the prep, then G2. An eval hand-off (rows 0 and 5-90, the
    offline curve, read (10) at 0 / 30 / 55 / 75 / 90).
  - Tests, a manifest, a fresh detached run worktree.
COST: prep ~5-10 min; G2 ~17 h (DS2's pace); the build and its audit loop
first.
BUILT 2026-10-27 (pre-audit; the audit loop follows before any launch):
  - experiments/distill.py: master_spec. A STUDENT master (an arm with
    `_student`) runs as its snapshot's spec when every key but
    student_arm's (STUDENT_KEYS) equals its arm's, its vocabulary is a
    student vocabulary of the arm's (or of its own `_master_vocab`) and its
    ceiling is the constant `_student_ceiling` of a `_ceiling_from_prep`
    arm. A non-student master runs as its arm, as before. Also:
    master_offline_steps (its final snapshot = offline steps + iterations x
    2,000); installed (ARMS[arm] swapped for the prep's recording and
    searches only, restored even on failure); the header's
    master_offline_steps.
  - experiments/ablation.py: the -gen2-distill arm (DS2's arm, with
    `_master_arm` = DS2's arm). student_master_vocab: when the master is
    itself a student, student_arm checks the vocabulary against the prep's
    master_vocab (kept as `_master_vocab`), and student_gate checks the
    set's master vocabulary against it (TK's 36 for generation 1, as
    before).
  - tests/test_gen2.py (25 tests; two slow: a real prep from a DS2-like
    master at a constant ceiling 3.0 after 4 offline steps, then G2's
    offline phase on it, whose row 0 a generation-3 prep could read back);
    test_pbn.py's flag-on arm set gains G2.
  - benchmarks/supervisors/supervise_gen2.py (from supervise_gen1.py: one
    variant g2; the master DS2 in Y:\SwitchPlayer-gen1b-run, waited on for
    its marker and rows 0 and 5-90 of both probe jsonls; the NEAR RANDOM
    gate on its deployed 60-90; set_intact also requires
    master_offline_steps 20,000; 22 suites; gen2_manifest.json) and
    tests/test_supervise_gen2.py (130 cases, from test_supervise_gen1.py:
    fs2's tests dropped, G0's moved to DS2; six seeded supervisor faults
    were each caught by the targeted tests).
  - Scratchpad: make_post_gen2.py -> post_gen2.py (sha256 0a487c93699933e5..., from
    post_gen1b.py: one run, rows 0 and 5-90, the offline curve 1-4, read
    (10) at 0 / 30 / 55 / 75 / 90); make_check_gen2.py ->
    check_post_gen2.py (CHECK OK); launch_gen2.ps1 (worktree
    Y:\SwitchPlayer-gen2-run).
  - On the real DS2 snapshot (CPU, under the lock): a one-game prep (600
    base steps) read DS2 as its spec (200,000 steps = 20,000 offline + 90 x
    2,000; temperature 0.125, long-token bias 0). On that game S(c1) was
    0.191 -> c2 = 21.76 -> 0.096, with 18 of 21 tokens kept, and ARMS was
    restored after. The reward supports allow ceilings up to 400 units.
    supervise_gen2's gate on the real DS2: nothing missing; MASTER
    1656.524; its snapshot present.
  - The 22 suites: pass in this tree (one failure on the first run: test_harness's augmentation-off arm list lacked G2's arm, which was added; test_harness then passes).
AUDIT r1 (two fresh-eyes auditors on ba269387): A (the core) nothing above
LOW; B (the orchestration) nothing above LOW. Verified in r1:
  A: on the real DS2 snapshot master_spec passes (its spec differs from
  the static arm only in student_arm's six keys, types included). The
  prep's world equals DS2's training world rebuilt through student_arm on
  its own prep: every muzero global, ACTIONS 21, temperature 0.125, bias
  0, ceiling 11.2583, PER beta 1.0. An instrumented real prep (1,500 base
  steps) recorded on the acting states at step 200,000, and every
  birth-search head call ran in eval mode on its step set (per_step 5);
  S(c1) 0.266 -> c2 19.99 -> 0.133; 18 of 21 tokens kept; ARMS and the
  globals were restored after. The G2 student ran end to end on that prep:
  student_arm, every startup gate with --ray --cpu-actor, the prefill born
  at 20,000, the offline phase, both probes on its row 0, and a
  generation-3 master_spec on it. halved_ceiling matched an exact rational
  solver (3,000 cases, non-integer c0). Generation 1 is unchanged.
  B: every fs2 removal and G0 -> DS2 change checked. Names and paths agree
  across the supervisor, the hand-off, the launcher and ablation's naming.
  The manifest covers everything the run, the prep, the probes and the 22
  suites execute (an AST import closure). The derivations regenerate byte
  for byte; check_post_gen2 prints CHECK OK. The 130 tests read no live
  state, and six seeded supervisor faults were caught.
FIXES r1 (LOW / NIT; a fresh-eyes r2 check follows):
  - A-L1 / B-L2: nothing before the prep tried the real DS2 snapshot
    against this tree's arm, so an edit to DS2's arm (its _doc included)
    would surface only as three failed preps. The prep's master checks are
    now distill.master_checks, and the preflight's quick probe ends with
    it on the real MASTER_SNAPSHOT. It passes on DS2; seven refusal cases
    are tested on DS2-like snapshots in tmp.
  - A-L2: master_spec requires `_master_vocab` exactly when the master's
    own master is a student, and checks that it is a student vocabulary
    of the arm's (+3 cases; DS2 carrying one is refused).
  - A-L3: changes to the arm's underscore keys are tested (+4 cases).
  - B-L1: the suites' import pinning reads every imported name (ast).
  - B-L3: set_intact also requires vocab.json's master_snapshot to be
    MASTER_SNAPSHOT (+2 cases).
  - NITs: ABANDON names the prep; the fresh-restart rule gives its reason
    (the set's sha256); the quick-probe want list gains the master's
    _ceiling_from_prep; the launcher refuses a missing hand-off script;
    two test comments / asserts (A-NIT-2).
ACCEPTED r1: A-NIT-1 (the prep's vocabulary check is a tautology for a
student master; kept for G0's path). A-NIT-3 (an equivalent mutant). B-NIT-4's
commit check: the manifest is committed after its audited commit, so
_commit is not the worktree's HEAD by design; the supervisor's hash check
is the gate. B-NIT-5: the date 2026-10-27 is used everywhere and is kept
even if the launch slips.
AUDIT r2 (a fresh-eyes check of FIXES r1 at 4d36e6a9): nothing above LOW.
THE AUDIT LOOP IS CLOSED. Verified: prep is unchanged for G0, DS2 and
G2-like masters (check order, out-dir, master_vocab, header); the deployed
quick probe passes on the real DS2 in 2 s, and an edit to DS2's arm in a
copy (its _doc) fails it with the prep's message as stderr's last line,
which preflight() reports; the `_master_vocab` rule agrees with
student_arm's; master_snapshot round-trips byte for byte (argv -> header
-> vocab.json -> set_intact); ten mutants of the fixes each fail their
tests; the four touched test files pass in an archive copy, slow tests
included; the launcher parses and its new refusal fires.
ACCEPTED r2: LOW-1 -- every supervisor start, a resume included, now reads
DS2's real snapshot (the quick probe), so Y:\SwitchPlayer-gen1b-run must be
KEPT until G2 completes (a pruned tree would stall a resume at PREFLIGHT,
hourly, without an ALERT). Procedure, not code: that tree stays (it also
holds DS2's evals). NIT-1 (master_spec and student_master_vocab test "the
master is a student" by truthiness / `is None`: equal for every real
value); NIT-2 (a test checks the message is in stderr, not its last line:
it is the last line, checked by hand).
LAUNCHED 2026-10-27 11:21 from Y:\SwitchPlayer-gen2-run (detached, CRLF,
at ce5e1488; gen2_manifest.json written from the audited 48716683, 52
AUDITED files). In that tree the manifest check, the real quick probe (the
prep's master checks on DS2's snapshot) and the DS2 gate (MASTER 1656.524,
nothing missing) pass. launch_gen2.ps1 started post_gen2.py (sha256
0a487c93...; log scratchpad post_gen2.log) and supervise_gen2.py g2 0
(the preflight's 22 suites in that tree, then the master gate, the prep,
G2). KEEP Y:\SwitchPlayer-gen1b-run until G2 completes (ACCEPTED r2).
Rerun launch_gen2.ps1 after a reboot or a pause (idempotent; PAUSE also
kills a running preflight or prep).
PREFLIGHT ok 11:32 (manifest + quick probe + 22 suites in the run tree);
MASTER DS2 weights-iter090 (deployed 60-90 1656.52, g 1.385); PREP done
11:34 in one attempt (4 min). The prep, descriptive (read (7)'s first
half):
  recording: 36 games of DS2 at temperature 0.125 (long-token bias 0), 108
      lives, 31,862 base steps; game returns mean 2127.8 (840-5000). G0's
      recording in Test 2's prep: 1176.0.
  THE CLIP: DS2's ceiling c1 = 11.258 removes S(c1) = 0.273 of the recorded
      7,660 reward units. The halved-clip rule gives c2 = 22.961 units
      (229.6 points per base step: the first ghost, 200, now passes whole;
      the later ghosts and fruit are cut), which removes S(c2) = 0.136.
      Base steps clipped: 0.33% -> 0.13% of all base steps (2.6% -> 1.0%
      of the 3,976 reward-earning ones).
  VOCABULARY: 18 of 21 tokens kept. Dropped: (NOOP, 16), (NOOP, 32) and
      (DOWNLEFT, 16), which held 1.5% of DS2's 5,572 decisions. Largest
      kept: (LEFT, 16) 0.236, (DOWNLEFT, 32) 0.183, (UPRIGHT, 32) 0.158,
      (DOWNRIGHT, 32) 0.107. 5,999 re-tokenized records (85 dropped-token
      decisions split into 319 extra 4-frame records); 0 uniform
      fallbacks; mean master visit mass on dropped tokens 0.045.
G2 LAUNCHED 11:35 10-27 (attempt 1, fresh): the prefill, then the 20,000
offline steps (~1.5 h to row 0), then 90 iterations (~16 h).
G2 ROW 0 (interim, 13:56 10-27):
  READ (0) transfer: row 0 (20,000 offline steps, before G2 played) 1350.3
      deployed (SD 681.7; g 1.071), raw 1526.3, against MASTER 1656.52:
      -306.2, descriptive (+-464); NOT TRANSFER FAILED. (DS2's row 0 was
      1065.0 against its master's 1113.86.) The offline curve so far,
      deployed / raw: 1722 / 1398 @4k, 1498 @8k (the rest of the curve to
      come: the hand-off's interim probes are waiting for free memory).
  READ (10) row 0: p1_ratio 0.88 -> FIXED (18 tokens, ceiling 22.96).
  Frames per decision, deployed: 23.2 at row 0.
  HOST MEMORY: the supervisor ALERTed at row 10 (14:59): host commit free
      3.0 GB < 4 GB. The trainer holds steady at ~7.7 GB; the drop is other
      applications (a browser ~3.5 GB across processes, Steam, Discord) on
      a 36.8 GB commit limit. The hand-off's interim probes (>= 6 GB) are
      starved and will catch up in the final pass (>= 4 GB).
G2 ROWS 5-45 (interim, 21:15 10-27; not read against the pre-registration
until row 90):
  Deployed (exact, acting): 1925 / 1798 / 2244 / 1686 / 1824 / 2040 /
      1828 / 1968 / 1997 at rows 5-45; mean 1923.3. DS2 at the same rows:
      1226 / 1200 / 1295 / 1533 / 1397 / 1944 / 1722 / 1385 / 1161, mean
      1429.2. Raw rows 5-40: 1268 / 1624 / 1610 / 1772 / 2001 / 1900 /
      1868 / 1846.
  The offline curve (deployed / raw): 1722 / 1398 @4k, 1498 / 1682 @8k,
      1510 / 1725 @12k, 1645 / 1674 @16k; row 0 (20k) 1350 / 1526.
  Env steps at row 45: 37,138. One attempt; no deaths; the interim probes
      run whenever free commit allows.

### RESULT -- Test 2: the first swap under the per-generation clip, DS2 against FS2 (seed 0), 2026-10-27

(Pre-registration: "Next experiment (pre-registered) -- Test 2", below PB's
sections; its interim notes there.)
RUNS (from the run worktree Y:\SwitchPlayer-gen1b-run, detached at
0467dd90, manifest from the audited fe4ef2ad; one attempt each, no ALERTs,
both supervisor.err empty): the prep 22:25 -> 22:29 10-25; DS2 22:30 10-25
-> 15:28 10-26 (61,050 s; the offline phase ~1 h 33 min, then ~10.3
min/iteration); FS2 15:39 10-26 -> 06:10 10-27 (52,230 s, 9.7
min/iteration); the hand-off's last evals 06:15 10-27 -- every row in both
variants (DS2 0 and 5-90 plus its offline curve 1-4, FS2 5-90), all nine
read (10) rows, no capped game.
MASTER = G0 weights-iter090 (deployed 60-90 1113.86, g 0.828). THE PREP
(read (7)'s first half, above in the interim notes): 45 G0 games, 135
lives, 31,784 base steps, game returns mean 1176.0; S(1) = 0.344 -> c1 =
11.258 units -> S(c1) = 0.172; 21 of 36 tokens kept (15 dropped: 278 of
G0's 5,615 decisions, 5.0%; the master's search put 0.177 of its visits on
them).
EXACT eval (30 noop starts, prior argmax); DEPLOYED primary, RAW
secondary, IN = the run's 8-game eval, maxp = the prior's mean max
probability:
  iter        0    5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  DS2 dep  1065 1226 1200 1295 1533 1397 1944 1722 1385 1161 1418 2064 1525 1798 1448 1514 1636 1825 1851
  DS2 raw  1119 1023 1638 1317 1155 1297 1309 1938 1361  909 1616 1709 1928 1534 1769 1365 1234 1742 1629
  DS2 in      - 1125 1156 1315 1882 1452 1955 1609 1382 1226 1260 1982 1261 1888 1506 1276 1615 1665 2265
  DS2 env     -  3.1  6.8 10.5 14.1 17.5 22.0 25.9 30.4 34.1 37.7 42.0 46.3 50.6 55.5 60.2 65.3 68.0 71.4
  DS2 maxp    - 0.71 0.69 0.69 0.65 0.67 0.67 0.68 0.69 0.43 0.65 0.67 0.65 0.67 0.65 0.65 0.64 0.71 0.67
  FS2 dep     -  348  379  430  426  473  805  697  568  678  695  714  672  846  827  849 1062  982 1042
  FS2 raw     -  474  331  385  465  439  565  620  537  712  600  639  643  898  996  860 1093 1046 1074
  FS2 in      -  370  349  415  446  441  800  748  491  548  565  601  918  781  825  712 1024  785  944
  FS2 env     -  2.8  5.4  8.9 11.8 14.7 17.7 20.2 22.7 25.5 28.3 31.3 34.4 38.0 40.9 44.6 48.6 51.6 55.1
  FS2 maxp    - 0.58 0.55 0.59 0.57 0.71 0.66 0.72 0.70 0.61 0.65 0.68 0.69 0.67 0.67 0.63 0.63 0.67 0.66
(env in thousands of the student's own base steps; both at the constant
ceiling 11.258 over the same 21 tokens, TK's schedules on the online
steps.)
NEAR RANDOM: neither (DS2 60-90 g 1.385, FS2 0.606).
READ (0) transfer: row 0 1065.0 deployed (raw 1119.3) against MASTER
  1113.86: -48.9, descriptive (+-464); NOT TRANSFER FAILED. Offline curve,
  deployed / raw: 948 / 1115 @4k, 942 / 1149 @8k, 1146 / 1035 @12k, 1326 /
  1226 @16k, 1065 / 1119 @20k.
READ (1) level 60-90: DS2 1656.52 (SD 167.2) against FS2 897.19 (SD
  138.8): +759.3 -> BETTER (band 232). Raw 1600.05 against 944.33: +655.7
  -> BETTER (band 205).
READ (2) head start 5-25: DS2 1330.27 against FS2 411.27: +919.0 ->
  BETTER (band 274.4). Raw 1286.00 against 418.60 (+867.4).
READ (3) against the master: DS2 1656.52 against 1113.86: +542.7 -> ABOVE
  (g 1.385; EfficientZero's 1281.2 is g 1). FS2 -216.7 -> SAME.
READ (4) time to level (the first row >= 881.86): DS2 row 0 (before it
  played); FS2 row 80 (48,563 env steps). DS2's lowest row in 5-25 is
  1200.0 (@10) against row 0's 1065.0: no forgetting.
READ (5) tokens: the 15 dropped tokens held 5.0% of G0's decisions. Play
  shares (4 / 8 / 16 / 32 frames): DS2 0.05 / 0.04 / 0.27 / 0.64 @5 ->
  0.10 / 0.03 / 0.22 / 0.64 @90; FS2 0.17 / 0.04 / 0.11 / 0.69 @5 -> 0.17
  / 0.00 / 0.00 / 0.83 @90 (FS2 all but stops using the 8- and 16-frame
  tokens). Frames per decision, deployed, rows 60-90: DS2 22.6-24.5, FS2
  23.1-28.6 (2.4-2.7 and 2.1-2.6 decisions per second; G0 2.6-2.9). FS2
  against TK-90 (the same recipe over 36 tokens, the growing clip) at
  60-90: +40.6, inside the band.
READ (6) stability, the largest adjacent deployed drop from row 45: DS2
  -539 (2064 @55 -> 1525 @60), FS2 -80 (1062 @80 -> 982 @85). Raw: DS2
  -403, FS2 -136.
READ (7) the clip: S(1) 0.344, c1 11.258, S(c1) 0.172 on G0's games (the
  prep). reward_kept by row: DS2 0.83 @5 -> 0.78 @90, falling steadily;
  FS2 0.81 @5, 0.94 @60, 0.86 @90 (more of a stronger player's reward is
  the capped kind).
READ (8) prior: R = maxp 75-90 / 30-55: DS2 0.670 / 0.633 = 1.06 (0.995
  without row 45's one-row dip), FS2 0.645 / 0.670 = 0.96 -> both HELD
  (>= 0.8). Fresh-target entropy without trend in either (DS2 0.49-1.65,
  FS2 0.46-1.66).
READ (9) the plateau rule (Step 0 (ii), one parameter set), after the
  fact: on deployed it never fires on either (DS2 sigma 235.5, best
  window 55-85 1687.0, last 1656.5; FS2 sigma 53.1, still rising, the last
  window its best). On raw it fires on DS2 at row 90 (its last row;
  best window 55-85 1611.5), never on FS2. The rank-based variant fires on
  DS2 deployed at 55 (its later rows averaged 1656.5 against the window
  then, 1584.5: a false plateau) and raw at 85.
READ (10) mechanism: FIXED at every row in both students. DS2 p1_ratio
  0.84 / 0.73 / 0.72 / 0.46 / 0.52 at rows 0 / 30 / 55 / 75 / 90; FS2 0.69
  / 0.40 / 0.60 / 0.88 at 30 / 55 / 75 / 90. Against the pooled statistics
  the step-1 offset would be DS2 1.78 / 6.88 / 5.30 / 4.88 / 4.79 SDs,
  FS2 0.40 / 0.80 / 3.64 / 6.31: without PB's fix both students' searches
  would have carried the artifact (DS2 from row 30, FS2 from ~75). No
  caveat on reads (1)-(9).
PRE-REGISTERED DECISION:
  - (1) BETTER -> the distillation step JOINS P3's swap.
  - TEST 3 (a seed-1 repeat of the pair) runs: read (1) and read (2) are
    BETTER. Read (10) FIXED throughout -> no diagnosis before it.
Beyond the letter (one seed, within-run bands -- "consistent with"):
  - The effect is large: +759 deployed is 3.3 bands (the distillation
    step's DS - FS was +197, inside its band). Two things changed between
    the two pairs. Here the students differ ONLY in the distillation step
    (the earlier pair also differed in clip and temperature schedules),
    and the master is stronger (G0 1114 against TK-90's 857).
  - Part of the gap is FS2 being slow: 897, below G0 (1114, -217), the
    earlier FS (1034) and PB (1032), above TK-90 (857) -- all inside the
    band -- and it reached level only at row 80. FS2
    differs from G0 in the vocabulary (21 tokens against 36) and the clip
    (a constant 11.26 against the sign clip; the fix is NO DIFFERENT by
    PB). Consistent with: a looser clip from scratch slows a fresh
    learner (G0, at the sign clip, is the best fresh run so far). That
    fits Kai's design: loosen the clip only at a swap, where the student
    starts from the master's play.
  - DS2 passed its master by 543 (g 1.39 against G0's 0.83) on 71.4k env
    steps of its own, beyond G0's 90 iterations; its curve is still level
    to rising at row 90 (the plateau rule does not fire on deployed).
  - FS2's play collapses onto the 32-frame tokens (0.83 at row 90; the
    16-frame ones 0.002), while DS2 keeps the 16-frame tokens (0.22-0.31;
    G0's most-used token was (LEFT, 16) at 0.278).
    The same 21-token menu, very different use: the distilled start
    carries the master's token habits.
  - Row 45's one-row diagnostics spike in DS2 (mean |prefix| 4.01, H(pi)
    1.90; losses unchanged) is the kind G0 and the earlier FS each showed
    once; it did not repeat.
  - Generation 2 (if Kai takes the loop on): DS2's weights-iter090 is the
    natural next master (training_steps 200,000 = 20,000 offline +
    180,000 online: the prep must count the offline steps, as noted in
    Test 2's AMENDMENTS), and the prep must read c0 and the vocabulary
    from the master's spec (c1 = 11.258 and DS2's 21 tokens become the
    generation-2 baseline).
KAI 2026-10-27, after this RESULT: "Generation 2 next" -- TEST 3 IS NOT RUN
(the proposal Kai approved framed the seed-1 repeat as "if it lands near
its bands"; this result is 3.3 bands out). Recorded for a later repeat:
a fresh seed-1 prep (Kai's answer). Generation 2 is pre-registered above.

### RESULT -- PB: per-step prefix BatchNorm statistics on the token arm (seed 0), 2026-10-25

RUN (mspacman-tokens-rgrow-pbn-s0-2026-10-25, from the run worktree
Y:\SwitchPlayer-pb-run, detached at 8ad72517): 05:23 -> 20:35 10-25 in ONE
attempt (54,781 s, 10.1 min/iteration; no deaths, no ALERTs,
supervisor.err empty; RUN COMPLETE). The hand-off ran the evals one at a
time while it trained and the last three by 20:41: all 18 rows in both
variants and read (1) at all four snapshots; every row 4/4 phases, no
capped game. Row 90 at 65.7k env steps (TK-90 58.5k, G0 59.2k). The
ceiling followed TK's schedule (1.57 @5, 26.3 @30, 160 from row 50;
reward_kept 1.0 from row 35).
EXACT eval as TK's (30 noop starts, prior argmax); DEPLOYED (acting states)
primary, RAW secondary, IN = the run's 8-game eval, maxp = the prior's mean
max probability, tgtH = fresh_target_entropy / ln 36, pfx1 / pfx5 = the
logged eval-mode prefix error at LSTM step 1 / step 5 (PB's with the
per-step sets, TK-90's pfx5 with the pooled statistics):
  iter          5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  PB dep      530  837  597  570  779  873  778  617  825  932 1131 1001  967  869 1150 1096 1016 1129
  PB raw      616  670  568  543  763  868  856  725  815  955 1153  959  957  986 1025  892 1059 1246
  PB in       709  702  562  482  979  750  759  692  876  681 1034  876  938 1112 1062 1321 1216 1120
  maxp       0.67 0.59 0.66 0.68 0.75 0.75 0.75 0.69 0.69 0.69 0.69 0.66 0.67 0.63 0.68 0.67 0.66 0.64
  tgtH       0.33 0.37 0.31 0.34 0.37 0.32 0.45 0.38 0.41 0.36 0.33 0.41 0.31 0.33 0.44 0.28 0.26 0.23
  pfx1       0.07 0.04 0.11 0.08 0.08 0.08 0.08 0.09 0.19 0.12 0.11 0.09 0.11 0.09 0.19 0.10 0.33 0.45
  pfx5       0.08 0.07 0.17 0.05 0.17 0.16 0.14 0.25 0.73 0.27 0.45 0.61 0.86 0.70 1.03 0.78 1.19 1.25
  TK-90 dep    60   60  271  482  665  524  913  742  732  659  879  780 1033  863  920  825  885  691
  TK maxp    0.27 0.19 0.61 0.60 0.65 0.65 0.66 0.62 0.66 0.64 0.61 0.57 0.46 0.47 0.40 0.35 0.27 0.20
  TK tgtH    0.34 0.41 0.43 0.39 0.35 0.32 0.47 0.36 0.40 0.39 0.69 0.79 0.78 0.85 0.93 0.93 0.46 0.92
  TK pfx5    0.50 0.03 0.07 0.07 0.14 0.17 0.27 0.23 0.47 0.92 1.18 0.73 1.95 1.32 2.61 3.91 2.65 3.45
NEAR RANDOM: no (deployed 60-90 g 0.745).
READ (1) MECHANISM: FIXED at all four snapshots. p1_ratio (eval |p1| /
  train-BN |p1|, with the arm's own step-1 set) 0.68 / 0.40 / 0.41 / 0.47 at
  rows 30 / 55 / 75 / 90 (FIXED <= 2; TK-90 8.20 / 20.95 / 23.63 at rows 55
  / 75 / 90 -- 8-24x, not the "8-27x" the pre-registration quoted, A-N1).
  Both directions (A-L5): the eval error errs LOW, |p1| 0.21-0.43 eval
  against 0.53-0.81 train-BN, as G0's did on its pooled statistics
  (p1_ratio 0.65-0.72). The step-1 batch sits 1.84 / 4.95 / 4.84 / 4.72
  pooled SDs off the pooled statistics and 0.18 / 0.40 / 0.31 / 0.36 off
  its own set. From row 55 the pooled statistics sat nearly as far off as
  TK-90's (7.09 / 6.49 / 6.79 at rows 55 / 75 / 90); the per-step set did
  not. Records: benchmarks/probes/step0_prefix_bn/pb_prefix_bn.jsonl (the
  hand-off's probe_prefix_bn-mspacman-tokens-rgrow-pbn-s0-2026-10-25.jsonl).
READ (2) TARGETS: mean tgtH over rows 60-90 0.324 -> HELD (<= 0.55; TK-90
  0.81). Rows 30-55 0.375 (TK-90 0.44). PB's targets sharpened late (0.44
  @75 -> 0.23 @90), where TK-90's went 0.39 @50 -> 0.79 @60 -> 0.93 @75.
READ (3) PRIOR: R = mean maxp 75-90 (0.6628) / mean maxp 30-55 (0.7082) =
  0.94 -> HELD (>= 0.8; TK-90 0.48, G0 0.99). policy_entropy / ln 36
  0.31-0.36 over rows 60-90 against TK-90's 0.45 @60 -> 0.87 @90.
READ (4) LEVEL: deployed mean 60-90 1032.43 (SD 99.6, g 0.745) against
  TK-90's 856.62: +175.8 -> NO DIFFERENT (band 232). Raw 1017.71 (SD 113.8,
  g 0.729) against 830.81: +186.9 -> NO DIFFERENT (band 205). Thirds 5-25 /
  30-55 / 60-90: deployed 663 / 859 / 1032 (TK-90 308 / 742 / 857); raw 632
  / 895 / 1018.
READ (5) descriptive:
  - The prefix errors. Step 1 (set 1): 0.04-0.19 through row 80, 0.33 @85,
    0.45 @90, while the step-1 targets grew (mean |target| 0.28-0.53 to row
    30, 0.68-0.88 at rows 35-60 and 70, 1.58-2.25 at 65 and from 75).
    Relative to the target that is 0.05-0.24 at every row, with no trend:
    the per-step set follows the scale. Step 5 (set 5): 0.05-0.27 to row 40,
    0.27-1.25 after, against TK-90's (pooled) 0.73-3.91 over rows 60-90 --
    a comparison that includes the fix (A-N7).
  - Tokens. Play shares over rows 60-90: 4-frame 0.010-0.035, 8-frame
    0.03-0.18, 16-frame 0.42-0.62, 32-frame 0.32-0.40. That is long tokens,
    as in G0 (32-frame 0.40-0.64), not TK-90's late shift to short ones
    (4-frame 0.32-0.51, 32-frame 0.10-0.14). Search shares @90 0.10 / 0.09 /
    0.37 / 0.45 (TK-90's near-even 0.28 / 0.27 / 0.24 / 0.21). Frames per
    decision, deployed (probe), rows 60-90: 19.6-22.1, so 2.7-3.1 decisions
    per second (TK-90 3.8-5.2, G0 2.6-2.9). Decisions per life 44-58
    (TK-90 57-84, G0 37-44).
  - The plateau rule (Step 0 (ii), one parameter set), after the fact: it
    never fires, on deployed (sigma 118.1; the best 7-row window is rows
    55-85 at 1032.8, the last 1032.4) or on raw (sigma 109.7; the best
    window is the last); the rank-based variant never fires either.
  - Stability: the largest adjacent deployed drop from row 45 is -131 (1131
    @55 -> 1001 @60; TK-90 -194, G0 -179).
PRE-REGISTERED DECISION:
  - (1) FIXED and (3) HELD -> the fix joins the base for every later arm:
    Test 2 is rebuilt with PREFIX_BN_PER_STEP in both students (an amendment
    to Test 2, re-audited) and launched.
  - (4) reported: NO DIFFERENT on both variants (+176 deployed, +187 raw).
Beyond the letter (one seed, within-run bands -- "consistent with"):
  - The one change removed the flattening. Under the same moving ceiling
    that flattened TK-90's targets from row 55 and its prior after them,
    PB's targets and prior held to row 90. With G0 (no artifact at the
    sign clip, nothing flattened) this is consistent with Step 0 (iii)'s
    reading: once the reward scale was large, the pooled prefix statistics
    flattened the search, and the flat targets flattened the prior. The
    tokens did not.
  - It did not buy a level by itself. PB's 1032 sits with FS (1034), above
    TK-90 inside the band and below G0 (1114; -82, inside the band). The
    finer reward scale (the ceiling at 160 from row 50) has still not paid
    for itself on Ms. Pac-Man. Test 2 asks whether a looser clip helps a
    student that starts from a good master.
  - The token play followed the search. With the targets sharp, 16- and
    32-frame tokens took 0.80-0.95 of play over rows 60-90, as in G0, while
    TK-90's late shift to 4-frame tokens came with near-even (flat) search
    shares. Consistent with: that shift was a symptom of the artifact, not
    a preference the tokens learned.
  - Longer decisions took more env steps per iteration: 65.7k at row 90
    against TK-90's 58.5k (+12%), so per env step PB's lead over TK-90 is
    a little smaller than per iteration.
  - For Test 2's rebuild (A-NIT-3): the master G0 is an off arm, so the
    prep's master search keeps the pooled statistics. G0's exploratory read
    shows them clean at G0's own scale (p1_ratio 0.65-0.72 at rows
    30-90), so the master's search needs no re-estimation. The students
    at c1 carry the fix.

### Next experiment (pre-registered) -- PB: per-step prefix BatchNorm statistics on the token arm (Kai, 2026-10-24)

DECISION: Kai, 2026-10-24, after Step 0 (iii) found that the value-prefix
head's BatchNorm flattens the search. Kai chose "test the fix on TK first",
then Test 2 with the fix, and the fix "per-step running stats". Test 2 is
STOPPED until PB reads: its waiting supervisors and hand-off were killed
2026-10-24 23:43 (first recorded as ~17:20; corrected by PB audit r2 B-L2)
before anything launched, and Test 2 is rebuilt with the fix after PB. G0
(Test 1) runs on unchanged.
QUESTION: does the fix keep the token arm's search targets and prior sharp
through 90 iterations (TK-90 flattened from row 55), and what does it do
to the score?
THE FIX (PREFIX_BN_PER_STEP, default off = bit for bit): every BatchNorm
layer in the value-prefix head (norm_in, norm_out and, when present,
fc_hidden_norm) keeps a separate set of running statistics for each LSTM
step since the last reset (1..PREFIX_LSTM_HORIZON). Training is unchanged:
the same train-mode calls on batch statistics, the same affine parameters,
and the pooled running statistics still updated. In addition, step k's
batch statistics update step k's set. Every eval-mode call (each search
engine's expansion, birth search, refresh / reanalyze, the prep's master
search, the logged diagnostics) normalizes each row with the set for its
own step: 1 after a fresh or reset state, otherwise steps since the reset
+ 1. A call without a step index is refused when the flag is on. Logged in
addition, for every arm: the eval-mode step-1 prefix error beside the step-5
one ("prefix"), since Step 0 (iii) showed the step-5 error hides step 1.
ARM (seed 0, 90 iterations, the TK recipe, --ray --cpu-actor): PB = TK
(effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow:
36 tokens, the clip growing 1 -> 160 by half the run, TK's schedules) with
PREFIX_BN_PER_STEP on. That is the one change. It runs after G0 completes,
from its own frozen run worktree. Reference: TK-90 itself (seed 0,
2026-10-19).
READS (EXACT eval as before: whole games, 30 noop starts, prior argmax;
rows 5-90; DEPLOYED primary, raw secondary). NEAR RANDOM first (deployed
60-90 < 502.08).
  (1) MECHANISM: the Step 0 (iii) probe at PB's snapshots 30, 55, 75 and 90,
      with the arm's own step-1 statistics. FIXED where the eval-mode step-1
      |prefix| <= 2 x its train-mode-BN value (TK-90: 8-27x from row 55).
      Not FIXED at any of 55-90 -> BUILD FAILED: diagnose, fix, rerun.
  (2) TARGETS: the mean of fresh_target_entropy / ln 36 over rows 60-90.
      HELD <= 0.55, FLAT >= 0.70, PARTLY between (TK-90: 0.81; its rows
      30-55: 0.44).
  (3) PRIOR: R = mean policy_maxp over rows 75-90 / over rows 30-55. HELD >=
      0.8, FLATTENED <= 0.6 (TK-90: 0.48).
  (4) LEVEL: deployed mean 60-90 against TK-90's 856.62: BETTER / NO
      DIFFERENT / WORSE at +-232; raw against 830.81 at +-205.
  (5) the logged step-1 prefix error by row, against the step-5 one;
      tokens' play share; the general plateau rule (Step 0 (ii)) after the
      fact, descriptively.
DECISION AFTER: (1) FIXED and (3) HELD -> the fix joins the base for every
later arm: Test 2 is rebuilt with it in both students (an amendment to
Test 2, re-audited) and launched; (4) is reported. (1) FIXED but (3)
FLATTENED or between -> the search artifact was not the only cause:
diagnose (the probes again on PB's snapshots); Test 2 with or without the
fix is Kai's call. (1) not FIXED -> BUILD FAILED as above. (4) WORSE
while (3) HELD -> Kai's call.
BUILD (audited like every build): the per-step statistics in
networks.ValuePrefixNetwork behind the flag, with a per-row step index (a
batched expansion mixes rows at different depths); the step index through
every caller (training unroll, both muzero_vtree engines, the sequential
engine, refresh, the diagnostics); the new logged field; the PB arm; a
supervisor (supervise_gen0.py's, waiting for G0's completion marker before
its preflight); the eval hand-off; tests (off: bit for bit; on: train-mode
outputs identical, eval-mode per-row statistics, every caller passes the
index).
COST: ~16 h (TK-90: 15.4 h) after G0 (~06:30 10-25); Test 2 then ~33 h.
BUILT 2026-10-25 02:45 (eb94fd6e, pre-audit).
AUDIT r1 (2026-10-25, two fresh-eyes auditors on eb94fd6e). A (the fix):
nothing above LOW. Verified: the per-step fold (BN's own update: unbiased
variance, its momentum, the cumulative average when None) and the eval
normalization (BN's eval formula with set k); no AMP anywhere; buffers
persistent (state_dict, .to(), deepcopy), strict loads, the target model
built and synced with them; EVERY value-prefix caller passes the right step
-- the training unroll i (a fresh state, never reset inside), the
sequential expand_prefixes and both vtree engines 1 after a reset else
prefix_len + 1 (root, chain restarts, horizon resets; chain nodes carry no
state), ReDo k + 1, the diagnostics i, the inference server per payload;
birth search, the prep's master search, SVE and refresh ride on those
engines. OFF bit for bit on real weights (TK@75: VectorizedPUCT 12 roots x
36 tokens x 50 sims and the sequential engine -- visits, policies, values
and the RNG draws identical to the parent commit). The diagnostics block
is already in eval mode in production (the forced eval is a no-op: the
step-5 "prefix" keeps its meaning for every off arm). probe_prefix_bn.py
reproduces xrun_results.jsonl at TK@55/75/90 in every field; a simulation of
the step-1 set PB's training would estimate (train-mode upstream latents)
gives 1.06 / 1.02 / 0.83 at TK@55/75/90 (pooled: 8.2 / 20.9 / 23.6). B (the
orchestration): one MEDIUM -- read (1) had no producer (the hand-off left
it "by hand", i.e. from a tree that will have moved on).
FIXES r1:
  - B-M1: read (1) runs IN the eval hand-off, from the frozen run tree:
    benchmarks/probes/probe_prefix_bn.py --states
    benchmarks/probes/step0_prefix_bn/states.npz --out
    benchmarks/probes/probe_prefix_bn-mspacman-tokens-rgrow-pbn-s0-2026-10-25.jsonl
    <snapshot> at 30, 55, 75, 90 -- each as it settles (after its exact
    evals), any missing in the final pass (counted with the missing rows).
    The supervisor refuses a fresh start beside a leftover read (1) jsonl
    as it does the exact-eval jsonls.
  - B-L2: the any-trainer guard also runs once BEFORE the preflight (a
    resume beside an orphaned trainer, or G0's trainer still shutting down,
    would otherwise run the 21 suites beside a live trainer).
  - B-L3: Test 2's stopped launcher renamed launch_gen1.ps1.stopped (and
    Test 2's LAUNCH note marked STOPPED).
  - B-L4: states.npz (read (1)'s input) joins AUDITED.
  - B-N1 / N2: two supervisor tests made able to fail / bounded; B-N4: the
    EVAL line shows fresh_target_entropy, prefix and prefix_step1 (reads (2)
    and (5) live); B-N5: the quick probe has a 900 s timeout.
  - A-L2 / L3 / L4: the probe test distinguishes step sets (set 1, not set
    2, not pooled, not an off-by-one), the diagnostics test checks
    prefix_step1's value independently, and one flag-on test runs a real
    Run's birth search checking every head call's step.
  - A-N1: "8-27x" was wrong: TK-90 read 8.20 / 20.95 / 23.63 at rows 55 /
    75 / 90, so 8-24x. The FIXED line (<= 2x) is unchanged.
ACCEPTED r1 (LOW / NIT):
  - B-L1: G0's ABANDONED file also releases PB's wait (PB needs only the
    GPU). Procedure: abandon G0 only after its trainer is stopped (the
    guard before the preflight covers a slip); to un-abandon G0, stop PB's
    supervisor and trainer first.
  - A-L1: the Step 0 search-side probes (step0_prefix_bn/xrun_prefix.py,
    probe_qspread.py) do not run on this tree (their _expand_forward
    overrides take no step), and on a PB snapshot their recalibration would
    reset only the pooled statistics -- the per-step sets eval mode reads
    would stay (silently). They matter only on the "(1) FIXED but (3)
    FLATTENED -> diagnose" branch; they are ported (a step parameter and a
    per-step-set reset) and audited before any such use.
  - A-L5: read (1) is one-sided (<= 2x); clean snapshots read 0.12-2.00 and
    an eval prefix that errs LOW passes. Reported descriptively at every
    snapshot, both directions.
  - A-N7: PB's logged step-5 "prefix" is normalized with set 5, TK-90's
    with the pooled statistics: a cross-arm comparison of that field
    includes the fix (read (5) says so).
  - A-N2 / N3 (a small per-expansion overhead on and off; no number
    changes), A-N4 / N5 / N6, B-N3 / N5-inherited: noted.
AUDIT r2 (2026-10-25, two fresh-eyes auditors on 7c97b48b): A (the r1 tests
and a second look at the fix) nothing above LOW; B (the orchestration)
nothing above LOW. THE AUDIT LOOP IS CLOSED. Verified in r2: every r1 test
fails under the bug it targets (mutants: truth[:, 0:1], the step-1 field
taken at step K or 2, the probe at step 2, the offsets read from set s,
the train-mode copy in eval, step.repeat for repeat_interleave in either
engine, a biased-variance fold -- all caught); TK's behaviour is unchanged
from a81bcf39 (TK-90's tree) to eb94fd6e apart from the flag
(SCHEDULE_OFFSET_STEPS 0 is the identity, the new arms copy TK, the
gates and banners take TK's branches); every net rebuild / load / copy
carries the step sets and nothing resets or copies BN statistics by name
past them (ReDo's recycle handles them); probe_prefix_bn peaks at 2.4 GB
commit (7 s) on a 36-token snapshot and fits the interim 6 GB gate; the
hand-off regenerates byte-identically and its read (1) runs from the
frozen tree on CPU, never two on one jsonl.
ACCEPTED r2 (LOW / NIT):
  - A-L1: tests/test_pbn.py's pb_runs fixture lowers torch's thread count
    to 2 and never restores it, so the preflight's later suites run on 2
    threads -- slower, not wrong (four later suites set 2 themselves).
  - B-L1 (fixed in the scratchpad hand-off, text only; confirmed by a
    fresh-eyes r3 check): the UN-ABANDON / FRESH-restart procedures and the
    abandon pass's log line name all three probe jsonls (read (1)'s too).
  - B-L2: Test 2's stop time corrected (23:43, not ~17:20) here and in
    Test 2's section.
  - B-L3: a FAILED G0 final eval pass retries at +10 / +20 / +30 min and
    could start a probe beside PB's young trainer at G0's 4 GB gate. Watch:
    if post_gen0.log shows a final-pass ALERT after G0 completes, watch
    PB's first attempt's memory.
  - A-NIT-3, for the Test 2 rebuild (not a PB defect): the distillation
    prep's master search runs under the MASTER's arm (distill.py
    arm_config(blob["arm"])), and G0 is an off arm with no step sets, so
    DS2's prefill targets from G0's search keep the pooled statistics even
    with the fix in both students. The Test 2 amendment must say so, or
    re-estimate the master's per-step statistics.
  - A-NIT-1 / 2, B-N1-N9: noted (B-N9: memory note corrected).
LAUNCHED 2026-10-25 03:55 from Y:\SwitchPlayer-pb-run (detached, CRLF, at
8ad72517; manifest written from the audited commit b992b6fb, 51 AUDITED
files; the 21 preflight suites pass in that tree: 943 passed, 3 skipped,
CPU): launch_pb.ps1 started post_pb.py (sha256 58b483a8368c1fe8...09782e20,
regenerated byte-identically by make_post_pb.py; log scratchpad
post_pb.log) and supervise_pb.py pb 0, which WAITS for G0's completion
marker (first line: "WAITING for G0 before pb s0 [0 h]", no ALERT), then
the any-trainer guard, the preflight and the run. Rerun launch_pb.ps1 after
a reboot or a pause (idempotent).
TRAINING since 05:23 10-25 (G0's marker 05:15; preflight ok: quick probe +
21 suites; attempt 1). INTERIM (rows 5-55; not read against the
pre-registration until row 90): the ceiling on TK's schedule (1.57 @5,
26.3 @30, 160 from 50); 10.0 min/iteration at row 35; no ALERT.
  read (1) so far: p1_ratio 0.68 @30, 0.40 @55 (TK-90 8.20 @55) -> FIXED at
      both; the step-1 batch sits 1.84 / 4.95 pooled SDs off the pooled
      statistics at 30 / 55 and 0.18 / 0.40 off its own step set.
  targets H/ln 36 0.31-0.45 through 55 (TK-90 0.69 @55); prior maxp
      0.69-0.75 from row 25 (0.69 @55).
  deployed  530  837  597  570  779  873  778  617  825  932 1131 (rows
      5-55; TK-90 60 60 271 482 665 524 913 742 732 659 879); raw 616 670
      568 543 763 868 856 725 815 955 1153.

### Next experiment (pre-registered) -- Test 2: the first swap under the per-generation clip (Kai, 2026-10-24)

DECISION: Kai's "Let's start with running the tests you proposed" (Test 1's
DECISION has the proposal and its defaults). Test 2 runs after Test 1 (G0)
completes and its evals land.
QUESTION: under Kai's swap design -- one clip per generation, loosened only
at the swap -- does the distillation step give the student a head start,
or a higher level, than a fresh student with the same vocabulary AND the
same clip? (The distillation step's DS and FS also differed in their clip
and temperature schedules; here the distillation step is the only
difference.)
THE SWAP (generation 0 -> 1):
  MASTER = G0's final snapshot (weights-iter090, its acting states). G0
    NEAR RANDOM (Test 1's first read) -> no Test 2 until diagnosed (Kai's
    call); G0 ABANDONED -> Kai's call.
  RECORDING, VOCABULARY, RE-TOKENIZATION, TARGETS: the distillation step's
    (the master's fresh games at its final temperature 0.125 through its
    acting states, at raw-step resolution, whole games until >= 31,250 base
    steps; the 1% rule; dropped tokens as raw 4-frame records; the masked
    read-out of the master's 50-sim birth search). The student's value is
    its own TD on the recorded rewards clipped at ITS ceiling (no value
    distillation, so the master's sign-clip value scale needs no
    rescaling).
  THE STUDENT'S CLIP (Kai's "decrease the amount clipped", across
    generations; rule (a)): on the recorded games' raw per-base-step reward
    magnitudes m (in REWARD_UNITs), the share a ceiling c removes is
    S(c) = 1 - sum(min(m, c)) / sum(m). The master's ceiling c0 = 1 removes
    S(1); the student's ceiling c1 is the smallest c with S(c) <= S(1) / 2
    (S is continuous and non-increasing: c1 is solved exactly on the sorted
    magnitudes). No game constant beyond REWARD_UNIT (the smallest positive
    reward, which makes c0 = 1 the sign clip); each later swap halves again.
    c1 past the reward supports (the growth gate's check) -> PREP FAILED,
    Kai's call. Logged: S(1), c1, S(c1), the share of base steps clipped.
ARMS (seed 0, the TK recipe, 90 iterations each, one after the other):
  DS2 = the TK arm over the student vocabulary at the CONSTANT ceiling c1,
    with the distillation step's prefill and 20,000-step offline phase,
    then TK's within-run schedules (prior-acting temperature 0.5 / 0.25 /
    0.125 staged, long-token bias 0.5 -> 0 by 25%, PER beta 0.4 -> 1, the
    off-policy horizon decay) KEYED TO ITS ONLINE STEPS: the offline 20,000
    steps do not advance them, so DS2 and FS2 see the same settings at
    every iteration.
  FS2 = the same vocabulary at the same constant c1, from scratch, TK's
    schedules.
READS. EXACT eval as before (whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4); DS2 rows 0 and 5-90 plus its offline
curve, FS2 rows 5-90; DEPLOYED primary, raw secondary; MASTER = G0's 60-90
deployed mean. The distillation step's reads (0)-(6) with DS2 / FS2 for DS
/ FS (NEAR RANDOM first; (0) transfer at row 0 and the offline curve; (1)
level 60-90 +-232, raw +-205; (2) head start 5-25 +-274.4; (3) DS2 against
MASTER +-232; (4) time to level; (5) tokens; (6) stability), plus:
  (7) the clip: S(1), c1, S(c1) on the master's games; reward_kept by row.
  (8) the prior: R = mean policy_maxp over rows 75-90 / over rows 30-55 per
      student (Test 1's read (1)), and the search targets' entropy by row.
  (9) the general plateau rule (Step 0 (ii)) applied after the fact to each
      curve, descriptively.
DECISION AFTER: the distillation step's table, unchanged ((1) BETTER ->
the distillation step joins P3's swap; (1) NO DIFFERENT + (2) BETTER ->
it joins for the time it saves, Kai confirms; (1) NO DIFFERENT + (2) NO
DIFFERENT -> by read (0): TRANSFER FAILED -> fix and rerun DS2; forgetting
(DS2's rows 5-25 below row 0) -> the annealed kickstarting variant; else
Kai's call; (1) NO DIFFERENT + (2) WORSE or (1) WORSE -> diagnose). TEST 3
(a seed-1 repeat of the pair) runs when read (1) or (2) is BETTER, or NO
DIFFERENT with its deployed difference beyond half its band in either
direction; Kai's call otherwise.
BUILD (before launch; audited like every build): the clip rule in the
prep (experiments/distill.py: S, c1 into vocab.json); the generation-1
student arms taking --token-vocab and the prep's ceiling; DS2's schedule
offset (the offline steps excluded from every within-run schedule); a
supervisor that waits for G0's completion marker and both G0 probe jsonls'
rows 5-90, then the prep, DS2, FS2 (the distillation step's, the master
fixed to G0); the eval hand-off.
COST: prep ~0.5 h; DS2 ~17.5 h; FS2 ~15 h; after G0 (~16 h).
AMENDMENTS (audit r1 A, the core, before launch): (M1) the off-policy
horizon decay is age-based, so the offset cancels for each student's own
data, but the prefill was born at step 0 and so aged through the offline
phase (horizon 4 instead of 5 over online steps ~34k-54k). It is now born at
SCHEDULE_OFFSET_STEPS, so its age is 0 offline and k at online step k, as
written above. (L1) the c1 solver now takes the first segment whose
breakpoint reaches the target and clamps into it. The old window test could
miss both neighbours of a duplicated non-integer magnitude and return "clip
nothing"; Ms. Pac-Man's integer units could never trigger it, but other
games could. (L2) the slow prep tests now record rewards past one unit, so
c1 > 1 is exercised end to end. (L3) both students check vocab.json's
master_arm and master_iterations against new arm keys (_master_arm G0,
_master_iterations 90); FS2 has no set header. Logged too:
clipped_base_share_master/student, the share of ALL base steps clipped (the
"share of base steps clipped" above), beside the reward-earning-step
shares. NOT offset, by design: LR warmup (1,000 steps; offsetting it would
hold the offline phase at LR 0) and every iteration-keyed cadence (refresh,
eval, checkpoint). Noted for later generations: distill.prep needs the
master's training_steps to be iterations x 2,000, so a DS2 master (200,000
steps after 90 iterations) needs the prep to count its offline steps first.
AMENDMENTS (audit r1 B, the orchestration: nothing above LOW): a failed
prep attempt's line now carries prep.log's last line, its reason (a
deterministic failure costs ~33 min: 3 tries, 15 min apart). fs2's hourly
ALERT for a stopped DS2 supervisor says "Kai's call" when that stop was the
master gate ("PREP FAILED: MASTER ..."), since rerunning the launcher only
repeats it. The reboot note now reads: rerun launch_gen0.ps1 until G0's
evals have landed. A record reward past the reward support's edge (400
units) ALERTs again, beside the 8 x c1 bound. Accepted as they stand: three
host kills before DS2's first checkpoint (~1 h 20 min: the prefill and the
offline phase) stop it with NO PROGRESS, and rerunning the launcher recovers
with nothing lost, so pause before Switch sessions; an ABANDONED DS2 leaves
fs2 waiting with hourly ALERTs (Kai's call); the master gate is not
re-checked between a finished prep and the first launch; a manifest read
error inside the prep is fatal (inherited); set_intact pins G0's arm and
iterations, not its snapshot path (the supervisor passes the one path).
AUDIT r2 A (the core, after the r1 fixes): nothing above LOW. The fixes
were checked against exact rational arithmetic (996 cases, 0 failures) and
every born_step reader (horizon, refresh rotation, priorities, eviction,
resume, Ray). The old DS arm is bit for bit unchanged. Noted for
generation 2 (no effect on Test 2): the prep reads the master's ceiling c0
and vocabulary from ARMS[its arm], not from the snapshot's spec. A
generation-1 master's ARMS entry still holds G0's ceiling 1, since
--token-vocab is never applied in the prep. So before the next swap the
prep must take c0 (START == CEILING) and the vocabulary from blob["spec"];
until then it must refuse a _ceiling_from_prep master.
AUDIT r2 B (the orchestration, after the r1 fixes): nothing above LOW. In a
CRLF export of 59932305, the 19 preflight suites gave 845 passed and 4
skipped (CUDA-only, plus a git-history test that needs .git and was checked
by hand), and both quick probes passed. The AUDITED list matched every repo
file the suites open. Accepted as they stand: after an un-abandoned G0 and
a relaunch, the waiting ds2 supervisor prints only WAITING lines, so its
old "PREP FAILED: MASTER" line still reads as the newest give-up. fs2 then
ALERTs "gave up" hourly until DS2 launches, which misleads but never
starts a wrong run. Other accepted points: an attempt killed before it
writes can show the previous attempt's last prep.log line; 973.9 is
EfficientZero's score minus random, not a human score (the 502.08
threshold is as pre-registered); there is no timeout on the quick probe
(inherited; post_gen1's 4-h silence ALERT covers a hang). Host commit fell
to 1.9 GB while three audit processes ran beside G0, so avoid parallel
suite runs while a run trains. THE AUDIT LOOP IS CLOSED (r1: one MEDIUM,
fixed; r2: nothing above LOW).
LAUNCHED 2026-10-24 16:29 from Y:\SwitchPlayer-gen1-run (detached, CRLF; manifest
commit a852bb1b, 46 AUDITED files): launch_gen1.ps1 started post_gen1.py and
the ds2 and fs2 supervisors, which WAIT (ds2 for G0's completion marker and
its evals' rows 5-90; fs2 for DS2). Nothing trains or preflights until G0
is done (~06:30 10-25) and evaluated.
STOPPED 2026-10-24 23:43 (Kai's PB decision, the PB section above): all six
processes -- post_gen1.py and the ds2 / fs2 supervisors, each a venv
launcher plus its child -- were killed by exact PID before anything
launched (their logs end with the 23:30 WAITING lines; a process-table
query found none since; PB audit r2 B-L2 corrected the time, first
recorded as ~17:20). The scratchpad launcher is
renamed launch_gen1.ps1.stopped (PB audit r1 B-L3), so a reboot procedure
cannot re-arm the pre-fix Test 2 beside PB; Test 2 is rebuilt with the fix
(a new manifest, worktree and launcher) after PB reads.
REBUILD WITH THE FIX (pre-registered 2026-10-25, before the rebuild; PB's
DECISION AFTER: read (1) FIXED and (3) HELD, so the fix joins the base for
every later arm). Everything above stands except:
  - BOTH STUDENTS carry PREFIX_BN_PER_STEP, so the pair still differs only
    in the distillation step. The two arm keys stay (-gen1-distill /
    -gen1-fresh) with the flag added: they never trained, because the
    pre-fix Test 2 stopped before its prep. The runs are dated 2026-10-25
    and run from a new worktree (Y:\SwitchPlayer-gen1b-run), so nothing
    from the stopped tree is reused.
  - THE MASTER is unchanged: G0's weights-iter090, an off arm with no
    per-step sets. The prep's master search (distill.py, under G0's arm)
    therefore keeps G0's pooled statistics (PB audit r2 A-NIT-3). Test 1's
    exploratory read shows them clean at G0's own scale (p1_ratio
    0.65-0.72 at rows 30-90, under PB's 2x line), so the master's policy
    targets need no re-estimation. No G0 value target reaches a student
    (each student's value is its own TD, recomputed at batch time); G0's
    search, through its pooled prefix head, shapes only DS2's prefill:
    its policy targets and, through the stored root values, its initial
    replay priorities (corrected by audit r1 A-NIT-1).
  - The master gate is settled: G0's deployed mean 60-90 is 1113.86, not
    NEAR RANDOM; MASTER for read (3) = 1113.86.
  - NEW READ (10) MECHANISM, per student: probe_prefix_bn.py (PB's read
    (1)) at rows 30, 55, 75 and 90 (DS2 also at row 0), run by the eval
    hand-off from the frozen tree on Step 0 (iii)'s 480 states. FIXED as
    in PB (p1_ratio <= 2), reported in both directions (PB A-L5). A
    student not FIXED at one or more of rows 55, 75 and 90 -> its search
    is not clean: reads (1)-(9) are reported with that caveat, and the fix
    is diagnosed before any Test 3 (Kai's call). (Worded so by audit r2
    A-L1: "not FIXED at any of" read both ways.)
  - The preflight adds tests/test_pbn.py (the fix's suite); the probe and
    its states join AUDITED.
  COST unchanged: prep ~0.5 h, DS2 ~17.5 h, FS2 ~15 h (PB ran at TK's
  pace, 10.1 min/iteration).
REBUILT 2026-10-25 (35982db6, pre-audit): the flag in both student arms;
the supervisor's runs and prep dated 2026-10-25, tests/test_pbn.py in its
SUITES (20 suites), probe_prefix_bn.py and states.npz in AUDITED, read
(10)'s jsonl refused like the exact evals' on a fresh start. The eval
hand-off post_gen1b.py (scratchpad; make_post_gen1b.py derives it from
post_gen1.py's audited derivation plus PB's read (1) machinery) runs read
(10); launch_gen1b.ps1 starts it and both supervisors from the new tree.
AUDIT r1 (two fresh-eyes auditors on 35982db6). A (the core): one HIGH,
already found by the lead's full preflight run and fixed in f4907d9d --
test_pbn's off-arms test still required PB to be the only flag-on arm, so
the preflight could never pass (loud, no wrong data); now it requires
exactly PB and the two students. Otherwise nothing above LOW. Verified:
every value-prefix call on the student paths passes its step (the offline
phase through train_step, birth search on the deep-copied nets, refresh on
the target nets, the diagnostics, resume; the prefill makes no head
calls); no master net or BN statistic enters a student (--init-weights is
refused; G0 loads strictly under its own off arm, and a flag-on build
refuses G0's state strictly); the flag reaches every weights file's spec,
the Ray overrides and both probes (synthetic flag-on DS2 / FS2 snapshots:
probe_prefix_bn A = 31, per_step 5, the globals restored; the exact-eval
probe's strict loads); BN recalibration and ReDo leave the step sets
alone; seven mutants each caught by the new tests; G0's p1_ratio
0.65-0.72 and deployed mean 1113.86 confirmed from the records. B (the
orchestration): one MEDIUM -- with G0 complete, the stopped pre-fix tree's
supervisors would no longer wait: started by hand or after a reboot (its
docstring still named launch_gen1.ps1), they would run an UNFIXED DS2,
and launch_gen1b.ps1 would skip its own as "already running".
FIXES r1:
  - B-M1: the stopped tree is disarmed -- an ABANDONED file in both of its
    run folders (Y:\SwitchPlayer-gen1-run\benchmarks\results\mspacman-
    gen1-{distill,fresh}-s0-2026-10-24; its supervisors exit before any
    wait); the supervisor's PROCEDURES name launch_gen1b.ps1 and
    post_gen1b.py, drop the launch_gen0.ps1 clause and name read (10)'s
    jsonl in UN-ABANDON and FRESH.
  - B-L1 (PB's r1 B-L2): the any-trainer guard runs once before the
    preflight too (a relaunch beside an orphaned trainer or prep would
    otherwise run the 20 suites beside it); tested.
  - B-L2: the hand-off checker pins read (10)'s "exited 0 but no row"
    branch (judged on read (10)'s own jsonl) and the final pass's missing
    count (n == 10 in the abandon case, read (10)'s row 0 among them), and
    that the offline curve's abandon line names no read (10) jsonl.
  - B-L3: fs2 waits on DS2's exact-eval rows, not read (10): the
    predecessor comment says so, and the hand-off's final-pass-FAILED line
    reads "fs2 waits on the exact-eval rows among them".
  - A-L1 / B-N2: the quick probe asserts the flag on, the module default
    off and G0 off; G0's arm fails both students' probes (tested).
  - B-N3 (PB's r1 B-N4 / N5): the EVAL line shows freshH, prefix and
    prefix_step1 (read (8) live); the quick probe has a 900 s timeout
    (tested).
  - B-N1: the hand-off's vanish guard is documented as watching the exact
    evals' rows (a moved read (10) jsonl is just re-probed).
  - B-N5: launch_gen1b.ps1 refuses a manifest that does not pin
    probe_prefix_bn.py (the pre-fix one), before anything starts.
  - A-NIT-1: the MASTER bullet above corrected; A-NIT-2: the DS2 offline
    test checks all three step-set BatchNorms (fc_hidden_norm too).
ACCEPTED r1: B-N4 (the suite count: 20, not 22 as first said). B's
auditor ran two no-op commands with the system python3, outside the lock
(nothing imported).
AUDIT r2 (two fresh-eyes auditors on 06d8d126): A (the core) nothing above
LOW; B (the orchestration) nothing above LOW. THE AUDIT LOOP IS CLOSED.
Verified in r2: every r1 fix in code and tests (core mutants: either
student's flag off or not True, G0's on, student_arm dropping it, the
offline phase in eval mode, no fc_hidden_norm fold -- each caught;
supervisor mutants: the guard missing or before the prime, the quick
probe unbounded or its timeout taken as success, read (10)'s jsonl
dropped from the refusal, freshH missing, SUITES / AUDITED / dates wrong
-- each caught; 17 hand-off mutants, all caught by the checker); an
end-to-end student run with every head call and fold instrumented (the
prep under G0's off arm folds nothing; every fold inside train_step on the
live head, each set exactly once per step; refresh, birth search and the
diagnostics in eval mode with their steps; resume restores the step sets
exactly; a fresh-interpreter --cpu-actor SelfPlayActor builds per_step 5
and strict-loads; both probes on row 0 and the offline curve); the
disarmed tree's supervisor (a852bb1b) returns "not launching" before any
wait in a trapped copy, and reaches the preflight with ABANDONED moved
away (the r1 MEDIUM was real); a CRLF export: all 49 AUDITED files hash
as here (states.npz is -text), a manifest from this tree passes there,
217 targeted tests pass; the G0 wait and the master gate (1113.857) pass
on the real G0; the derivations regenerate byte for byte.
FIXES r2 (text and tests, LOW / NIT; confirmed by a fresh-eyes r3 check,
as PB's r2 B-L1 was):
  - A-L1: read (10)'s caveat rule worded "not FIXED at one or more of rows
    55, 75 and 90" (above; "not FIXED at any of" read both ways).
  - B-L1 (inherited): PAUSE also kills a running preflight (python -m
    pytest in the run tree): the guard does not count it, and a quick
    resume would run the 20 suites twice. B-N1: the docstring names the
    guard before the preflight.
  - B-N2: the hand-off's docstring and final-pass comment say fs2 waits on
    DS2's exact evals only.
  - A-NIT-1: each of the quick probe's three flag asserts alone fails it
    on an otherwise real student arm (6 cases, with the unbroken probe as
    the control); A-NIT-2: the DS2 offline test requires each step set
    folded exactly once per offline step; A-NIT-3: a log line's count.
ACCEPTED r2: both auditors ran one no-op command with the system python
outside the lock (json only, nothing written); r1's "seven mutants" are
the r1 auditor's own set (r2's are listed above).
AUDIT r3 (a fresh-eyes check of FIXES r2 at 99c4db3e): nothing above NIT.
Verified: the read (10) rule reads one way; the docstring matches the
guard's three sites; each of the three flag cases trips only its own
assert (all 13 asserts given messages in a patched copy), 9 vacuous
rewrites x 2 variants all caught, the control passes; the exact fold
count catches a double fold, a skipped first fold and a skipped step 5;
post_gen1b.py differs from the r1 derivation by the two intended texts
only (byte-identical regeneration, <= 79 columns, compiles), its checker
passes; 193 targeted tests pass. ACCEPTED r3 (NIT): a flag assert
weakened to spec.get(..., True) passes the new test (a student arm
without the key is still caught in the preflight by test_gen1 and
test_pbn); the test does not pin WHICH assert fired; the guard runs
before the first preflight attempt, not before each 10-min retry (CPU
contention at worst); PAUSE's "by command line" for the preflight is its
-m pytest argv over the 20 suites (the tree is its working directory);
two ragged lines.
LAUNCHED 2026-10-25 22:16 from Y:\SwitchPlayer-gen1b-run (detached, CRLF,
at 0467dd90; gen1_manifest.json written from the audited commit fe4ef2ad,
49 AUDITED files; in that tree the manifest check, both quick probes, the
G0 wait -- nothing missing -- and the hand-off's code check pass):
launch_gen1b.ps1 started post_gen1b.py (sha256 51c36cdcb3055517...,
regenerated byte-identically by make_post_gen1b.py; log scratchpad
post_gen1b.log), supervise_gen1.py ds2 0 (the preflight's 20 suites in
that tree, then the master gate, the prep ~0.5 h, DS2) and
supervise_gen1.py fs2 0 (WAITING for DS2). The stopped pre-fix tree stays
disarmed. Rerun launch_gen1b.ps1 after a reboot or a pause (idempotent;
PAUSE also kills a running preflight).
PREFLIGHT ok 22:26 (manifest + quick probe + 20 suites in the run tree);
MASTER G0 weights-iter090 (deployed 60-90 1113.86, g 0.828); PREP done
22:29 in one attempt (4 min). The prep, descriptive (read (7)'s first
half):
  recording: 45 games of G0 at temperature 0.125, 135 lives, 31,784 base
      steps; game returns mean 1176.0 (490-1910).
  THE CLIP: G0's sign clip (c0 = 1) removes S(1) = 0.344 of the recorded
      5,292 reward units; the halved-clip rule gives c1 = 11.258 units
      (112.6 points per base step: pellets, power pellets and the cherry
      pass whole, ghosts and the later fruit are cut), which removes
      S(c1) = 0.172. Base steps
      clipped: 0.42% -> 0.19% of all base steps (3.9% -> 1.7% of the
      3,469 reward-earning ones).
  VOCABULARY: 21 of 36 tokens kept (every 4-frame token, by design, plus
      the 12 longer ones at >= 1% of G0's 5,615 decisions; the largest
      (3,16) 0.278, (7,32) 0.149, (5,32) 0.146, (6,32) 0.079); 15 dropped.
      6,607 re-tokenized records (278 dropped-token decisions split into
      857 extra 4-frame records); 21 uniform fallbacks; mean master visit
      mass on dropped tokens 0.177.
DS2 LAUNCHED 22:30 10-25 (attempt 1, fresh): the prefill, then the 20,000
offline steps (~1 h 20 min to the first checkpoint), then 90 iterations.
DS2 ROW 0 (interim, 00:25 10-26; the offline phase took ~1 h 33 min):
  READ (0) transfer: row 0 (20,000 offline steps, before DS2 played)
      1065.0 deployed (SD 413.7, min 570; g 0.778), raw 1119.3, against
      MASTER 1113.86: -48.9, descriptive (+-464); NOT TRANSFER FAILED. The
      offline curve, deployed / raw: 948 / 1115 @4k, 942 / 1149 @8k,
      1146 / 1035 @12k, 1326 / 1226 @16k, 1065 / 1119 @20k (row 0) --
      every row within ~170 of MASTER from 4k steps on; G0 scored 1176.0
      in the recorded games. (The distillation step's DS: 660 at row 0
      against its master's 857, 492 at 4k.)
  READ (10) row 0: p1_ratio 0.84 -> FIXED (|step-1 prefix| 0.611 in eval
      mode against 0.725 with batch statistics); the post-LSTM BN's step-1
      input offset is 0.25 running SDs against its step set (1.78 against
      the pooled statistics).
  Frames per decision, deployed: 22.5 at row 0 (21.3-22.4 offline).
DS2 ROWS 5-30 (interim, 05:12 10-26; 10.1-10.4 min per iteration):
  deployed 1226 @5, 1200 @10, 1295 @15, 1533 @20, 1397 @25, 1944 @30
      (SD 301-510 per row; g 1.68 at row 30, one row); raw 1023, 1638,
      1317, 1155, 1297, 1309.
  rows 5-25 (read (2)'s window): deployed 1330.3, raw 1286.0 (FS2's to
      come). Read (4): DS2 is at level (>= 881.86) from row 0; its lowest
      row in 5-25 is 1200.0, above row 0's 1065.0: no forgetting as it
      began to play.
  READ (10) row 30: p1_ratio 0.73 -> FIXED (0.649 eval against 0.884);
      the step-1 offset is 0.12 running SDs against its step set and 6.88
      against the pooled statistics (row 0: 1.78): pooled statistics
      would already be far off at step 1 by row 30.
  Tokens (the training evals' play shares, rows 15-30): 32-frame 0.57-
      0.64, 16-frame 0.24-0.33, 4-frame 0.07-0.11; 23.7-25.0 frames per
      decision (~2.4-2.5 decisions per second).
DS2 ROWS 35-55 (interim, 09:37 10-26): deployed 1722 @35, 1385 @40, 1161
  @45, 1418 @50, 2064 @55 (SD 229-677); raw 1938, 1361, 909, 1616, 1709.
  READ (10) row 55: p1_ratio 0.72 -> FIXED (0.682 eval against 0.941);
      step-1 offset 0.18 running SDs against its step set, 5.30 against
      the pooled statistics. FIXED at rows 0, 30 and 55 so far.
  Row 45, descriptive: the training eval's diagnostics spiked for one row
      (mean |prefix| 4.01 against 0.11-0.24 at every other row, value
      8.2, H(pi) 1.90, maxp 0.43), with the training loss, the gradient
      norm (0.175) and the nonfinite count (0) unchanged; back at row 50.
      Its exact evals (1161 / 909) are DS2's lowest since rows 10 / 5.
      Earlier runs show the same one-row kind: G0 at row 60 (prefix 2.52,
      H 2.09), the distillation step's FS at rows 60 and 80-90.
DS2 COMPLETE 15:32 10-26 (RUN COMPLETE, rc 0 after 61,050 s, attempt 1;
  every exact eval and read (10) row in; fs2's supervisor released 15:31).
  rows 60-90, deployed: 1525, 1798, 1448, 1514, 1636, 1825, 1851; raw
      1928, 1534, 1769, 1365, 1234, 1742, 1629. No game capped.
  NEAR RANDOM: no. DS2 60-90: deployed 1656.52 (SD 167.2; g 1.385), raw
      1600.05 (SD 241.8; g 1.327).
  READ (3) against the master: 1656.52 against MASTER 1113.86: +542.7 ->
      ABOVE (band 232); EfficientZero's 1281.2 is g 1. (The distillation
      step's DS: 1231.38, +374.8 over its master.)
  READ (4): at level from row 0 (no forgetting, as above).
  READ (6) stability, the largest adjacent drop from row 45: -539 (2064
      @55 -> 1525 @60); raw -403 (1769 @70 -> 1365 @75).
  READ (8) prior, DS2: R = mean policy_maxp 75-90 / 30-55 = 0.670 / 0.633
      = 1.06 (0.995 without row 45's one-row dip) -> the prior HELD; the
      fresh targets' entropy 0.49-1.65 by row, no trend.
  READ (10): FIXED at all five rows -- p1_ratio 0.84 / 0.73 / 0.72 /
      0.46 / 0.52 at rows 0 / 30 / 55 / 75 / 90; the step-1 offset
      against the pooled statistics 1.78 / 6.88 / 5.30 / 4.88 / 4.79 SDs.
  Tokens and clip: the 32-frame tokens' play share 0.64 @5 -> 0.64 @90
      (16-frame 0.27 -> 0.22, 4-frame 0.05 -> 0.10); 21.6-25.2 frames per
      decision at 60-90 (deployed probe); reward_kept 0.83 @5 -> 0.78 @90
      (the constant ceiling 11.26 cuts a growing share: more of DS2's
      reward comes from the capped ghosts and later fruit).
  Reads (1), (2), (5)'s FS2 half and (9) wait for FS2 (~15 h).
FS2 LAUNCHED 15:39 10-26 (preflight ok, attempt 1, fresh). ROWS 5-30
  (interim, 20:10): deployed 348 @5, 379 @10, 430 @15, 426 @20, 473 @25,
  805 @30; raw 474, 331, 385, 465, 439, 565.
  READ (2) head start 5-25 (final: both arms' rows are in): DS2 1330.27
      against FS2 411.27: +919.0 -> BETTER (band 274.4). Raw 1286.00
      against 418.60 (+867.4). (The distillation step: DS 935.00 against
      FS 772.07, +162.9, NO DIFFERENT.)
  READ (4), FS2 so far: not at level (>= 881.86) by row 30 (17,661 env
      steps); DS2 was at level from row 0.
  READ (10) row 30, FS2: p1_ratio 0.69 -> FIXED; step-1 offset 0.13
      running SDs against its step set, 0.40 against the pooled statistics
      (DS2 at row 30: 6.88).
  Descriptive: FS2 plays the 32-frame tokens 0.69 @5, 0.80-0.93 @10-30
      (DS2: 0.57-0.65), 27.7-30.5 frames per decision (deployed probe);
      proj_erank 4.1-6.6 at rows 15-25, 18.5 at row 30 (DS2 13-27 there).
FS2 ROWS 35-55 (interim, 00:28 10-27): deployed 697 @35, 568 @40, 678
  @45, 695 @50, 714 @55 (SD 145-346); raw 620, 537, 712, 600, 639. Not at
  level (>= 881.86) by row 55 (31,281 env steps). (The distillation
  step's FS: 60-90 1034.38.)
  READ (10) row 55, FS2: p1_ratio 0.40 -> FIXED; step-1 offset 0.17
      against its step set, 0.80 against the pooled statistics.
  Descriptive: the 32-frame play share stays 0.75-0.85 (rows 35-55);
      proj_erank 12.8-27.2.

### RESULT -- Test 1: a generation-0 token master at a constant sign clip (G0, seed 0), 2026-10-25

RUN (mspacman-tokens-gen0-s0-2026-10-24, from the run worktree
Y:\SwitchPlayer-gen0-run, detached at the manifest commit): 13:58 10-24 ->
05:15 10-25 in ONE attempt (54,540 s, 10.1 min/iteration; no deaths, no
ALERTs, supervisor.err empty; RUN COMPLETE). The hand-off ran the evals one
at a time while it trained and the last two by 05:19: all 18 rows in both
variants, every row 4/4 phases, no capped game. Row 90 at 59.2k env steps
(TK-90 58.5k, the twin 53.4k). The ceiling held at 1.0 throughout.
EXACT eval as TK's (30 noop starts, prior argmax); DEPLOYED (acting states)
primary, RAW secondary, IN = the run's 8-game eval, maxp = the prior's mean
max probability, tgtH = fresh_target_entropy / ln 36:
  iter      5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  G0 dep  162  583  530  583  640  740  851  799  895  875  806 1154 1097  992 1116  937 1311 1190
  G0 raw  217  499  496  630  710  594  796  856  734  789 1182  386 1041  994 1226 1199  894 1527
  G0 in   173  691  578  418  713  714  845  553  859  824  845 1360  956 1171 1173  963 1346 1274
  maxp   0.68 0.69 0.85 0.60 0.82 0.78 0.79 0.77 0.80 0.71 0.76 0.42 0.79 0.72 0.79 0.72 0.77 0.75
  tgtH   0.36 0.21 0.21 0.38 0.30 0.26 0.20 0.21 0.28 0.37 0.44 0.33 0.26 0.40 0.34 0.27 0.32 0.31
  TK-90 dep 60  60  271  482  665  524  913  742  732  659  879  780 1033  863  920  825  885  691
NEAR RANDOM: no (deployed 60-90 g 0.828).
READ (1) the prior: R = mean maxp 75-90 (0.7569) / mean maxp 30-55 (0.7676)
  = 0.99 -> HELD (TK-90 0.48 and FS 0.37 FLATTENED; the twin 0.88, RG
  0.92, DS 0.96 HELD). policy_entropy / ln 36 stayed 0.14-0.39 at every row
  but 60 (0.58), against TK-90's 0.45 @60 -> 0.87 @90. The search targets
  never went flat: tgtH 0.20-0.44 over the whole run, against TK-90's 0.39
  @50 -> 0.79 @60 -> 0.93 @75.
READ (2) level against TK-90: deployed mean 60-90 1113.86 (SD 124.4) against
  856.62: +257.2 -> BETTER (band 232). Raw 1038.19 (SD 353.2, g 0.750)
  against 830.81: +207.4 -> BETTER (band 205, by 2.4).
READ (3) tokens against no tokens at the sign clip: against the twin's
  649.14: +464.7 -> BETTER (band 232); raw against 585.62: +452.6 -> BETTER
  (band 138).
READ (4) the clip, descriptive: reward_kept 0.94 @5 -> 0.64 @10 -> 0.73-0.76
  @20-45 -> 0.70 @50 -> 0.61 @75 -> 0.585 @90, so by row 90 the sign clip
  removes 41% of the replay's raw reward units (it removes more as the
  agent reaches ghosts and fruit). reward_max (a token's record, the sum of
  its inner steps' clipped rewards) 3.0 from row 5. Test 2's c1 halves the
  removed share, measured on the master's recorded games.
READ (5) tokens, descriptive: play shares (4 / 8 / 16 / 32 frames) 0.07 /
  0.13 / 0.27 / 0.53 @5 -> 0.10 / 0.06 / 0.33 / 0.51 @90; over rows 60-90
  the 32-frame tokens took 0.40-0.64 of play, the 4-frame ones 0.07-0.16
  (TK-90, under the moving ceiling: 32-frame 0.10-0.14, 4-frame 0.32-0.51 --
  the opposite end). Search shares @90 0.12 / 0.08 / 0.27 / 0.53. Frames
  per decision, deployed (probe), rows 60-90: 20.9-23.0, so 2.6-2.9
  decisions per second (TK-90 11.5-15.7, 3.8-5.2 per second). Decisions per
  life 37-44 (TK-90 57-84).
READ (6) stability: the largest adjacent deployed drop from row 45 is -179
  (1116 @75 -> 937 @80; TK-90 -194, DS -275, FS -260). Thirds 5-25 / 30-55
  / 60-90: deployed 500 / 828 / 1114 (TK-90 308 / 742 / 857); raw 510 / 825
  / 1038. THE ROW-60 JOLT: at the row-60 snapshot every eval-mode reading
  moved at once -- maxp 0.76 -> 0.42, the logged prefix error 0.29 -> 2.52,
  value error 1.43 -> 3.49, proj_erank 42.7 -> 18.1 -- and the RAW probe
  scored 386 (1182 @55, 1041 @65), while the behaviour that row did not
  move: the actor's prior maxp 0.59 (0.59 @55, 0.63 @65), the in-run eval
  1360, DEPLOYED 1154. The acting states are the trainer's weights with
  BN statistics recalibrated on the newest replay frames, so raw and
  deployed differ only in the BN statistics: the jolt sat in the trainer's
  running BatchNorm statistics at that snapshot (what the eval-mode
  diagnostics and the raw probe use), not in the weights or the play. Gone
  by row 65; one row, not repeated.
READ (7) the plateau rule (Step 0 (ii), one parameter set), applied after the
  fact: it never fires on G0, on deployed (sigma 104.8; the best 7-row
  window is the last one, rows 60-90, 1113.9) or on raw (sigma 140.5); the
  rank-based variant never fires either. G0 was still rising at row 90.
PRE-REGISTERED DECISION:
  - Read (1) HELD -> consistent with: the moving ceiling TOGETHER WITH the
    tokens drove the flattening (the r1 amendment's reading); the
    per-generation clip removes it.
  - Reads (2) / (3) reported (neither gates Test 2): BETTER against both.
    The heavy generation-0 clip cost nothing measurable here.
  - Not NEAR RANDOM -> G0 becomes Test 2's MASTER (its weights-iter090).
Beyond the letter (one seed, within-run bands -- "consistent with"):
  - G0 is the best FRESH Ms. Pac-Man run so far: deployed 60-90 1114
    against FS 1034, TK-90 857, RG 650 and the twin 649; only DS (1231,
    which started from a master) is higher. One row reached
    EfficientZero's 1281.2 (row 85, 1311); raw row 90 scored 1527.
  - This fits Step 0 (iii): the search's step-1 prefix artifact appears
    once the reward scale is large (the moving ceiling at 40-160 units), and
    a token run at the sign clip never reaches that scale. G0's targets
    stayed sharp and its prior followed them. Consistent with: the tokens
    do not flatten by themselves; a large reward scale under pooled prefix
    BN statistics does (one seed per arm). PB (TK + per-step statistics,
    training since 05:23 10-25) tests the fix under the moving ceiling;
    Test 2's students at c1 are exposed to the larger scale, which is why
    Test 2 waits for PB.
  - Under the sign clip the token play moved to LONG tokens (32-frame about
    half of play), the opposite of TK-90's late shift to short ones; at
    2.6-2.9 decisions per second G0 plays fewer, longer decisions than
    TK-90, DS or FS (3.8-5.2; Pong's P2a run 2.5). Which clip-dependent
    effect drives that is not
    separated here (the prior's sharpness, the clipped token reward
    summing at most one unit per inner step).
  - The sign clip scores a ghost (20+ units) like a pellet, yet G0 beat
    TK-90, which trained toward a 160-unit ceiling; without tokens the clip
    growth made no difference either (RG = the twin). So far a finer reward
    scale has not paid for itself on Ms. Pac-Man, and with tokens it cost
    more through the search artifact than it gained. One seed; PB reads
    whether the fixed search changes that, and Test 2 whether a looser
    clip helps once the master is already good.
EXPLORATORY (post hoc, not a pre-registered read; probe_prefix_bn.py on Step
0's 480 states, CPU, 05:24 10-25): G0's eval-mode step-1 prefix is NOT
inflated at rows 30 / 55 / 75 / 90 -- p1_ratio (eval |p1| / train-BN |p1|)
0.65 / 0.69 / 0.68 / 0.72 (|p1| 0.28-0.49 eval against 0.43-0.69 train-BN),
against TK-90's 8.20 / 20.95 / 23.63 at rows 55 / 75 / 90. That is under
PB's FIXED line (<= 2) on the pooled statistics alone, the error erring LOW
(DS's harmless direction). Step 0 (iii)'s prediction for the sign clip
holds to the end of the run. For Test 2's rebuild (A-NIT-3): the master's
own searches (G0's weights, pooled statistics) are clean at G0's scale; the
students at c1 are the exposed ones. Records:
benchmarks/probes/step0_prefix_bn/g0_prefix_bn.jsonl.

### Next experiment (pre-registered) -- Test 1: a generation-0 token master at a constant sign clip (Kai, 2026-10-24)

DECISION (Kai, 2026-10-24): "I think reward clipping is implemented
incorrectly. I was thinking that we would only change the clipping between
swaps of student to master." His 10-17 rule ("clip heavily in the
beginning and over time decrease the amount clipped") is ACROSS
GENERATIONS: one clip level per generation, constant while that network
trains, loosened only at a student / master swap -- as the vocabulary
(which never changes while a network trains). The REWARD_GROWTH built for
P2a (the ceiling 1 -> 160 log-linear WITHIN each run, the replay re-clipped
every iteration) misread it. Every Ms. Pac-Man run so far but DS trained
under a moving ceiling; DS alone held one (160). Also (Kai, same message):
the swap trigger ("a plateau") must be a GENERAL rule, not thresholds
tuned to one game's reward scale. Kai then: "Let's start with running the
tests you proposed" -- the proposal's defaults for the questions it left
open: (a) the clip loosens at each swap so that the share of the points it
removes HALVES (measured on the master's recorded games), generation 0 at
the sign clip; (b) the other within-run schedules (prior-acting
temperature stages, long-token bias, PER beta) stay as built -- the clip
is the one change; (c) the order Step 0 -> Test 1 -> Test 2 (-> a seed-1
repeat of Test 2 if it lands near its bands), Test 1 a fixed 90
iterations.
STEP 0 (no GPU, beside Test 1): (i) the prior-flattening diagnosis from the
logged rows (the search targets' entropy against the prior's, by row,
against the ceiling); (ii) GENERAL plateau rules -- thresholds in units of
the run's own measured noise, or a rank-based trend test -- backtested on
the Ms. Pac-Man AND Pong curves (two reward scales), on the raw game score
(a clip change at a swap must not move the yardstick); (iii) Test 2's
build (pre-registered before it is built).
QUESTIONS (Test 1): (A) does a token network trained at ONE constant clip
avoid the prior flattening that TK-90, TKX and FS (all under the moving
ceiling) showed? (B) how good is a sign-clipped generation-0 token master
-- the master Test 2's swap starts from?
ARM (seed 0): G0 = effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen0
  = TK (effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow)
  with REWARD_GROWTH_START = REWARD_GROWTH_CEILING = 1.0: every base
  step's reward / REWARD_UNIT 10 (Ms. Pac-Man's smallest reward, a pellet)
  clipped at 1 for the WHOLE run -- the sign clip (Ms. Pac-Man has no
  negative reward), a token's reward the sum of its inner base steps'
  clipped rewards (as TK). The constant-ceiling path is DS's (START ==
  CEILING), already run. Everything else TK's: 36 tokens, the long-token
  bias 0.5 -> 0 by 25%, the staged temperature, PER beta 0.4 -> 1, the TK
  recipe (MsPacmanNoFrameskip-v4 --frameskip 4, 90 x 2000 train steps, 3
  episodes per iteration, --episode-cap 3000, eval every 5,
  --checkpoint-every 2, --snapshot-weights, --ray --cpu-actor; TK's compute
  knobs).
COMPARED WITH: TK-90 (mspacman-tokens-rgrow-s0-2026-10-19: the same arm
under the moving ceiling; deployed 60-90 856.62, raw 830.81) and the twin
(mspacman-s0-2026-10-16: sign clip, no tokens; deployed 649.14, raw 585.62).
READS. EXACT eval as TK's (whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4), rows 5-90; DEPLOYED (acting states)
primary, raw secondary; g = (score - 307.3) / 973.9.
  NEAR RANDOM first: a 60-90 deployed mean with g < 0.2 (< 502.08) is
      diagnosed before any other read.
  (1) the prior, read (A): R = the mean policy_maxp (the prior's mean max
      probability, logged per row) over rows 75-90 divided by its mean over
      rows 30-55 -- a ratio to the run's own earlier sharpness (scale-free;
      the token count cancels): HELD >= 0.8, FLATTENED <= 0.6, else
      PARTIAL. On the finished runs: TK-90 0.48, FS 0.37 (FLATTENED); TKX
      0.81 (it flattened after row 90), RG 0.92, DS 0.96 (HELD). Beside it,
      descriptively: policy_entropy against ln 36, and the search targets'
      entropy (fresh_target_entropy) by row.
  (2) level against TK-90: G0's deployed mean 60-90 against 856.62, +-232
      (the twin's row spread, as every Ms. Pac-Man read): BETTER / NO
      DIFFERENT / WORSE; raw against 830.81, +-205.
  (3) tokens against no tokens at the sign clip: against the twin's
      649.14, +-232 (raw against 585.62, +-138).
  (4) the clip, descriptive: reward_kept by row (the share of the replay's
      raw reward units the clip keeps, per base step; 1 - kept = the share
      it removes -- the quantity Test 2's rule halves), reward_max.
  (5) tokens, descriptive (as TK's read 4), and (6) stability: the largest
      adjacent deployed drop from row 45; the thirds 5-25 / 30-55 / 60-90.
  (7) the plateau rule Step 0 settles on, applied after the fact to G0's
      curve -- descriptive (Test 1 runs a fixed 90 iterations).
DECISION AFTER. Single seed, within-run bands: BETTER / WORSE read
"consistent with"; the DEPLOYED reads decide, raw beside them.
  - Read (1) HELD -> consistent with: the moving ceiling drove the
    flattening (the same arm flattened under it, R 0.48); the
    per-generation clip removes it. PARTIAL / FLATTENED -> not the moving
    ceiling alone: Step 0's diagnosis continues (the temperature stages,
    PER beta's anneal, the token count) -- it does not hold up Test 2.
  - Read (2) / (3) are reported; neither gates Test 2 (Kai's design starts
    generation 0 at the sign clip whatever it scores against a moving
    ceiling). A WORSE against TK-90 is the price of generation 0's heavy
    clip that the swaps are meant to recover.
  - G0 becomes Test 2's MASTER unless NEAR RANDOM (then diagnose; Kai's
    call).
TEST 2 (outline; its full pre-registration lands before its build, while
G0 trains): MASTER = G0's final snapshot; the student vocabulary by the 1%
rule; the student's constant ceiling chosen on the master's recorded games
so the share of points it removes is HALF the sign clip's; two students at
that ceiling, identical but for the distillation step -- DS2 (the prefill
+ the offline phase, as DS) and FS2 (fresh) -- both on TK's within-run
temperature / bias / PER schedules, 90 iterations each, seed 0. The clean
version of the DS / FS question, whose two arms differed in the clip
schedule and the temperature schedule too.
ORDER / COST: G0 ~16 h (TK-90's 15.4 h) from a detached run worktree, its
evals one probe at a time on the CPU while it trains (the hand-off), the
rest after. The pace rule as before (past 25 min/iteration: Kai's call).
Kai's Switch-control sessions kill GPU pythons: pause the supervisor and its
trainer by command line, resume after.

AMENDMENTS (audit r1, before launch: A -- the run code -- and B -- the
supervisor / hand-off -- NOTHING ABOVE LOW; the LOWs fixed):
  - CORRECTION (A): the twin (mspacman-s0-2026-10-16) trained at a CONSTANT
    sign clip too (no REWARD_GROWTH): it is every Ms. Pac-Man TOKEN run but
    DS that trained under a moving ceiling. Read (1)'s references gain the
    twin: R 0.88 (HELD; no tokens, constant clip). So read (1) HELD reads
    "consistent with: the moving ceiling TOGETHER WITH the tokens drove the
    flattening" (RG -- moving ceiling, 9 actions -- and the twin -- constant
    clip, 9 actions -- both HELD); PARTIAL / FLATTENED reads "tokens flatten
    without the moving ceiling".
  - supervise_gen0.py: NO PROGRESS -- 3 long-lived attempts (>= 10 min) in a
    row that die without writing a new checkpoint stop it (a slow
    deterministic crash never reached CRASH LOOP; B-L3); the trainer guard
    also sees ablation.py / distill.py run from experiments/ or as a module
    (B-L4; a python whose command line cannot be read is not counted, so a
    foreign process cannot block every launch). Tests pin the child's
    environment (ABLATION_RESULTS, no CUDA_VISIBLE_DEVICES), cwd and
    --resume (B-L2), and that the P3 student gate passes the arm through
    and adds nothing to its resume recipe (A-LOW). The startup banner says
    "CONSTANT ceiling" when START == CEILING (A-NIT). The hand-off's
    docstring restores the un-abandon procedure (B-L1).
  - Accepted NITs: a normal completion logs "DEATH ... rc=0" before RUN
    COMPLETE (inherited); the fake-tree checker leaves its temp trees; the
    PREFLIGHT counter ignores the suites' own ~12 min; the AUDITED self-test
    scans only top-level imports (B verified coverage dynamically: every
    module the 17 suites open is hashed).

AMENDMENTS (audit r2, the confirmation round: ONE MEDIUM, in the hand-off,
fixed): the eval hand-off now treats the supervisor's NO PROGRESS stop as a
failed run -- echoed as an ALERT, a give-up ALERT at once -- instead of a
misleading 4-h "silent log" ALERT (MEDIUM-1: the r1 stop was new to it).
NO PROGRESS counts deaths after >= 5 min (was 10) without a new checkpoint,
so no lifetime falls between it and CRASH LOOP (which reaches ~465 s under
the quick-death backoffs; 466-599 s ran to MAX ATTEMPTS; LOW-1). Accepted:
a healthy attempt killed by the host before its first new checkpoint
counts as a stall -- three stop the supervisor, which loses nothing (the
idempotent launcher resumes from the checkpoint). The 17 suites pass in a
CRLF worktree of d8f35a9c (663 passed, 3 CUDA-only skips).

AUDIT + LAUNCH: r1 (A run code, B supervisor / hand-off) nothing above
LOW -> fixed (d8f35a9c); r2 ONE MEDIUM (the hand-off blind to the new NO
PROGRESS stop) + one LOW -> fixed (703857b3); r3 NOTHING ABOVE LOW (it
simulated every no-checkpoint death lifetime 60-1500 s: each stops by CRASH
LOOP or NO PROGRESS, never MAX ATTEMPTS; mutation-checked the hand-off's
NO PROGRESS handling; 17 suites in a CRLF worktree: 666 passed, 3 CUDA-only
skips) -- the loop closes. r3's LOW (three host kills of 5-25 min in a row
stop the supervisor; the launcher resumes it -- pause training before a
Switch session) and NITs accepted. Manifest 8e370b78 (_commit 523ef49b);
RUN WORKTREE Y:\SwitchPlayer-gen0-run (detached, CRLF; never edited).
LAUNCHED 2026-10-24 13:58 (scratchpad launch_gen0.ps1: the hand-off
post_gen0.py, then supervise_gen0.py g0 0). Expected done ~06:30 10-25.

STEP 0 FINDINGS (2026-10-24, from the logged rows and probe curves; no
GPU):
  (i) the prior flattening starts in the SEARCH. The birth-search targets'
      entropy (fresh_target_entropy / ln A) went near-uniform 10-15 rows
      BEFORE the prior's: TK-90 0.39 @50 -> 0.69 @55 -> 0.79 @60 -> 0.93
      @75 (prior 0.45 @60 -> 0.87 @90); FS 0.59 @45 -> 0.76 @50 -> 0.95 @60
      -> 0.99 @70 (prior 0.57 @60 -> 0.97 @90); TKX 0.42 @85 -> 0.83 @90 ->
      0.93 @100 (prior 0.52 @90 -> 0.98 @125). The KL from the targets to the
      prior then fell as the prior caught up (FS 0.65 @50 -> 0.08 @90). DS's
      targets held 0.54-0.65 throughout (its prior sharpened to them, 0.92 ->
      0.55); RG's 0.56-0.79. So the prior follows its targets; the question
      is why the 50-simulation birth search stops telling the tokens apart.
      The onset matches no single schedule landmark: TK-90 and FS at rows
      50-55 (just after their ceiling reached 160 and their temperature's
      0.25 stage, both at 46), TKX at row 90 (ceiling 28, temperature 0.5,
      ~45 rows before either). Test 1 reads whether the moving ceiling is
      involved. Next diagnostic (CPU, on existing snapshots; not yet run):
      the birth search's per-child Q spread against its exploration term at
      FS iter 45 and 60.
  (ii) a GENERAL plateau rule, backtested with ONE parameter set on Ms.
      Pac-Man (6 runs, deployed probes) and Pong (24 runs, the in-run eval):
      M = the mean of the last 7 evals; the noise sigma estimated from the
      run itself (1.4826 x the median absolute difference between adjacent
      evals / sqrt 2); PLATEAU when M has not risen 1 x sigma / sqrt 7 above
      its last reference for 30 iterations; DECLINE when M stays 2 x sigma x
      sqrt(2/7) below its best for two checks in a row. No game constant. It
      fires on TKX at row 125 (best window then rows 70-100, 937; TKX's later
      rows averaged 503), on the flat runs (the twin and RG at 65, Pong's
      edge-s0 at 75: their later rows within noise of the best window), on
      DS at 90 (its last row); never on TK-90, FS or 23 of the 24 Pong runs
      -- every Pong breakout included (their pre-breakout phase rises slowly,
      not flat). A one-check decline misfired on a single outlier eval (Pong
      sve2-s0); the two-check version above does not. A rank-based
      Mann-Kendall variant (8 evals) was less consistent: silent on the
      flat RG, firing on TKX at 110, the twin at 70 and four Pong runs (two
      at their last row). Remaining refinements: a noise floor from each
      eval's own game spread where the probe records it; the swap trigger's
      other conditions (a minimum budget, the student against its master)
      are P3's loop design, not this rule.
  (iii) WHY THE SEARCH FLATTENS (2026-10-24, CPU probes on the existing
      snapshots; scripts and records in benchmarks/probes/step0_prefix_bn/):
      the VALUE-PREFIX HEAD'S BatchNorm, not the Q spread against the
      exploration term. Training calls the prefix head once per unroll step
      in train mode, so each step is normalized with ITS OWN batch
      statistics. The search runs it in eval mode, on running statistics
      pooled over all 5 steps, and the first step after a fresh or reset
      LSTM state has a very different input distribution. In eval mode that
      step-1 prefix comes out far too high, while the same weights with
      train-mode BN predict it correctly. On FS, for a true token reward of
      ~0.4, it reads 0.32 @45, 16.4 @60 and 30 @75. In the tree, a root edge
      gets +p1 and the next edge about -p1. A child's second visit backs up
      through that negative edge and its Q drops (~-11 raw @60), so visited
      children look worse than unvisited ones. PUCT then spreads the visits
      over nearly every token (6.9 of 14 visited @45, 13.1 @60), and the
      inflated min-max range compresses the normalized Q. Root noise is not
      it (the noise-free search flattens as much). Instrumented copy of the
      engine (identical visit counts), 480 states:
        FS target H/lnA, eval BN -> step-1 BN re-estimated: @45 0.47 ->
        0.48; @60 0.84 -> 0.42; @75 0.95 -> 0.50 (prefix off: 0.58 @60).
      ACROSS RUNS (26 snapshots in 7 runs, each on its own code and spec):
      the step-1 artifact (eval |p1| 8-80x the train-BN value) is present at
      every snapshot whose targets went flat, and re-estimating the step-1
      BN statistics restores sharp targets in every one of them:
        TK-90 @55 0.71 -> 0.33, @75 0.90 -> 0.31, @90 0.92 -> 0.16;
        TKX @100 0.92 -> 0.45, @140 0.93 -> 0.48, @200 0.72 -> 0.15;
        FS as above.
      Clean before the flattening (TK @30, TKX @55, FS @45) and in every
      held run: RG (no tokens, small per-decision rewards), the twin (sign
      clip) and G0 @5-15 (tokens, sign clip; too early to call, since TK's
      artifact came between rows 30 and 55). DS shows the BN offset too, but
      its eval-mode step-1 prefix errs LOW (0.10-0.28 against 0.28-0.63),
      which does not penalize a second visit. So DS held because the error
      points the harmless way; the probe does not show why. The artifact
      precedes the prior's fall (TK @55: artifact 3.8, prior maxp still
      0.61; then 0.40 @75, 0.20 @90). The pattern across runs fits reward
      scale: it appears in the token runs once the ceiling reaches the full
      reward scale (TKX ~41 by row 100, TK and FS 160). The logged `prefix`
      error is measured only at unroll step 5 (FS 0.25 -> 1.81), so it hid
      the step-1 error (0.27 -> 16.1). EfficientZero has the same layout (its
      bn_value_prefix), but under the sign clip the error stays small.
      Caveats: one seed per arm, so the link across runs is correlational.
      The causal result covers the search targets at each snapshot, not
      whether a fix in training keeps the prior sharp. All arms were
      searched on FS@45's states, and the per-run searches used 48 states
      (SE ~0.02). Under Kai's plan every later generation loosens the clip,
      i.e. raises the reward scale, which is where this appears. Test 2's
      students at c1 are exposed to it; G0 at the sign clip should not be.

### RESULT -- P3's distillation step: DS against FS on Ms. Pac-Man (seed 0), 2026-10-24

RUNS (from the run worktree Y:\SwitchPlayer-distill-run, detached at
e2de0c29; one attempt each, no ALERTs, supervisor.err empty): the prep
10-23 00:17 -> 00:23; DS 00:23 -> 17:56 10-23 (63,210 s, the offline
phase included); FS 18:08 10-23 -> 08:58 10-24 (53,430 s, 9.9
min/iteration); the hand-off's last evals 09:02 10-24 -- all rows in both
variants (DS 0 and 5-90, FS 5-90, DS's offline curve 1-4), no capped game.
PREP: MASTER = TK-90 (weights-iter090; TKX's rows 240-270 523.24 < 856.62).
The master's recorded play: 44 games, 132 lives, 31,620 base steps, game
returns mean 810.2. VOCABULARY: 14 of 36 tokens -- the nine 4-frame tokens,
LEFT and UPLEFT x16, NOOP, DOWN and DOWNRIGHT x32; ALL nine 8-frame tokens
dropped. The 22 dropped tokens held 10.1% of the master's decisions (1,113
decisions, split into 3,293 extra 4-frame records; 14,451 records in all;
no uniform fallback). The master's search put 0.569 of its visits on them
(the masked read-out's discarded mass; A's ~40% estimate was low).
EXACT eval as TK's (30 noop starts, prior argmax); DEPLOYED (acting
states) primary, RAW secondary, IN = the run's 8-game eval, maxp = the
prior's mean max probability:
  iter      0    5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  DS dep  660  635  841 1415  833  951 1184  972 1327  973 1310 1391 1192 1018 1239 1338 1336 1061 1436
  DS raw  579  700  740 1315  971 1421  923  967 1048 1334 1461 1022 1073  966 1084 1478 1158 1190 1756
  DS in     -  484 1002 1065  825  912 1299  921 1229 1061 1182 1188 1359 1095 1415 1176 1202 1208 1401
  DS env    -  3.3  6.1  9.7 13.3 17.6 21.8 25.8 29.4 32.8 36.7 40.3 44.7 48.7 52.3 56.2 60.4 64.3 68.2
  DS maxp   - 0.19 0.22 0.31 0.36 0.45 0.49 0.55 0.54 0.56 0.58 0.54 0.58 0.57 0.50 0.55 0.52 0.52 0.50
  FS dep    -  567  808  868  797  820  705  748 1079  851 1068  937 1035 1097 1090  830 1141  961 1087
  FS raw    -  594  963  780  775  813  788  751  780  959 1015  939  992  937  894 1173  847  998 1247
  FS in     -  399  820  751  788  785  750  740 1124  978  986 1151 1099  985 1346  845 1152  996 1068
  FS env    -  3.4  6.0  9.0 12.4 15.8 19.1 22.6 25.6 29.2 32.9 37.0 40.2 44.3 47.7 51.1 54.5 58.3 61.6
  FS maxp   - 0.60 0.55 0.62 0.61 0.55 0.58 0.58 0.57 0.63 0.60 0.58 0.55 0.45 0.38 0.29 0.23 0.19 0.16
  TK-90 dep -   60   60  271  482  665  524  913  742  732  659  879  780 1033  863  920  825  885  691
(env in thousands of the student's own base steps; FS's ceiling 2 @5 ->
26 @30 -> 143 @45 -> 160 from 50, T 0.5 / 0.25 from 50 / 0.125 from 75;
DS's ceiling 160 and T 0.125 throughout.)
NEAR RANDOM: neither (DS 60-90 g 0.949, FS 0.747).
READ (0) transfer: row 0 (20,000 offline steps, before DS played) 660.0
  deployed (579.3 raw, g 0.362) against MASTER 856.62: -196.6,
  descriptive (+-464); NOT TRANSFER FAILED. The offline curve, deployed /
  raw: 492 / 274 @4k, 747 / 395 @8k, 520 / 362 @12k, 803 / 795 @16k, 660 /
  579 @20k -- within ~110 of MASTER at 8k and 16k steps, noisy; the
  master scored 810.2 in the recorded games.
READ (1) level 60-90: DS 1231.38 (SD 153.1) against FS 1034.38 (SD 106.6):
  +197.0 -> NO DIFFERENT (band 232). Raw 1243.43 against 1012.48: +231.0
  -> BETTER (band 205). A deployed / raw disagreement: reported, not
  acted on.
READ (2) head start 5-25: DS 935.00 against FS 772.07: +162.9 -> NO
  DIFFERENT (band 274.4). Raw 1029.47 against 784.87 (+244.6).
READ (3) against the master: DS 1231.38 against 856.62: +374.8 -> ABOVE
  (g 0.949; EfficientZero's 1281.2 is g 1). FS +177.8 -> SAME.
READ (4) time to level (the first row >= 624.62): DS row 0 (before it
  played); FS row 10 (5,950 env steps); TK-90 row 25 (15,589). DS's
  lowest row in 5-25 is 635.0 against row 0's 660.0 (-25, inside one
  row's noise; the 5-25 mean is 935): no forgetting when it began to play.
READ (5) tokens: per-duration play shares (4 / 16 / 32 frames) DS 0.51 /
  0.40 / 0.08 @5 -> 0.45 / 0.31 / 0.24 @90; FS 0.64 / 0.05 / 0.31 @5 ->
  0.58 / 0.18 / 0.24 @90. Frames per decision, deployed (probe), rows
  60-90: DS 14.3, FS 12.9 (4.2 and 4.7 decisions per second; TK-90
  11.5-15.7). FS against TK-90 (the same recipe over 36 tokens) at 60-90:
  +177.8 deployed, inside the band: dropping the master's least-used
  tokens did not hurt a fresh start, and may have helped (one seed).
READ (6) stability, the largest adjacent drop from row 45: DS -275 (1336 @80
  -> 1061 @85), FS -260 (1090 @70 -> 830 @75).
PRE-REGISTERED DECISION: (1) NO DIFFERENT + (2) NO DIFFERENT -> no
  measurable effect at these bands; by read (0) the transfer did not fail,
  and DS did not forget as it began to play -> Kai's call.
Beyond the letter (one seed, within-run bands -- "consistent with"):
  - Both deployed reads favour DS and fall just inside their bands (+197
    of 232, +163 of 274); raw is BETTER. DS led FS at 17 of the 18 shared
    rows (all but row 65), and in every third: 935 / 1193 / 1231 against
    772 / 898 / 1034. Seven DS rows reached EfficientZero's 1281.2 (15,
    40, 50, 55, 75, 80, 90); no FS row did. Adjacent rows are correlated,
    so the row count is not an independent test; a seed-1 pair is the
    test the bands ask for.
  - The swap with distillation started AT the master's level (row 0) and
    ended 375 above it; the swap without started at 567 and ended at the
    master's level (+178, SAME). The data bill differs: DS also used the
    master's 58.5k training steps and the 31.6k recorded ones.
  - PRIOR FLATTENING (TKX's diagnosis, updated): FS flattened too -- maxp
    0.63 @45 -> 0.16 @90, entropy 1.24 -> 2.55 of ln 14 = 2.64 -- so the
    vocabulary-size reading TKX's RESULT proposed is NOT supported (14
    tokens flatten as 36 do). DS did not (maxp 0.50-0.58 from row 30).
    Across the six Ms. Pac-Man runs the prior flattened in TK-90, TKX and
    FS -- tokens with the clip ceiling growing during training -- and in
    neither RG (growth, 9 actions) nor DS (tokens, ceiling 160 from the
    start, a sharp distilled start). Its onset does NOT keep a fixed lag
    behind the ceiling: TKX's coincides with the ceiling passing ~30-40
    units (maxp 0.53 @90 -> 0.25 @100, ceiling 28 -> 41), TK-90's and
    FS's comes ~25-30 rows after it (ceiling ~40 @34; maxp falling from
    ~60-65, after their temperature's 0.25 stage). CORRECTION to TKX's
    RESULT: "it began ~10 rows after the ceiling passed ~30-40 units, and
    so did TK-90's" is wrong on both counts -- the clip-growth link is
    weaker than written there. FS's and TK-90's
    deployed scores held through 90 while their priors flattened; TKX's
    collapsed once maxp reached ~0.05. Not tested: the clip growth's
    timing against the temperature stages and PER beta's anneal, which
    differ between these arms too.
  - DESIGN NOTE for P3's swap trigger (Kai's question 10-24, "a plateau"),
    backtested on the six deployed curves: a 7-row moving mean of the
    deployed rows; PLATEAU when it has not risen 75 (~1 SE of a 7-row mean)
    above its last reference for 30 iterations; DECLINE when it falls 232
    below its best. It fires on TKX at row 125 (the best window then rows
    70-100, 937; TKX's rows after 125 averaged 503), on the twin and RG at
    row 65 (both flat; their later rows averaged 685 / 694 against best
    windows of 649 / 586 -- within noise), and never on TK-90, DS or FS
    within 90 (still rising). A slope test (OLS over the 7 rows, upper
    1-SE bound x 30 iterations < 75) fires earlier and noisier (row 40 on
    the twin and RG). Not adopted; Kai's call with the loop's design.

### Next experiment (pre-registered) -- P3's distillation step: master -> student on Ms. Pac-Man (Kai, 2026-10-21)

DECISION (Kai, 2026-10-21, after the P2a Ms. Pac-Man RESULT -- TOKENS HOLD
UP): "I'd like the distillation to be implemented and ran after the current
run" -- the distillation step he proposed on 10-20 ("a distillation step
between the two networks ... before starting the student"), built and run
once TKX (the 270-iteration token run) has completed and its evals have
landed. P2b (the composition loss, the HOLD UP verdict's next step) waits
behind it. GROWTH (NO DIFFERENT) stays Kai's call; meanwhile both student
arms keep the token arm's clip machinery (below).
QUESTION: does a distillation step at the generation swap give the student
a head start, or a higher level, than a student that starts fresh with the
same vocabulary?
THE SWAP (one generation, built for this run; deficit exploration and
promotion are P3's later steps):
  MASTER = the best token network so far: the final snapshot of whichever
    of TKX (weights-iter270, deployed mean of rows 240-270) and TK-90
    (weights-iter090, rows 60-90 = 856.62) has the higher 7-row deployed
    mean; TK-90 if TKX did not complete (ABANDONED / crash loop). The
    snapshot carries the master's acting states (its BN-recalibrated
    representation / prediction).
  RECORDING: the master plays fresh games with its own training-time
    acting policy -- the prior at its final temperature (0.125 =
    0.5 x the staged 0.25; the long-token bias long gone) through its acting
    states, NoopResetEnv's random starts, --episode-cap 3000 per life --
    and every game is recorded at RAW-STEP resolution: each decision's
    stack, token and raw per-base-step rewards (as the replay already
    keeps) PLUS the preprocessed frame of every inner base step, which the
    replay never stored (the completion marker also deletes a finished
    run's replay, so the master's training data is gone: TK-90's already
    is). Whole games until >= 31,250 base steps (the replay cap; ~40
    games, ~120 lives). Self-check: replaying each decision's inner frames
    onto its stack reproduces the next decision's stack exactly.
  VOCABULARY (Kai's 10-17 rule, "drop the least used"): a master token
    whose share of the recorded decisions is below 1% is dropped; the nine
    4-frame tokens are never dropped (every dropped token must be
    expressible as raw 4-frame steps). The student's vocabulary is the
    remaining tokens in the master's slot order (slot 0 stays NOOP x4, the
    placeholder). Fixed slots: a dropped slot is EMPTY, and with no new
    token to put in it an empty slot is implemented as its absence -- a
    shorter vocabulary -- since a fresh student inherits nothing per slot;
    slot-preserving masks become necessary only once tokens are added
    (P5e) or weights are inherited.
  RE-TOKENIZATION: each recorded life becomes a student trajectory -- a
    kept token stays one record; a dropped token (a, f) becomes its m
    played base steps as m records of (a, 4), each on the stack at that
    base step (rebuilt from the inner frames) and each carrying its own
    raw base-step reward (m < f / 4 when a life loss cut it short).
  TARGETS: the master's 50-sim birth search (the arm's own target recipe:
    fresh root, Dirichlet noise, raw weights) on every student decision
    state, over the MASTER's 36 tokens, its visit fractions renormalized
    over the student's tokens (a masked read-out: visits to dropped tokens
    are discarded); the tail as the arm's (terminated: uniform, masked;
    truncated: one greedy search). The master's root values set only the
    insertion priorities. The student's value / value-prefix targets are
    its own (n-step TD through ITS target nets on the recorded rewards,
    clipped at 160) -- no value distillation.
ARMS (seed 0, Ms. Pac-Man, the TK recipe: MsPacmanNoFrameskip-v4
--frameskip 4, 90 x 2000 train steps, 3 episodes per iteration,
--episode-cap 3000, eval every 5, --checkpoint-every 2, --snapshot-weights,
--ray --cpu-actor; TK's compute knobs VTREE_REFRESH_BATCH 64 /
VTREE_COMMUNAL_TRAJS 1):
  DS (distilled student) = the TK arm over the student vocabulary, with
    (a) the replay pre-filled with the re-tokenized master games (born at
    step 0: the off-policy horizon shrinks as they age; FIFO-evicted as
    the student's own games arrive, ~48 iterations; the communal refresh
    rotates through them oldest first, re-searching them with the
    student's own nets -- so the master's targets fade out without an
    explicit schedule), (b) an OFFLINE phase before iteration 1: 20,000
    gradient steps (10 iterations' worth) on that replay, no collection,
    no refresh, the usual losses / LR warm-up / target syncs; snapshots
    every 4,000 steps (the 20,000-step one is ROW 0), and (c) the
    master's END-STATE schedules from the start -- reward ceiling 160
    constant (REWARD_GROWTH_START 160), prior-acting temperature 0.125
    constant (PRIOR_ACT_STAGED_TEMP off, ACTING_TEMPERATURE 0.25 x the
    arm's 0.5), long-token bias 0, PER beta 1.0. Its training-step
    counter runs 0 -> 200k (the offline 20k included); the off-policy
    decay keeps the 90-iteration value (54k steps).
  FS (fresh student) = the TK arm over the SAME student vocabulary, from
    scratch, with TK's schedules (ceiling 1 -> 160 by 50%, temperature
    0.5 / 0.25 / 0.125 staged, bias 0.5 -> 0 by 25%) -- the swap without
    the distillation step.
  DS vs FS is the question. It compares two PACKAGES -- a distilled
  student continues in the master's end-state regime, a fresh one needs
  the beginning-of-learning regime (Kai's "clip heavily in the beginning")
  -- not the offline phase alone. FS vs TK-90 (36 tokens, same recipe)
  reads, descriptively, what dropping the master's least-used tokens does
  from scratch.
ORDER: prep (record, vocabulary, re-tokenize, master search: ~0.5 h on
the GPU), then DS, then FS, one at a time (the sequential rule). The DS
supervisor waits for TKX's completion marker AND both TKX probe jsonls
holding rows 5-270, then for no trainer at all.
READS. EXACT eval as TK's: whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4; DS rows 0 and 5-90 (19), FS rows 5-90
(18); DEPLOYED (acting states) primary, raw secondary; g = (score - 307.3)
/ 973.9; MASTER = the selected master's 7-row deployed mean.
  NEAR RANDOM first, per arm: a 60-90 deployed mean with g < 0.2
      (< 502.08) is diagnosed before any other read of that arm.
  (0) transfer, row 0 (the distilled student before it plays): its
      deployed score against MASTER, descriptively (one row against a
      7-row mean: 2 SE = +-464, too wide to verdict), with TRANSFER FAILED
      if row 0 is near random (< 502.08); the offline curve (4k, 8k, 12k,
      16k, 20k steps) and the master's own scores in the recorded games
      beside it.
  (1) level: DS's deployed mean 60-90 against FS's, +-232 (two 7-row
      means, the twin's row spread, as every Ms. Pac-Man read): BETTER /
      NO DIFFERENT / WORSE. Raw likewise (+-205, as TKX's raw band).
  (2) head start: DS's deployed mean of rows 5-25 against FS's (5-row
      means: 2 x 216.9 x sqrt(2/5) = +-274.4): BETTER / NO DIFFERENT /
      WORSE.
  (3) against the master: DS's deployed mean 60-90 against MASTER, +-232
      -- ABOVE / SAME / BELOW (does a swap with distillation keep the
      master's level within 90 iterations, or pass it?).
  (4) time to level, descriptive (the real-time quantity): per arm, the
      first row whose deployed score is >= MASTER - 232, in iterations and
      in the student's own env steps; DS's lowest deployed row in 5-25
      against row 0 (forgetting when it starts to play).
  (5) tokens, descriptive: the dropped tokens and their share of the
      master's recorded play; the student's per-duration play / search
      shares, frames per decision, the deployed decisions per second; FS
      against TK-90 rows 60-90.
  (6) stability: each arm's largest drop between adjacent rows from row 45.
DECISION AFTER. Single seed, within-run bands: BETTER / WORSE read
"consistent with"; the DEPLOYED reads decide, raw is reported beside them.
  (1) BETTER -> the distillation step joins P3's generation swap; a seed-1
    replicate before it is permanent.
  (1) NO DIFFERENT + (2) BETTER -> it buys a head start, not a higher level
    within 90 iterations: it joins P3 for the time it saves (the real-time
    case), Kai confirms; a seed-1 replicate.
  (1) NO DIFFERENT + (2) NO DIFFERENT -> no measurable effect: by read (0)
    -- TRANSFER FAILED -> fix the transfer (targets, offline steps, the BN
    states) and rerun DS; otherwise, if DS's rows 5-25 fell from row 0, the
    student forgot as it began to play -> the annealed KICKSTARTING variant
    next (a KL to the master's prior, decaying to 0 over the first ~25% of
    the run); else Kai's call.
  (1) NO DIFFERENT + (2) WORSE -> diagnose (the end-state regime: the
    0.125 temperature's exploration, the off-policy master data).
  (1) WORSE -> negative transfer: diagnose the same before anything else.
  A near-random arm, or an ABANDONED one (stopped under the pace rule or a
  crash loop), is read descriptively; no verdict that needs it.
  Read (3) is reported with every outcome; BELOW with (1) BETTER still
  counts for the swap (the comparison is the swap without distillation).
COST: prep ~0.5 h; DS ~17 h (the offline 20k steps ~1 h + 90 iterations at
TK-90's ~10.3 min, fewer tokens searching a little faster); FS ~16 h;
evals as TKX's (one probe at a time on the CPU while training, the rest
after). Expected: TKX done ~01:00 10-23, DS ~01:30 -> ~19:00 10-23, FS ->
~11:00 10-24. PACE RULE: a row past 25 min/iteration is an ALERT and Kai's
call. Kai's Switch-control sessions kill GPU pythons: pause the running
supervisor and its trainer by command line, resume after.
AMENDMENTS (audit r1, before launch: A -- the run code -- nothing above LOW;
B -- the supervisor / hand-off -- two MEDIUM, fixed):
  - B-M1: the prep (experiments/distill.py) counts as a trainer for the
    any-trainer guard, and writes per-process tmp files: an orphaned prep
    (its supervisor killed) can never run beside a second one.
  - B-M2: once DS was launched the prep is NEVER redone (DS's checkpoint is
    bound to the set's sha256, FS to its vocabulary); the launch gate
    ALERTs and waits instead. Before that, a transient read error is
    retried, and damaged prep files are moved aside, not overwritten.
  - Read (0) adds, descriptively, the mean visit mass the masked read-out
    discards (A's note: on TK-90's late rows its 50-sim search put ~40% of
    its visits on the tokens the 1% rule would drop, ~5% of its play, so
    the student's targets come out sharper than the master's own; logged
    by the prep as discarded_visit_mass_mean).
  - The recording overshoots 31,250 base steps by up to a game, so the
    prefill's FIFO eviction drops the oldest 1-4 lives before iteration 1
    (logged). A dropped long token that crossed the master's 3000-step cap
    is split in full (up to 7 base steps past it). The offline curve's
    probe rows are numbered by chunk (row k = k x 4,000 offline steps).
  - Smaller fixes (A / B LOWs): DS refuses to start without
    --snapshot-weights and with a distillation set other than the one
    vocab.json was written with; the master snapshot must be a final one
    (its iteration and step count); master.json is written atomically; a
    TKX that completed with rows 240-270 incomplete stops the prep (PREP
    FAILED) rather than falling back to TK-90, and a tie keeps TK-90; fs
    also waits for DS's offline-curve rows and checks before each launch
    that vocab.json is the vocabulary DS trained with; only DS, and only
    for its row 0 and offline curve, may restart over its own
    pre-checkpoint snapshots; the prep is watched (an ALERT when silent
    past 30 min); ABANDONED is checked while waiting and before the prep;
    ds --no-wait needs TKX's completion marker or ABANDONED file; the
    hand-off makes up to 4 final passes over a run's missing rows before
    moving on.
  - PROCEDURES: pause = kill the supervisor, then its trainer (ablation.py)
    or prep (distill.py), by command line. After a reboot rerun BOTH
    launchers (launch_rgrowx.ps1 while TKX is unfinished;
    launch_distill.ps1). Retracting a --no-wait also moves the prep folder
    aside (its master.json holds the first choice).
AMENDMENTS (audit r2, before launch: A and B NOTHING ABOVE LOW -- the loop
closes; its LOWs fixed, then a confirmation round):
  - Once DS was LAUNCHED no prep is ever started again, whatever is missing;
    an orphaned prep that finished while the next attempt waited counts;
    stale prep tmp files are swept; prep attempts are 15 min apart.
  - NO PROGRESS: 3 consecutive long-lived DS / FS attempts (>= 10 min) that
    die without writing a new checkpoint stop the supervisor (DS's attempts
    each cost the ~1 h offline phase until its first checkpoint).
  - fs checks DS's vocabulary in a child process (the long-lived fs
    supervisor holds no torch); a NO-WAIT line alone skips the wait only
    while the --no-wait preconditions still hold. RETRACTING a --no-wait =
    the fresh-restart procedure (move the DS run folder AND the prep folder
    aside). UN-ABANDONING ds: stop the hand-off and its probes, drop the
    probe rows its abandon pass logged as newer than the checkpoint, remove
    ABANDONED, relaunch launch_distill.ps1.
  - Known, accepted: the offline curve's snapshot path is ~289 characters
    (Windows long paths are enabled on this machine; tools that are not
    long-path aware may fail to copy the folder).
AMENDMENTS (audit r3, the confirmation round: nothing above LOW; its two
LOWs -- both behind a --no-wait slip -- fixed): a logged NO-WAIT is judged
on TKX's persistent state (its marker or ABANDONED file), not on whether a
TKX script is briefly alive after a reboot; a master.json chosen while TKX
had no marker stops the prep (PREP FAILED) once TKX has completed (the
retraction procedure moves it aside); a launched DS resumes even under
-NoWait. GPU smoke (2026-10-21, beside TKX, no Ray): prefill from a tiny
TK-90-master set, 20 offline steps, row-0 and offline-curve snapshots with
acting states, a student life with birth search over its 13 tokens and a
communal refresh, all on CUDA -- OK (28 s, peak 1.0 GB). The Ray path is
TK's, unchanged but for the vocabulary size; DS's first iteration is its
first Ray run.
AMENDMENTS (audit r4, confirmation: nothing above LOW; its LOW fixed): the
stale-master.json stop also covers a FINISHED prep, and is re-checked after
every guard wait and before DS's first launch; --no-wait is also refused
while any trainer or prep lives.
AMENDMENTS (audit r5, confirmation: ONE HIGH, in the tests -- fixed): the
supervisor's tests read the live TKX run's real paths, so they passed only
while TKX was unfinished -- DS's own preflight (which runs them after TKX
completes, or with TKX abandoned) could never have passed. Every live
TKX / TK-90 path is now redirected in the tests' fixture, and the suite is
run under simulated TKX states (live, completed, abandoned: all pass). Also
a finished prep without a readable master.json stops the prep (a
half-retraction).
AMENDMENTS (audit r6, confirmation: ONE HIGH, in a test -- fixed): the run
worktree checks out CRLF (autocrlf) while the dev tree is LF; the manifest
test flipped LF -> CRLF, which on a CRLF file writes \r\r\n -- DS's
preflight would have failed in the run worktree (r5's "all pass" was in
the LF dev tree only). The test now flips whichever way differs, and the
preflight SUITES are verified in a CRLF export of the commit (git archive)
under the completed and abandoned TKX states. Its LOW (the master.json read
retried like the integrity check) fixed.
AUDIT + LAUNCH: seven rounds. r1 -- A nothing above LOW; B two MEDIUM (an
orphaned prep invisible to the guard; a resumed DS could redo its prep) ->
3ad39e42. r2 -- A and B nothing above LOW (the loop closes) -> its LOWs
36d41b56. Confirmation rounds on each fix: r3 (LOWs) -> 406ece2f; r4 (LOW)
-> f358b2d2; r5 ONE HIGH in the tests (they read the live TKX state) ->
27e05228; r6 ONE HIGH in a test (CRLF run worktree) -> 1e1b7ce2; r7 nothing
above LOW (its LOW in this file only). The preflight suites pass in a CRLF
export of the commit under the completed and abandoned TKX states. GPU
smoke (no Ray) OK. Manifest e2de0c29 (44 AUDITED files of 835dc1a9). RUN
WORKTREE Y:\SwitchPlayer-distill-run (detached at e2de0c29; never edit).
LAUNCHED 2026-10-21 19:53 (WMI, scratchpad launch_distill.ps1): the hand-off
post_distill.py, then `supervise_distill.py ds 0` (WAITING for TKX's marker
and its 108 evals) and `supervise_distill.py fs 0` (WAITING for DS).
Expected: TKX done ~00:45 10-23, DS preflight + prep ~01:00-02:30, DS
~02:30 -> ~20:00 10-23, FS -> ~13:00 10-24 (+ evals). Accepted NITs: the
r6 LOW fix has no dedicated test (r7 probed it by hand); test_rgrowx's
line-ending flip does nothing in an LF tree.

### RESULT -- the token arm on Ms. Pac-Man, 3x longer (TKX, seed 0), 2026-10-23

TKX (mspacman-tokens-rgrow-x270-s0-2026-10-20) trained from its run
worktree 10-21 01:08 -> 10-23 00:07 in one attempt (169,171 s, 10.44
min/iteration; no ALERTs, supervisor.err empty). The hand-off ran its last
evals by 00:11: all 54 rows in both variants, no capped game. EXACT eval,
30 noop starts, prior argmax; DEPLOYED (acting states) primary, RAW
secondary, IN = the run's 8-game eval; maxp = the prior's mean max
probability, ceil = the reward-clip ceiling (units), T = the acting
temperature:
  iter       5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  dep      354  692  506  527  748  550  791  591  782  770  761  680  824  898  914  828  798  877
  raw      446  655  587  499  611  655  730  818  657  753  788  659  783  844  924  835  765  959
  in       408  642  360  444  745  499  612  604  921  669  719  582  849 1118  816  888  852  878
  env k      3    6    9   12   16   20   23   26   30   34   37   41   45   48   52   55   59   63
  maxp    0.62 0.62 0.63 0.66 0.73 0.73 0.75 0.69 0.66 0.66 0.71 0.67 0.61 0.61 0.53 0.61 0.62 0.53
  ceil       1    1    2    2    2    3    4    4    5    6    8    9   11   13   16   19   24   28
  iter      95  100  105  110  115  120  125  130  135  140  145  150  155  160  165  170  175  180
  dep     1034 1213  761  774  939  580  621  569  575  296  292  484  447  524  619  382  305  295
  raw     1195 1088  634  932  939  722  688  457  276  362  694  300  525  590  436  330  413  522
  in      1222 1244  731  804  721  479  675  674  590  266  270  398  706  496  596  421  300  298
  env k     66   70   74   77   79   82   85   88   90   92   95   98  101  103  106  108  111  114
  maxp    0.40 0.25 0.21 0.16 0.12 0.13 0.06 0.06 0.05 0.05 0.05 0.07 0.08 0.08 0.09 0.11 0.13 0.13
  ceil      34   41   50   60   73   88  106  128  154  160  160  160  160  160  160  160  160  160
  iter     185  190  195  200  205  210  215  220  225  230  235  240  245  250  255  260  265  270
  dep      558  356  546  608  382  506  491  475  783  740  685  463  423  510  857  460  419  530
  raw      430  544 1186  479  884  761  402  372  555  404  386  891  472  765  695  689  666  464
  in       624  371  332  631  426  765  485  459  700  909  599  404  344  431 1295  486  414  505
  env k    116  119  122  124  127  130  133  137  140  143  146  149  152  155  158  161  164  167
  maxp    0.10 0.13 0.17 0.18 0.23 0.16 0.24 0.15 0.12 0.10 0.09 0.10 0.11 0.12 0.16 0.14 0.13 0.12
(T 0.5 to row 135, 0.25 from row 140, 0.19 @205, 0.125 from row 210;
the ceiling 160 from row 140; TK-90's row 90 at 58.5k env steps.)
NEAR RANDOM: no -- the last-7 deployed mean 523.24 (g 0.222) is above
  502.08, by 21.
READ (1) level, rows 240-270: deployed 523.24 (SD 152.9) against TK-90's
  856.62: -333.4 -> WORSE (<= 624.6). Raw 663.10 (SD 153.0) against
  830.81: -167.7 -> NO DIFFERENT (band 205). A deployed / raw
  disagreement: reported, not acted on. Against EfficientZero's 1281.2:
  g 0.222 deployed, 0.365 raw.
READ (2) trend: rows 240-270 against rows 150-180 (436.62): +86.6 ->
  PLATEAU (raw 445.43 -> 663.10, +217.7).
READ (3) data: the first row past 100k env steps is row 155 (100,639);
  the 7-row window 140-170 scored 434.8 deployed (462.7 raw) -- g 0.13,
  against EfficientZero's 1281.2 at 100k. TKX gathered 167k env steps,
  2.9x TK-90's 58.5k.
READ (4) same iterations: TKX's rows 60-90 scored 831.24 deployed (824.14
  raw) against TK-90's 856.62 (830.81): -25.4 -- the stretched schedule
  cost nothing over the first 90 iterations.
READ (5) tokens, descriptive: play shares (4 / 8 / 16 / 32 frames) 0.11 /
  0.04 / 0.58 / 0.27 @90 -> 0.45 / 0.27 / 0.20 / 0.08 @140 -> 0.51 /
  0.40 / 0.07 / 0.03 @270: play moved to short tokens, as TK-90's did
  late. Frames per decision, deployed (probe): 17.9 @90, 5.7 @140, 11.0
  @180, 8.0 @270; acting 18.7 @90 -> 7.2 @270. Decisions per life 94-134
  from row 220.
READ (6) stability: the largest drop from row 140, -397 (857 @255 -> 460
  @260). The first row >= 881 was row 70; 15 rows fell below 502.08 after
  it (140, 145, 150, 155, 170, 175, 180, 190, 205, 215, 220, 240, 245,
  260, 265). The best 7-row deployed window is rows 70-100, 937.4; the
  peak row 100, 1213.
READ (7) clip growth: the ceiling followed its stretched schedule (28 @90,
  88 @120, 160 from 140). reward_kept 0.98-0.998 to row 115, 1.000 from
  row 120; reward_max sat AT the ceiling through row 115 (an 80-unit
  record clipped), 80 for rows 120-145, 40-42 from row 150: non-binding
  from row 120, as RG's was from row ~30.
PRE-REGISTERED DECISION: WORSE + PLATEAU -> WORSE (any trend): diagnose
  before anything else. 90 iterations stay the Ms. Pac-Man length for
  P2b / P3 comparisons; the 3x stretch does not pay at this recipe.
DIAGNOSIS, descriptive (one seed -- "consistent with", not a cause):
  - The fall is rows 100 -> 140 (1213 -> 296 deployed) at a CONSTANT
    temperature (0.5 to row 135): not the temperature stages (140, 205).
  - Over the same rows the prior went flat: maxp 0.53 @90 -> 0.25 @100 ->
    0.05 @135-145, policy_entropy 1.85 -> 3.49 (ln 36 = 3.58) -- the
    deployed player's argmax over a near-uniform prior. It never
    re-sharpened (maxp <= 0.24 to row 270).
  - It began ~10 rows after the ceiling passed ~30-40 units (28 @90, 41
    @100), and so did TK-90's: maxp held 0.66 to row 45 (ceiling 143 @45)
    and fell 0.61 @55 -> 0.27 @85 -> 0.20 @90 (its prior-flattening note,
    READ (4) of the 90-iteration result). RG -- the same clip growth over
    9 actions -- never flattened (maxp 0.47-0.56, entropy 1.30-1.55 of
    ln 9 = 2.20, every 10th row 5-85). Consistent with: once large rewards pass the
    clip, a 50-simulation birth search over 36 tokens spreads its visits
    and the prior trained on them flattens; TK-90 stopped at row 90 before
    its argmax lost its level, TKX ran 180 rows past it.
  - Already in flight: the distillation step's 14-token student
    vocabulary. DS (the end-state ceiling 160 from row 0) sharpens (maxp
    0.19 @5 -> 0.54 @40); FS (the same 14 tokens from scratch, TK's clip
    growth) is the cleaner check of the vocabulary-size reading.
  - Not checked yet: the visit distributions themselves (the birth
    targets' entropy by row), the value scale against the ceiling.
  - UPDATED 2026-10-24 (the distillation RESULT): FS -- 14 tokens, TK's
    clip growth -- flattened too, so the vocabulary-size reading is not
    supported; and the "~10 rows after the ceiling passed ~30-40 units,
    and so did TK-90's" lag above is wrong (TKX's onset coincides with
    that passage, TK-90's comes ~25-30 rows after it). See that RESULT.

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

### RESULT -- P2a on Ms. Pac-Man: tokens (TK) and reward-clip growth (RG), seed 0, 2026-10-21

TK: see TK DONE in the pre-registration below (15.4 h, one attempt). RG
(mspacman-rgrow-s0-2026-10-19) trained from the run worktree (detached at
a81bcf39) 10-20 07:35 -> 10-21 00:47 in one attempt (61,951 s, 11.5
min/iteration; no deaths, no ALERTs, supervisor.err empty). The hand-off
ran its evals by 01:03: all 18 rows in both variants. EXACT eval, 30 noop
starts, prior argmax; whole games, no capped game in any row of either
arm; every row 4/4 phases. DEPLOYED (acting states) primary, RAW
secondary, IN-RUN = the run's 8-game eval; TWIN = mspacman-s0-2026-10-16:
  iter      5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90
  RG dep  423  780  475  489  374  692  435  478  853  521  609  516  563  676  816  752  606  621
  RG raw  618  460  439  512  458  548  623  785 1350  483  969  563  498  612  803  883  685  815
  RG in   450  654  460  560  378  565  396  571  761  530  708  489  591  494  748  668  603  605
  RG env  2.9  5.9  8.6 11.5 14.1 17.6 20.4 23.4 26.4 30.1 33.4 36.8 40.5 43.5 46.5 49.8 53.0 56.4
  TK dep   60   60  271  482  665  524  913  742  732  659  879  780 1033  863  920  825  885  691
  TK raw   60   60  212  597  542  629  627  604  747  863  526  731  597 1028  767  823  721 1148
  twin dep 462 1109  453  606  495  684  490  706  588  643  508  781  338  361  730  816  875  642
(env in thousands of base steps; TK's row 90 at 58.5k, the twin's 53.4k.)
READ (1) level, deployed mean 60-90 (bands +-232; raw +-138):
  - RG 650.10 (SD 105.6, g 0.352) against the twin's 649.14: +1.0 -> NO
    DIFFERENT. Raw 694.05 (g 0.397) against 585.62: +108 -> NO DIFFERENT.
  - TK 856.62 (SD 108.3, g 0.564): against the twin +207.5 -> NO
    DIFFERENT (raw +245.2 -> BETTER); against RG +206.5 -> NO DIFFERENT
    (raw +136.8 -> NO DIFFERENT, 1.2 short of the band).
  - Neither arm near random (both > 502.08).
  - By thirds (rows 5-25 / 30-55 / 60-90), deployed: RG 508 / 598 / 650;
    TK 308 / 742 / 857; the twin 625 / 603 / 649 (504 without its @10
    outlier). TK started slowest (rows 5 and 10 scored 60, a stuck
    policy) and climbed most (+549 from the first third to the last; RG
    +142, the twin +24).
READ (2) data: matched within 10% (RG 56.4k, TK 58.5k, the twin 53.4k
  env steps at row 90). TK's deployed row beat RG's at every row from 50
  to 90 (9 of 9); RG led at rows 5-20, 30 and 45.
READ (3) clip growth, descriptive: the ceiling followed the schedule
  (1.6 @5, 2.8 @10, 142.9 @45, 160 from row 50). reward_kept 0.80 @5 ->
  0.97 @20 -> 1.000 from row 30: once the ceiling passed ~26 it clipped
  nothing, because RG's buffer never held a record above 40 units (a
  second ghost, 400 points) -- the 80 / 160 ghost values the end ceiling
  was built for never occurred. TK's buffer held an 80-unit record from
  row 50 (most likely a third ghost; a token's record sums its inner
  steps). The row value (clipped units) ran 0.5-1.1 to
  row 50 and 1.2-3.6 after.
READ (4) tokens (TK), descriptive:
  - Play shares over rows 60-90: 4-frame 0.32-0.51, 8-frame 0.02-0.03,
    16-frame 0.34-0.51, 32-frame 0.10-0.14 -- unlike Pong (32-frame 0.64
    @90), play moved TOWARD short tokens late; the birth search spread
    more evenly (row 90: 0.28 / 0.27 / 0.24 / 0.21).
  - Frames per decision, deployed (probe) 11.5-15.7 over rows 60-90 ->
    3.8-5.2 decisions per second against the twin's 15; acting 11.5-14.3.
    Decisions per life 57-84.
  - Payoff by duration (count-weighted, clipped units at the ceiling 160
    from row 50): small and without a consistent sign (row 60: +0.35 /
    -1.70 / -0.47 / -0.11 for 4 / 8 / 16 / 32 frames; row 90: -0.21 /
    +0.08 / -0.40 / +0.29).
  - policy_entropy rose from 1.6 @60 to 3.1 @90 (ln 36 = 3.58): the
    prior flattened late while its argmax kept its level.
READ (5) stability, largest deployed drop from row 45: RG -332 (853 @45 ->
  521 @50; raw -866, 1350 -> 483), TK -194 (885 @85 -> 691 @90), the twin
  -443.
PRE-REGISTERED DECISION:
  - TOKENS: HOLD UP (TK g >= 0.2, not WORSE than the twin, not WORSE than
    RG) -> P2b (the composition loss) on the TK arm; RG is not WORSE, so
    the base question does not arise.
  - GROWTH: NO DIFFERENT -> Kai's call (keep it for its intent, or retune
    the end ceiling / schedule).
Beyond the letter (one seed, within-run bands -- "consistent with"):
  - RG = the twin, so TK's lead over the twin is the tokens' (plus their
    interaction with the clip), not the clip growth's. TK leads RG by
    +207 deployed and +137 raw, each just inside its band.
  - The clip growth as built is non-binding after row ~30: no RG game
    reached a third ghost. An end ceiling of ~40 units would have run the
    same RG; a schedule that matters for Ms. Pac-Man would grow slower
    early (it was ~1 for 5 rows only) or tie the ceiling to what the
    buffer holds.
  - P2b runs after the extended token run (TKX, live on the GPU; 270
    iterations, ~01:00 10-23 at 10.8 min/iteration): its length and base
    follow TKX's decision table and Kai's GROWTH call.
TKX INTERIM (rows 5-80 settled; not read against its pre-registration until
row 270): deployed 354 / 692 / 506 / 527 / 748 / 550 / 791 / 591 / 782 / 770
/ 761 / 680 / 824 / 898 / 914 / 828 (rows 5-80), at the ceiling 19.5 @80
and 16-frame tokens 0.64 of play (the long-token bias runs to row 69);
row 80 at 55.5k env steps, 10.2 min/iteration so far.

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
