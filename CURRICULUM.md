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

### RESULT -- E3a: duplicate spellings are COMMON in strong play and latents do not group them (T2, seed 0), 2026-11-05

(Pre-registration: E3a in "Queued experiment (pre-registered) -- E3",
below, with amendments A1-A14 and its audit loop, closed at r3.)
RUN: 07:36-07:56 EST 11-05 (1,164 s), CPU, under the lock, on T2's
iteration-42 snapshot; tests/test_e3a.py's 55 tests passed first. Output:
benchmarks/probes/probe_e3a_sequences-mspacman-two-t2-s0-2026-10-29.jsonl
(summary), -meta.jsonl and -decisions.jsonl.
VALIDITY: main_line_ok -- all 30 games MATCH the exact eval (mean 1699.0);
the replay and reached-stack checks held at all 4,454 candidate plays; no
branch lost a life inside a candidate's inputs.
INPUTS: T2's log 126 lives, 6,548 decisions; 32 candidates (keyed by raw
inputs, A13); tau_L 0.030-0.037 by length. Eligible runs on the main
line 3,798 / 3,774 / 3,735 (2 / 3 / 4 decisions), of which candidate plays
1,796 / 1,719 / 939.
READ (1) the DUPLICATE-SPELLING RATE (A11): 65.3% of the 3,815 candidate
  plays with another same-length candidate had one landing on identical
  frames (all 4,454 plays: 55.9%). By length: 64 frames 79% (12
  candidates), 48 52% (8), 36 38% (3), 20 9% (3); the single-candidate
  lengths 0 by construction. 14.1% of same-length candidate pairs were
  equivalent; 5.35 outcome classes per (t, length) against 8.57
  candidates.
READ (2) PREDICTED latents: precision 0.177, recall 0.269 (all one class:
  0.141); the plain pair rule 0.207 / 0.176; tau x 0.5 0.190 / 0.065, x 2
  0.159 / 0.671. Median distance: equivalent pairs 0.071, non-equivalent
  0.094. Above the trivial rule, far below the 0.7 bar: FAILS.
READ (3) REAL-OUTCOME latents (descriptive): precision 0.381 at recall 1.0
  (by construction); at tau x 0.5 0.596. Equivalent pairs sit at distance
  0 exactly, non-equivalent at a median 0.093, but many below tau_L.
VERDICT (A8): KAI_CALL -- duplicates are common (65% >= 10%) and grouping
  by predicted latents fails.
BEYOND THE LETTER:
  - Kai's assumption ("raw token sequences from high level gameplay will
    have little duplicate spellings") does not hold on T2: even after
    merging identical inputs (A13), most long candidates have a
    same-outcome sibling in the state where they are played. The
    duplicates grow with length (corridor travel: several directions
    have the same effect between junctions) -- E0's finding (3.28 outcome
    classes among 18 tokens) at sequence scale.
  - The equivalence is STATE-dependent: two spellings are one move in a
    corridor and two at a junction. A global merge in the vocabulary
    would be wrong at junctions; the redundancy is a search problem (one
    expanded child per outcome class, E2), which needs a per-state
    equivalence model -- and the agent's own model cannot yet supply one
    (read (2); E1 likewise).
  - Even real-outcome latents separate outcomes only partly at tau_L
    (precision 0.38), so "identical latents" as a criterion needs a much
    tighter threshold than the model's own prediction error.
  Kai's call among: E3 mining WITHOUT grouping (tokens are distinct input
  sequences; their state-dependent redundancy measured in E3 -- search
  visits split across equivalent children, slot use); variable-length
  "until the next decision point" tokens (absorbing corridor travel);
  back to E1 with diverse own play for an equivalence model first.
DECISION: Kai 2026-11-05: "Let's go with 1 for now and backlog the other
2 as possible things to try later" -> E3 MINES WITHOUT GROUPING (A8's
fallback, taken despite the 65% duplicate rate; E3 measures the
redundancy's cost). BACKLOGGED: variable-length "until the next decision
point" tokens; E1 again on diverse own play (an equivalence model). Also
still in the backlog: Test D (the swap trigger) and its offline D0.

### RESULT -- Test B2W: Test B2's lineage with BBF's AdamW -- STOPPED in W1 by Kai, AdamW dropped (seed 0), 2026-11-05

(Pre-registration: "Next experiment (pre-registered) -- Test B2W", below,
with its amendments, audits and run notes.)
DECISION: Kai 2026-11-05 07:30, at W1's row 35: "Is this worth
continuing. I think we should probably just drop adam and move onto the
next thing." Stopped at 07:32 (the supervisors w1 / w2 and W1's trainer by
exact PID; no Ray process left); W1 and W2 ABANDONED at 07:33, so the
hand-off ran W1's abandon pass over its snapshots. W2 never ran, so read
(1) has no window and no pre-registered verdict.
WHAT RAN: W0 complete (iteration 52, 33,592 env steps); the swap to W1
(the prep from W0's own play); W1 to iteration 35 (25,573 of its 33,075
env steps). 0 non-finite steps; no alert but the advisory memory ones.
DEPLOYED / trainer-BN exact eval (T = Test B2's control at the same row):
  W0 rows 5-final: 334 / 753 / 310 / 401 / 1327 / 612 / 980 / 431 / 694 /
     739 / 630 (iteration 52); every row 655.5 (n 11) against T0's 709.1
     (n 10): -53.6, band +-189.5 -> no different (read (3)'s W0 against
     T0, descriptive). Far more volatile than T0 (adjacent swings up to
     +-926; T0's largest drop -67).
  W1 row 0 520 (T1 892), rows 5-35: 846 / 396 / 508 / 277 / 245 / 245 /
     311 (T1 730 / 656 / 850 / 901 / 1045 / 1849 / 1123): by row 20 at or
     below a random policy (307), where it stayed (trainer-BN 290 at 35).
  W1's TRAINER-BN nets were degenerate from the offline phase on: exactly
     333.3 at row 0, row 5 and three offline points, exactly 240 from row
     10 -- a fixed policy whose score did not move as the weights did.
     W0's trainer-BN ran far below its deployed throughout; T0's and T1's
     tracked theirs.
MECHANISM, as far as the logs show (one seed):
  - Decisions collapsed to the shortest token. Frames per decision 16.9
    at W0's row 5 -> 5-7 from row 10 (T0's recipe ~19-27); W1 4.3-6.3
    (4-frame tokens 85-96% of play, 82-86% of the search mass).
  - W0's prior never sharpened (policy_maxp 0.38-0.50, entropy rising to
    2.1; T0's sharpened). The swap's prep then dropped 19 of 36 tokens --
    nearly every 16- and 32-frame token -- and discarded 38% of the
    master's search mass on them; W1's distillation set was 3.5x T1's
    (W0 made ~3x the decisions).
  - The birth search that labels new data runs on the trainer's nets in
    eval mode (their BN running statistics); those were the degenerate
    nets, so W1's new targets kept pushing toward 4-frame tokens.
  - Not separated: AdamW's own effect in this stack against an
    interaction of AdamW with our BN setup (BBF has no eval-mode birth
    search). A CPU diagnostic on W1's snapshots (eval-mode running
    statistics against the acting statistics on the same states) would
    separate them; not run, as AdamW is dropped (Kai's call).
DECISION AFTER (the pre-registration's NO DIFFERENT / WORSE branches):
  EfficientZero's SGD stays in the base. E3 runs on Test B2's recipe. Next
  (Kai: "the next thing"): E3a's tests and probe, then E3.

### Queued experiment (pre-registered) -- E3: tokens mined from the agent's own play, added at every swap, with E3a (does grouping by latents make sense?) first (Kai, 2026-11-04)

DECISION: Kai 2026-11-04, with Test B2W ("plan out a queued test that
will depend on the result of this test"). Test D is backlogged and the
generations keep a constant env-step length. Kai's answers to the three
design questions:
  - GROUPING (E1 found that a model trained on the agent's own play cannot
    yet tell which tokens are equivalent): "Before deciding on this option,
    I'm wondering if we can run some sort of test to see if grouping by
    latents even makes sense? If not, we can probably mine without
    grouping for now and just assume that raw token sequences from high
    level gameplay will have little duplicate spellings since it will
    hopefully be somewhat optimized towards a consistently correct
    solution." -> E3a below, before E3.
  - TOKEN FORM: "I'm fine with fixed length as long as it can be a mix of
    tokens so it isn't always the exact same (for example, in a different
    game where tokens are a mix of inputs at the same time or in sequence,
    I'd like this to be able to learn those tokens)."
  - WHEN: at every swap.
Background: E0 (T2's 18 tokens: 3.28 outcome classes per decision);
E1 (LOW: T2's own forward model grouped single 4-frame tokens at chance,
P 0.536 against the trivial 0.530, because T2's play is narrow -- 2 of its
9 four-frame actions made 90% of its 4-frame plays). E1 never tested
multi-token sequences, latents of REAL outcomes, or how often strong play
spells one move two ways. This is E0's planned E3 ("tokens from successful
play"), its grouping step now settled by E3a instead of E1; E2 (merged
search) is not a prerequisite and is skipped.
DEPENDS ON:
  - Test B2W: E3 runs on the recipe B2W leaves as the base (BBF's AdamW if
    B2W reads BETTER, else EfficientZero's SGD = Test B2's recipe). Its
    control is that lineage's own run (W0-W2 or T0-T2), already run.
  - E3a: whether, and how, mined sequences are grouped.

E3a -- does grouping by latents make sense for mined sequences? (CPU; the
emulator scores only, per Kai's long-term rule)
  DATA: Test B2's T2 (its iteration-42 snapshot, deployed 1699.0) on the 30
  exact-eval games E0 and E1 used; candidates mined from T2's own-play log.
  CANDIDATES: contiguous runs of 2-4 decisions in T2's own play, each a
  fixed sequence of (action, frames) tokens (mixed actions allowed), total
  length at most 64 frames, counted over the HIGH-ADVANTAGE half of its
  decisions (the decision's realized discounted return minus its value
  estimate, ranked within the log); the top 32 by count, single-action
  runs that equal an existing duration token removed.
  TRUTH (the emulator, as E0's census): at each live decision of the 30
  games, every candidate of the same total length is executed from the
  same state; candidates landing on identical frames form one outcome
  class.
  READS:
  - (1) the DUPLICATE-SPELLING RATE: of the candidate plays T2 actually
    made at those decisions, the share for which another candidate of the
    same length lands on identical frames (Kai's assumption: low).
  - (2) grouping by PREDICTED latents (what works without a simulator):
    T2's dynamics model unrolled over each candidate from the decision's
    latent; pairs within a distance threshold grouped; precision / recall
    against the truth, beside the trivial rules, as E1.
  - (3) grouping by the latents of the REAL outcomes (an upper bound):
    T2's representation of each candidate's resulting frames, the same
    scoring. If (3) cannot separate outcomes, latent grouping is not worth
    pursuing.
  DECISION:
  - (2) reaches precision 0.7 at recall 0.2 (E1's bar) -> E3 groups mined
    sequences by predicted latents.
  - else, (1) under 10% -> E3 mines WITHOUT grouping (Kai's fallback).
  - else (duplicates common, latents fail) -> Kai's call.
  COST: a probe and its tests (built and audited while B2W trains), about
  1-3 h of CPU after B2W, under the lock.
  E3a AMENDMENTS (the design, 11-04, before the build):
  (A1) SOURCE: Test B2's T2 -- its iteration-42 snapshot (E0's and E1's;
      deployed exact eval 1699.0) and its own-play log over its generation
      (iterations 1-42, at raw-step resolution with token boundaries).
  (A2) ADVANTAGE of a logged decision: the life's discounted return from
      it (T2's reward clip and unit, gamma per frame, as the prep's value
      check computes it) minus T2's value estimate on the decision's stack
      (the deployed acting states); ranked over the whole log; the HIGH
      half is the top 50%.
  (A3) CANDIDATES: every contiguous run of 2, 3 or 4 decisions inside one
      logged life whose FIRST decision is in the high half; its key is its
      token tuple ((action, frames), ...), its LENGTH the sum of its
      frames, at most 64. A run of one action whose length equals a token
      already in T2's vocabulary is dropped (it already exists as a
      duration token). The 32 most frequent keys are the candidates (ties
      by first occurrence).
  (A4) TRUTH (E0's rule, the emulator only scoring): the main line is the
      exact eval of the 30 noop starts, each game validated against its
      deployed exact-eval row (VOID otherwise, as E0). At each live
      decision t, the run T2 actually plays next (its own next 2, 3 and 4
      tokens on the main line) is a CANDIDATE PLAY when its key is a
      candidate. For each candidate play, every candidate of the same
      LENGTH is played from t's saved emulator state, then NOOP to a common
      window of (length + 60) frames; two candidates are EQUIVALENT at t
      when every raw RGB frame of the window is identical. Each (t,
      length) is scored once. Also checked: the played candidate replays
      to the main line's own frames (branch-order independence).
  (A5) READ (1), the DUPLICATE-SPELLING RATE: the share of candidate
      plays for which another same-length candidate is equivalent at t.
      Descriptive: the equivalent share of all same-length candidate
      pairs scored, and the classes per (t, length) against the
      candidates per (t, length).
  (A6) READ (2), PREDICTED latents: T2's representation of t's stack
      (deployed acting states), its dynamics unrolled over each
      candidate's tokens, the final latent projected (E1's forward model,
      cosine distance in the projection space). A pair is GROUPED when its
      distance is below tau_L: the median L-length prediction error over
      T2's logged runs of that length (the distance between the unrolled,
      projected latent and the target projection of the stack the run
      actually reached; E1's one-step tau, by length). Scored as pairs
      against the truth: precision and recall, beside the trivial rules
      (all one class; all distinct), as E1.
  (A7) READ (3), REAL-OUTCOME latents (an upper bound): T2's
      representation of the stack each candidate actually reached in the
      emulator (before the NOOP tail), projected; grouped below the same
      tau_L; scored the same way. Descriptive: the median distance of
      equivalent against non-equivalent pairs.
  (A8) DECISION, as above: (2) precision >= 0.7 at recall >= 0.2 and above
      every trivial rule's precision -> E3 groups by predicted latents;
      else (1) under 10% -> E3 mines without grouping; else Kai's call.
      Too few candidate plays to read (under 100) -> Kai's call.
  BUILD: benchmarks/probes/probe_e3a_sequences.py (E0's main line and
  branching, E1's forward model and calibration, reused by import) and
  tests/test_e3a.py; audited like every build; tests and the run after
  B2W (they need the host's memory), CPU, under the lock.
  BUILT 2026-11-04 (pre-audit, uncommitted at first): the probe and its
  tests, written by a build agent from A1-A8 (py_compile only: nothing ran
  while B2W trains; the agent once ran a read-only `python -c` to print a
  prep's vocab.json fields, against its brief). Its judgment calls, now
  amendments:
  (A9) A2 in detail: the return discounts per BASE STEP with T2's clip
      and unit, exactly as distill.discounted_return_max (T2's own value
      targets discount per token edge, so returns and values are slightly
      offset; the ranking only uses their difference); a life cut short
      (not terminated) is not bootstrapped -- such decisions are counted
      and reported; T2's iteration-42 snapshot (deployed acting states)
      values every logged decision, early iterations included; the high
      half is the floor(n / 2) largest advantages, ties to the earlier
      decision.
  (A10) A3 / A4 in detail: runs containing a life's last, cut-short token
      are excluded; a run is dropped as a duration token only when it is
      ONE action and (that action, its length) is in T2's vocabulary. On
      the main line a run is eligible only inside one life, every token
      playing its full frames, its last token allowed to lose a life but
      not to end the game (EpisodicLife's extra NOOP between lives;
      MaxAndSkip stops mid-skip at game over). Branches run once the next
      tokens are known, then the emulator is restored to the main line.
      The replay check is two VOID-making checks: E0's replay digest of
      the played candidate, and the stack its branch reaches equal to the
      main line's own next stack. Liveness comes from E0's census file
      (required, as for E1).
  (A11) READ (1)'s DECISION RATE counts only the candidate plays whose
      length holds at least one other candidate (a play with no
      same-length alternative cannot be a duplicate; counting it would
      dilute the rate toward "rare"). The rate over all candidate plays
      is reported beside it, descriptive.
  (A12) READS (2) / (3) in detail: the representation runs with the
      deployed acting statistics; tau_L pools every full logged run of
      length L (any start, any key); a pair is grouped strictly below
      tau_L, classes are the connected components (E1's; the plain pair
      rule reported beside it); the real-outcome latent of read (3) is
      the TARGET projection; the trivial bar of A8 is strict, and "all
      distinct" (no precision) sets none; the under-100 rule comes first;
      tau x 0.5 and x 2 are reported, descriptive. CAVEATS: read (2)
      compares two ONLINE predictions against an online-to-target tau_L
      (as E1's forward read): a pair below it means the two predictions
      sit closer than the model's own prediction error. Read (3)'s recall
      is near 1 by construction (frame-equivalent candidates reach
      identical stacks), so only its precision informs; it stays
      descriptive and never decides.
  AUDIT r1 (two fresh-eyes auditors, 11-04, read-only): r1a (the probe)
  nothing HIGH -- the main line is undisturbed (branches touch only the raw
  ALE from one cloned state, then restore), the reached-stack check matches
  atari_wrappers' MaxAndSkip, the reads follow A1-A12; one MEDIUM (below,
  A13) and LOWs. r1b (the tests) three HIGH -- no verdict case pinned
  A8's precedence (grouping before the fallback); every fixture had read
  (3) equal to read (2), so a verdict reading (3) would pass; every
  scripted candidate play stood first in its group, so a wrong played
  index would pass -- and MEDIUMs (no fast main() test on fakes; the
  game-ending guard untested; the under-100 count unregistered; one tau
  for every length), LOWs.
  (A13) CANDIDATES ARE KEYED BY THEIR RAW INPUTS (audit r1a M1): a run's
      key is its per-base-step input sequence (each token expanded to its
      frames / 4 base steps of its action), not its token tuple, so
      re-tokenizations of the same inputs -- ((a,32),(b,4)) and
      ((a,16),(a,16),(b,4)) -- are ONE candidate (their counts summed; the
      spelling played most often stands for it, and read (2) unrolls that
      spelling). Such pairs are equivalent wherever played and would
      otherwise inflate read (1) with duplicates that an exact,
      model-free, game-agnostic canonicalization removes. The duration-
      token drop of A3 / A10 then reads: a key whose inputs are one
      action held for a length that is a token of T2's vocabulary.
      E3 canonicalizes the same way.
  (A14) THE UNDER-100 RULE of A8 counts the plays of A11's decision rate
      (candidate plays whose length holds another candidate).
  FIXES r1 (c4375b4e): A13 and A14 built; truncated decisions counted;
  each branch's life loss recorded (descriptive); the encode-side h
  transform checked; tests for every r1 finding (verdict precedence, read
  (3) never deciding, a played index > 0, main() on fakes over two games,
  the game-end guard, per-length taus, the NOOP tail's delayed effect,
  A13's merges).
  AUDIT r2 (two fresh-eyes auditors, 11-04): r2a (the probe) nothing above
  LOW -- played runs map one-to-one to candidates by inputs, branches play
  the main line's own frames, the main line is undisturbed, the right rate
  and count decide; L: a few stale game-end targets entered tau_L's
  calibration. r2b (the tests) three MEDIUM -- no fixture with mixed taus
  (a pooled tau would pass), read (2)'s unrolled spelling pinned only
  where it equals the played one, main()'s advantage-to-high-half wiring
  unpinned -- and LOWs.
  FIXES r2: the calibration drops runs ending at a game-ending life's last
  decision (counted); the restore after branching in a finally; whole base
  steps asserted; tests for each r2 finding (per-group tau, the
  representative spelling unrolled, the advantage array reaching the high
  half, an unrounded recall edge, argmax mismatches voiding).
  (Also r2: the slow smoke's empirical checks -- the return / value
  correlation, the p50 ratio, at least one candidate play in game 1's
  first 80 decisions -- relabelled DIAGNOSTICS; the last skips rather than
  fails.)
  AUDIT r3 (one fresh-eyes auditor, 11-04, on 98b4d67b): nothing above LOW
  -> THE E3a AUDIT LOOP IS CLOSED. Accepted, not fixed: L, a capped life's
  last runs (kept in the calibration, correctly) are not pinned by a test;
  L, the slow smoke's skip on no candidate play could hide an eligibility
  bug -- the full run would show it plainly (zero candidate plays, read as
  KAI_CALL_TOO_FEW), so a full-run result with almost no eligible runs is
  checked by hand before it is read; NITs.
  NOT YET RUN: tests/test_e3a.py (fast and slow) and the probe, after B2W
  (the host's memory), CPU, under the lock. A failure there is fixed and
  re-audited before the probe runs.
  RAN 11-05 (B2W stopped, the host free): tests/test_e3a.py, all 55 passed
  on the first run (07:36, 42 s, the slow real-data smoke included); the
  probe launched 07:37 on T2's iteration-42 snapshot (CPU, under the lock).

E3 -- the lineage with mined tokens (after B2W and E3a)
  THE LINEAGE: the base recipe's three generations at ~33.3k env steps
  each (constant), seed 0, every env step counted, mined tokens added at
  BOTH swaps.
  A MINED TOKEN is a fixed sequence of steps, each step an input state
  held for a number of frames: on ALE an action (UPLEFT is up and left at
  once), on a controller a set of buttons. So the same machinery covers
  inputs at the same time and in sequence. Like a duration token it is ONE
  dynamics transition and one search edge; its reward is the sum over its
  frames, its edge discounted gamma_frame^frames.
  AT EACH SWAP, in the prep (the master's own-play log, which the prep
  already reads):
  - the vocabulary keeps its fixed number of slots; the 1% rule drops the
    least-used tokens as now; mined tokens fill freed slots (at most 8 per
    swap);
  - mining: the candidates as in E3a, from the master's own play, scored
    by frequency, by compression (decisions saved), and by EFFECT
    CONSISTENCY (low variance of the master's latent change across the
    sequence's occurrences: game-agnostic), grouped or not per E3a;
  - the distillation set relabels the master's stretches that match a
    mined token as one decision of that token, so the student starts with
    the new tokens in its targets (the exact target rule is a build
    amendment, audited);
  - a mined token the student stops using is dropped at the next swap by
    the 1% rule, like any token.
  READS (EXACT eval as before; DEPLOYED primary):
  - (1) PRIMARY: the last generation's rows in [90k, end] against the
    base lineage's (W2 or T2), band 2 x 216.9 x sqrt(1/n + 1/n_c):
    BETTER / NO DIFFERENT / WORSE. Trainer-BN likewise.
  - (2) the mined tokens: which, their play share and search share per
    generation, their effect consistency, how many survive the next swap.
  - (3) decision speed: frames per decision and decisions per second of
    play (the real-time goal), against the base lineage.
  - (4) per generation and swap as Test B2's reads; the prior;
    probe_prefix_bn; stability.
  DECISION AFTER:
  - (1) BETTER: mining joins the swap.
  - (1) NO DIFFERENT: if (3) shows longer decisions at the same score,
    Kai's call (a speed gain at no cost); otherwise mining stays out.
  - (1) WORSE: diagnose from (2) (unused or inconsistent tokens) before
    any rerun.
  CAVEAT: one seed. The mined tokens come from the master's own play
  only (no watched footage, so P5c's barrier question does not arise).
  BUILD: the prep's mining and relabeling, the student vocabulary's mined
  entries (validate_student_vocab, the token executor and its inner
  frames, the own-play log at raw-step resolution), the supervisor's
  lineage, tests, tooling; audited like every build. Pre-registered in
  detail (amendments) before the build starts.
  COST: the build and its audit loop during or after E3a, then about 1.3
  days of training.
  E3 AMENDMENTS (the design, 11-05, after E3a and Kai's decision to mine
  WITHOUT grouping; the token map of the code is in the build notes):
  (E3-A1) THE BASE: Test B2's recipe (EfficientZero's SGD; Test B2W
      dropped AdamW). THE FIRST GENERATION IS TEST B2'S T0 ITSELF: its
      recipe has no mining, so E3's generation 0 would be a rerun of it.
      E3 starts at the first swap from T0's final snapshot (iteration 47)
      and own-play log in the frozen tree Y:\SwitchPlayer-testb2-run
      (read-only; T0 passes the master gate: its last two rows average
      904). E3's M1 and M2 are T1's and T2's recipes plus mining at both
      swaps; the control is T1-T2 themselves, and the two lineages share
      generation 0 exactly. Saves ~9 GPU hours.
  (E3-A2) TOKENS: a duration token stays (action, frames), bit for bit; a
      MINED token is a tuple of steps ((a1, f1), (a2, f2), ...), at most
      64 frames (16 base steps), canonical as nested tuples (JSON lists
      converted on read; a one-step sequence collapses to (a, f)). It is
      one slot, one dynamics transition (the slot's own learned
      embedding row), one search edge; its reward is the sum over its
      base steps, its edge discount gamma_frame^(total frames), its
      frame-time n-step targets by its total frames. play_token plays
      each base step with that step's action. Two tokens with the same
      per-base-step inputs are one token (A13).
  (E3-A3) MINING, in the prep, from the master's own-play log: E3a's
      candidates (runs of 2-4 decisions inside one life, the first in the
      high-advantage half, full tokens, length <= 64, keyed by raw inputs,
      the most-played spelling standing for each key), the advantage from
      the master's own value on each decision's stack (deployed acting
      states). Dropped: a key equal in inputs to a token of the student's
      vocabulary; a key whose inputs are a PREFIX of a higher-scored
      selected key's, or extend one (prefix-free, so mined paths are
      disjoint, E3-A5). SCORE = count x (decisions saved per use, k - 1)
      x effect consistency c, c = 1 - the mean cosine distance of each
      occurrence's latent change (the master's representation of the
      stack after the run minus before, deployed acting states) from the
      candidate's mean change. NO GROUPING (Kai 11-05).
  (E3-A4) THE VOCABULARY: the 1% rule as before; then the top-scored
      candidates are APPENDED after the kept tokens, at most 8 and at most
      the slots the rule freed (the slot count never grows past the
      master's). vocab.json records the mined tokens, their spellings,
      counts, scores and consistency.
  (E3-A5) THE TARGETS (the key rule): at EVERY distillation record -- not
      only where a mined token was played -- each mined token m = (t1,
      ..., tk) gets the mass P(m) = q(t1) x prod_{j>=2} pi(tj | the
      master's latent after t1..tj-1), where q is the master's search
      target at the record and pi the master's prior on its own
      dynamics-unrolled latents; P(m) is moved from t1's slot (prefix-free
      keys keep the moved masses disjoint, so they never exceed q(t1));
      then the dropped tokens' mass is discarded and the rest
      renormalized, as now. So a mined slot carries the master's own
      preference for the whole sequence everywhere, never a forced 0 (a
      0 elsewhere would train it down and the 1% rule would drop it at the
      next swap).
  (E3-A6) RELABELING: when re-tokenizing the master's log into the
      student's vocabulary, at each decision boundary the longest mined
      token whose inputs equal the next k decisions' concatenated inputs
      (all played in full, inside one life) replaces them with ONE record
      (the start stack, the mined slot, the summed raw rewards); greedy,
      left to right. Its target is E3-A5's at that record. The interior
      stacks leave the set. Relabeled counts are recorded.
  (E3-A7) THE LONG-TOKEN BIAS (TOKEN_LONG_BIAS 0.5, inherited from TK): a
      mined token's bias weight is capped at the longest duration token's
      (32 frames), so length alone does not advantage it.
  (E3-A8) THE SECOND SWAP: the master (M1) holds mined tokens; mining,
      the 1% rule, relabeling and E3-A5 treat its mined tokens as tokens
      (a kept mined token stays; a dropped one is replayed as its raw
      base steps, each with its own action); new candidates are mined
      from M1's own play and appended likewise. Every vocabulary check
      (validate_student_vocab, student_arm, master_spec) accepts a mined
      tail declared in vocab.json.
  (E3-A9) SUPPORTS AND ALERTS: the reward and value-prefix support check
      (ceiling_fits_supports) and the supervisor's MAX_TOKEN_STEPS bound
      use the vocabulary's longest token (16 base steps), not 32 frames.
  (E3-A10) READS, as above: (1) PRIMARY M2's [90k, end] against T2's
      (1629.5, n 4), band rule; (2) per mined token: its play share and
      search share per generation, its effect consistency, whether it
      survives the next swap's 1% rule; the REDUNDANCY E3a predicts: the
      search visits split across same-outcome children (descriptive:
      the share of each record's search mass on mined tokens whose inputs
      overlap the played token's); (3) frames per decision; (4) Test B2's
      reads per generation against T1-T2.
  (E3-A11) DECISION AFTER: (1) BETTER -> mining joins the swap; NO
      DIFFERENT -> Kai's call (with (3): longer decisions at the same
      score are a real-time gain); WORSE -> diagnose from (2).
  BUILT, PART 1 (11-05, pre-audit; a build agent; nine suites pass, 703
  tests: test_e3 67 new, test_tokens 72 (+1 CUDA skip), test_distill 58,
  test_gen0 37, test_rgrow 72, test_pbn 30, test_e3a 55, test_testb 67,
  test_supervise_testb 245): muzero's token helpers (token_is_mined,
  token_steps, token_total_frames, token_inputs, canon_token, canon_vocab,
  token_json, token_key; token_frames and token_edge_discounts by total
  frames); ablation's stepwise play_token, token_gate, the bias cap, the
  support check by the longest token, the vocabulary checks with a
  declared mined tail, the mined row diagnostics; distill's mining (ported
  from probe_e3a), E3-A5's targets (mined_path_weights, masked_targets),
  relabeling (retokenize), the second swap, the `--mine` prep flag; a
  slow real-env test of both swaps. Every existing arm, vocab.json,
  snapshot and checkpoint bit for bit (parity tests).
  (E3-A12) the build's judgment calls, now amendments: a token is mined
      when its first element is itself a step (numpy ints stay duration
      tokens); the candidate pool is every key (no top-32 cut), k the
      standing spelling's decisions, only scores > 0 selected; an
      occurrence ending at a game-ending life's last record counts but is
      left out of c (E3a's stale-stack rule); E3-A5's prior uses the nets
      the master's search ran with (its raw BN states; q came from them),
      E3-A3's values and latents the acting states; E3-A5 and E3-A6 apply
      to the tokens mined AT THIS SWAP -- the master's own mined tokens get
      its q directly (it searched them) and are never relabeled; a k = 1
      relabel fires only where a dropped token equals a new mined token in
      inputs; spellings keep their form (duplicates judged by inputs); the
      E3-A9 support check applies only to vocabularies holding a mined
      token (duration-only ones keep the per-base-step check bit for bit);
      mined tokens have their own row bucket; "overlap" (read (2)) = two
      tokens agreeing on every base step they share, the played slot
      excluded; at most 8 mined per swap is enforced on the student, not
      on a master (its mined tokens may span swaps).
  AUDIT r1 of part 1 (two fresh-eyes auditors on 743c925a, 11-05): r1a
  (the core) nothing above LOW -- every duration-token arm bit for bit
  (play, discounts, bias weights, gate, vocab checks, resume equality, row
  keys); mined tokens play, discount and canonicalize correctly; LOWs: the
  E3-A7 cap took the student vocabulary's longest duration token, not the
  spec's 32 frames; probes outside E3's reads unpack (a, f); the slow
  lineage test never asserted a mined token was played; NITs. r1b (the
  distill side) nothing HIGH -- the port matches probe_e3a, the target
  rule moves mass before discarding dropped tokens, relabeling aligns
  records and rewards, the second swap splits dropped mined tokens by
  their own actions, a prep without --mine is bit for bit; MEDIUM: E3-A5
  credited a mined token only along its most-played spelling (below,
  E3-A13); MEDIUM: the prep's wiring untested (acting vs raw states, the
  master's ceiling, the discount, the tail's order, weights on the
  relabeled records); LOWs.
  (E3-A13) E3-A5 SUMS OVER EVERY SPELLING: P(m) is the sum of the path
      products over every logged spelling of m's inputs that is a path of
      master tokens -- a one-token spelling (the master's own token with
      the same inputs, e.g. a dropped (a, 32) or a dropped master mined
      token re-mined) contributing its q directly. Spellings of the same
      inputs are never slot-prefixes of one another, so the masses stay
      disjoint and below q. Otherwise the master's mass on another
      spelling of m's exact inputs never reached m -- small precisely
      where m is the relabeled label -- pushing mined slots toward the
      self-fulfilling 0. And E3-A7's cap is the constant 32 frames, as
      written.
  FIXES r1 (a build agent; nine suites pass: test_e3 77, test_tokens 72,
  test_distill 58, test_supervise_testb 261, test_testb 67, test_pbn 30,
  test_e3a 55, test_gen0 37, test_rgrow 72): E3-A13 built
  (target_spellings: the master's own token with exactly the mined inputs,
  listed first, played or not, contributing its whole q; and every logged
  run of master decisions with those inputs, any start, one life, all
  full; each spelling's mass moved from its own first slot; the products
  from any one slot asserted <= 1); the overlap assert before the clamp;
  the support check whenever a ceiling is set; latents kept only for the
  counted runs' end stacks (two passes, float32); the prep's steps in
  distill_lives(), pinned on fakes with acting != raw states (four
  mutations each failing); the k = 1 relabel on a valid vocabulary;
  TOKEN_BIAS_CAP_FRAMES 32; strict token numbers; load_student_vocab
  deleted; the slow lineage test forces a mined slot to be played (and the
  never-played ones are dropped at the second swap); probe_action_census
  and probe_qspread refuse a mined vocabulary by name.
  BUILT, PART 2a (feb257fd): arms M1 / M2 (T1 / T2 plus _mine; student_arm
  refuses a non-mined prep for them); the supervisor's lineage "m"
  (generation 1's master T0 read from Test B2's frozen tree as the bound
  pseudo-variant t0, first = 1, absolute paths only for it; --mine; the
  reward_max bound from the run's longest token, 16 base steps at most;
  the MINED line; quick() pins); SUITES 28 (tests/test_e3.py).
  AUDIT r2 (two fresh-eyes auditors on 199720df / feb257fd, 11-05): the
  core / distill side nothing above LOW (E3-A13's masses disjoint; a prep
  without --mine bit for bit; strict token numbers accept every valid
  form) -- L: a float32 prior could sum just above 1 and trip the stricter
  of two overlap asserts, stopping the prep (PREP FAILED) though nothing
  overlapped; test gaps. The supervisor's lineage m nothing above LOW
  (T0's real files match; stops 33,051; --mine for m only; every other
  lineage bit for bit) -- L: a lapsed test redirect could let a test write
  into the frozen tree; T0's recorded spec not pinned against the arm;
  the wait's advice wrong for a frozen master; NITs.
  FIXES r2: the prior in float64; one overlap tolerance (1e-9, relative to
  each slot's q) raising ValueError; the overlap-at-q=0 and one-token-
  spelling-through-distill_lives tests; inf refused cleanly; the test
  fixture asserts every m path lands in tmp; quick() compares T0's
  recorded spec key by key with this tree's T0 arm (T0's real snapshot
  passes); the frozen-master ALERT text ("run nothing in that tree");
  the MINED line on restarts; an ALERT when a mined prep appends nothing;
  run_token_steps capped; other lineages' quick() strings pinned;
  smoke_teste3.py counted as a trainer. Suites: test_e3 77, test_distill
  58, test_supervise_testb 265, test_tokens 72, test_testb 67, test_pbn
  30.
  AUDIT r3 (fresh eyes on 143229b3, 11-05): nothing above LOW -- the
  round-2 fixes correct and complete (within one slot the credited paths
  are leaves of one prefix tree, so a genuine non-overlap cannot raise;
  quick()'s spec comparison cannot fail spuriously: T0's arm block
  unchanged since a8c83a58; nothing writes into the frozen tree). L: the
  new overlap check let a NaN path weight through silently (the record
  would fall back to a uniform target); NITs: disjointness rested only on
  select_mined's prefix-freeness; a deterministic overlap would be retried
  three times as a full prep; "unreadable" for incomplete frozen files.
  FIXES r3: non-finite path weights raise; the overlap check written so a
  NaN fails it; check_disjoint_paths (exact: no credited path a slot-prefix
  of another) before any search; both errors in PREP_CHECKS (stop at once,
  Kai's call); "unreadable or incomplete". Suites test_e3, test_distill,
  test_supervise_testb, test_tokens pass; the new tests fail on the old
  code.
  AUDIT r4 (fresh eyes on fa8c30de, 11-05): NOTHING ABOVE LOW -- the
  in-tree AUDIT LOOP IS CLOSED. check_disjoint_paths cannot raise on a
  legitimate set (a one-token spelling and a one-decision run are one
  key; select_mined keeps inputs prefix-free); it runs before any search
  on exactly the credited paths; the non-finite check cannot fire on
  legitimate data; a prep without --mine bit for bit; the PREP_CHECKS
  strings exact and unique; the tests catch the obvious mutations.
  NITs accepted (no behaviour at stake): a NaN in the master's policy
  outside the credited slots is reported as an overlap (still a stop);
  numpy's repr in the message; PREP_CHECKS pinned against hand-copied
  messages.
  PRE-LAUNCH, 11-05 (code at 6ee611c1): ALL 28 SUITES PASS (11:16-11:35).
  RUN TOOLING (scratchpad, from Test B2W's): post_teste3.py (+ checker),
  launch_teste3.ps1, smoke_teste3.py, prep_dryrun_teste3.py (NEW: M1's
  real prep on T0's real snapshot and own-play log, --out in the
  scratchpad), prelaunch_teste3.py (requires the smoke's AND the dry
  run's OK logs), score_teste3.py (+ checker). Tooling audits: r1 HIGH
  (the smoke compared prep entries with tokens: could never pass) and
  MEDIUMs (the scorer read a dropped-and-re-mined token as surviving; the
  eval probes never run on a mined snapshot); r2 MEDIUM (the launcher's
  path classes lost their backslash to a shell heredoc: its own tree's
  processes never recognised); r3 MEDIUM (the dry run had no
  provenance); r4 MEDIUM (the dry run crashed in a fresh run tree);
  each fixed and re-audited.
  SMOKE r1 (GPU, 11:37-11:56, 18 min): SMOKE OK at 6ee611c1 -- prep1 on a
  tiny T0-arm master mined 4 (4 slots freed); M1 killed after its
  iteration-2 checkpoint (mined tail in it) and resumed; prep2 from M1
  dropped all four of M1's mined tokens (three played: split; one
  unplayed) and mined 8 new; M2 played them (one at 32% of decisions);
  both eval probes ran on M1's and M2's mined snapshots.
  DRY RUN r1 of M1's REAL prep (GPU, 11:57-12:00, 3.8 min, peak 2.69 GB
  host, 4.3 GB GPU): PREP DRY RUN OK at 6ee611c1; T0's run folder
  unchanged. On T0's real play it MINED 8 (cap 8: 14 slots freed;
  vocabulary 36 -> 30 slots), 6,297 records, 1,580 decisions relabeled:
  by score 3,16x3 (48 frames of one action; count 219, c 0.156), 7,32x2
  (64 frames; count 353, c 0.167), 7,32;5,16;0,16 (count 16, c 0.883),
  5,16;0,16;7,32 (14, 0.952), 5,16x3 (36, 0.351), 7,32;3,32 (41, 0.562),
  7,32;5,16;5,16 (22, 0.494), 5,16;5,16;7,32 (18, 0.501) -- the top two
  by count are single-action holds LONGER than the 32-frame duration menu
  (low consistency, chosen on count), the rest turn sequences with high
  consistency; every mined token's target nonzero at some record (means
  0.0004-0.060).
  TOOLING AUDIT LOOP CLOSED (r5 one MEDIUM: a misleading "absorbed" label
  in the smoke's COVERAGE line -- fitting inside a new token does not
  prove a play was absorbed; fixed; r6 nothing above LOW). SMOKE r2
  (12:01-12:21) and r3 on the final script (12:21-12:43): SMOKE OK at
  ae6cdea2.
  LAUNCHED 12:44 11-05: manifest a6aefbdb (from ae6cdea2; 62 AUDITED
  files, test_e3.py and probe_e3a_sequences.py added); run worktree
  Y:\SwitchPlayer-teste3-run at a6aefbdb; PRELAUNCH OK (smoke r3 at
  ae6cdea2, dry run r1 at 6ee611c1: AUDITED identical; T0's arm = its
  snapshot's spec, _doc included; m1's stop 33,051 = 66,667 - 33,616);
  launch_teste3.ps1 started the hand-off post_teste3.py (cmd pid 22172)
  and the supervisors m1 (cmd pid 26884; STOP 33051, preflighting) and m2
  (cmd pid 13592; WAITING for M1). KEEP Y:\SwitchPlayer-testb2-run (T0 is
  generation 0; T1-T2 the control).
  PAUSED 12:57 11-05 (before anything trained): m1's PREFLIGHT failed
  (12:56) -- tests/test_bnrecal.py::test_run_one_snapshots_carry_the_
  pipelines_acting_states, which passes alone in the run tree and passed
  in the pre-launch pass: that pass ran each suite in its OWN pytest
  process (run_suites_b2.sh), the preflight runs all 28 in ONE, so a
  suite leaks state into it (not test_e3.py alone: test_e3 + the test in
  one process pass). Stopped the m1 / m2 supervisors by exact pid (no
  child was running); the hand-off left waiting. Diagnosis: the
  preflight's own order in one process, run in the run tree.
  LEFT FOR PART 2: the supervisor's lineage (E3-A1: generation 1's master
  is T0 in Test B2's frozen tree), arms M1 / M2 (T1 / T2 plus mining: the
  prep's --mine), MAX_TOKEN_STEPS 16 for it (a 16-step mined token's
  reward), nested tokens in the supervisor's vocab.json readers, the run
  tooling (hand-off, launcher, smoke, pre-launch check, scorer).

### Next experiment (pre-registered) -- Test B2W: Test B2's lineage with BBF's AdamW in place of EfficientZero's SGD (Kai, 2026-11-04)

DECISION: Kai 2026-11-04, after Test C2's RESULT. Offered B2 with Adam
(BBF's settings) rather than C with Adam, Kai answered "Yeah I'd like to
try that as the next test". In the same message:
  - Test D (the smart swap trigger) is BACKLOGGED: "skip D for now (add
    it to the backlog) and stick with a constant number of steps between
    swaps". Generations keep a fixed env-step length, as in Test B2.
  - The test after this one is the queued E3 (tokens mined from the
    agent's own play, at swaps), which depends on this result: it runs on
    whichever recipe this test leaves as the base.
Background:
  - The optimizer was never tested on its own. SGD (lr 0.2, momentum 0.9,
    coupled weight decay 1e-4, 1,000-step warmup) entered on 09-24 as one
    of seven flags in the EfficientZero-fidelity bundle, applied at once
    and never bisected. That bundle fixed Pong's projector collapse, but
    the collapse's mechanism was later found to be the shared
    augmentation draw (probe_shift_collapse), not the optimizer. Before
    the bundle, the repo ran AdamW (lr 3e-4 on the edge arms, weight
    decay 1e-4).
  - Test B2's T2 is the best 100k result: 1629.5 deployed at [90k, end].
    C0 (T0's recipe, no swaps) reached 1262.4, WORSE than T2. Test C's
    resets (NO DIFFERENT from C0) and Test C2's ramps (WORSE than C) are
    out of the base. BBF itself runs both with AdamW; the optimizer is the
    one part of BBF's training kit not yet tried here.
  - Weight decay strength. The repo's old AdamW default (lr 1e-3 or
    3e-4, decay 1e-4) shrank weights by lr x wd = 1e-7 to 3e-8 per step:
    almost none. EfficientZero's coupled SGD shrinks by lr x wd = 2e-5
    per step (more with momentum). BBF's AdamW shrinks by 1e-4 x 0.1 =
    1e-5 per step, the same order as our SGD. So with BBF's settings the
    test is mostly about Adam's per-parameter step sizes, not about
    dropping weight decay.
QUESTION: on Test B2's lineage and budget (two swaps, three generations
of ~33.3k env steps, 100k in all), does BBF's AdamW reach a higher
deployed score at 100k than EfficientZero's SGD?
BBF'S OPTIMIZER (verified 11-04 against the released
bigger_better_faster/bbf/configs/BBF.gin and bbf/agents/spr_agent.py
create_scaling_optimizer):
  - optax.adamw: learning_rate 0.0001 (BBFAgent.learning_rate, and
    encoder_learning_rate the same), eps 0.00015, weight_decay 0.1
    (decoupled; both values "from DER (van Hasselt et al, 2019)"), b1
    0.9, b2 0.999 (the function's defaults).
  - decay_bias False: rank-1 parameters (biases; in our nets also the BN
    affines) are not decayed.
  - warmup 0: no learning-rate warmup.
  - No gradient clipping in the optimizer.
THE LINEAGE (seed 0; three generations W0-W2; every env step counted):
  W0-W2 are Test B2's T0-T2 with one change, the optimizer:
    - OPTIMIZER "adamw", LEARNING_RATE 1e-4, WEIGHT_DECAY 0.1, eps 1.5e-4,
      betas (0.9, 0.999), rank-1 parameters excluded from the decay,
      LR_WARMUP_STEPS 0. torch.optim.AdamW applies the decay as
      p <- p - lr x wd x p, as optax.adamw does.
    - Everything else is T0-T2's: TK's 36 tokens at the start,
      policy-only acting, PREFIX_BN_PER_STEP, the own-play log, the sign
      clip c0 = 1 in W0, DISCOUNT 0.98805 per base step at every
      generation, the swap (the master's own play over its whole
      generation, the halved-clip rule, the 1% vocabulary rule, the
      master's 50-sim birth search targets, 20,000 offline steps, then
      online play), the value check, the stop (round(100,000 x (k+1) /
      3) minus the earlier generations' final env steps), the schedules
      (nominal 50 iterations, cap 100) and the master gate (the master's
      last two deployed rows averaging < 502.08 stop the chain; Kai's
      call).
    - The offline distillation steps run under the same AdamW (one
      optimizer per network per generation, fresh at each generation,
      as in B2).
    - Kept from B2 although tied to SGD: EfficientZero's gradient
      placement (EFFZERO_LOSS_SCALING) and the gradient-norm clip at 5.
      Changing them too would make this more than an optimizer test.
  The CONTROL is Test B2's T0-T2 (10-29/30, frozen tree
  Y:\SwitchPlayer-testb2-run): same seed, same budget, already run.
READS. EXACT eval as before (whole games, 30 noop starts, prior argmax,
PROBE_ENV=MsPacmanNoFrameskip-v4); DEPLOYED primary, trainer-BN
secondary. Rows every 5 iterations of each generation, plus each
generation's final snapshot and the students' row 0 and offline curves.
Cumulative env steps are the lineage's own online env steps.
  (1) PRIMARY at 100k: the mean deployed score of W2's rows in [90k, W2's
      end] (its final included; n rows) against T2's (1629.5, n 4). The
      band is 2 x 216.9 x sqrt(1/n + 1/4): BETTER / NO DIFFERENT /
      WORSE. Trainer-BN likewise (T2 1605.2).
  (2) descriptive: the deployed mean per 20k window of cumulative env
      steps, beside Test B2 (622 / 801 / 1071 / 1484 / 1588), C and C0,
      each window against B2's with the same band rule.
  (3) per generation and per swap, as Test B2's reads: the OLS
      improvement rate within each generation, the transfer (student
      row 0 against its master's final), the jump, the offline curves;
      W0 against T0 over the first 33k env steps (the optimizer is the
      only difference there, before any swap).
  (4) mechanism: the gradient norm (step_grad_norm_mean), the value loss
      and the value targets' mean by row, the prior (policy_maxp, the
      search targets' entropy), and the plasticity read (dormant
      fractions by net, weak first-layer filters, the projector's erank)
      by row. Each is set against T0-T2 where Test B2's rows hold the
      field and reported alone otherwise.
  (5) probe_prefix_bn FIXED at each student's row 0 and each
      generation's final row.
  (6) stability: the largest adjacent deployed drop; non-finite steps.
  STOP EARLY on loss divergence or repeated non-finite steps, or at the
  master gate.
DECISION AFTER:
  - (1) BETTER: AdamW with BBF's settings joins the base. E3 runs on the
    W recipe.
  - (1) NO DIFFERENT: SGD stays, as the recipe behind every positive
    result so far. E3 runs on Test B2's recipe. (Kai's call if reads (2)
    to (4) disagree strongly with (1).)
  - (1) WORSE: SGD stays. E3 runs on Test B2's recipe.
CAVEAT:
  - One seed, against a control run on 10-29/30 from an earlier tree:
    the same seed but different async trajectories. The band is the
    within-run row noise, not the seed noise.
  - The optimizer is tested as BBF configures it (lr, eps, decay, the
    bias mask, no warmup) as one unit. A verdict is on BBF's settings, not
    on Adam in general. No learning rate is tuned.
  - The kept gradient placement and clip were set for SGD. Under Adam a
    uniform loss scale cancels except against eps, so the 1/K scaling
    makes eps (1.5e-4) relatively larger than in BBF.
BUILD (before launch; audited like every build):
  - muzero.make_optimizer gains two knobs whose defaults leave every
    existing arm bit for bit unchanged: ADAM_EPS (None = torch's 1e-8)
    and DECAY_RANK1 (True = decay every parameter, as now; False = BBF's
    mask, rank-1 parameters in a no-decay group).
  - The W arms (W0 from T0, W1 / W2 from T1 / T2) with the optimizer
    settings above; supervise_testb.py's lineage table gains the "w"
    variant; the prep, student and offline paths take the arm's
    optimizer unchanged.
  - Tests: the new knobs (defaults unchanged, the mask's groups, eps
    and decay reaching torch's AdamW), the W arms' specs against T's
    (only the optimizer keys differ), and the preflight suites' arm
    lists.
  - In the scratchpad: the hand-off, launcher, checker, smoke and scoring
    script, as for Test B2 and Test C2.
  - A manifest and a fresh run tree. No change to any existing arm.
COST: the build and its audit loop, then about 1.3 days of training (Test
B2 took 27.2 h).
AMENDMENTS (the build, before launch):
  (A1) READ (4)'s PLASTICITY READ is limited to what both lineages log:
      the projector's erank and dead fraction (proj_erank, proj_dead_frac)
      by row, against T0-T2's. The dormant fractions by net and the weak
      first-layer filters come only from DORMANT_DIAGNOSTIC, which T0-T2
      ran without; W0-W2 keep it off so that they stay T0-T2 but for the
      optimizer. (A post-hoc probe on both lineages' snapshots could add
      them later, descriptive only and Kai's call.) Audit r1a M1.
  (A2) CAVEAT, added: the control ran from Test B2's tree (a8c83a58);
      W0-W2 run from a tree that also holds Test C's and C2's code (the
      resets, the ramps, the warmup origin), all off in these arms and
      pinned off by tests/test_testc.py and tests/test_testc2.py ("every
      other arm, bit for bit"). Audit r1a L1.
  (A3) The scoring script (score_testb2w.py) and its check are built and
      audited while the run trains, before any scoring. They read the
      frozen trees only. Audit r1b L4.
BUILT 2026-11-04 (pre-audit, 4a21836f): muzero.make_optimizer's ADAM_EPS
and DECAY_RANK1 (defaults unchanged; refused under SGD); the W arms
(ablation.ADAM_ARM, BBF_ADAMW); supervise_testb.py's lineage "w" with the
optimizer pins (OPTIM_BBF for "w", OPTIM_SGD for every other lineage) and
the smoke counted as a trainer; tests/test_testb2w.py (new) and the
supervisor's, PBN's and the harness's tests; in the scratchpad the
hand-off post_testb2w.py and its checker (generated from Test B2's by
make_post_testb2w.py / make_check_testb2w.py), launch_testb2w.ps1,
smoke_testb2w.py and prelaunch_testb2w.py. The 27 suites ran once
(26 passed; the supervisor's usage-message test needed the new variants)
and the checker printed CHECK OK.
AUDIT r1 (two fresh-eyes auditors, 11-04): nothing HIGH.
  - r1a (the core): M1 read (4)'s plasticity fields are never logged (A1);
    LOWs: the control's older tree (A2), the quick probe checked the spec
    but not the optimizer it builds, the checkpoint round-trip test
    bypassed the checkpoint's path and compared groups after the load,
    no CPU test of the student path (the GPU smoke runs it); NITs.
  - r1b (the tooling): M1 the smoke's 900-env-step stop made iteration 2
    the LAST iteration, which the trainer never checkpoints, so the kill
    could never land; LOWs: the PAUSE listing matched any distill.py, the
    smoke did not show the student's offline steps ran under the same
    optimizer, its non-finite check passed on a missing field, the
    scorer was not yet built (A3); a NIT.
FIXES r1: all of them. The quick probe now builds one net's optimizer
under the arm's config and checks its type and groups (and pins flags by
identity); the round-trip test goes through _cpu_state and a protocol-5
pickle and compares the fresh groups before the load; the smoke's stop is
3,000 env steps, it checks every parameter's Adam step count (W1: its 40
offline steps plus its online ones), that a resume never reruns the
offline phase and that the non-finite field is present and 0; the PAUSE
listing matches only distill.py of a W arm; the launcher records A3; the
usage and smoke-as-trainer tests know the w variants.
AUDIT r2 (two fresh-eyes auditors, 11-04): r2a (the core) nothing above
LOW -- the round-trip test copied the checkpoint path rather than calling
it, no CPU check covered Run.__init__'s own optimizer lines, read (4)'s
projector dead fraction is partly confounded (under W the BN gammas are
not decayed; T's coupled SGD decays them), NITs. r2b (the tooling) one
HIGH in the smoke: its "offline phase not rerun" check matched the W1 /
W2 arms' _doc banner ("... its offline steps under the same optimizer"),
which ablation prints at startup, so it could never pass on w1b (and its
positive check on w2 passed vacuously). The smoke running at the time
(b2w_smoke_r1.log, at 0bfa45a2) failed there as predicted, after W0's and
W1's iteration-2 checkpoints had passed every optimizer check -- W1's
Adam step count 80 = 40 offline + 40 online: the offline phase ran under
the run's own optimizers -- and w1b's log holds no "] offline step"
progress line (w1a's holds two): the resume did not rerun it. LOWs:
provenance checked only at the smoke's start; the PAUSE listing's pytest
alternative also matches a hand-run of the suites; NITs.
FIXES r2: the smoke matches the trainer's own progress line ("] offline
step N / M"; absent from w1b, "40 / 40" in w2) and re-checks the
provenance (the same HEAD, AUDITED clean) before SMOKE OK;
prelaunch_testb2w.py requires the smoke log's "start:" line to name
smoke_testb2w.py; the launcher's PAUSE notes that a child counts only
under a listed supervisor; the round-trip test calls
save_run_checkpoint / load_run_checkpoint; the five-optimizer test reads
Run.__init__'s source for the five make_optimizer lines; read (4)'s
confound goes into the scorer (A3).
AUDIT r3 (one fresh-eyes auditor, 11-04): nothing HIGH; the r2 fixes
correct. M: the smoke's end check compared HEAD for equality, stricter
than the pre-launch check (the AUDITED files identical between the smoke's
commit and the run tree's HEAD), so a CURRICULUM.md commit during a smoke
would fail it after every phase passed. L: the PAUSE note's
"ParentProcessId" rule was ambiguous for the child's second pid. NITs:
the pre-launch check accepted any later "rc=0"; three test NITs (a
no-op training_steps line, env_steps not asserted, a source check that
an imported class could slip past) and a message wording. (The auditor
once ran the venv's `python --version` against the brief; nothing else.)
THE SMOKE passed at 666d00e9 (b2w_smoke_r2.log, 11:06-11:30): SMOKE OK --
W0 / W1 / W2 to their stops (iterations 6 / 6 / 7, 3,055 / 3,005 / 3,250
env steps), BBF's AdamW in W0's and W1's iteration-2 checkpoints (two
groups each, W1's Adam step count 80 = 40 offline + 40 online), both
resumes after iteration 2, no offline phase rerun, W2's whole offline
phase, every row's non-finite count present and 0.
FIXES r3 (the scratchpad tooling only; no AUDITED file changes, so the
smoke stands): the smoke's end check compares the AUDITED files with the
start's commit (git diff --quiet) and fails with its own message; the
pre-launch check takes the FIRST rc line after SMOKE OK; the PAUSE note
says the child's first pid has a supervisor as its parent. The test NITs
are left: fixing them would change an AUDITED file after the smoke.
AUDIT r4 (one fresh-eyes auditor, 11-04): nothing above LOW; the r3
fixes correct (the smoke's end check passes a CURRICULUM.md commit and
fails on any AUDITED change; the pre-launch check takes b2w_smoke_r2.log
and refuses r1's, a non-zero rc, a foreign smoke's log). L: the PAUSE
rule told the operator to leave a W trainer whose supervisor was already
dead; NITs. FIXES r4 (scratchpad): every ablation.py / distill.py line of
a tokens-adam-w arm is the run's, killed whatever its parent; the
pre-launch check's rc line must come before any later "start:"; the
smoke's end-check message carries git's stderr.
AUDIT r5 (one fresh-eyes auditor, 11-04): nothing above LOW -> THE AUDIT
LOOP IS CLOSED. Accepted, not fixed (wording; the operator in practice
notes the supervisor pids before killing): L, a preflight pytest started
between the listing and the supervisor's kill has a dead parent in step
3's relisting, so the literal rule says to leave it (it is CPU only and
ends); NITs, the preflight's own quick-probe test spawns a "-c" child that
names the arm (CPU, seconds), two message / docstring wordings.
THE SUITES: all 27 passed at 47cf92b0 (11:32-11:44 11-04, CPU, under the
lock; its AUDITED files identical to the smoke's 666d00e9).
LAUNCHED 11:46 EST 11-04: the manifest at ed359242 (written at
e75e44d3); the run worktree Y:\SwitchPlayer-testb2w-run detached at
ed359242; prelaunch_testb2w.py PRELAUNCH OK there (11:45, with
b2w_smoke_r2.log); launch_testb2w.ps1 started the hand-off post_testb2w.py
(cmd 20112) and the supervisors w0 / w1 / w2 (cmd 8188 / 24844 / 10176).
w0 preflights (~25 min), then trains W0; w1 and w2 wait for their
masters. Keep Y:\SwitchPlayer-testb2-run, -testc-run and -testc2-run until
B2W is scored.
THE SCORER (A3), built during the run: scratchpad score_testb2w.py
(stdlib, python -I; reads the frozen trees) and its fake-tree check
check_score_testb2w.py (SCORE CHECK OK). Every derived read is computed for
both lineages by the same code, and T's are asserted on every run against
Test B2's RESULT (T2's window 1629.5 / 1605.25, n 4; the rates +11.6 /
+24.5 / +5.2; transfers -169 / +245; jumps -211 / +133; the windows; the
drops; T0's mean 709.2, which B2 printed for the exact 709.13).
SCORER AUDITS s1-s5 (one fresh-eyes auditor each, 11-04): s1 HIGH x2 --
the jump used row 0 (B2's jump is rows 5 and 10 against the master's last
two rows) and the rate dropped row 0 -- and MEDIUMs (read (4) not by row;
a final verdict possible before the final row's evals; a checker unable
to catch definition errors); s2 MEDIUM (read (3)'s W0 vs T0 with no
PROVISIONAL tag); s3 MEDIUM (a row lost into a malformed line could leave
read (1) final on n-1 rows); s4 MEDIUM (read (1)'s tag ignored W0's and
W1's integrity, though their final rows set W2's cumulative base); all
fixed, each with its checker case. s5: nothing above LOW -> CLOSED.
Accepted LOWs, checked BY HAND at scoring instead: (a) each generation
reached its stop (its final cumulative env steps >= round(100,000 x
(k+1) / 3)); (b) each master's final row is its true final iteration (the
supervisor's MASTER line); (c) a probe fragment from a killed probe keeps
read (1) PROVISIONAL (safe side) -- check it by hand if it happens.
W0 (11:59 launched after the preflight; ~10 min / iteration). Rows 5 /
10 / 15 / 20 (exact eval, deployed / trainer-BN): 334 / 156, 753 / 392,
310 / 139, 401 / 358 at 3,085 / 6,430 / 9,643 / 12,793 env steps; T0 at
the same rows: 400 / 601 / 714 / 690 deployed. Two in-run reads differ
from T0's recipe from row 10 on: frames per decision 16.9 -> 7.9 -> 7.2
-> 6.4 (4-frame tokens 15% -> 67% -> 65% -> 70% of play; T0's recipe ran
~19-27 in C0) and policy_maxp 0.46-0.50 (T0 0.59-0.80). One reading,
not yet tested: at BBF's lr 1e-4 the prior sharpens more slowly than
under SGD at 0.2. Single rows, far inside the one-row band; the primary
read is W2's [90k, end].
W0 rows 25-45 (deployed / trainer-BN): 1,327 / 467, 612 / 391, 980 / 535,
431 / 385 (row 45's evals pending at the pause); the final row iteration
52 at 33,592 env steps. W0 COMPLETE 21:13 11-04 (rc 0, 33,301 s). Frames
per decision 5.4-7.3 from row 25 on (4-frame tokens 59-86% of play);
policy_maxp 0.38-0.41, the prior's entropy rising to 2.1 (T0's
sharpened). PAUSED 21:16 EST 11-04 at Kai's request ("a couple hours"),
between W0 and W1: the supervisors w1 (25812 / 25968) and w2 (29136 /
29572) stopped by exact PID while w1 waited for W0's final evals; no
trainer, prep or preflight child was running. The hand-off runs on
(W0's final evals). Resume by rerunning launch_testb2w.ps1.
RESUMED 21:47 EST 11-04 when Kai said to continue (the 2-h timer cancelled): launch_testb2w.ps1 skipped the running hand-off and restarted the supervisors w0 (it logs RUN COMPLETE and exits), w1 and w2.
W0's last rows (deployed / trainer-BN): 45 694 / 821, 50 739 / 1,067,
final 52 630 / 681; W0's every row 655.5 (n 11) against T0's 709.1.
THE FIRST SWAP (22:00-22:12): w1's preflight ok; MASTER W0 iteration 52
(last two rows 684.5, the gate passed); the prep kept 17 tokens (T1: 22),
dropping 19 -- nearly every 16- and 32-frame token -- the clip 9.93 (T1
7.26), the discount kept; 25,374 records (T1's set 7,174: W0 made ~3x the
decisions per game), with 38% of the master's search mass on dropped
tokens discarded. W1 launched 22:12; its 20,000 offline steps took ~2 h.
W1's offline curve (deployed / trainer-BN): 661 / 333, 873 / 333, 555 /
333, 536 / 403 at 4k-16k steps; row 0 520 / 333 (T1: 710-745 rising, row
0 892; its trainer-BN 633 -> 898). Rows 5 / 10 / 15 / 20: 846 / 333, 396
/ 240, 508 / 240, 277 / 240 (T1 730 / 656 / 850 / 901). Transfer -110
(T -169); jump (rows 5, 10 against W0's last two) -64 (T -211).
W1's TRAINER-BN NETS ARE DEGENERATE: the raw exact eval reads exactly
333.3 (sd 99.8) at row 0, row 5 and three offline points, then exactly
240 at rows 10-20 -- a fixed, poor policy of ~4-frame tokens whose score
does not move as the weights do -- while the deployed policy (the
recalibrated acting statistics) varies. W0's trainer-BN already ran far
below its deployed (156 / 334, 139 / 310 at rows 5 / 15); T0's and T1's
tracked theirs. This matters for training, not only for the secondary
read: the birth search that labels new data runs on the trainer's nets in
eval mode (ablation.py, the search keeps `nets`; only the logged prior
uses the acting statistics), so W1's policy targets come from the
degenerate nets: 4-frame tokens took 84-86% of the search mass and
90-95% of play by rows 20-25 (frames per decision 4.5). Not yet known:
whether it is AdamW's own effect in this stack or an artifact of AdamW
with our BN setup (BBF has no such eval-mode birth search) -- a CPU
diagnostic on W1's snapshots after B2W. THE MASTER GATE at W1's end (its
last two deployed rows >= 502.08) may stop the chain (Kai's call).

### RESULT -- Test C2: BBF's n-step and discount ramps on top of the resets, against C (seed 0), 2026-11-04

(Pre-registration: "Next experiment (pre-registered) -- Test C2", below
E1's; the interim notes, pauses and resumes are at its end.)
RUN (from the run worktree Y:\SwitchPlayer-testc2-run, detached at
541a19ee, manifest from the audited cbfd9af5): C2 (a0) launched 12:35
11-02. It was paused at 12:50 and relaunched fresh at 14:50 (no checkpoint
yet), then paused three more times at Kai's request (20:51-22:52 11-02,
13:47-15:14 11-03, 18:55 11-03 to 00:56 11-04), each resumed from the last
checkpoint (iterations 36 / 120 / 138). It was complete at 05:51 11-04:
iteration 165, 100,489 env steps, 330,000 train steps. The final evals and
read (6) landed at 06:09.
  - 6 resets, after iterations 20 / 40 / 60 / 80 / 100 / 120 (C had 5).
    The rule resets every 20 iterations while env steps <= 80,000. C2
    gained fewer env steps per iteration than C (609 against 753), so it
    was still under 80k at iteration 120 (71,299). 24 held ships.
  - Pace: 3.28 gradient steps per env step, against C's 2.66 (C reached
    100,849 env steps at iteration 134, 268,000 train steps).
  - supervisor.err is empty. Every row had 0 non-finite steps. No
    exact-eval game was capped (33 rows, both evals).
  - Host commit ran at 0.6-4.4 GB free while training (Firefox was open
    at times). Only the advisory "memory creep?" ALERTs fired, and no
    allocation failed. The interim probes for rows 140-160 waited for
    memory and ran after the trainer exited.
EXACT eval (whole games, 30 noop starts, prior argmax). Every score is
unclipped. DEPLOYED is primary; trainer-BN is secondary. env = thousands
of env steps. C's rows are in Test C's RESULT.
       row     5    10    15    20    25    30    35    40    45    50    55
  C2   env   2.6   5.6   8.7  11.7  14.4  17.0  19.7  23.6  26.4  29.1  31.8
       dep   502   400   682   653   858   500   808   579   515   792   659
       tBN   269   586   536   701   607   420   540   719   687   735   577
       row    60    65    70    75    80    85    90    95   100   105   110
  C2   env  34.8  37.5  40.3  43.6  47.0  49.6  52.4  55.6  58.5  62.5  65.6
       dep   635   655   758   773   746   798  1005   805   903   836   948
       tBN   570   581   966   580   838   952   774   716  1117  1037   886
       row   115   120   125   130   135   140   145   150   155   160   165
  C2   env  68.5  71.3  74.8  78.5  81.5  85.3  88.1  90.9  94.3  97.1 100.5
       dep   772  1085   863   998   930   732   893   879   892   994   885
       tBN   886  1066   846   905  1008   777   918   803   909   810   851
  (Row 165 is the final row.)
READ (1) PRIMARY at 100k: C2's rows in [90k, end] are 150 / 155 / 160 /
  165 (n = 4): 912.6 deployed, against C's 1411.0 (n 3). C2 - C = -498.4,
  band +-331.3 -> WORSE. Trainer-BN: 843.3 against 1358.1, -514.8 ->
  WORSE.
READ (2) per 20k window of env steps (n), deployed:
      window              0-20k    20-40k    40-60k    60-80k    80-end
      C2                  629 (7)   639 (6)   827 (7)   917 (6)   886 (7)
      C                   754 (6)  1028 (5)  1067 (5)  1101 (6)  1423 (5)
      C2 - C             -125      -389      -240      -184      -537
      band (+-)            241       263       254       250       254
      C0                  564 (6)  1015 (5)  1160 (5)  1290 (6)  1300 (6)
      C2 - C0             +65      -376      -333      -373      -414
      band (+-)            241       263       254       250       241
      Test B2             622       801      1071      1484      1588
  C2 is below C in every window, beyond the band at 20-40k and 80-end. It
  is below C0 (no resets, no ramps) beyond the band in every window from
  20k on. C2's [90k, end] against the other references:
  - C0's 1262.4 (n 4): -349.8, band +-306.7 -> WORSE.
  - Test B2's T2 1629.5 (n 4): -716.9, band +-306.7 -> WORSE.
  C2 levelled off from 60k: 917 at 60-80k, 886 at 80-end. Its best row
  was row 120 (1085, the last before reset 6), and no later row reached
  it.
READ (3) per reset. The dip is row 25 of the cycle against row 20. The
  rate is OLS deployed per 1k env steps over the cycle's rows 25-40.
      reset  rows      C2 dip   C dip   C2 rate  C rate  recovered (C2)
      1      20->25     +205     +165    -17.7   -12.9   row 25
      2      40->45      -64     +208     +7.6   -21.1   row 50
      3      60->65      +20      +82     +8.7    +4.0   row 65
      4      80->85      +52     +154     +3.1   -16.2   row 85
      5     100->105     -67     +228    +19.1   +30.4   row 110
      6     120->125    -222       -      -13.4      -    never
  (Reset 6 has no counterpart in C. Over the same rows C, unreset, moved
  -194 and rose +5.9 per 1k env steps.) Every one of C's resets "dipped"
  upward. Three of C2's dipped down (resets 2, 5 and 6). Reset 6, the one
  C did not have, cost the most and never recovered.
  The start (rows 5-20, before any reset, so the ramp is the only
  difference from C): C2 mean 559.3, rate +24.3 per 1k env, against C's
  642.1 and +47.0 (n 4 each; -82.8, band +-306.7, inside).
READ (4) mechanism:
  - The ramp: 33 rows and 165 trace entries checked against the
    schedule, n exactly and gamma within 1e-12: ALL ON IT. The ramp
    windows' steps matched the schedule at all 33 rows.
  - The ramp windows (value loss = the trainer's value cross-entropy per
    unroll position, in nats; z = the batch's mean position-0 target): in
    every cycle the value loss in the ramp was above the loss out of it,
    by +0.008 to +0.023 nats, and the ramp's targets were smaller (mean z
    0.85-3.16 lower: n 10 at gamma 0.97, against n 5 at 0.988). The
    value loss fell steadily over the run, in and out (0.140 in cycle 1's
    ramp, 0.039 at row 165).
  - A1: the step loss on the rows ending a ramp window, intact in both
    runs (rows 5 / 45 / 65 / 85 / 105): C2 +0.430 against C +0.341 (n 5),
    C2 - C +0.089. C2 was higher at all five.
  - The value |V - z| and the targets' mean |z| by row, over the 26
    common rows: C2 1.275 / 5.99, C 1.555 / 7.18. The relative error is
    the same (0.213 against 0.216). C2's targets are smaller because it
    scores less.
  - Plasticity: dormant fraction (<= 0.025), by net, representation /
    dynamics / prediction:
      row    C2                    C
      20     14.2 / 17.2 / 31.3%   27 / 21 / 26%  (before reset 1)
      40     17.5 /  8.2 /  8.3%   0.6 / 0.4 / 0%
      60      0.2 /  0   /  0%     1.3 / 0.4 / 0.5%
      80      0.2 /  0   /  1.0%   0 / 0.4 / 0%
      100     0.4 /  0   /  0%     0.2 / 1.6 / 1.0%
      120     0.4 /  0.8 /  1.0%   0.4 / 2.0 / 0.5%
      final   1.0 / 12.5 /  0.5%   4.4 / 9.4 / 2.1%
    C2's first reset left part of the trunk dormant for its whole cycle
    (rows 30-40: representation 17.5-17.9%, dynamics 8.2-8.6%, prediction
    8.3-9.4%). From reset 2 on, every net stayed at or under 1% within
    cycles. After reset 6 the dynamics net re-accumulated, 0 to 12.5%
    over 45 iterations, like C's after its last reset. The projector
    re-collapsed to 31-67% within every cycle, as in C. value_prefix was
    0 throughout. Weak first-layer filters: 22-28% at rows 15-40, 0-3%
    from row 45 to 145, rising to 16% at the final row. erank 2.5-58.
READ (5) the prior: policy_maxp C2 0.44-0.76, below C's at 25 of 26
  common rows; C 0.53-0.77. The search targets' entropy: C2 0.71-1.82;
  C 0.86-1.91. Neither prior collapsed.
READ (6) mechanism: probe_prefix_bn FIXED at C2's final row 165 (p1_ratio
  0.86).
READ (7) stability: the largest adjacent deployed drop is -359 for C2 (row
  25 -> 30, in the cycle after reset 1) and -228 for C (row 130 -> 134).
  The reset dips are in read (3).
PRE-REGISTERED DECISION: (1) WORSE -> the ramps hurt. Reads (3) and (4)
  say why: the dips, the value loss in the ramp windows. Test D is next
  unless Kai decides otherwise.
  What reads (3) and (4) show (one seed, so "consistent with"):
  - The ramp windows trained worse. The value loss was higher inside the
    ramp than outside it in every cycle, and the step loss at the rows
    ending a ramp window was above C's at all five comparable rows. The
    ramp switches to a shorter-horizon target (n 10 at gamma 0.97) for
    10,000 steps after each reset, then moves it back, so the value head
    is retrained on a moving target exactly while the reset heads
    relearn.
  - The resets cost more with the ramps. None of C's resets dipped; three
    of C2's did, and the extra reset 6 dipped -222 and never recovered.
  - The cost is not only at the resets. In the start (rows 5-20, the ramp
    without a reset), C2 rose at half C's rate (inside the band), and C2
    sat below C0, which has neither, from 20k on.
Beyond the letter:
  - C2 used more gradient steps per env step than C (3.28 against 2.66)
    and still scored less, so the gap is not a compute shortfall.
  - C's final-cycle rise (+30.4 per 1k env) did not repeat. C2's last
    cycle (rows 125-165, 45 iterations, no reset after) was flat to
    falling (-13.4 over rows 125-140). Test C's reading that "resets help
    once a cycle runs long enough" gets no support here, but the two
    cycles are not alike: C2's followed an extra reset at 71k.
  - The BBF bundle has now not paid in our stack, in either step: resets
    NO DIFFERENT from no resets (Test C), and the ramps on top WORSE than
    resets alone (here). It stays out of the base.
KAI 2026-11-04, after this RESULT: Test D is BACKLOGGED (generations keep
a constant env-step length). Next is Test B2W (Test B2 with BBF's AdamW),
then the queued E3 (mined tokens), which depends on it. See Test B2W's
pre-registration, above.
  - Decision speed: frames per decision in the in-run evals 16.7-23.7;
    36 tokens throughout.

### RESULT -- E1: a model trained on T2's own play does NOT recover E0's outcome classes (seed 0), 2026-11-02

(Pre-registration: "Next experiment (pre-registered) -- E1", directly
below, with its amendments A1-A8 and the audit loop, closed at r4.)
RUN:
  - 12:02-12:04 EST 11-02 (157 s), CPU (2 threads), under the lock.
  - The probe is unchanged since the audited 967da7c7.
  - Output: benchmarks/probes/probe_e1_equivalence-mspacman-two-t2-s0-
    2026-10-29.jsonl (summary), -meta.jsonl and -decisions.jsonl.
VALIDITY: main_line_ok. All 30 games MATCH the exact eval (mean 1699.0),
  with every token and argmax equal to the census's. 4,636 decisions
  scored; 3,798 live, of which 1,182 are junctions.
INPUTS:
  - tau = 0.0260: the median one-step consistency error over T2's 6,548
    logged decisions. By token length (4 / 8 / 16 / 32 frames): 0.027 /
    0.032 / 0.024 / 0.029.
  - IDM: 1,970 training samples and 72 held out. Labels by action 0-8:
    13 / 59 / 85 / 97 / 161 / 190 / 568 / 52 / 745 (UPLEFT and DOWNLEFT
    dominate).
  - Accuracy: train 0.61, held out 0.875. 49 of the 72 held-out samples
    are UPLEFT, so the held-out accuracy is mostly the majority class.
VERDICT: KAI_CALL_LOW. Neither candidate reaches precision 0.7 with
  recall 0.2 on the live decisions. A7's rule and the all-decision rule
  also read LOW.
  READS on live decisions, by precision / recall (junction precision):
      reading                           P       R     junction P
      (A) forward, tau                0.536   0.816   0.306
      (A) all one class (trivial)     0.530   1       0.301
      (A) tau x 0.5                   0.539   0.556     -
      (A) tau x 2                     0.531   0.955     -
      (B) IDM, 0.1 x max              0.434   0.161   0.221
      (B) all nine (trivial)          0.611   1       0.239
      (B) 0.05 / 0.2 x max      0.550 / 0.486   0.447 / 0.052   -
      state-blind reference (A)       0.906   0.333     -
      keyed (heading) reference (A)   0.902   0.547   0.722
      keyed (heading) reference (B)   0.923   0.544   0.605
  - (A) predicts 1.90 classes per decision against the truth's 2.33.
    Its precision equals the all-one rule's at every tau. Over the
    9 x 4-frame tokens, the forward model's predicted next latents carry
    essentially no information about which tokens are equivalent.
  - (B) is below its trivial floor on precision. It puts the chosen
    action in its own predicted class 91% of the time, but the other
    members it adds are mostly wrong.
  - The secondary groups are no better. The 16-frame group: P 0.42
    against the trivial 0.38. The 32-frame group: 0.27 against 0.23.
  - By rarity (A1): the one non-rare pair (UPLEFT-4 with DOWNLEFT-4) has
    P 0.868, against the trivial 0.861. The 35 pairs with a rare token
    have P 0.530, against the trivial 0.520. Neither split beats
    chance by much.
BEYOND THE LETTER:
  - The information is in the game, not in the models. A fixed rule
    keyed on Pac-Man's heading, fitted on the census truth, reaches P
    0.90 / R 0.55. Neither model, trained only on T2's own play, gets
    near it.
  - The likely cause is E0's known trap at full strength. In T2's last
    20 iterations, 2 of its 9 four-frame actions make 90% of its 4-frame
    plays (UPLEFT-4 232 and DOWNLEFT-4 115, of 384).
    Its dynamics embeddings for the other seven are barely trained, and
    the IDM's labels follow the policy. Own play at this volume is not
    diverse enough to learn what the untried inputs do.
  - The forward test asks a model trained to predict VALUE-relevant
    latents (MuZero's consistency loss on its own trajectories) to
    separate outcomes it never needed to separate. A near-uniform
    distance structure is what that predicts.
DECISION (pre-registered, LOW): Kai's call among:
  - more diverse own play (exploration aimed at untried tokens, so the
    models see their effects), then E1 again;
  - identical latents as the criterion, Kai's later plan;
  - stopping the line.

### Next experiment (pre-registered) -- E1: can a model trained on the agent's own play recover E0's outcome classes? (Kai, 2026-11-02)

DECISION: Kai 2026-11-02. E1 runs alongside Test C2 ("You can do them at
the same time"): CPU only, under the lock, while C2 trains. E0 read E1:
T2's 18 tokens give 3.28 frame classes per decision (ratio 0.18).
QUESTION: at the decisions of T2's own exact-eval play, does a model
trained only on the agent's own play group T2's tokens the way the
emulator census does? Kai's rule holds throughout: the emulator's
branches only score the models and never train them.
SCOPE: tokens of the SAME length. A model of one decision compares
outcomes at the token's end. Tokens of different lengths end at
different times, and the census merges them only through the NOOP tail of
its 92-frame window (a common horizon). That needs a multi-step rollout
through the NOOP token, which T2 barely plays (below). So cross-length
pairs are left for later.
  - PRIMARY GROUP: T2's nine 4-frame tokens, one per action. The census
    gives 2.09 classes among them per decision (E0 read (2)).
  - SECONDARY: the 32-frame group (5 tokens) and the 16-frame group (3
    tokens).
  - TRUTH: E0's partition at each of the 4,636 decisions, restricted to
    the group (the decisions jsonl; GRID indices of T2's slots [0, 4, 8,
    12, 14, 15, 16, 19, 20, 22, 23, 24, 28, 31, 32, 33, 34, 35]).
THE STATES: the census main line replayed exactly: the same snapshot,
contexts, acting BN states and thread count, and every decision's token
checked against the jsonl's. The replay is deterministic (main_line_ok).
The 30 games fall into about 8 near-duplicate families, so the decisions
are correlated; the reads are also given per family.
CANDIDATES:
  (A) FORWARD, no new training: T2 iteration 42's own dynamics model and
      consistency head, trained by the trainer on T2's own play.
      - For each token i of the group at state s: z_i = the online
        projection with prediction of g(h(s), i). h uses the trainer's
        BN statistics, as the birth search and the consistency loss do.
        This is the space the consistency loss trains to match the next
        observation's target projection.
      - Two tokens are predicted equivalent when 1 - cos(z_i, z_j) <= tau.
        The classes are the connected components.
      - tau is noise-scaled from own play only: the median one-step
        consistency error of T2 iteration 42 over T2's own-play log,
        1 - cos(online projection of g(h(s_t), a_t), target projection of
        h(s_t+1)), over every logged decision.
  (B) INVERSE, a new model trained on own play only: an IDM q(action |
      s_t, s_t+1) over the 9 actions.
      - It is a small conv net trained on the 4-frame decisions of the
        lineage's own-play logs (T0, T1 and T2; Test B2's run tree). The
        label is the decision's action.
      - The stacks are the logged network inputs; the inner frames are
        not needed for 4-frame tokens.
      - Held-out: the last 10% of T2's lives, for the training read only.
      - At a census decision whose chosen token is 4 frames long, the
        predicted class of the chosen action is {a : q(a | s, s') >= 0.1
        x max_b q(b | s, s')}, where s' is the main line's next stack.
        Ratio-based, so it carries to other games.
      - (B) is scored on the chosen action's class only. That is all an
        IDM sees: the transition that happened.
  KNOWN TRAP (E0's pre-registration): "equivalent" against "never tried
  here". T2's final row played six of its 18 tokens zero times, all six
  of them 4-frame tokens (NOOP, UP, LEFT, DOWN, UPRIGHT and DOWNRIGHT at 4
  frames); 4-frame tokens were 15% of its play there (tok_counts,
  tok_duration_play_share). The dynamics embeddings of rarely played tokens are barely
  trained, and an IDM's posterior follows the behaviour policy (q is
  proportional to pi(a | s) x p(s' | s, a); the log keeps no pi). (B)
  trains on T0's and T1's logs too, which played more of the grid. The
  reads split by how often each token was played.
READS:
  (1) PRIMARY, (A) on the 4-frame group: pair-counting precision, recall
      and F1 over the 36 token pairs per decision, pooled over the 4,636
      decisions, against the census truth. The floor is the better of the
      two trivial rules (all one class; all distinct). Also reported: the
      mean predicted class count against 2.09.
  (2) (B) on the chosen 4-frame actions: the precision and recall of the
      predicted class's members against the census class; the same floor
      rules; the IDM's held-out accuracy.
  (3) secondary: (A) on the 32- and 16-frame groups; (A) and (B) split by
      token play frequency (zero in T2's last 20 iterations against the
      rest); per game family.
  (4) robustness: (A) at tau x 0.5 and tau x 2, and (B) at 0.05 and 0.2,
      reported, not used for the decision.
DECISION AFTER (on the better F1 of (1) and (2), each only where it beats
its floor):
  - F1 >= 0.8: a model trained on own play recovers same-length outcome
    classes. Next: E2 (merged search) with that model, pre-registered on
    its own, and the cross-length question.
  - F1 < 0.5, or neither beats its floor: own play at this volume does
    not recover them. Kai's call among: more diverse own play
    (exploration aimed at untried tokens), identical latents as the
    criterion, or stopping the line.
  - In between: Kai's call.
CAVEAT: one agent, one game, one snapshot. The scoring states are greedy
play by the final weights, while the logs are sampled play over 42
iterations of changing weights, so there is a distribution shift. The
census truth is frames over 92 frames: two tokens identical at their end
but different later (a buffered input) count as different. E0 found no
hidden state (strict = frames everywhere).
BUILD: a new probe (benchmarks/probes/probe_e1_equivalence.py) and its
tests. It reuses the census's setup and hook pattern and
probe_exhaustive_eval's loaders. No trainer file changes; the live C2 run
tree is untouched. Audited like every build; the tests and the probe run
on CPU under the lock, one process at a time, only with at least 8 GB
free commit.
COST: about a day of build and audit around C2's launch. The run: (A)
about the census's replay time without the branches (well under an hour);
(B) training a small IDM on CPU, about an hour. Results about 11-03 or
11-04.
AMENDMENTS (the build, before any real run):
  (A1) THE FREQUENCY SPLIT. "Zero in T2's last 20 iterations" leaves one
      token: only NOOP-4 has no play in T2's own-play log over iterations
      23-42. The "six zero" were T2's final row alone. Plays there: UP-4
      1, RIGHT-4 4, DOWNRIGHT-4 4, UPRIGHT-4 5, LEFT-4 8, DOWN-4 15. The
      split is now the lineage's 1% rule (the vocabulary rule): a token
      is RARE when it is under 1% of T2's decisions in iterations 23-42,
      against the rest. Reported, not used for the decision, as before.
      Counted from the log (3,141 decisions; 1% = 31.4): RARE = NOOP-4,
      UP-4, RIGHT-4, LEFT-4, DOWN-4, UPRIGHT-4 and DOWNRIGHT-4, all
      4-frame tokens. Only UPLEFT-4 (7.4%) and DOWNLEFT-4 (3.7%) are not,
      so (A)'s 4-frame split is 35 rare pairs against 1. Most of what
      the primary group asks is about tokens T2 seldom plays.
  (A2) THE SCALE OF tau. tau is the median distance from an online
      prediction to its target projection. The pairs compare two online
      predictions. If two tokens truly reach the same next observation,
      both predictions are trained toward the same target, so their
      distance can be up to about the sum of the two errors. Read (4)'s tau
      x 2 covers that case. The decision stays on tau, as registered.
  (A3) THE DATA AND THE COST. The logs hold 484 / 596 / 962 4-frame
      decisions (T0 / T1 / T2), so the IDM trains on about 2,040 samples
      (the 20,000 cap never binds). (B) applies at 506 census decisions,
      where the chosen token is 4 frames long (UPLEFT-4 386, DOWNLEFT-4
      98, LEFT-4 12, UP-4 5, RIGHT-4 5). The full run is estimated at
      about 5 minutes and about 1.3 GB, not an hour.
  (A4) CONVENTIONS fixed before the run:
      - F1 = 2tp / (2tp + fp + fn), and 1.0 when that is 0/0;
      - "beats its floor" is strictly greater;
      - (B) counts the chosen action in both classes (the version without
        it is reported only);
      - game families are the games' returns (8 families);
      - the IDM's hyperparameters were fixed before any data was seen:
        Nature-DQN convolutions, fc 256 -> 9, Adam lr 3e-4, batch 64, 10
        epochs, seed 0, plain cross-entropy;
      - the held-out set is the last ceil(n / 10) of T2's lives (13), and
        its accuracy selects nothing.
  (A5) THE DECISION METRIC (audit r1, before any real run). On the census
      truth, the trivial rule "all one class" already scores pair F1 0.761
      on the 4-frame group: most of its pairs ARE equivalent (2.09 classes
      among 9). "All nine actions" scores 0.837 for (B). So the registered
      rule could call E2 on a margin of 0.001 over the floor, and its
      "< 0.5" branch could never fire. These floors come from the census
      truth alone; no full-run output of either model exists. The smoke's
      120 decisions gave (A) F1 0.74; I saw it, and it is not a result.
      The decision now reads MERGE PRECISION and MERGE RECALL, which are
      what E2 needs. A false merge hides a real choice from search; a
      missed one only spends simulations.
      - (A): precision is the share of the pooled pairs predicted
        equivalent that are truly equivalent; recall is the share of truly
        equivalent pairs that are predicted.
      - (B), over the predicted class's members other than the chosen
        action: precision is the share truly in the chosen action's class;
        recall is the share of the true class's other members that are
        predicted. The chosen action's own presence in its predicted class
        is reported (chosen_in_pred), not scored. This replaces A4's
        "counts the chosen action in both classes".
      - E2: precision >= 0.9 and recall >= 0.5 for (A) or for (B), pooled
        and in at least 6 of the 8 game families.
      - Kai's call (low): neither candidate reaches precision >= 0.7 with
        recall >= 0.2, pooled.
      - Kai's call (mid): otherwise.
      Neither trivial rule can pass. All one class has precision 0.61 on
      (A); all nine has 0.68 on (B); all distinct has recall 0. F1 and
      the floors stay reported.
  (A6) (B)'S RARE SPLIT. As written it is degenerate: every chosen
      action's class contains a rare action, so all 506 decisions fall on
      the rare side. It now splits the predicted and true members by
      whether the member action is rare, which tests the IDM-follows-
      the-behaviour-policy trap directly. It also reports by the chosen
      action's rarity (22 decisions against 484).
  (A7) THE DECISION ON LIVE DECISIONS (audit r2, before any real run).
      18.1% of the census decisions are dead: all 36 tokens give
      identical frames (the start jingle, the death animations). Every
      pair there is equivalent, and they carry 29% of (A)'s true pairs.
      Pooling them lets a rule with one bit of state reach E2. That rule
      is all one class when dead, and one fixed grouping of the nine
      actions otherwise. On the census truth it gives pooled precision
      0.907 and recall 0.569, but live only 0.825 and 0.390.
      On live decisions alone, no fixed partition reaches E2. The best
      recall at precision >= 0.9 is 0.33 for (A) and 0.03 for (B). This is
      E0's (1b) effect again.
      - A decision is LIVE when its census grid partition has more than
        one class (E0's definition).
      - The A5 bars now apply to the live decisions: pooled, and in at
        least 6 of the 8 families. The low bar does too.
      - The all-decision reads stay reported.
      - Also reported as a reference: the best state-blind partition's
        live precision and recall, the best fixed grouping at precision
        >= 0.9.
      - Diagnostics: tau's calibration percentiles per token length.
      - A meta line (tau, the rare set, the IDM's record) is written
        before the replay, so they survive a failed summary write.
      - A short smoke of the fixed probe runs before the full run.
  (A8) SAFETY AT JUNCTIONS (audit r3, before any real run). A lookup keyed
      only on the PREVIOUS token (Pac-Man's heading, which the stack
      shows) reaches E2 under A7. It was fitted on the census truth with
      each family left out:
      - (A) live: P 0.911, R 0.547, 7 of 8 families;
      - (B) live: P 0.923, R 0.544, 6 of 8 families.
      Its partitions are joystick logic: NOOP continues the heading, and
      the diagonals act like it. Its false merges sit at junctions:
      - corridor decisions (at most 2 true classes among the group): P
        0.987;
      - junction decisions (3 or more): P 0.729, and 90% of all its false
        merges fall there.
      Junctions are where a merged search would hide a real choice. So E2
      now ALSO needs merge precision >= 0.9 on the live JUNCTION decisions,
      pooled. A junction is a decision whose census partition, restricted
      to the nine 4-frame tokens, has 3 or more classes. The same
      definition holds for (B), at its decision.
      Also reported, not decided on:
      - the probe's precision and recall split into corridor and junction;
      - the keyed reference beside the state-blind one, fitted on the
        census truth with each family left out: per previous token for
        (A), per chosen action x previous token for (B);
      - reads (3) and (4) on the live decisions too.
      A model that only knows the heading would now read MID, not E2.
      This is the third change to the decision rule (A5, A7, A8). All
      three came from audits of the census truth and of rules fitted to it;
      no model's full-run output exists.
      The numbers above are audit r3's tuned fit. The probe's own keyed
      reference (per key, the partition with the highest training recall
      at training precision >= 0.9) gives:
      - (A): P 0.902, R 0.547, 5 families, junction P 0.722;
      - (B): P 0.923, R 0.544, 6 families, junction P 0.605.
      Under A7 alone it reads E2 on (B); under A8 it reads MID.
      CONVENTION (audit r4): a candidate with no predicted merge at any
      live junction has no junction precision (None), and None fails the
      junction bar, as the probe implements it. This is the conservative
      reading: E2 must show safe merges at junctions, not only avoid them.
      With 1,182 (A) and 112 (B) live junctions it is unlikely to matter.
      In a VOID summary only "verdict" is VOIDed; the reported verdicts
      (A7's, all decisions) keep their values and are not to be read.
AUDIT LOOP (E1):
  - r1: two MEDIUMs, giving A5 and the (B) scoring.
  - r2: one MEDIUM, giving A7.
  - r3: one HIGH in the decision rule, giving A8.
  - r4 (fresh eyes on 967da7c7): nothing above LOW.
  - Job 2 of r4 tried 16 keyed rules (the previous token, previous two
    tokens, chosen token, game phase, life and position in life, and
    crossings of them), fitted left-one-family-out, in-sample, and with a
    Lagrangian. None reaches E2 under A8. Even overfitted, no two-feature
    key reaches R >= 0.5 in 6 families once junction P >= 0.9 is required.
    The loop is CLOSED at 967da7c7: 62 tests passed, and the smoke ran
    valid.
BUILT 2026-11-02 (pre-audit): benchmarks/probes/probe_e1_equivalence.py
and tests/test_e1_equivalence.py (40 passed). A smoke run (2 games, 60
decisions each, small caps) matched the exact eval in both games
(main_line_ok); its numbers are not results.

### Next experiment (pre-registered) -- Test C2: BBF's n-step and discount ramps on top of the resets (Kai, 2026-11-02)

DECISION: Kai 2026-11-02, after Test C read NO DIFFERENT. Asked which
test runs next: "Ramps bundle only" (Test D waits). E1 is built and run
alongside it ("You can do them at the same time").
Background:
  - Test C: C's resets reached 1411.0 at [90k, end] against C0's 1262.4
    (NO DIFFERENT, band 331.3). They kept the trunk plastic (C0 went
    25-29% dormant, C stayed under 2% within cycles) without a score gain.
    C's only gain came in its last cycle, which was 30 iterations long.
  - BBF pairs its resets with two schedules that restart at every reset:
    the n-step horizon falls from 10 to 3 and the discount rises from
    0.97 to 0.997 over the 10k gradient steps after it. The reading: just
    after a reset the fresh heads bootstrap badly, so the targets lean on
    real rewards (a long n) over a short horizon (a low gamma), then hand
    over to the long-horizon, low-bias setting as the network recovers.
QUESTION: on top of C's resets, do BBF's n-step and discount ramps reach a
higher deployed score at 100k env steps than the resets alone?
BBF'S RAMPS (verified 11-02 against the released bbf/agents/spr_agent.py
and bbf/configs/BBF.gin):
  - Both are keyed on cycle_grad_steps, the gradient steps since the last
    reset (0 at the start of training; zeroed in reset_weights).
    cycle_steps = 10,000.
  - n: exponential_decay_scheduler(cycle_steps, 0, 1, update_horizon /
    max_update_horizon), times max_update_horizon, then int(onp.round(.)).
    With update_horizon 3 and max_update_horizon 10, n falls geometrically
    from 10 to 3.
  - gamma: exponential_decay_scheduler(cycle_steps, 0, min_gamma, gamma,
    reverse=True). 1 - gamma falls geometrically from 1 - 0.97 to
    1 - 0.997: log(1 - gamma) moves linearly over the cycle.
  - Both are passed to the replay sampler on every batch, so every target
    uses the current values. BBF's discount is per agent step at frameskip
    4, which is our base step.
THE RUN (seed 0; one run; every env step counted):
  C2 is C (Test C's "-reset-r0", every reset knob unchanged) plus the two
  ramps. The control is C itself, the run of 10-30 to 11-01. No new
  control is run.
  THE RAMPS, keyed on the gradient steps since the last reset (since the
  start before the first), over 10,000 gradient steps (5 iterations of
  2,000, the same window as the actor hold), BBF's shapes:
    - n(f) = round(10 x 0.5^f), f = min(steps since reset / 10,000, 1),
      rounded as onp.round (half to even): 10 -> 5 in base steps;
    - 1 - gamma(f) = 0.03^(1-f) x (1 - 0.997^4)^f: 0.97 -> 0.98805 per
      base step;
    - f by iteration of the cycle: 0, 0.2, 0.4, 0.6, 0.8, 1 at +0 ... +5;
      n 10, 9, 8, 7, 6, 5 and gamma 0.970, 0.975, 0.979, 0.983, 0.986,
      0.988.
  THE END VALUES ARE C'S, not BBF's (n 5 with EfficientZero's age shrink,
    gamma 0.98805), and only the start values are BBF's (n 10, gamma
    0.97 per base step). This is my choice, open to Kai's change before
    launch. BBF's own end values (n 3, gamma 0.997) would add two more
    changes to the ramps: a horizon four times longer (333 base steps
    against 84) and a shorter n. One comparison could not separate those
    three changes, and 0.98805 is the discount behind every positive token
    result so far (Test B2's note).
  WHERE THEY APPLY: every reader of the discount and the horizon. All run
    in the learner process (mapped 11-02):
    - the batch-time value targets (the frame horizon of n base steps;
      every token edge discounted by gamma ** (frames / 4));
    - the birth search and the refresh (the search's edge discounts);
    - the insertion and refresh PER priorities;
    - the row diagnostics.
    EfficientZero's age shrink applies on top: the horizon is clip(n -
    age // OFFPOLICY_DECAY_STEPS, 1, n). The actors play the prior and
    read neither value. The value prefix is undiscounted, so it is
    unchanged.
  After the last reset (after iteration 100) the ramp completes by
  iteration 105, and the end values hold to the run's end.
READS. EXACT eval as before (whole games, 30 noop starts, prior argmax,
PROBE_ENV MsPacmanNoFrameskip-v4); DEPLOYED primary, trainer-BN secondary.
Rows every 5 iterations, plus the final.
  (1) PRIMARY at 100k: C2's deployed rows in [90k, C2's end] against C's
      (1411.0, n 3). The band is 2 x 216.9 x sqrt(1/n_C2 + 1/n_C):
      BETTER / NO DIFFERENT / WORSE. Trainer-BN likewise (C 1358.1).
  (2) descriptive: the deployed mean per 20k window of cumulative env
      steps, beside C, C0 and Test B2. C2's [90k, end] window against C0
      (1262.4, n 4) and T2 (1629.5, n 4), with the same band rule.
  (3) per reset, as in Test C:
      - the dip: row 25 of the cycle against row 20;
      - the recovery: the first row at or above row 20;
      - the cycle's rate: OLS deployed per 1k env steps over the cycle's
        rows 25-40, against C's over the same rows;
      - the start: rows 5-20 (the first cycle, before any reset) against
        C's, the only stretch where the ramps differ from C without a
        reset.
  (4) mechanism:
      - the ramp's n and gamma at every row (logged), checked against the
        schedule;
      - the value loss and the value targets' mean by row, against C's;
      - the plasticity read as in Test C (dormant fractions, first-layer
        filters, erank).
  (5) the prior: policy_maxp and the search targets' entropy by row.
  (6) probe_prefix_bn FIXED at C2's final row.
  (7) stability: the largest adjacent deployed drop. The reset dips are
      reported separately, in (3).
DECISION AFTER:
  - (1) BETTER: BBF's ramps with the resets help in our stack. Kai's call
    among: resets and ramps inside the swap lineage (B2 plus BBF); Test
    D; BBF's own end values.
  - (1) NO DIFFERENT: the ramps do not pay at this resolution either.
    The BBF bundle stays out of the base, and Test D is next unless Kai
    decides otherwise.
  - (1) WORSE: the ramps hurt. Reads (3) and (4) say why: the dips, the
    value loss in the ramp windows. Test D is next unless Kai decides
    otherwise.
CAVEAT:
  - The control is C, a run from before this build. The ramp knobs
    default to off, and the tests pin C's path unchanged, but C2 and C do
    not share an async timing: same seed, different trajectories. The
    band is the within-run row noise, not the seed noise.
  - One seed.
  - The end values are C's, not BBF's (above).
  - The birth search runs at collect time, partway into the next
    iteration, so it takes the ramp's values at collect time, not at
    request time. BBF's sampler also runs two batches ahead.
  - The replay keeps targets stored under earlier values: the birth and
    refresh search policies and root values, and the PER priorities. As
    in Test C's A1, they stay until the refresh re-searches them or they
    are evicted. The value targets themselves are recomputed at batch time
    with the current n and gamma, as in BBF.
BUILD (before launch; audited like every build):
  - muzero.py: the RAMP_* knobs (all off by default): the window
    (10,000), the start values (10, 0.97) and the end values (the arm's
    TD_STEPS and DISCOUNT); ramp_values(steps since reset); a helper
    that sets TD_STEPS and DISCOUNT from it.
  - experiments/ablation.py:
    - the arm C2 (C plus the RAMP_* knobs);
    - a gate (the end values equal the arm's TD_STEPS and DISCOUNT; the
      run shape C's);
    - the values set wherever the step count or the reset origin changes:
      at the run's start, after every gradient step, at a reset, and at a
      checkpoint's resume, so every reader sees the current values;
    - the rows' "ramp" (n, gamma, steps since reset).
  - benchmarks/supervisors/supervise_testb.py: C2's lineage (one run,
    alone); quick() pins the ramp knobs; a per-row check of the logged
    ramp against the schedule.
  - Tests; a scorer (C2 against C, C0 and T2); a hand-off; a launcher; a
    smoke (a tiny ramp run with resets, killed mid-ramp and resumed: the
    ramp's values continue exactly); a manifest and a fresh run tree.
COST: the build and its audit loop today, then about 28 h for the run
(Test C's C took 27-28 h): results about 11-04. E1 is pre-registered,
built and audited while C2 trains, and run on CPU under the lock when
memory allows.
BUILT 2026-11-02, the trainer core (71a28e99, pre-audit):
  - muzero.py: RAMP_* (off by default); ramp_values, BBF's scheduler
    with exact end values; apply_ramps.
  - experiments/ablation.py: the arm ...-tokens-ramp-a0; ramp_gate;
    apply_ramps after every step-count or reset-origin change; ramp_step,
    a per-step check that fails loudly on a stale value; the rows' "ramp"
    and "ramp_trace" (one entry per iteration, checkpointed);
    ramp_globals, which restores both globals after a run.
  - Tests: tests/test_testc2.py (66). 24 suites passed.
  - The first step at each n: 10 at 0, 9 at 741, 8 at 2,345, 7 at
    4,151, 6 at 6,215, 5 at 8,625.
AUDIT r1 A (the trainer core, fresh eyes, static): nothing above LOW.
  - LOW-1: ramp_gate does not refuse search acting, whose actors would
    keep the static discount. C2 acts from the prior, so there is no live
    impact.
  - LOW-2: rows fall where the ramp has ended (cycle steps that are
    multiples of 10,000), so read (4)'s row value loss never sees a ramp
    window.
  - LOW-3: the real readers (the birth search, the refresh, the
    priorities) are not exercised under a ramp in the tests.
  - NITs: a test name overclaims its gate coverage; the token payoff in
    rows 5, 25, ... 105 is computed at ramped values (diagnostic only);
    the tripwire counts mutation sites in ablation.py only.
FIXES r1 (9f817c06):
  - ramp_gate refuses search acting;
  - the rows' "ramp_windows" ({in, out}: steps, scored steps, the value
    cross-entropy in nats, the mean value target z at position 0; per
    gradient step, checkpointed, each step scored once);
  - a test that spies on the real readers under a ramp (the birth search,
    the insertion and refresh priorities, the refresh search);
  - NITs.
CHAIN BUILT (39be296e, pre-audit): supervise_testb.py's lineage "a" (a0);
  quick() pins the ramp knobs (on for "a", off for the others);
  ramp_alert checks each row's ramp and its per-iteration trace against
  the schedule and the reset log. Scratchpad: post_testc2.py,
  launch_testc2.ps1, prelaunch_testc2.py, score_testc2.py,
  smoke_testc2.py. SMOKE OK on 71a28e99, killed twice (mid-ramp, and at
  a reset's own checkpoint): the trace was exact. The smoke is re-run on
  the final code before launch.
AUDIT r2 A (the trainer core at 9f817c06, fresh eyes, static): nothing
  above LOW. This closes the core's loop: the core is frozen from here.
  - LOW-1 (read (4) has no C counterpart) is amendment A1 below.
  - LOW-2: ramp_gate alone would accept the out-of-process refresh
    (REFRESH_OFFLOAD, --ray-workers), but reset_gate refuses both and
    always runs first. Documented here, not changed.
  - NITs: a skipped non-finite step can still be scored in ramp_windows
    (diagnostic only); a docstring's reason for the exact end values; a
    reset row's cycle_steps reads 40,000 (pre-reset, as intended).
  - Test gaps: noted, none blocking. The GPU smoke covers run_one end to
    end.
AMENDMENTS (before launch):
  (A1) READ (4) AGAINST C. C's rows have no ramp_windows. Their value
      panel always falls where any ramp would have ended, and their
      step_loss_mean is the total loss. So:
      - against C: C2's step_loss_mean on the rows that end a ramp window
        (rows 5, 25, 45, 65, 85, 105) against C's on the same rows. Both
        include the post-reset recovery, so the difference is the ramps'
        effect on the total loss;
      - C2's ramp_windows (in against out, per cycle) are descriptive and
        C2-internal: "in" is always the 10,000 steps after a reset, so it
        mixes the reset's recovery with the ramp.
      - (audit r2b) step_loss_mean is not checkpointed. After a resume, a
        row covers only the iterations since the resume. C's row 25 holds
        iterations 23-25 only (C resumed after iteration 22), so it misses
        the first 4,000 steps after reset 1. A1 is read over the rows
        intact in BOTH runs, and the excluded rows are listed.
CHAIN FIXES r1 (63a5a9b3): ramp_alert checks the ramp_windows against the
  schedule (in and out steps, every step scored, finite means). A running
  smoke now counts as a trainer. The smoke asserts the windows, the
  checkpoint state at both kills, and its commit (the AUDITED files
  clean). The PAUSE procedure lists the preflight's pytest and the quick
  probe.
SMOKE OK at 63a5a9b3 (11:28-11:56, the 57 AUDITED files clean):
  - final iteration 26; resets after 4, 8, 12, with the cutoff biting at
    16;
  - the trace's n by iteration 10, 8, 6, 5 after every reset;
  - the windows by row: 50/30 at rows 4-16, 0/80 at 20 and 24, 0/40 at
    26, every step scored;
  - the checkpoint state exact at both kills (mid-ramp, and at a reset's
    own checkpoint).
AUDIT r2 B (the chain at 63a5a9b3, fresh eyes): everything that launches,
  guards, pauses, resumes or alerts is clean. One MEDIUM, in the scorer's
  A1 read (above). LOW-1: prelaunch must tie the launch to the smoked
  commit. NITs: a stale smoke docstring; a race in the watcher; the
  procedures to be given as cmd commands; a possible false ALERT if a kill
  lands between the final row and the completion marker (a few seconds;
  documented, not changed). The fixes are scratchpad-only, so the smoked
  commit stays the launch commit.
CHAIN FIXES r2 (scratchpad only):
  - A1 is read over the rows intact in both runs: resumes come from the
    supv2 logs, cross-checked against eager_collect_steps. C's row 25 is
    excluded.
  - prelaunch_testc2.py requires the smoke's "SMOKE OK at HEAD <sha>",
    and requires the AUDITED files identical between that sha and the
    run tree's HEAD.
  - The procedures are given as cmd commands.
  - The watcher re-checks for an ALERT 90 s after "evals done".
  - The hand-off's probes ran cleanly on a C2-arm smoke snapshot: both
    exact evals and probe_prefix_bn (p1_ratio 1.63, FIXED).
AUDIT r3 B (the changed scripts, fresh eyes): nothing above LOW. The
  chain's loop is CLOSED.
  - LOW-1: prelaunch accepts any later rc=0 after the last SMOKE OK.
    c2_chain_smoke_r2.log is one invocation (one start line, then SMOKE OK
    and rc=0 on the next line), checked by hand.
  - LOW-2: A1 keeps a row whose eager list proves an unlogged resume
    (only a note). Unreachable under the supervisor, and A1 is
    descriptive.
  - NITs: an A1 row without step_loss_mean is skipped rather than listed;
    the no-log fallback; the watcher prints "C2 DONE" for an abandoned run
    (the tail shows ABANDONED); the PAUSE listing also shows the
    preflight's own quick-probe pairs.
ALL 26 SUITES passed in the build tree at c6dc8ac6, 12:12-12:30 EST 11-02,
  one at a time under the lock. The AUDITED files are identical to the
  smoked 63a5a9b3.
LAUNCHED 12:35 EST 11-02:
  - The manifest is 541a19ee (written at cbfd9af5).
  - The run worktree Y:\SwitchPlayer-testc2-run is detached at 541a19ee.
  - The in-tree prelaunch printed PRELAUNCH OK: the smoke's commit with
    identical AUDITED files, the arm's knobs and gates, and the schedule
    against the table.
  - launch_testc2.ps1 started the hand-off post_testc2.py and the
    supervisor a0 (run mspacman-ramp-a0-s0-2026-11-02). The supervisor
    preflights (the 26 suites) before training.
  - The end values stay C's (n 5, gamma 0.98805); Kai raised no change
    before launch.
  - Frozen while it runs: the run tree, the AUDITED files in the build
    tree, and Y:\SwitchPlayer-testc-run and Y:\SwitchPlayer-testb2-run
    (the references).
PAUSED 12:50 EST 11-02 at Kai's request (2 hours), before the first  checkpoint: the supervisor pair and then the trainer pair were killed by  exact PID, the hand-off left running. Resume about 14:50 by rerunning  the launcher. With no checkpoint the run restarts fresh (iter-001.pkl is  truncated on its first write), so about 15 minutes of training are lost.

RESUMED 14:50 EST 11-02 (the 2 hours were up; Kai did not ask for longer) by
rerunning the launcher. The supervisor logged RESUMING (no checkpoint, so the
trainer starts fresh: attempt 2) and is preflighting again (about 25 minutes).
The hand-off (post_testc2.py) ran through the pause. CS2 was open on the GPU
at the resume (about 6 of 12 GB in use), and the trainer shares the card.

Rows 5-25 (trainer-BN returns from the training log; the exact evals wait
for memory, as in Test C, so deployed scores come later). C at the same
row in brackets:

| row | env steps | C2 | C |
|---|---|---|---|
| 5 | 2,599 | 500 | 393 |
| 10 | 5,636 | 291 | 394 |
| 15 | 8,725 | 728 | 575 |
| 20 | 11,691 | 696 | 868 |
| 25 | 14,387 | 751 | 965 |

These are single rows, so no read is possible yet. The first reset fired after
iteration 20 (gradient step 40,000), as in C. The ramp trace restarts with
it: iterations 21-25 run n 10, 9, 8, 7, 6 and gamma 0.97 to 0.9856,
identical to iterations 1-5, and the cycle ends at (5, 0.988054). The
"in" window holds the first 10,000 steps of each cycle (value loss 0.140
in cycle 1, 0.131 in cycle 2). Host commit free is about 4.9 GB, so the
interim probes wait (the hand-off's hourly ALERT, advisory); there are no
orphans from the 12:50 kill.

Row 30 dipped (456; H(pi) 1.91, prefix 0.80 against control 0.37) and row 35
recovered (1,026; H 1.05, prefix 0.15; C 783 at row 35): one row, no read.
Firefox opened at 20:14 and took host commit free from 4.3 to about 2 GB.
The supervisor's advisory "memory creep?" ALERT fired at row 35; the trainer
itself held steady at 7.7 GB.

PAUSED AGAIN 20:51 EST 11-02 at Kai's request (2 hours), in iteration 37,
after the iteration-36 checkpoint: the supervisor (13544; its child 24280
exited with it), then the trainer (10436; 18800 exited with it), killed by
exact PID. No Ray processes were left. The hand-off was left running; it has
started an interim exact eval now that commit has freed (CPU only). Resume
about 22:51 by rerunning the launcher; it will continue from the
iteration-36 checkpoint.

During the pause the hand-off ran the interim exact evals for rows 5-35
(commit free 6-24 GB, about 1 minute each, as in C). Deployed:
502 / 400 / 682 / 653 / 858 / 500 / 808. Raw (trainer-BN, exact): 269 / 586 / 536 /
701 / 607 / 420 / 540. RESUMED 22:52 EST 11-02 by rerunning the launcher
(supervisor cmd wrapper 22960); it preflights, then resumes from the
iteration-36 checkpoint. The trainer resumed after iteration 36 (20,340 env steps, 72,000
gradient steps).

Rows 40-50 (trainer-BN, from the training log; C at the same row in brackets):

| row | env steps | C2 | C |
|---|---|---|---|
| 40 | 23,582 | 504 | 1,044 |
| 45 | 26,406 | 520 | 1,253 |
| 50 | 29,085 | 879 | 1,240 |

Reset 2 fired after iteration 40 (gradient step 80,000), as in C. Cycle 3's
ramp trace (iterations 41-45) is identical to cycles 1 and 2. The "in"
window's value loss falls cycle by cycle: 0.140, 0.131, 0.122. C2 trails C
by about 250 on the rows 20-40 average (687 vs 937) and by more at rows
40-50. Its games are shorter: 29,085 env steps at row 50 against C's
32,343. This is still trainer-BN and single rows; the read is the deployed
90k+ window.

Rows 55-80 (trainer-BN; C in brackets):

| row | env steps | C2 | C |
|---|---|---|---|
| 55 | 31,793 | 614 | 991 |
| 60 | 34,843 | 706 | 965 |
| 65 | 37,454 | 765 | 871 |
| 70 | 40,315 | 675 | 1,080 |
| 75 | 43,563 | 688 | 1,505 |
| 80 | 46,971 | 676 | 965 |

Resets 3 and 4 fired after iterations 60 and 80, as in C. Cycle 4's ramp
trace is identical to the earlier cycles; the "in" window's value loss is
0.101 (0.140 / 0.131 / 0.122 before). From row 50, C2 has stayed at
610-880 while C has held 870-1,505. C2's policy entropy runs higher (1.17-1.47
against C's 0.87-1.16), and it plays fewer env steps per row (46,971 against
C's 56,313 by row 80). With 20 rows left this points toward WORSE, but the
read is still the deployed 90k+ window.

Rows 85-120 (trainer-BN; C in brackets):

| row | env steps | C2 | C |
|---|---|---|---|
| 85 | 49,614 | 988 | 1,210 |
| 90 | 52,382 | 1,119 | 949 |
| 95 | 55,607 | 713 | 1,115 |
| 100 | 58,549 | 795 | 875 |
| 105 | 62,474 | 614 | 1,313 |
| 110 | 65,638 | 975 | 1,125 |
| 115 | 68,526 | 764 | 1,329 |
| 120 | 71,299 | 1,019 | 1,646 |

Reset 5 fired after iteration 100, as in C. C2 then got a reset C never had:
reset 6 after iteration 120, at 71,299 env steps, still under the
pre-registered 80,000 cutoff (C was at 88,447 by then). It is the rule's
outcome, not a fault; C2 reaches 80k env steps later in iterations because
its games are shorter. Cycle 5's ramp trace is exact, and the "in"
window's value loss is 0.088 (0.140 / 0.131 / 0.122 / 0.101 before).

The dormant fraction is about 0 five iterations after every reset and
0.21-0.34 five iterations later, in both C and C2. The ramps leave it
unchanged.

Memory: host commit free fell to 0.6 GB at 11:53 (Firefox reopening on a
trainer peak). The commit limit is fixed (34.5 GB; the pagefiles are not
system-managed). The replay buffer is capped and has been full since early in
the run (31,125 base steps). A 2-minute sampler shows the trainer
oscillating within each iteration between 8.3 and 11.0 GB, without growth,
so free commit cycles between 1.6 and 4.4 GB. There is no creep; the risk
is a new app opening on a peak.

PAUSED A THIRD TIME 13:47 EST 11-03 at Kai's request (a couple of hours),
in iteration 121, after the iteration-120 checkpoint (which holds reset 6
and its ramp origin). The supervisor (26552; 23932 exited with it), then the
trainer (17792; 9680 exited with it), were killed by exact PID; no Ray
processes were left. The hand-off was left running. Resume about 15:48 by
rerunning the launcher.

During the pause the hand-off finished the interim exact evals through row
120 (24 rows each, all rc=0). Provisional read from score_testc2.py at
14:30:
  (2) deployed 20k windows:
      C2 629 / 639 / 827 / 910 (n 4)
      C  754 / 1,028 / 1,067 / 1,101 / 80-end 1,423
      C0 564 / 1,015 / 1,160 / 1,290 / 1,300
    C2 trails C in every window. The gap is beyond the band only at 20-40k
    (-389, band 263); at 60-80k it is -191 (band 280). (1) is not yet
    readable (last row 120 at 71,299 env steps).
  (3) within-cycle deployed rates after resets 2-5: C2 +7.6 / +8.7 /
    +3.1 / +19.1 per 1k env; C -21.1 / +4.0 / -16.2 / +30.4.
  (4) all 24 rows and 120 trace entries are on the schedule. A1 (rows 5,
    45, 65, 85, 105, intact in both): step_loss_mean C2 +0.430 vs C
    +0.341. The "in" windows run 0.012-0.023 nats above the "out" windows in
    every cycle. Mean value targets are lower inside the ramp (as designed:
    lower gamma, longer n).
  DORMANCY BY NET (correcting an aggregate read given to Kai the same day):
    after the first reset, representation, dynamics and prediction stay
    near 0 dormant (0.000-0.02) in both C and C2. The per-cycle regrowth
    (0.29-0.64) is almost all in the PROJECTOR. Once resets stop (C,
    rows 105-134) the trunk re-accumulates slowly (representation 0.044,
    dynamics 0.094 at row 134). The resets keep the policy/value path alive
    under SGD, so the dormancy argument for an AdamW C3 does not hold.

RESUMED 15:14 EST 11-03, early, when Kai said to continue, by rerunning
the launcher (supervisor cmd wrapper 21628). It preflights, then resumes
from the iteration-120 checkpoint. The trainer resumed after iteration 120 (71,299 env steps). Rows
125-135 (trainer-BN from the training log): 1,063 / 1,165 / 974 at 74,795 /
78,526 / 81,531 env steps. By env steps, C was at 1,313 (76,202) and 1,125
(79,888). Resets stopped after reset 6 (past 80k env steps). 143 stale
tail/grep processes from expired log watches (this session's) held about
0.8 GB of commit; they were stopped by PID, and the watches now end themselves.

PAUSED A FOURTH TIME 18:55 EST 11-03 at Kai's request (no duration given),
in iteration 139, after the iteration-138 checkpoint: the supervisor (28024;
27208 exited with it), then the trainer (1680; 8600 exited with it), were
killed by exact PID. No Ray processes were left. The hand-off is running an
interim exact eval. Resume when Kai says, by rerunning the launcher.

RESUMED 00:56 EST 11-04 when Kai said to continue ("Can we continue?"), by
rerunning the launcher. It skipped the still-running hand-off and started
the supervisor (8780), which logged RESUMING a0 s0, passed its preflight
(manifest + quick probe + 26 suites) and relaunched the trainer (27840 ->
5896). The trainer resumed after iteration 138 (83,695 env steps, 276,000
train steps). The hand-off's 4-6 h "supervisor.log silent" alerts during
the pause were expected. Rows 140-150 (trainer-BN from the training log):
595 / 790 / 938 at 85,251 / 88,149 / 90,858 env steps; row 150 is C2's
first row in the [90k, end] window. By env steps, C was at 1,329 (83,844),
1,646 (88,447) and 1,111 (92,607).
Rows 155-165: 780 / 1,105 / 770 at 94,345 / 97,136 / 100,489 env steps
(C: 1,458 at 97,314, 1,180 at 100,849). RUN COMPLETE 05:51 EST 11-04: the
trainer passed 100k env steps in iteration 165 and exited rc=0 (16,979 s
since the last resume). The hand-off then started the queued interim probes
(rows 140-165) and the final pass.

### RESULT -- E0: the action-equivalence census on T2's own play (seed 0), 2026-11-02

(Pre-registration: "Queued experiment (pre-registered) -- E0", below
Test C's RESULT; the build, the audits and the launch are there.)
RUN:
  - 00:39-01:35 EST 11-02 (3,392 s), CPU (2 threads), under the lock.
  - From the curriculum tree. probe_action_census.py,
    test_action_census.py, probe_exhaustive_eval.py, ablation.py and
    muzero.py are unchanged since the audited ac947976.
  - On Test B2's T2 iteration-42 snapshot (18 tokens): 30 games, 4,636
    decisions, every one censused; the sequence census ran at 478 of
    them (every 10th).
VALIDITY (A3): main_line_ok TRUE.
  - All 30 games equal the exact eval's games ([return, base steps,
    terminated, decisions]), and the mean is 1699.0, as in the eval.
  - There were no prior-argmax mismatches and no replay mismatches, and
    no void reasons.
  - Games 1-9 are near-repeats in a cycle of four (1700 / 1880 / 1580 /
    1690, each noop start shifting a game by one base step). Games 10-30
    differ more (790 to 2170). Decisions across games are therefore
    correlated; the census counts every decision once.
READ (1) PRIMARY, the effective branching factor. Frame-equivalence
  classes per decision:
      tokens               mean   median  p10  p90   ratio to nominal
      T2's 18 (vocab)      3.28     3      1    5    0.182
      the 36-token grid    3.56     4      1    6    0.099
  - The distribution over T2's 18 tokens, as decisions per class count:
    1: 842, 2: 323, 3: 1,367, 4: 1,181, 5: 652, 6: 254, 7: 17. No
    decision had more than 7 classes (8 for the grid).
  - (1b) (A6), over the 81.9% of decisions where some input changes the
    frames: 3.79 classes (median 4, p10 3, p90 5), ratio 0.210.
  - Both are far below one half -> E1 (the verdict and verdict_1b agree).
READ (2) by duration. Classes among the 9 actions at each duration: 2.09
  (4 frames), 2.27 (8), 2.67 (16), 3.50 (32). Each moving action (NOOP
  left out) gives 1.32 classes over its 4 durations on average. How often
  the shorter token, with no input after it, equals the longer one:
      4=8 95.5%   4=16 86.5%   4=32 70.1%   8=16 91.0%   8=32 73.6%
      16=32 81.9%
  So the duration choice is mostly redundant here. In Ms. Pac-Man no
  input continues the current motion, so a short press and a long one
  usually play out the same.
READ (3) sequences of 1-3 base steps (4 frames each), distinct outcomes:
  k_1 = 2.08 of 9, k_2 = 4.53 of 81, k_3 = 10.18 of 729. That is a
  per-step growth of 2.18 / 2.25: about two real choices per 4-frame
  step, against 9 nominal.
READ (4) free decisions:
  - 18.2% of decisions had every token equivalent (T2's vocabulary;
    18.1% for the grid): the choice changed nothing.
  - 25.1% had at most two classes.
  - Over the grid, the agent's chosen token shared its outcome with 19.1
    of the 36 tokens on average.
READ (5) the prior's split: the entropy of T2's prior over its 18 tokens
  averages 1.556 nats. The same mass summed over the frame classes
  averages 0.564 nats. The gap of 0.99 nats, 64% of the prior's entropy,
  is spread over tokens that play out identically. That blur is what a
  merged search would remove.
READ (6) frames against emulator state: none. At all 4,636 decisions the
  strict classes (frames and the end-of-window emulator state) equal the
  frame classes, for T2's vocabulary and for the grid, and so do the
  sequence counts. No hidden state (a buffered input, a random byte)
  separated frame-identical branches within the 92- and 72-frame windows.
PRE-REGISTERED DECISION: (1) at most half the nominal size (0.18 for
  T2's 18; 0.21 on the live decisions) -> grouping can at least double
  what search spends per distinct choice. Next is E1: can a learned
  model, trained on the agent's own play alone, recover these classes?
  Then E2. E1 is designed but not pre-registered. It is pre-registered on
  its own before any build, and the emulator only scores it (Kai's
  rule).
Beyond the letter (one snapshot, one game):
  - The ratio is far below the threshold, not near it. About 3-4
    outcomes per decision against 18 tokens means the 50-simulation birth
    search spreads its visits over roughly five tokens per real choice.
  - The collapse is in direction and in duration. Duration is measured
    (read (2)) and fits Ms. Pac-Man's controls, where no input continues
    the motion. For direction, only about 2 of the 9 actions differ at 4
    frames. The likely cause, not measured here, is that a diagonal or a
    press into a wall plays as a cardinal move or as no input. If the
    classes come from the controls and the maze, they may be learnable
    from state alone. Whether anything like them holds in Tetris, where
    holding an input repeats it, is open.
  - Equivalence here is over a 1-s window after the token; a longer
    window could split some classes (no read covers that).
  - The decisions file (probe_action_census-mspacman-two-t2-s0-2026-10-29
    -decisions.jsonl, 4,636 lines with the 36 tokens' partitions) is
    committed for E1's scoring.
  - Test B2's run tree (Y:\SwitchPlayer-testb2-run) is no longer needed
    by any queued read; it stays until Kai says otherwise.

### RESULT -- Test C: BBF-style resets in one 100k run against the no-reset twin C0 (seed 0), 2026-11-02

(Pre-registration: "Next experiment (pre-registered) -- Test C", below
Test B2's RESULT; the interim notes for both runs are there.)
RUNS (from the run worktree Y:\SwitchPlayer-testc-run, detached at
d682fa5e, manifest from the audited 2a51e527):
  - C (r0) launched 18:29 10-30 and was complete at 22:38 10-31: iteration
    134, 100,849 env steps, 268,000 train steps. Its last evals landed at
    23:06. There were 5 resets, after iterations 20 / 40 / 60 / 80 / 100,
    and 20 held ships. It was paused twice at Kai's request (22:02 10-30
    for 2 h; 11:36 10-31, with a reboot, resumed 14:39).
  - C0 (n0) launched 23:15 10-31 and was complete at 00:32 11-02 (final
    evals included): iteration 136, 100,173 env steps, 272,000 train
    steps. It was paused once at Kai's request (11:46-13:58 EST 11-01).
  - Each pause resumed from the last checkpoint in the same attempt, and
    no reset was repeated or skipped.
  - Both supervisor.err files are empty. Every row had 0 non-finite
    steps, and no exact-eval game was capped (27 / 28 rows, both evals).
  - Host commit dipped in both runs: C to 2.4-3.3 GB over its last 4 h,
    C0 to 0.6-1.0 GB on the morning of 11-01. Only the advisory "memory
    creep?" ALERTs fired, and no allocation failed. Delayed rows were
    scored later from their snapshots.
EXACT eval (whole games, 30 noop starts, prior argmax). Every score is
unclipped. DEPLOYED is primary; trainer-BN is secondary. env = thousands
of env steps.
       row     5    10    15    20    25    30    35    40    45    50
  C    env   2.6   5.6   8.6  11.9  15.0  18.5  21.8  25.1  29.0  32.3
       dep   473   481   741   873  1038   917   726   959  1167  1172
       tBN   341   499   544   984  1007   950   882   861  1076  1323
  C0   env   2.6   6.0   9.1  12.5  15.9  19.3  22.9  26.3  29.9  33.9
       dep   499   450   402   655   476   899  1343   699   996   785
       tBN   305   350   328   436   776   759   935   731  1103   838
       row    55    60    65    70    75    80    85    90    95   100
  C    env  36.1  40.5  44.4  48.2  52.5  56.3  60.7  64.4  68.2  72.3
       dep  1115   925  1007  1165  1186  1052  1207  1052  1095   984
       tBN   927  1039  1133  1172   985  1171  1173  1046  1026  1167
  C0   env  38.2  42.1  45.4  49.0  52.9  57.0  60.6  64.7  67.6  71.4
       dep  1252  1035  1124  1325  1143  1175  1349  1168  1125  1363
       tBN   630   953  1077  1019   942  1021  1315  1500  1223  1153
       row   105   110   115   120   125   130  135   final
  C    env  76.2  79.9  83.8  88.4  92.6  97.3    -   100.8 (134)
       dep  1212  1057  1373  1512  1318  1572    -    1344
       tBN  1076  1232  1099  1293  1275  1245    -    1554
  C0   env  75.7  79.7  83.3  87.1  91.2  95.5  99.4  100.2 (136)
       dep  1294  1441  1287  1464  1211  1239  1317   1283
       tBN  1582  1420  1291  1242  1232  1250  1403   1096
  C and C0 ran at similar paces, within about 2k env steps of each other
  at every row.
READ (1) PRIMARY at 100k: C's rows in [90k, end] are 125 / 130 / 134 (n =
  3): 1411.0 deployed. C0's are 125 / 130 / 135 / 136 (n = 4): 1262.4.
  C - C0 = +148.6, band +-331.3 -> NO DIFFERENT. Trainer-BN: 1358.1
  against 1245.3, +112.8 -> NO DIFFERENT.
READ (2) per 20k window of env steps (n), deployed:
      window              0-20k    20-40k    40-60k    60-80k    80-end
      C                   754 (6)  1028 (5)  1067 (5)  1101 (6)  1423 (5)
      C0                  564 (6)  1015 (5)  1160 (5)  1290 (6)  1300 (6)
      C - C0              +190       +13       -93      -189      +123
      band (+-)            250       274       274       250       263
      Test B2             622       801      1071      1484      1588
      Test B              503       881       845      1003      1023
      current lineage     540       896      1107        -       1164
  No window differs between C and C0. About half of C's 0-20k lead comes
  from rows 5-20, which precede the first reset and are the same recipe
  in both runs (C 642.1 against C0 501.5, n 4 each, inside the band); the
  rest from rows 25-30 (C 1038 / 917 against C0 476 / 899).
  C0's [90k, end] (1262.4, n 4) against the references:
  - Test B's 1015.9 (n 4): +246.5, band +-306.7 -> NO DIFFERENT.
  - The current lineage's 1163.8 (n 3): +98.6, band +-331.3 -> NO
    DIFFERENT.
  - Test B2's T2 1629.5 (n 4): -367.1, band +-306.7 -> WORSE.
  - Trainer-BN: +89.7 / -14.6 / -359.9 against the same three.
  C's own [90k, end] against T2 was -218.5 (band +-331.3, inside).
READ (3) per reset (C). The dip is row 25 of the cycle against row 20.
  The rate is OLS deployed per 1k env steps over the cycle's rows 25-40.
  C0's rate and mean are over the same rows.
      reset  rows      C dip (dep / tBN)  C rate  C0 rate  C mean  C0 mean
      1      20->25    +165 /  +23         -12.9    +32.9     910      855
      2      40->45    +208 / +215         -21.1    +14.8    1095     1017
      3      60->65     +82 /  +94          +4.0     -1.0    1102     1192
      4      80->85    +154 /   +2         -16.2     -0.3    1084     1251
      5     100->105   +228 /  -91         +30.4     +9.7    1341     1317
  (Reset 5's means are over rows 105-134 / 105-136, since no reset
  followed.) Every reset "dipped" upward: row 25 of each cycle was at or
  above row 20, so each recovered at once. In cycles 1-4, C's rates were
  flat or falling while C0's were positive or flat. Only cycle 5, the one
  cycle that ran 30 iterations past its reset, rose faster in C.
READ (4) plasticity. Dormant fraction (<= 0.025), by net, representation
  / dynamics / prediction:
      row    C0                    C
      20     7.3 /  4.7 / 21.9%    27 / 21 / 26%  (before reset 1)
      40     9.4 /  6.6 / 25.0%    0.6 / 0.4 / 0%
      60    13.1 / 10.5 / 25.5%    1.3 / 0.4 / 0.5%
      80    15.8 / 12.5 / 26.0%    0 / 0.4 / 0%
      100   18.1 / 16.4 / 26.6%    0.2 / 1.6 / 1.0%
      120   23.3 / 21.1 / 27.6%    0.4 / 2.0 / 0.5%
      final 26.5 / 24.6 / 29.2%    4.4 / 9.4 / 2.1%
  (The <= 0.1 fractions are within 2 points of these. value_prefix is 0
  in every row of both runs.)
  - C0's trunk dormancy rose steadily through the whole run, with no
    plateau. C's was cleared by each reset and stayed under 2% within
    every 20-iteration cycle. It climbed only over the final
    30-iteration stretch.
  - Weak first-layer filters: C0 25% at row 20, 47% from row 90 to the
    end. C: 53-56% at rows 15-20, 0 after each reset, 9-16% late in
    cycles.
  - The projector: C0 59-66%. C re-collapsed to 44-63% within 5-10
    iterations of every reset.
  - erank: C0 27.5-43.2 after row 20. C 5.4-68.7, swinging with the
    resets.
  - So the resets did what they are for. They kept the trunk's units
    alive, and C0's units died steadily without them. That did not show
    up as a score difference at 100k.
READ (5) the prior: policy_maxp C0 0.62-0.80 except row 55 (0.43, back to
  0.72 at row 60); C 0.53-0.77. The search targets' entropy: C0 0.45-1.60,
  with its lows late (0.59 / 0.45 / 0.52 at rows 115 / 130 / 135; 1.30 at
  the final row); C 0.86-1.91. Neither prior collapsed.
READ (6) mechanism: probe_prefix_bn FIXED at C0's final row (p1_ratio
  0.54) and at C's rows 20 / 25 / 100 / 105 / 134 (0.40 / 0.77 / 1.11 /
  1.04 / 0.53).
READ (7) stability: the largest adjacent deployed drop is -644 for C0 (row
  35 -> 40, after a single high row 35 of 1343; outside the one-row band
  +-464) and -228 for C (row 130 -> 134). C0's later rows never dropped
  by more than 253 (row 120 -> 125).
PRE-REGISTERED DECISION: (1) NO DIFFERENT -> resets alone do not pay at
  this resolution (one seed, n 3 against 4). Next is Kai's call between:
  - the ramps bundle: BBF's n-step and discount ramps, on top of the
    resets;
  - Test D: a smart swap trigger, rate-based and game-agnostic (the rate
    falling below a fraction of the generation's own peak, or the
    marginal-value rule; see "LATER (Kai 2026-10-27)" at the end of
    Test B's pre-registration notes).
  E0 (the action-equivalence census) runs first, in the gap, as Kai
  decided 10-30.
Beyond the letter (one seed, within-run bands, so "consistent with"):
  - Resets bought plasticity, not score. C's trunk stayed alive and C0's
    trunk lost a quarter of its units, but C0 matched C in every window
    (inside the bands) and was ahead at 40-80k. At this budget, with this
    replay ratio (~2.7 gradient steps per env step), dormancy at this
    level did not limit learning.
  - Where C did gain: the last stretch. After its final reset C rose
    +30.4 per 1k env steps, against C0's +9.7 over the same rows, and its
    80-end window is +123 above C0's (inside the band). One reading is
    that resets help once a cycle runs long enough to use the fresh
    capacity; 20-iteration cycles were flat. One seed cannot separate
    that from noise.
  - C0 is the first 0-swap seed of the swap-count curve. A single long
    run at T0's recipe reached 1262 at [90k, end], level with Test B
    (4 swaps) and the current lineage, and 367 below B2's T2 (2 swaps;
    outside the band). The clip also differs (C0 keeps the sign clip;
    T2 played at clip 17.4), and the vocabulary (36 tokens against 18),
    so this is not yet the swap count alone.
  - Decision speed: frames per decision in the in-run evals, C0
    18.9-26.9 and C 21.8-30.2. Both keep 36 tokens throughout.

### Queued experiment (pre-registered) -- E0: an action-equivalence census on the agent's own play (Kai, 2026-10-30)

DECISION: Kai, 2026-10-30. He asked whether an inverse dynamics model on
the agent's own play can find which actions actually have consequences:
  "actions 'left - up - left - no action - no action - up' might result
  in the exact same game play as 'left - no action - no action - no
  action - no action' due to the dynamics of the game and a bunch of the
  inputs not actually making a difference depending on what the current
  state of the game is. I want to figure out if there is a way to use the
  inverse dynamics model on its own gameplay to then discover what tokens
  might overall simplify search."
  "I don't want to reinforce behavior but I do want to take successful
  behavior and see if it generalizes into other places."
Kai's answers (10-30) to the three questions:
  - When: in the gap after C0 completes, before the next BBF test.
  - Ground truth: the emulator's state save/restore is "ok for now as a
    test to see if the theory holds". It does not work long term: it
    would train the IDM faster than real-time play allows in environments
    with no simulator.
  - Equivalence: "identical frames work for now". Later it may have to be
    identical latents.
THE IDEA: at a given state many inputs, or input sequences, are
outcome-equivalent: they lead to the same game. Formally this is an MDP
homomorphism (Ravindran & Barto): (state, action) pairs with identical
transitions and rewards merge. Search needs one representative per
outcome class. Today 50 simulations spread over 18-36 children, many of
them possibly duplicates. Duplicates split the visits and blur the policy
targets. Merging them lets the same simulations go deeper, or lets fewer
simulations do the same work: the real-time lever. The deployed agent
plays the prior and does not search; search builds the training targets
(the 50-sim birth search, the refresh).
QUESTION (E0, measurement only): on the agent's own state distribution,
how many distinct outcomes do the tokens, and short input sequences,
actually produce per state? That is the action space's effective
branching factor, against its nominal size. If it is close to the
nominal size, grouping cannot pay on this game, and the line stops here
cheaply.
METHOD:
  - THE MAIN LINE: Test B2's final T2 snapshot (iteration 42) plays
    whole games as the exact eval plays them: deployed acting states,
    prior argmax, MsPacmanNoFrameskip-v4, noop starts 1..N, the 3000
    base-step per-life cap. N = 30 games, the exact eval's count.
  - AT EVERY DECISION of the main line, the emulator state is saved and
    branches are played from it. After each branch the state is
    restored, and the main line plays the agent's own token.
  - TOKEN BRANCHES: all 36 grid tokens (9 actions x 4, 8, 16 and 32
    frames), not only T2's 18. Each token is followed by no input (NOOP)
    until a common window ends 60 frames (1 s) after the longest token
    (32 + 60 = 92 frames from the decision). A short token with no input
    after it is thereby compared with a long one: Kai's example.
  - SEQUENCE BRANCHES: every sequence of L = 1, 2, 3 base steps (4
    frames each; 9, 81 and 729 sequences), each followed by NOOP to a
    common window of 12 + 60 frames. Expanded breadth-first, with
    identical emulator states merged: identical states have identical
    futures under the same inputs, so the classes are exact.
  - EQUIVALENCE (Kai): two branches are equivalent when every raw frame
    of the common window is identical. Secondary, stricter: identical
    emulator state at the window's end, which also catches state the
    screen does not show yet (a buffered input, a random-number byte).
  - THE GAME END: a branch in which a life is lost is compared like any
    other (its frames show the loss).
READS:
  (1) PRIMARY: the effective branching factor of T2's own vocabulary,
      the mean over decisions of the number of frame-equivalence classes
      among its 18 tokens, against 18. Also for the full 36-token grid,
      against 36. With the distribution over decisions (median, the 10th
      and 90th percentiles).
  (2) by duration: classes among the 9 actions at 4, 8, 16 and 32 frames.
      Across durations: how often a short token followed by no input
      equals a longer token of the same action (is the duration choice
      redundant?).
  (3) sequences: the number of distinct outcomes k_L for L = 1, 2, 3,
      against 9^L, and the per-step growth k_L / k_(L-1).
  (4) free decisions: the share of decisions where every token is
      equivalent (the choice changed nothing), and the share with at most
      two classes (plausibly "continue" against "reverse").
  (5) the prior's split: at each decision, the entropy of T2's prior over
      its tokens against the entropy of the same mass summed over the
      frame classes. The gap is the uncertainty spent on choices that
      change nothing: the blur that merged search would remove.
  (6) frames against emulator state: classes under the strict definition
      against the frame definition. A large gap means hidden state (a
      buffered input or a random-number byte) separates choices the screen
      cannot yet tell apart.
DECISION AFTER (ratios, so the rule carries to other games):
  - (1) at most half the nominal size: grouping can at least double what
    search spends per distinct choice. Next: E1 (can a learned model,
    trained on the agent's own play alone, recover these classes?), then
    E2.
  - (1) above three quarters of the nominal size: grouping does not pay
    on Ms. Pac-Man. Kai's call (another game, Tetris, or stop).
  - In between: Kai's call.
THE LINE AFTER E0 (designed, not pre-registered; each is pre-registered
on its own once the one before it reads):
  - E1: a learned equivalence model, scored against E0's census. Two
    candidates: an IDM p(token | s_t, s_t+k) trained on the agent's own
    logged play (tokens it cannot tell apart are equivalent), and the
    existing dynamics model (children whose predicted next latents
    coincide). Training data is the agent's own play only; the emulator's
    branches only score the model, per Kai's long-term rule. The IDM's
    known trap is confusing "equivalent" with "never tried here"; the
    census exposes it.
  - E2: merged search. At each node, children are grouped by predicted
    outcome and one per class is expanded; the class's visits go to a
    canonical token (the simplest input). Two arms against the
    unmerged recipe: the same simulations (deeper search), and fewer
    simulations (the same play for less compute).
  - E3: tokens from successful play. Recorded sequences are canonicalized
    by the equivalence model, so that Kai's two sequences become one
    effect. Frequent effects are mined from high-advantage stretches and
    added as tokens at a swap, in the freed vocabulary slots. The student
    learns when to use them unaided: vocabulary, not policy, as in P5. The
    likely Ms. Pac-Man token is variable-length, "this way until the next
    decision point"; there the dynamics model must also predict the
    token's length, for the discount. This is P5e with the IDM trained on
    the agent's own play, brought forward.
  - Later (Kai): equivalence by identical latents instead of frames, for
    environments with no simulator.
PRIOR ART: action elimination (Zahavy et al. 2018, irrelevant actions
dropped in a text game's large action space); abstraction inside tree
search (Jiang, Singh & Lewis 2014, approximate homomorphisms in UCT;
Anand et al. 2015, ASAP-UCT); an inverse model on the agent's own play to
find what it controls (Pathak et al. 2017; Choi et al. 2019 on Atari).
BUILD: a new probe (benchmarks/probes/probe_action_census.py) and its
tests. No trainer file changes; the live Test C run tree is untouched.
Audited like every build, the tests run after C0.
COST: about 1-3 h of one process (the policy on the GPU, the emulator on
the CPU), in the gap after C0, before the next BBF test.
AMENDMENTS (the build, before any run):
  (A1) The sequence census runs at every 10th decision of each game
      (decisions 0, 10, 20, ...), not at every decision. Its worst case,
      when nothing merges, is about 50,000 frames per decision (9 + 81 +
      729 branches, each continued to 72 frames). At every decision that
      could take over 12 h. The token census still runs at every decision.
  (A2) CPU only, as the exact eval (probe_exhaustive_eval's settings).
  (A3) VALIDITY: the census's main line must reproduce the snapshot's
      deployed exact-eval row (T2 iteration 42: 1699.0 over noop starts
      1..30). A difference means a branch leaked into the main line, and
      the census is VOID. The probe records the check (main_line_ok).
  (A4) Strict equality compares ALE's full serialized state (CPU, RAM and
      the rest), not RAM alone.
  (A5) The sequence census expands breadth-first. A node is the emulator
      state together with the frames played so far. Nodes equal in both
      have identical futures under identical inputs and are expanded once.
      The tests pin the counts against brute force.
BUILT 2026-10-30 (pre-audit): benchmarks/probes/probe_action_census.py,
tests/test_action_census.py. The probe imports the exact eval's module
only when it runs as a script, so importing it changes nothing
process-wide.
AUDIT r1 (two fresh-eyes auditors, static: host memory stayed below the
test floor while C trains):
  - r1a (the probe): nothing HIGH. Two MEDIUMs:
      - M1: the A3 check looked up its reference row only after the
        census, and was silently skipped without one.
      - M2: decisions where no input matters (the start jingle, the death
        animations) are one class by construction. They pull (1) toward
        E1.
    LOWs:
      - A3 compared only the 30-game mean.
      - The duration read included NOOP, which is equal by construction.
      - Strict was keyed on the end state alone, so it was not a
        refinement of the frames.
      - Branch determinism rested on full-frame redraws.
      - No per-line flush.
  - r1b (the tests): one HIGH: a test that could never pass (the hidden
    counter offsets the merges exactly, 9 against 9). Three MEDIUMs, all
    about test strength:
      - The end-to-end test's fresh net plays NOOP throughout, since the
        policy head is zero at init.
      - The vocabulary's classes were never told apart from the grid's.
      - The prior's alignment with the classes was not pinned.
    LOWs: the windows, the duration read, Census.decision, the 0.75
    boundary, the overwrite refusal, the validity logic, and the real
    emulator's asserts (identical input streams only).
FIXES r1 (all of them):
  - A3 runs before anything plays: a full run refuses without its
    reference row. It compares every game ([return, base steps,
    terminated, decisions]), not the mean.
  - Per decision: the census count against the game's decisions; the
    hook's prior argmax against the agent's token; and the agent's own
    grid token replayed after all the other branches, against its census
    digest (branch-order independence). Any failure: VOID.
  - Strict keys are (frames, end state), a refinement of the frames.
  - The duration reads leave NOOP out. The verdict uses the unrounded
    ratio. Each decision line is flushed.
  - The tests:
      - exact counts for the different-paths case (frames 3 / 9, strict
        4 / 15);
      - a frame-dependent policy in the end-to-end test (the policy head
        re-drawn), 60 censused decisions past the intro, the whole game
        tuple compared;
      - vocabulary-against-grid and misaligned-prior cases;
      - windows pinned; reveal at the window's last frame and after it;
      - the duration read on a scripted 36-token census;
      - a scripted Census.decision (the A1 phase, the restore, the lines,
        the argmax check) and a branch-order-dependent emulator that the
        replay check catches;
      - the 0.75 boundary, the overwrite refusal, the validity logic;
      - a real-emulator search for a state where LEFT or RIGHT changes
        the frames.
  (A6) A secondary read (1b): (1) over the decisions where some input
      changes the frames (classes_grid > 1), with their share. The
      verdict is (1)'s, as registered. When (1b) falls on another side of
      a threshold, the decision is Kai's call, and both are reported. The
      no-control decisions are real for search: the birth search runs on
      those states too, and there merging saves all of it. But they
      should not carry the verdict alone (audit r1a M2).
AUDIT r2 (fresh eyes, static, on 0e429c6f):
  - r2a (the probe): nothing above LOW. Every r1 fix is correct, with no
    regression; the main line still reproduces the exact eval bit for bit.
    LOW: a mismatch in game 1 shows only after all 30 games. NITs: the
    first matching reference row wins; the count check is skipped under a
    census cap; a negative cap slips through; VOID leaves the sub-verdicts
    live; the thread count is not recorded.
  - r2b (the tests): two MEDIUMs.
      - The end-to-end test's 60 censused decisions start at decision 0,
        and with 4-frame tokens they may all fall inside the start jingle:
        one token, one class. The FIXES r1 text above ("past the intro")
        was wrong.
      - main()'s A3 wiring is untested: the full-run refusal, the VOID
        override.
    LOWs: (1b) never separates the vocabulary from the grid; the rounding
    inside summarize; strict-on-state-only in the token census; the
    import subprocess dropped CUDA_VISIBLE_DEVICES (a GPU init beside the
    live run).
FIXES r2 (all of them):
  - The probe:
      - a full run stops at the first game that differs from the exact
        eval (VOID either way);
      - the last matching reference row wins;
      - the count check holds under a census cap (at most the cap);
      - negative or zero settings are refused;
      - VOID marks verdict_1 / verdict_1b too and keeps the would-be
        verdict apart;
      - the thread count is recorded.
  - The tests:
      - The end-to-end test biases the 4-frame tokens out, so its 60
        censused decisions reach past the jingle; it asserts a
        multi-class decision.
      - It records the plain game's every token and checks the census's
        tokens against them, decision by decision.
      - It runs main() four ways: the full-run refusal before anything
        plays; a smoke census without a row; with a matching row
        (main_line_ok True); with a newer differing row (VOID everywhere).
      - (1b) with a live decision whose vocabulary sits in one class.
      - The rounding inside summarize.
      - A wrap-around emulator for the token census's strict keys.
      - The import subprocess keeps CUDA_VISIBLE_DEVICES=-1.
  (Correction: "negative or zero settings are refused" means CENSUS_GAMES
  and CENSUS_SEQ_EVERY below 1 and CENSUS_MAX_DECISIONS below 0;
  CENSUS_MAX_DECISIONS 0 is the uncapped census.)
AUDIT r3 (fresh eyes, static, on ac947976): nothing above LOW -- THE AUDIT
LOOP IS CLOSED. Every r2 fix is correct; no regression; a full run still
reproduces the exact eval bit for bit; every new or changed test passes by
hand-trace and catches the mutation it targets.
ACCEPTED r3 (no edit, so no further round):
  - The end-to-end variety assert (more than one token over the plain
    game) rests on a random-init argmax changing with the frames: likely,
    and deterministic if it fails (a fixture fix, not a probe bug). Its
    torch.manual_seed(5) is dead (Run.__init__ reseeds).
  - The full-run fail-fast break and its "stopped after" reason are not
    exercised by a test (the end-to-end stages are smoke runs). Dropping
    the break would still VOID.
  - NITs: CUDA_VISIBLE_DEVICES=-1 passed into the import subprocess
    hides a leak of that one value; no validity case has a game shorter
    than the cap; the settings refusal is untested and CENSUS_GAMES above
    30 is not refused (smoke only); after a fail-fast break the would-be
    verdict covers only the games played; stage (e) does not assert
    verdict_1b.
  - NOT YET RUN: the suite (tests/test_action_census.py, its slow test
    included) runs in the gap after C0, under the lock, before the census.
    A failure there is fixed and re-audited before the census runs.
RUN 11-02 (the gap after C0; no other python alive, host commit ~24 GB
free):
  - SUITE 00:37-00:38 EST: tests/test_action_census.py, all 38 tests
    including the slow one, passed (CPU, under the lock, 62 s).
  - CENSUS launched about 00:40 EST from the curriculum tree at 7dfba61c,
    CPU, under the lock: probe_action_census.py on Test B2's T2 iteration
    42 snapshot (Y:\SwitchPlayer-testb2-run). The A3 reference row was
    found before anything played: deployed 1699.0 over 30 games.

### RESULT -- Test B2: two swaps in 100k env steps at the constant discount, three generations T0-T2 (seed 0), 2026-10-30

(Pre-registration: "Next experiment (pre-registered) -- Test B2", below
Test B's RESULT; the interim notes per generation are there.)
RUN (from the run worktree Y:\SwitchPlayer-testb2-run, detached at
a8c83a58, manifest from the audited c57d2a16): launched 14:03 10-29, T2's
last evals 17:13 10-30 (27.2 h). Each generation ran in one attempt with
no deaths. Every supervisor.err is empty, the hand-off raised no ALERT,
and no eval game was capped. Both master gates passed. Training steps per
generation were 94k / 112k / 104k, 310k in all, including the 2 x 20k
offline steps.
EXACT eval (30 noop starts, prior argmax). Every score is the unclipped
score of a whole game. DEPLOYED is primary; trainer-BN is secondary. cum =
the lineage's cumulative env steps, in thousands.
       row     0     5    10    15    20    25    30    35    40    45  final
  T0   cum     -   3.2   6.3   9.9  13.9  17.6  21.4  25.1  28.6  32.0  33.6 (47)
       dep     -   400   601   714   690   706   768   701   705   746  1062
       tBN     -   181   614   701  1184   731   933   743   739   754   759
  T1   cum  33.6  37.0  40.2  44.1  47.5  51.1  55.0  58.7  62.6  66.4  67.1 (46)
       dep   892   730   656   850   901  1045  1849  1123  1668  1406  1314
       tBN   719   696   697   766  1036  1175  1156  1750  1463  1474  1259
  T2   cum  67.1  70.6  74.7  78.6  82.9  86.5  90.4  94.4  98.4     - 100.1 (42)
       dep  1559  1517  1468  1457  1482  1528  1539  1592  1688     -  1699
       tBN  1207  1703  1392  1654  1421  1515  1490  1587  1681     -  1663
  The offline phases' checkpoints 1-4, deployed / trainer-BN: T1 710 / 633,
  706 / 736, 740 / 834, 745 / 898; T2 1655 / 1237, 1406 / 1461, 1391 /
  1296, 1431 / 1540.
READ (1a) PRIMARY at 100k: T2's rows in [90k, 100.1k] are rows 30 / 35 /
  40 / 42 (n = 4): 1539.3 / 1592.0 / 1687.7 / 1699.0. Their mean is 1629.5
  deployed (g 1.358), against Test B's F4 window of 1015.9 (n = 4): +613.6
  -> BETTER (band +-306.7). Trainer-BN is 1605.2 against 1155.6: +449.7 ->
  BETTER (band +-306.7).
READ (1b) against the current lineage (DS2 rows 0 / 5 / 10, 1163.8):
  +465.7 -> BETTER (band +-331.3). Trainer-BN 1605.2 against 1259.9:
  +345.3 -> BETTER (band +-331.3, a margin of 14). References, not
  verdicts: EfficientZero 1281.2, BBF 2008.3, EZ-V2 2251.0. B2's window is
  above EfficientZero and below BBF and EZ-V2 (one seed).
READ (2) per swap: transfer is row 0 against the master's final row; the
  jump is the student's rows 5 and 10 against the master's last two rows;
  the rate is the student's OLS deployed rate per 1k env steps.
      swap      transfer   jump   student rate
      T0 -> T1      -169   -211          +24.5
      T1 -> T2      +245   +133           +5.2
  T0's own rate was +11.6. T0's final row (1062) sits +316 above its row
  45, so the first swap's transfer and jump are measured against a high
  single row; against T0's rows 30-45 (730), T1's row 0 is +162.
READ (3) the clip, the discount, the vocabulary and the swap data:
      gen  clip   discount  tokens  reward_kept   set records  lives by temp
      T0   1      0.98805       36  0.49 -> 0.79            -  72 / 33 / 36
      T1   7.26   0.98805       22  0.84-0.93           7,174  72 / 33 / 33
      T2   17.40  0.98805       18  0.84-0.89           6,564            -
  The set records column is the distillation set built from that
  generation's master's log (141 / 138 lives over 33,616 / 33,436 base
  steps). A1's prefill cap evicted the oldest 13 lives (2,942 base steps)
  and 12 lives (2,831), so the students started from 128 / 126 master
  lives (31,126 / 31,079 base steps). "Lives by temp" counts that
  generation's own logged lives at temperature 0.5 / 0.25 / 0.125; T2's
  log was never read, since no swap followed it. The discount was kept at
  every swap (rule "keep"). Both value checks passed: the largest
  discounted return was 27.1 / 46.3 units, against a 5,000-unit support.
READ (4) the prior: policy_maxp is 0.59-0.80 in T0 (0.64 at row 5, 0.59
  at its final row), 0.64-0.72 in T1 and 0.63-0.70 in T2. The search
  targets' entropy is 1.00-1.45 / 1.14-1.76 / 0.87-1.74; the 0.87 is T2's
  final row alone. There is no collapse.
READ (5) mechanism: FIXED at every probed row. p1_ratio at row 0 / final:
  T0 - / 0.31, T1 0.64 / 0.30, T2 0.78 / 0.29.
READ (6) stability: the largest adjacent deployed drop within each
  generation (row 0 included) is -67 (T0, rows 30 -> 35) / -725 (T1, rows
  30 -> 35) / -49 (T2, rows 5 -> 10). T1's -725 is outside the one-row
  band (+-464). It follows a +803 rise into row 30 (1849), and trainer-BN
  rose over the same pair (1156 -> 1750), so row 30 is a single high row,
  not a collapse. Across the swaps, row 0 against the master's final row
  is -169 / +245. T2 never dropped by more than 49 and rose at every row
  from row 15 on.
READ (7) means per 20k window of cumulative env steps (every in-run row
  whose cumulative env step falls in the window, the students' row 0
  included; n per B2 window 5 / 7 / 6 / 7 / 6):
      window              0-20k  20-40k  40-60k  60-80k  80-100k
      Test B2 deployed      622     801    1071    1484    1588
      Test B deployed       503     881     845    1003    1023
      current deployed      540     896    1107       -    1164 (91-98k)
      Test B2 trainer-BN    682     763    1097    1450    1559
      Test B trainer-BN     660     812     934     951    1106
      current trainer-BN    524     790    1147       -    1260
READ (8) T0 against G0 over the first 33.6k env steps (rows 5-47 against
  G0's rows 5-50, n = 10 each): deployed 709.2 against 665.8 (+43.4,
  inside the band of +-194.0); trainer-BN 734.0 against 632.1. The shapes
  differ. G0 rose steadily (162 @2.8k -> 740 @19.8k -> 875 @32.9k). T0
  reached 714 by 9.9k and then stayed at 690-768 until its final row
  (1062), with its schedules compressed to 33k.
PRE-REGISTERED DECISION: (1a) BETTER -> fewer, longer generations beat
  four swaps; the generation length becomes an input to Test D's trigger
  (a minimum length). CAVEAT (pre-registered): against Test B two things
  changed, the swap count (4 -> 2) and the discount (Test B's ladder ->
  the constant 0.98805), so (1a) is on the pair. Kai's call (10-30) is
  already made: the Test C series (BBF's proposed changes) runs first,
  starting with C and its no-reset twin C0. Then, if B2 holds up, a
  multi-seed swap-count curve (0-4 swaps at one constant discount), which
  separates the swap count from the discount. C0 (T0's recipe to 100k,
  no swap) gives the first 0-swap seed.
Beyond the letter (one seed, within-run bands, so "consistent with"):
  - Longer generations improve within themselves. T1 rose from 892 at
    row 0 to 1463 over rows 40-46 (+24.5 per 1k env steps), and T2 rose
    steadily from row 15 to its end (1457 -> 1699). Test B's generations
    after F0 were flat or falling (+2.1 / +1.5 / -8.2 / -2.1).
  - Where the gains came: across the swaps, the jumps are -211 / +133.
    Within the generations, row 0 to the final row is +422 (T1) and +140
    (T2). Test B was the reverse: its gains came at the swaps.
  - The gap to Test B opens at 40-60k (1071 against 845) and widens to
    +481 at 60-80k and +565 at 80-100k. Against the current lineage, B2
    is level through 40-60k (1071 against 1107) and 424 higher in the last
    window (1588 against 1164), at no recording cost.
  - Decision speed (frames per decision in the in-run evals): T0 22.5-25.4,
    T1 19.7-23.4, T2 18.9-21.8, so about 2.4-3.2 decisions per second at
    60 fps from the start, like G0 (20.9-23.0 at rows 60-90). Test B's F0
    at discount 0.97 was at 6.8, and its tokens lengthened only over the
    generations. The vocabulary shrank 36 -> 22 -> 18 (Test B: 36 -> 18
    -> 16 -> 16 -> 16).
  - The distillation sets: 7,174 / 6,564 records from about 33.5k base
    steps each, against Test B's 10,418 -> 4,671 from about 20k each.
  - T0 against Test B's F0 (0-20k: 622 against 503) also differs in the
    discount (0.98805 against 0.97), so part of B2's lead over Test B
    may come from the discount itself. The swap-count curve at one
    discount separates the two.
  - Deployed and trainer-BN are level in B2's final window (1629.5
    against 1605.2).

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
AUDIT r3 (fresh eyes on dd5a77af, static): nothing affecting the runs.
  - A-MEDIUM: a TEST bug. The double-buffer test's 40,000 case could never
    pass (from the helper's sync at 39,900 no sync is due at 40,000), so
    the suite would fail red and block the launch's preflight.
  - B-MEDIUM: the smoke's margins (cutoff 6,000, stop 9,000) failed a
    correct build at a mean pace of 750 env steps per iteration or more.
  - LOW: no test of the current build has run yet (memory below the
    floor all afternoon); the supervisor checked resets only at row
    resolution (every 5th iteration).
  - NITs: a docstring still crediting BBF for the target reset; test
    holds without "from"; unequal shrink / perturb factors untested; the
    scorer's read (7) label, its env filter and read (2)'s other lineages;
    the smoke invisible to the supervisors' trainer guard; the hand-off's
    gave-up text.
FIXES r3 (a fresh-eyes r4 follows):
  - the test: the 40,000 case steps the stamp back to 39,800 first, as
    the run's last in-training sync, so the refresh's sync really
    snapshots the pre-reset nets before the reset;
  - the smoke: cutoff 8,000, stop 13,000 (two resets up to a mean of
    1,000 env steps per iteration; the cutoff bites while 4 iterations
    play under 5,000);
  - the supervisor reads the trainer's RESET lines from each attempt's
    log, echoes them to its log and ALERTs any reset off its exact
    iteration, step or cutoff (ResetLines, reset_line_alert); the hand-off
    echoes them;
  - the NITs above (a factor test at 0.8 / 0.2; the scorer prints Test
    B's, Test B2's and the current lineage's 20k windows beside C's and
    C0's).
AUDIT r4 (fresh eyes on c13ced2d, static; both auditors traced every Test C
test, the hand-off checker and the smoke by hand): nothing above LOW --
THE AUDIT LOOP IS CLOSED. By the traces:
  - all 71 test_testc items and the 12 new and 4 edited supervisor tests
    pass, none vacuous;
  - check_post_testc passes against the regenerated hand-off;
  - all three smoke steps pass at the measured pace;
  - no other suite breaks.
ACCEPTED r4 (no edit, so no further round):
  - RESET lines are echoed twice after a supervisor restart, because the
    attempt log is appended to and its read position starts at 0. Valid
    lines raise no ALERT.
  - An orphan trainer's RESET lines are not read. Its rows are still
    checked by reset_alert.
  - Coverage the GPU smoke carries instead of the suites:
      - no CPU end-to-end run_one test of a reset and its resume;
      - no main() test of r0 through an attempt;
      - the supervisor's RESET regex is not run on the smoke's real
        trainer lines.
  - The dormant test's dead-unit check is weak, and the seed half of the
    fresh draw's key is untested.
  - The smoke's margin ends at a mean pace of 1,000 env steps per
    iteration (measured: 500-690).
  - NITs: wording (docstrings, the smoke's "lost row" comment), the
    scorer's edge cases (a torn final line, a reset row without an eval),
    the smoke's relative out dir, check_post_testc not covering the RESET
    echo.
  - None of this has run yet. Host commit stayed below the 8 GB test
    floor all afternoon while T0 trains: the trainer holds about 7.8 GB.
    The suites (with C's preflight), check_post_testc and the GPU smoke
    run after Test B2, before the launch. A failure there is fixed and
    re-audited before any launch.
UPDATE (10-29 21:55-21:57, while host memory was free between T0 and
T1): test_testc, test_supervise_testb, test_pbn and test_harness all
pass on 61d4197a. check_post_testc prints CHECK OK. Still to run before
the launch: all 25 suites and the GPU smoke.
KAI (10-30, on the swap-count controls): "We can hold off on the single
swap run for now. If B2 holds up as the better algorithm, we'll want to
run repeat tests for all the swap values (0-4) to make a graph and prove
that this is an actual improvement. But that can hold off until all of
the C tests with BBF proposed changes are tested."
  - The order: the Test C series comes first, starting with C (resets
    alone) and its twin C0. BBF's other proposed changes follow as their
    own pre-registered tests. After that, if B2 holds up: the swap-count
    curve.
  - The swap-count curve: 0, 1, 2, 3 and 4 swaps in 100k env steps, each
    at several seeds, plotted as the score against the swap count. Every
    point uses one recipe at one constant discount. Test B (4 swaps) also
    changed the discount per generation, so its point is rerun.
  - C0 (T0's recipe to 100k env steps, no swap) is the 0-swap point's
    first seed. It is also the 0-swap comparator for B2 (B2's [90k, end]
    window against C0's), but only as a single-seed read.
  - C and C0 keep the built order (C first, then C0): no change to the
    audited supervisor.
PRE-LAUNCH (10-30, after Test B2's RESULT, nothing else running): the 25
suites all pass in the build tree (17:17-17:32). The GPU smoke
(smoke_testc.py, 20 gradient steps per iteration, a reset every 4
iterations, a 2-iteration hold, the cutoff at 8,000 env steps, the stop
at 13,000) passes every step:
  - r0: resets after iterations 4, 8 and 12 (2,562 / 5,290 / 7,521 env
    steps); the cutoff stops the fourth; the stop at iteration 23. Every
    row's resets and held ships pass the supervisor's reset_alert, and the
    RESET lines match.
  - kill: killed right after iteration 4's checkpoint (reset 1's own step,
    the hold open); the resume did not repeat reset 1, and resets 2 and 3
    followed after iterations 8 and 12.
  - n0: the control runs clean (the plasticity read, no reset fields, no
    RESET line).
LAUNCHED 18:17 10-30 from the run worktree Y:\SwitchPlayer-testc-run,
detached at d682fa5e: testb_manifest.json rewritten from the audited
build (2a51e527; seven AUDITED files changed against Test B2's, and
test_testc.py added). In that tree, before the launch: the manifest check
and the quick probes of r0 and n0 passed, both with a 100,000 env-step
stop. launch_testc.ps1 passed its checks and started the hand-off
post_testc.py and the supervisors r0 and n0. r0's preflight (manifest,
quick probe, the 25 suites in the run tree) passed. C LAUNCHED 18:29
(attempt 1, fresh). n0 waits for C's completion and evals. Expected: about
140-150 iterations and about 28 h per run. Reset 1 is due after
iteration 20.
C ROWS 5-20 AND RESET 1 (10-30 21:27). 8.9-9.6 min per iteration. Rows
5-20 come before any reset, so they are the C0 recipe on C's seed (no
twin read yet). Exact evals, 30 games (single rows; the band is +-464):
  row  env     deployed  trainer-BN
   5    2,571    473.3     340.7
  10    5,642    481.3     498.7
  15    8,633    741.0     544.0
  20   11,911    872.7     984.0
  Beside B2's T0 (same recipe; its schedules keyed to its own 33.6k):
  399.7/181.3, 600.7/614.3, 714.3/701.0, 689.7/1183.7 at 3,225-13,875 env
  steps.
RESET 1 after iteration 20 (step 40,000, 11,911 env steps), as scheduled.
The trainer's line: "representation+dynamics shrunk x0.5 + 0.5 x fresh;
prediction+value_prefix+projector re-initialized; target reset; warmup
restarts; the actor holds the pre-reset weights until step 50000". No
ALERT; the hand-off's row-20 evals ran on the pre-reset learner.
READ (4) so far: the dormant fraction at <= 0.025 per net (at <= 0.1, all
within 0.02), the representation's weak first-layer filters, and erank:
  row  repr  dyn   pred  vprefix  proj  all   conv1 weak  erank
   5   0     0     .177  0        0     .010  0           18.9
  10   .298  .219  .260  0        .398  .359  0           17.9
  15   .269  .207  .260  0        .605  .504  .563        16.0
  20   .271  .207  .260  0        .644  .532  .531         5.4
  - Dormancy set in between rows 5 and 10 (2.6k-5.6k env steps) in the
    representation, dynamics and prediction nets, then stayed flat
    (21-30%). value_prefix has none.
  - The projector kept rising (40% to 64%) as its erank fell (18.9 to
    5.4): the projector collapse seen in every run by iteration ~20 (B2's
    T0: erank 33.9 at row 10, 13.7 at row 20). The overall fraction is
    mostly the projector's.
  - Over half of the first-layer filters fell under 10% of a fresh draw's
    median norm between rows 10 and 15 (0 to 56%). On Pong, the faithful
    runs had lost up to half by iteration 85. Under SGD at lr 0.2,
    momentum 0.9 and coupled weight decay 1e-4, the decay's steady rate is
    0.2 x 1e-4 / (1 - 0.9) = 2e-4 per step, so a filter the gradients stop
    supporting falls to about e^-2 = 14% of itself in 10k steps (5
    iterations). The jump fits filters the gradients abandoned between
    rows 10 and 15.
  - Row 25, the hold's end, is the first read on what the reset does to
    these numbers.
PAUSED 22:02 10-30 (Kai asked, 2 h), in iteration 23: supervise_testb.py r0
and its trainer killed by exact PID; the Ray processes exited with the
trainer. The hand-off and n0 stayed up (n0's expected "R0 is unfinished
and nothing drives it" ALERT at 22:18). RESUMED: launch_testc.ps1 rerun at
00:03 10-31 started only r0; preflight ok; attempt 1 (resume) after
iteration 22 (12,958 env steps, 44,000 train steps). Iteration 23's
partial work was lost; reset 1 was not repeated.
ROW 25 (15,003 env steps; the hold's end, 4 held ships; the evals take
the learner's weights): deployed 1037.7, trainer-BN 1006.7.
  - Read (3), cycle 1: the dip, row 25 against row 20, is +165.0 deployed
    and +22.7 trainer-BN, so there is no dip (single rows, +-464). The
    recovery is immediate: row 25 is already at or above row 20.
  - Read (4): every net's dormant fraction is 0 at <= 0.025 (the
    projector's is 0.0004 at <= 0.1); weak first-layer filters 0; erank
    46.0 (5.4 at row 20).
  - Read (6): probe_prefix_bn FIXED at rows 20 and 25 (p1_ratio 0.40 and
    0.77).
  - Next: whether the dormancy and the filter loss come back within the
    cycle (rows 30-40), and reset 2 after iteration 40.
ROW 30 (18,513 env steps): deployed 917.3, trainer-BN 949.7 (row 25:
1037.7 / 1006.7; single rows, +-464). Read (4): the projector is dormant
again (0.293 at <= 0.025; erank 13.5, from 46.0), but representation,
dynamics, prediction and value_prefix are all still 0, and no first-layer
filter is weak. From scratch, those three nets were 21-30% dormant by row
10 (20k gradient steps); here, 20k steps after the reset, they are at 0.
ROWS 35-40 AND RESET 2 (10-31 03:10). ~10 min per iteration.
  row  env     deployed  trainer-BN  dormant repr/dyn/proj  all   erank
  35   21,815    725.7     882.3      .002/0/.442           .320  22.5
  40   25,142    959.0     861.0      .006/.004/.505        .367  24.8
  (prediction and value_prefix 0, no weak filter, at both rows.)
  - Cycle 1 in full (rows 25-40, deployed 1037.7 / 917.3 / 725.7 /
    959.0): read (3)'s OLS rate is -12.9 deployed per 1k env steps
    (trainer-BN -15.0). The cycle held its post-reset level but did not
    climb; row 35's 725.7 is the low point (-311.9 from row 25, inside
    the band). C0's rate over the same rows comes with C0.
  - Read (4): through the whole cycle the trunk stayed under 1% dormant
    and no filter went weak. Only the projector re-collapsed (to 51%),
    with its erank recovering to 22-25 after row 30's 13.5.
RESET 2 after iteration 40 (step 80,000, 25,142 env steps), as scheduled,
with the same trainer line; the hold runs to step 90,000 (row 45).
ROW 45 (28,986 env steps; 8 held ships in all): deployed 1167.3,
trainer-BN 1076.3. Read (3), cycle 2: the dip against row 40 is +208.3
deployed and +215.3 trainer-BN, so again no dip and an immediate
recovery. Read (4): dormant 0.0003 overall (prediction 0.005, the rest
0), no weak filter, erank 53.2.
ROWS 50-60 AND RESET 3 (10-31 06:32).
  row  env     deployed  trainer-BN  dormant repr/dyn/pred/proj  all   conv1 weak  erank
  50   32,343   1172.3    1322.7     .002/0/.005/.357           .259  .031         9.7
  55   36,069   1115.0     926.7     .008/0/.005/.423           .308  .031        11.5
  60   40,535    924.7    1039.0     .013/.004/.005/.475        .346  .156        22.9
  - Cycle 2 in full (rows 45-60, deployed 1167.3 / 1172.3 / 1115.0 /
    924.7): OLS -21.1 deployed per 1k env steps (trainer-BN -13.4). The
    cycle's mean, 1094.8 deployed, is 184.9 above cycle 1's 909.9: the
    level stepped up at reset 2 (row 40 to 45, +208.3), then drifted down
    within the cycle, as in cycle 1 (-12.9). So far the gains come at the
    resets, as Test B's came at its swaps. These are single rows, and C0's
    rates over the same rows decide it.
  - Read (4): the trunk stayed near clean (representation up to 1.3%),
    the projector re-collapsed to 47.5%, and, for the first time since a
    reset, first-layer filters went weak: 3% at rows 50-55, 16% at row 60.
RESET 3 after iteration 60 (step 120,000, 40,535 env steps), as
scheduled; the hold runs to row 65.
ROW 65 (44,398 env steps; 12 held ships in all): deployed 1006.7,
trainer-BN 1133.0. Read (3), cycle 3: the dip against row 60 is +82.0
deployed and +94.0 trainer-BN, a third reset with no dip. Read (4): every
net 0 dormant, no weak filter (16% at row 60), erank 68.7.
ROWS 70-80 AND RESET 4 (10-31 09:57).
  row  env     deployed  trainer-BN  dormant dyn/proj  all   erank
  70   48,209   1164.7    1172.3     0/.371            .268  10.6
  75   52,499   1185.7     985.0     .004/.418         .303  17.9
  80   56,313   1052.3    1171.3     .004/.461         .334  21.6
  (representation, prediction and value_prefix 0; no weak filter at any
  of the three rows.)
  - Cycle 3 in full (rows 65-80, deployed 1006.7 / 1164.7 / 1185.7 /
    1052.3): OLS +4.0 deployed per 1k env steps (trainer-BN -2.2), mean
    1102.3, against cycle 2's 1094.8 and cycle 1's 909.9. Cycle 3 is flat
    and does not drift down, but the level did not step up at reset 3.
    The cycle means so far: 910, 1095, 1102.
  - Read (4): the trunk under 0.5% dormant all cycle, no weak filter, the
    projector re-collapsed to 46%.
RESET 4 after iteration 80 (step 160,000, 56,313 env steps), as
scheduled. At ~3.8k env steps per 5 iterations, reset 5 (after iteration
100) should fall near 71k env steps, inside the 80k limit. Iteration 120
should be past 80k (~86k), so it would get none: 5 resets in all, not the
pre-registration's "about 6".
ROW 85 (60,666 env steps; 16 held ships in all): deployed 1206.7,
trainer-BN 1173.0. Read (3), cycle 4: the dip against row 80 is +154.3
deployed and +1.7 trainer-BN, a fourth reset with no dip. Read (4): every
net 0 dormant, no weak filter, erank 33.3.
PAUSED AGAIN 11:36 10-31 (Kai asked, 2 h), in iteration 89: the r0
supervisor and its trainer were killed by exact PID; Ray exited with them.
Kai rebooted at 11:45, which also ended the hand-off and n0's supervisor.
RESUMED: launch_testc.ps1 at 14:27 restarted all three; preflight ok
(manifest + quick probe + 25 suites); attempt 1 (resume) at 14:39 after
iteration 88 (63,110 env steps, 176,000 train steps). Iteration 89's
partial work was lost. The relaunched supervisor re-logs its old RESET
lines; that is cosmetic (the scorer does not parse them, and the hand-off
skips lines it has logged).
ROWS 90-95 AND RESET 5 (10-31 16:46).
  row  env     deployed  trainer-BN  dormant repr/dyn/pred/proj  all   erank
  90   64,420   1051.7    1045.7     0/0/.005/.384              .278  15.3
  95   68,239   1095.0    1026.3     .002/.004/.010/.419        .305  17.7
  (value_prefix 0, no weak filter.)
  - Row 100's exact evals (the pre-reset net) have not run yet. The
    hand-off starts an interim probe only with >= 6 GB of host commit
    free, and commit has stayed at 4.7-5.4 GB since about 16:45, so it
    logs an hourly "probes waiting for memory" ALERT. The trainer is
    steady at 7.9 GB private (11.5 GB peak); Kai's open applications hold
    about 3.5 GB. The row waits; the final pass (>= 4 GB, after the
    trainer exits) runs every missing row anyway.
  - Read (4) at row 100 (training side): dynamics 1.6% dormant, the
    projector 44%, representation and prediction 1% or under, and the
    first weak filters of cycle 4 (9.4%). As in cycles 2 and 3, the trunk
    stays nearly clean, the projector re-collapses, and weak filters come
    late in the cycle.
RESET 5 after iteration 100 (step 200,000, 72,293 env steps), as
scheduled. It should be the last: iteration 120 falls past 80k env steps.
Row 105 (76,202 env steps, 20 held ships, the hold's end), training side:
every net 0 dormant (prediction 0.5%), no weak filter, erank 44.7, 0
non-finite steps. Its exact evals wait behind row 100's.
C COMPLETE 22:38 10-31: iteration 134, 100,849 env steps, rc=0; 5 resets
(after iterations 20 / 40 / 60 / 80 / 100), none after iteration 120
(88,447 env steps, past the 80k limit); 20 held ships. Host commit fell to
2.4-3.3 GB in dips over its last 4 h (the trainer grew to 10 GB private;
the supervisor's advisory "memory creep?" ALERT at row 130), so every row
from 100 on was scored in the final pass (22:38-23:06; 27 rows each, plus
the 5 read (6) rows). 0 non-finite steps in every row.
ROWS 100-134 (score_testc.py, 23:06):
  row  env      deployed  trainer-BN  dormant repr/dyn/pred/proj   conv1 weak  erank
  100   72,293    983.7    1167.3     .002/.016/.010/.443          .094        15.4
  105   76,202   1212.0    1076.0     0/0/.005/0                   0           44.7
  110   79,888   1057.0    1231.7     .002/0/.005/.377             .031        43.8
  115   83,844   1372.7    1098.7     .002/0/.005/.475             .125        43.2
  120   88,447   1511.7    1293.0     .004/.020/.005/.557          .094        49.3
  125   92,607   1317.7    1275.3     .029/.055/.016/.597          .125        38.4
  130   97,314   1571.7    1245.0     .038/.070/.016/.614          .156        33.3
  134  100,849   1343.7    1554.0     .044/.094/.021/.625          .125        36.2
  (value_prefix 0 at every row.)
C'S READS (C0 is not run yet, so read (1) and C0's halves wait):
  (1) C's [90k, end] window, rows 125 / 130 / 134 (n = 3): deployed
      1411.0, trainer-BN 1358.1. Against C0's, when C0 is scored.
  (2) deployed per 20k window of env steps (n):
      window              0-20k   20-40k   40-60k   60-80k   80-end
      C                   754 (6) 1028 (5) 1067 (5) 1101 (6) 1423 (5)
      Test B2             622     801      1071     1484     1588
      Test B              503     881      845      1003     1023
      current lineage     540     896      1107     -        1164 (91-98k)
  (3) per reset: dip = row 25 of the cycle against row 20; the cycle rate
      is OLS deployed per 1k env steps over rows 25-40 of the cycle.
      reset  rows     deployed       dip     trainer-BN dip  cycle rate
      1      20->25    873 -> 1038   +165     +23            -12.9
      2      40->45    959 -> 1167   +208    +215            -21.1
      3      60->65    925 -> 1007    +82     +94             +4.0
      4      80->85   1052 -> 1207   +154      +2            -16.2
      5     100->105   984 -> 1212   +228     -91            +30.4
      Every reset recovered by its first row (row 25 at or above row 20,
      deployed). Cycle means, deployed: 910 / 1095 / 1102 / 1084 / 1341.
      The last is rows 105-134, 30 iterations with no further reset.
  (4) the trunk (representation, dynamics, prediction) was 21-30%
      dormant at rows 10-20 from scratch. After reset 1 it stayed under
      2% through every 20-iteration cycle. Only the final 30-iteration
      stretch climbed past that: dynamics 9.4% and representation 4.4% at
      row 134. The projector re-collapsed within 5-10 iterations after
      every reset (to 44-63%). Weak first-layer filters: 53-56% at rows
      15-20, 0 after each reset, back to 9-16% late in cycles 2, 4 and 5.
  (5) the prior did not collapse: policy_maxp 0.53-0.77, the search
      targets' entropy 0.86-1.91.
  (6) probe_prefix_bn FIXED at rows 20, 25, 100, 105 and 134 (p1_ratio
      0.40 / 0.77 / 1.11 / 1.04 / 0.53).
  (7) the largest adjacent deployed drop is -228 (row 130 -> 134); no
      reset produced a deployed dip (reset 5's trainer-BN -91 is inside
      the one-row band).
  Beyond the letter (C alone, one seed, not a verdict):
  - C rose over its last 17k env steps. Rows 115-134 average 1423.5,
    against 1084 for cycle 4. The rise began after the last reset, in the
    only cycle that ran past 20 iterations. C0 is needed to tell "the
    resets paid off late" from "any run climbs here at 80k+".
  - C's [90k, end] (1411.0, n 3) against the references, with read (2)'s
    band rule: Test B's 1015.9 (n 4) +395.1, band +-331.3, above; the
    current lineage's 1163.8 (n 3) +247.2, band +-354.2, inside; Test
    B2's T2 1629.5 (n 4) -218.5, band +-331.3, inside.
  - Against B2 by window: C led at 0-40k (inside the bands), was level at
    40-60k, fell 383 behind at 60-80k (n 6 against 7, band +-241), and
    ended 165 behind at 80-end (band +-263).
C0 LAUNCHED 23:15 10-31 (supervise_testb.py n0, once C was complete and
evaluated: "R0 complete and evaluated; n0 s0 may start" at 23:05); preflight
ok (manifest + quick probe + 25 suites); attempt 1 (fresh). ~10.5 min per
iteration, so C0 should end about 03:00 11-02.
C0 ROW 5 (2,625 env steps): deployed 498.7, trainer-BN 305.3 (C's row 5,
2,571 env steps: 473.3 / 340.7). Read (4): prediction 19.8% dormant, the
rest 0; no weak filter; erank 11.0; 0 non-finite steps.
C0 ROWS 10-20 (01:19 EST 11-01; the clocks went back an hour at 02:00 EDT,
so the hand-off log's times repeat 01:00-02:00).
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  5     2,625     498.7       305.3     |    473.3       340.7
  10    5,980     449.7       349.7     |    481.3       498.7
  15    9,105     402.3       328.3     |    741.0       544.0
  20   12,465     655.3       436.0     |    872.7       984.0
  - C's first reset falls after iteration 20, so rows 5-20 are the same
    recipe in both runs, and their gap is run-to-run noise: C0 501.5
    against C 642.1 deployed (n 4 each, -140.6, band +-306.7, inside).
    C0 starts 141 lower before any reset, so C's lead at 0-20k (754) is
    partly C's head start, not the resets.
  - Read (4), C0 row 20: representation 7.3%, dynamics 4.7%, prediction
    21.9% dormant (C at row 20: 27% / 21% / 26%); projector 58.6%; weak
    first-layer filters 25% (C 53%); erank 32.0. The same recipe gave
    two quite different dormancy levels by row 20, so a dormancy gap
    between the runs needs a margin this wide before it counts.
C0 ROWS 25-40 (04:49 EST 11-01).
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  25   15,875     476.3       776.0     |   1037.7      1006.7
  30   19,263     899.0       758.7     |    917.3       949.7
  35   22,933    1343.3       935.3     |    725.7       882.3
  40   26,250     699.3       731.0     |    959.0       861.0
  - Read (3), cycle 1 (rows 25-40): C0's OLS rate is +32.9 deployed per
    1k env steps (trainer-BN +1.6), against C's -12.9. The cycle means
    are 854.5 (C0) and 909.9 (C). C0's rate rests on noisy single rows:
    1343.3 at row 35, then 699.3 at row 40. That -644 drop is beyond the
    one-row band (+-464) and is C0's largest so far (read (7)).
  - Read (2), 0-20k: C0 564 (n 6) against C 754 (n 6), -190, inside the
    band (+-250). 20-40k is open.
  - Read (4): C0's trunk keeps creeping with no reset (representation
    9.4%, dynamics 6.6%, prediction 25.0% at row 40). C's stayed under
    1% in the same rows after reset 1. The projector holds at 61-63%,
    weak filters at 22-25%, erank 27.7-43.2. The prior is steady:
    policy_maxp 0.77-0.80. 0 non-finite steps.
C0 ROWS 45-60 (08:17 EST 11-01).
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  45   29,856     996.3      1103.3     |   1167.3      1076.3
  50   33,899     784.7       837.7     |   1172.3      1322.7
  55   38,167    1251.7       630.3     |   1115.0       926.7
  60   42,061    1035.3       953.0     |    924.7      1039.0
  - Read (3), cycle 2 (rows 45-60): C0's rate is +14.8 deployed per 1k
    env steps (trainer-BN -16.5), against C's -21.1. The means are 1017.0
    (C0) and 1094.8 (C).
  - Read (2): 20-40k is closed, C0 1015 (n 5) against C 1028 (n 5),
    level. 0-20k was 564 against 754. C's lead from its higher start has
    gone.
  - Read (4): C0's trunk keeps creeping (representation 13.1%, dynamics
    10.5%, prediction 25.5% at row 60), against C's 1.3% / 0.4% / 0.5% at
    its row 60. Weak filters 31% (C 16%), the projector 64%.
  - Read (5): a one-row drop in the prior's confidence at row 55
    (policy_maxp 0.43, against 0.72-0.80 at every other row), back to
    0.72 at row 60. C never went below 0.53.
  - Read (7): the largest adjacent drop is still -644 (row 35 -> 40).
C0 MEMORY, 10:00-11:45 EST 11-01: host commit free fell to 0.6-1.0 GB (the
trainer at 8.6 GB private, Kai's open applications about 7 GB). Windows
began growing the C: pagefile (16 -> 80 MB). The interim probes waited
from row 65 on, and the supervisor raised its advisory "memory creep?"
ALERT at row 65. No allocation failed.
C0 PAUSED 11:46 EST 11-01 (Kai asked, 2 h), in iteration 78: the n0
supervisor and its trainer were killed by exact PID; Ray exited with them;
the GPU is idle. The resume checkpoint is iteration 76's (11:25,
0.66 GB). The hand-off stays up and scores the settled rows (65-75) with
the memory freed. Resume: launch_testc.ps1 at about 13:46.
RESUMED: launch_testc.ps1 at 13:47 EST started the two supervisors (r0's
exited at once, RUN COMPLETE, before any preflight); n0's preflight ok;
attempt 1 (resume) at 13:58 after iteration 76 (53,842 env steps, 152,000
train steps). Rows 65-75 were scored during the pause.
C0 ROWS 65-75:
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  65   45,428    1124.0      1076.7     |   1006.7      1133.0
  70   49,041    1325.0      1019.0     |   1164.7      1172.3
  75   52,901    1143.3       942.3     |   1185.7       985.0
  - Read (2), 40-60k so far: C0 1156.9 (n 4, rows 60-75) against C's
    1067 (n 5).
  - Read (4): representation 15.0%, dynamics 12.9%, prediction 26.0%
    dormant at row 75 (C at row 75: 0 / 0.4% / 0); weak filters 37.5%;
    the projector 64%; erank 38-43; policy_maxp 0.76-0.78; 0 non-finite
    steps.
C0 ROW 80 (evaluated 15:01 EST 11-01):
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  80   57,013    1174.7      1021.3     |   1052.3      1171.3
  - Read (2), 40-60k complete: C0 1160 (n 5, rows 60-80) against C's
    1067 (n 5): +93, band 274 -> no difference.
  - Read (3), cycle 3 (rows 65-80): C0 OLS -1.0 per 1k env against C's
    +4.0. Both flat; C0's rates over C's cycles 1-3 are +32.9 / +14.8 /
    -1.0.
  - Read (4) at row 80: representation 15.8%, dynamics 12.5%, prediction
    26.0% dormant (C at row 80, before reset 4: 0 / 0.4% / 0); weak
    filters 43.8% (C 0); the projector 64% (C 46%); erank 37.7.
C0 ROWS 85-90 (row 90 evaluated 16:46 EST 11-01):
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  85   60,630    1349.0      1315.0     |   1206.7      1173.0
  90   64,702    1168.0      1500.0     |   1051.7      1045.7
  - Read (2), 60-80k so far: C0 1258.5 (n 2) against C's 1101 (n 6).
  - Read (3), cycle 4 so far (rows 85-90 only, n 2): C0 -44.4 per 1k
    env against C's -16.2 over rows 85-100. Two rows, inside the
    single-row band; not a rate yet.
  - Read (4) at row 90: representation 16.7%, dynamics 14.8%, prediction
    26.0% dormant (C at row 90, 10 iterations after reset 4: 0 / 0 /
    0.5%); weak filters 46.9% (C 0); the projector 64.7% (C 38%);
    erank 37.3 (C 15.3).
C0 ROWS 95-100 (row 100 evaluated 18:11 EST 11-01):
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  95   67,569    1124.7      1223.0     |   1095.0      1026.3
  100  71,351    1363.0      1153.3     |    983.7      1167.3
  - Read (2), 60-80k so far: C0 1251 (n 4, rows 85-100) against C's
    1101 (n 6): +150, band 280.
  - Read (3), cycle 4 (rows 85-100): C0 -0.3 per 1k env against C's
    -16.2. C0's rates over C's cycles 1-4: +32.9 / +14.8 / -1.0 / -0.3.
  - Read (4) at row 100: representation 18.1%, dynamics 16.4%,
    prediction 26.6% dormant (C at row 100, before reset 5: 0.2% / 1.6%
    / 1.0%); weak filters 46.9% (C 9.4%); the projector 65% (C 44%);
    erank 40.4 (C 15.4).
C0 ROWS 105-115 (row 115 evaluated 21:17 EST 11-01):
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  105  75,706    1293.7      1582.0     |   1212.0      1076.0
  110  79,738    1440.7      1419.7     |   1057.0      1231.7
  115  83,296    1286.7      1291.0     |   1372.7      1098.7
  - Read (2), 60-80k complete: C0 1290 (n 6, rows 85-110) against C's
    1101 (n 6): +189, band 250 -> no difference. 80k-end so far: C0
    1287 (n 1, row 115) against C's 1423 (n 5).
  - Read (3), cycle 5 so far (rows 105-115, n 3): C0 -0.1 per 1k env
    against C's +30.4 over rows 105-120.
  - Read (4) at row 115: representation 22.5%, dynamics 19.9%,
    prediction 27.1% dormant (C 0.2% / 0 / 0.5%); weak filters 46.9%
    (C 12.5%); the projector 65% (C 48%); erank 39.1 (C 43.2).
  - C0's [90k, end] rows for read (1) should be about 125, 130, 135 and
    the final row (~796 env steps per iteration over rows 100-115).
C0 ROWS 120-130 (row 130 evaluated 23:25 EST 11-01):
  row  env     C0 deployed  trainer-BN  | C deployed  trainer-BN
  120  87,101    1464.0      1241.7     |   1511.7      1293.0
  125  91,229    1210.7      1232.3     |   1317.7      1275.3
  130  95,466    1239.3      1250.0     |   1571.7      1245.0
  - Read (1), PROVISIONAL (C0 has 2 of its ~4 [90k, end] rows): C 1411.0
    (n 3) against C0 1225.0 (n 2): +186, band 396 -> no difference so
    far; trainer-BN 1358.1 against 1241.2 (+117).
  - Read (2), provisional: C0 [90k, end] against Test B 1015.9 +209 (no
    difference), the current lineage 1163.8 +61 (no difference), T2
    1629.5 -405 (band 376 -> WORSE). 80k-end so far C0 1300 (n 4)
    against C's 1423 (n 5).
  - Read (3), cycle 5 (rows 105-120): C0 +9.7 per 1k env against C's
    +30.4.
  - Read (4) at row 130: representation 26.3%, dynamics 23.8%,
    prediction 28.7% dormant (C 3.8% / 7.0% / 1.6%, 30 iterations after
    reset 5); weak filters 46.9% (C 15.6%); the projector 66% (C 61%);
    erank 36.3 (C 33.3).
C0 COMPLETE 00:32 EST 11-02 (iteration 136, 100,173 env steps, rc=0; the
hand-off's final evals and read (6) landed at 00:32, "all done"). Rows 135
and 136: deployed 1317.0 / 1282.7, trainer-BN 1402.7 / 1096.3. Final read
(1): C 1411.0 (n 3) against C0 1262.4 (n 4), +148.6, band 331.3 -> NO
DIFFERENT. The full reads are in "RESULT -- Test C" above.

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
T0 COMPLETE 21:50 10-29 (attempt 1, no deaths; 47 iterations, 94,000
gradient steps, 33,616 env steps, 7.7 h). Unclipped whole-game means,
deployed / trainer-BN, by row (env steps): 5 (3,225) 400 / 181; 10
(6,286) 601 / 614; 15 (9,909) 714 / 701; 20 (13,875) 690 / 1184; 25
(17,595) 706 / 731; 30 (21,381) 768 / 933; 35 (25,061) 701 / 743; 40
(28,606) 705 / 739; 45 (32,004) 746 / 754; 47 (33,616) 1062 / 759. The
0-20k window (rows 5-25): deployed 622, trainer-BN 682 (Test B 503 / 660,
the current lineage 540 / 524). Rows 30-47 (20-33.6k): deployed 796,
trainer-BN 786; the 20-40k window closes with T1's first rows. Row 47's
1062 is one row (the band is +-464). The hand-off's final evals (acting,
trainer-BN, read (5)) finished 22:00: all 10 rows of each.
SWAP 1 PREP 22:10: t1's preflight (manifest, quick probe, the 24 suites)
passed; MASTER T0 = iteration 47 (the deployed mean of its last two rows
904, g 0.612); PREP done on attempt 1; T1's stop 33,051 env steps (the
lineage's 66,667 less T0's 33,616). T1 LAUNCHED 22:13 10-29 (attempt 1,
fresh).
T1 ROWS 0-5 (00:49 10-30). The offline phase's checkpoints 1-4, deployed /
trainer-BN: 710 / 633, 706 / 736, 740 / 834, 745 / 898. Row 0 (the
student after distillation): 892 / 719, level with the master's 904. Row
5 (3,339 env steps): in-run return 726. Its projector erank is 4.06, the
lowest early value in these lineages. T0 went 16 -> 34 -> 7-8 (rows 25-30)
-> 20-24 at its end; G2 was 6.8 at row 10 and 32-40 from row 30 on; Test
B's F0-F4 stayed 16-59 throughout. One row of a diagnostic, not a read:
watched over T1's next rows.
T1 COMPLETE 07:45 10-30 (attempt 1, no deaths; 46 iterations, 33,436 env
steps, 9.5 h). The erank recovered: 11, 18, 19, 19, 30, 28, 33, 27, 34
(rows 10-46). Deployed / trainer-BN by row (cumulative env steps): 0
(33,616) 892 / 719; 5 (36,955) 730 / 696; 10 (40,232) 656 / 697; 15
(44,119) 850 / 766; 20 (47,527) 901 / 1036; 25 (51,139) 1045 / 1175; 30
(54,951) 1849 / 1156; 35 (58,743) 1123 / 1750; 40 (62,640) 1668 / 1463;
45 (66,396) 1406 / 1474; 46 (67,052) 1314 / 1259. Within T1, row 0's 892
-> rows 40-46's 1463 deployed: a rise inside the generation, unlike Test
B's flat-to-negative within-generation rates after F0. The 20k windows so
far, deployed (n) / trainer-BN: 0-20k 622 (5) / 682; 20-40k 801 (7) / 763;
40-60k 1071 (6) / 1097; 60-80k 1462 (3, open until T2's rows) / 1399.
Test B's: 503 / 881 / 845 / 1003 / 1023; the current lineage's: 540 / 896
/ 1107 / - / 1164. Read (5) at row 46: p1 0.225 eval vs 0.750 trainer-BN.
No read before the RESULT; the primary read is the [90k, end] window.
SWAP 2 PREP 08:04: t2's preflight passed; MASTER T1 = iteration 46 (the
deployed mean of its last two rows 1360, g 1.081); PREP done on attempt 1;
T2's stop 32,948 env steps (100,000 less 67,052). T2 LAUNCHED 08:07 10-30
(attempt 1, fresh).
T2 ROWS 0-5 (10:45 10-30). The offline phase's checkpoints 1-4, deployed /
trainer-BN: 1655 / 1237, 1406 / 1461, 1391 / 1296, 1431 / 1540. Row 0 (the
student after distillation, 67,052 cumulative env steps): 1559 / 1207,
above the master's 1360. Row 5 (70,573): in-run return 1716; the projector
erank 29.7 (no early dip this time; T1's was 4.06).

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
