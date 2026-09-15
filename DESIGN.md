# DESIGN notes

## vtree v2: communal chain trees (`muzero_vtree.VectorizedCommunalPUCT`)

Goal: the refresh topology of the repo's best run (archfix, +17.1 Pong,
arm `effzero-repro-refresh5x-communal`) -- `create_tree_from_game` +
`run_communal_mcts` -- computed by the vectorized engine, so the fast
driver can refresh with archfix's *target semantics*. The acceptance
criterion is bitwise parity with the sequential engine
(`tests/test_vtree_communal.py`), and the sequential engine's order of
operations was reproduced rather than approximated. Everything below is
what that required.

### Tensor layout (`ChainForest`)

One flat node table per forest (any number of trajectories; the
integration groups `muzero.VTREE_COMMUNAL_TRAJS` per forest, default 1
-- see Speed). Chain nodes of tree *t* sit at ids
`base[t] .. base[t]+L-1` in trajectory order, pre-linked by the behaviour
actions (`children[k, a_k] = k+1`, `beh[k] = a_k`); imagined nodes are
appended by expansion. Per node: `parent`, `children[A]`, `tree`, `cidx`
(chain index or -1), `owner` (chain index whose imagined subtree holds
the node; -1 = dead slot), `visit`, `wsum`, `reward` (edge INTO the node),
`prior`, `policy[A]`, `evalue`, `expanded`, `done`, `pref`/`pref_len`/
`has_state` (value prefix), `vl` (in-round virtual loss), `latent[C,H,W]`
float32, and the LSTM state `pref_h/pref_c[L,N,H]`. Every scalar
statistic is stored as **float64 value + NumPy type code** (see
"type-faithful arithmetic").

Seeding (`append_chain`): the Monte-Carlo backup is folded in Python
with the SAME scalar objects the sequential engine folds (`0.0` leaf at
a terminal; `expanded_value or 0.0` -- an np.float32, unless it is
exactly 0 -- at a truncated tail; `bv = reward + DISCOUNT * bv`), so
every chain node's value and type are reproduced by construction. One
visit per chain node = the bookkeeping visit the readout subtracts.

Expansion (`_expand`) creates all A children of a batch of parents in
one dynamics/prediction/value-prefix forward with `repeat_interleave`
row order; a chain parent keeps its gameplay child and the slot the
network computed for that action becomes a dead node (`owner=-1`,
dropped at release). Chain nodes never carry LSTM state, so expanding
one is always an `is_reset` boundary (edge reward = fresh prefix),
exactly as `batch_expand_nodes` skips gameplay children and chain nodes
keep `prefix_state=None`.

Release (`ChainForest.release`) is `release_span`: after a window with
last root *e*, keep chain nodes (their statistics are communal across
windows) and imagined nodes with `owner > e` (the next window's reuse),
compact the table, remap links. Memory per trajectory stays at chain +
carry + `W*(sims+1)*A` slots instead of the whole trajectory's imagined
nodes.

### The intra-round order that was reproduced

`run_communal_mcts` gathers one `selection_task_no_expand` per root and
then drains `expand_selected`. The selection coroutines never suspend
(uncontended `asyncio.Lock`, unbounded `Queue.put`), so `gather` runs them
**to completion in index order**. Consequences that decide targets:

1. Within a round, root *j*'s descent sees the `virtual_loss += 1` that
   roots `< j` deposited on shared nodes -- root *j-k* standing on root
   *j*'s depth-*d* node at its own depth *d+k* (it walked *k* gameplay
   edges along the chain). Virtual loss enters `total_visits`, the
   `u` denominator, and the `min(score, score*n/(n+vl))` clamp. Roots
   never see deposits from later roots, and never see same-round backups.
2. The drain backs up every path in root order; visit/value increments on
   a shared node land in root order and each root's own `MinMaxStats`
   update reads `q` right after ITS increment.
3. Expansion batches the unique unexpanded, non-terminal leaves in root
   order (first occurrence), one chunk when <= 40 leaves.
4. Noise draws (`np.random.dirichlet`) are consumed per root in index
   order; `seed_min_max_stats` iterates children in dict order (gameplay
   child first, then imagined by action); `random.choice` tie-breaks use
   the same dict order.

A caveat the parity gate stands on (audit round 1): this ordering is a
property of asyncio internals -- `Lock.acquire` on an uncontended lock
and `Queue.put` on an unbounded queue return without suspending. Any
future `await` inside `selection_task_no_expand` that does suspend
would interleave the roots' descents and silently change archfix's
target semantics; the gate (`tests/test_vtree_communal.py`) would fail
on the multi-root scenarios and is the alarm.

**The ordering rule** (`_ready`) reproduces (1) without serializing
roots. Let the window's roots of one tree sit at chain indices
`i_0 < i_1 < ...`. Root *k* can only reach root *j*'s region (the subtree
of chain node `i_j`) by walking `i_j - i_k` gameplay edges first, so it
stands on root *j*'s depth-*d* node at its own depth `d + (i_j - i_k)`;
that is the step at which it deposits on the child root *j* will score,
and the step at which it reads the child root *j* would deposit on. Root
*j* at depth *d* may therefore step in a wave iff for every earlier
same-tree root *k*: *k* has stopped, or *k* left the chain at an index
`< i_j` (its imagined subtree is disjoint from root *j*'s region -- a
root that left at an index `>= i_j` is INSIDE the region and still
counts), or `depth_k > d + (i_j - i_k)` (it has passed that step).
Readiness is evaluated from the state at the start of the wave. Two
roots stepping in one wave never touch the same node: *k* on *j*'s
depth-*d* node would be at depth exactly `d + (i_j - i_k)`, which the
rule excludes. Deposits from later roots never reach earlier roots'
reads for the same reason.

The **static chain-offset stagger** -- root *j* takes its depth-*d* step
at wave `d + 2(i_j - i_0)` -- is a special case: root *k*'s depth
`d + (i_j - i_k)` step ran at wave `d + 2 i_j - i_k - i_0 - ... =
d + 2(i_j - i_0) - (i_j - i_k)`, earlier by exactly the chain distance.
It bounds the wave count (`2(i_last - i_0) + MAX_DEPTH`) and is pinned
as exact by its own test. The original version staggered by WINDOW SLOT
(root *j* at wave `d + 2j`), which is only correct for contiguous
windows: with gaps (`REFRESH_NO_SHARPEN_GUARD`'s guarded records) root
*j-k*'s deposit lands at wave `d + (i_j - i_{j-k}) + 2(j-k)`, not
earlier than root *j*'s read whenever `i_j - i_{j-k} >= 2k`, and two
roots scored one node in one wave (audit F1; reproduced by
`test_slot_stagger_is_wrong_for_gapped_windows`). The adaptive rule is
never later than the static one (induction on waves: root *k* completes
its blocking step by wave `d + i_j + i_k - 2 i_0`, which is `< d +
2(i_j - i_0)`), and executed 24 waves per round (the WAVE_BLOCK-padded
count; the last ACTIVE wave was <= 23) against the static bound of 81
on the production shape, because roots that take an imagined action at
the root leave every later root's region at once.

**Rank-ordered backups** reproduce (2): per round, for every (root, path
node) pair compute the rank of that root among the roots whose paths
contain the node (sort by `node*R + root`), apply increments rank by
rank, and record the post-increment value each root's range update
reads. Range updates then run per root leaf-to-root, excluding the
root, with typed `max()/min()` replacement.

### Type-faithful arithmetic (NumPy 2 / NEP 50)

Chain nodes hold np.float32 scalars (`eval_obs` rows: `expanded_value`,
priors). Under NumPy 2 a Python float is a *weak* operand, so
`np.float32 + float` is a float32 op. In the sequential engine this means:
a backup whose leaf is a chain node (common -- every root's first walk
down the behaviour action) makes `value_sum` np.float32 on that path
for good; `q_value()` divides in float32; the composite
`reward + discount * onward` is float32; `MinMaxStats.maximum/minimum`
hold whichever object won the `max()/min()` and comparisons cast the
weak side DOWN to float32; `c_puct * prior` rounds through float32 for
un-noised priors. The engine stores every scalar as float64 value +
type code (WEAK / F32 / F64) and evaluates every op in both precisions,
keeping NumPy's. A negative-control test replaces this with
float64-everywhere and the 1e-12 value gate fails (~2e-6 drift), so it
is load-bearing, not cosmetic.

Open question for the audit: this precision behaviour of the sequential
engine is accidental (a Python-float `expanded_value` for chain nodes
would make it all float64). It does not matter for learning at 1e-6,
but the two engines are now pinned to each other at the bit level; any
"fix" to the sequential engine's scalar types must update both.

### What is NOT reproduced (explicit deltas)

* The Python `random` tie-break stream. `tie_break="first"` (parity mode,
  also used by the v1 gate) matches the sequential engine with
  `random.choice` patched to the first candidate; production
  `tie_break="random"` draws from torch. Exact ties are measure-zero
  with real heads.
* Windows wider than 40 roots would split the sequential expansion
  batch into 40-leaf chunks (batch-shape numerics only, ~1e-7 latent /
  ~1e-3 decoded value wobble, as documented for v1). COMMUNAL_WINDOW_ROOTS
  is 16.
* Multi-trajectory forests batch several trees' windows in the same waves
  (exact on the all-ties tree; real heads see the batch-shape wobble).
  `VTREE_COMMUNAL_TRAJS` (default 1) is that knob; see Speed below.
* Re-searching a released span (`search_window` asserts `idx >
  released[t]`): the sequential engine would also find those nodes'
  latents gone. Accepted reward scalars are exactly Python `float`,
  `int`, `bool` (weak), `np.float32` (F32) and `np.float64` (F64;
  `np.longdouble` aliases it on this Windows venv, so it is accepted
  here and refused only where it is the 80/128-bit type).
  `np.integer` (strong under NEP 50) and `np.float16` are refused
  (`_scalar_type` raises; `append_chain` asserts).
### Speed (audit F2) -- measured, RTX 4070 shared with a live run

Production shape: default nets (128x6x6 latents, value-prefix hidden
512, the repro arm's 16-channel / FC-32 prefix head), A=6, 50 sims,
MAX_DEPTH 50, 16-root contiguous windows, prefix on, noise on, random
tie-breaks; 2 windows per trajectory; positions/s over the whole call.

| engine                          | round-1 audit | after F2 | after F2 + adaptive readiness |
|---------------------------------|--------------:|---------:|------------------------------:|
| sequential `run_communal_mcts`  |   33.0 pos/s  |  33.0    |  32.1                          |
| vectorized `VTREE_COMMUNAL_TRAJS=1` |  3.6 (0.11x) | 16.6 (0.50x) | **18.8 (0.59x)**           |
| `VTREE_COMMUNAL_TRAJS=8`        |  24.9 (0.75x) | 97.8 (2.96x) | **110.7 (3.4x)**            |
| `VTREE_COMMUNAL_TRAJS=16`       |  44.9 (1.36x) | 153.2 (4.64x) | **169.5 (5.3x)**           |

(The "round-1 audit" column is this box's re-measurement of the
reviewed commit with the same script; the auditor's own run of that
commit read 0.83-0.95 s/window sequential vs 5.8-6.0 s vectorized --
same ratios.) What changed: the selection wave is one fixed-shape
masked body for all roots with no host syncs, captured once per window
as a CUDA graph and replayed in blocks of `WAVE_BLOCK` (one early-exit
sync per block); adaptive readiness cut the executed waves from ~50-81
to ~24 per round; the per-window noise application is one download and
one upload (numpy RNG still consumed per root in index order); leaf
dedup runs on the device; the backup's two path folds are bounded by
the round's longest path (two syncs) with fixed-shape ranks and
in-place MinMaxStats writes. Eager and graph paths are bitwise
identical on the GPU, and the sequential and vectorized engines agree
at 1e-12 on the GPU under deterministic cuDNN (benchmark "check" mode).

**TRAJS=1 does not reach parity, and the floor is understood.** Section
timing of one TRAJS=1 window (100 rounds, 1.72 s): backup 43% (two
sequential path folds of ~20 hops at ~30-55 tiny kernels each, launched
eagerly: ~7 ms/round), selection 28% (24 waves x 0.20 ms graph replay:
~140 kernels per wave, dominated by the typed two-precision
arithmetic), network expansion 23% (the same batches the sequential
engine runs), other 6%. The sequential engine's Python does the
equivalent bookkeeping in ~0.3 s/window. Levers NOT taken, each a real
refactor: a fixed-loop CUDA graph for the backup executes MORE kernels
than the dynamic loops (51 hops instead of ~20), so it does not pay;
caching backup graphs per (path length, max rank) across windows needs
call-invariant buffers (padded root count, no forest growth mid-call);
kernel fusion (`torch.compile`/inductor) needs Triton, which this
Windows venv does not have. `VTREE_COMMUNAL_TRAJS` is the working
lever: the wave and backup costs are nearly independent of how many
trees they serve, so 8 trajectories per forest give 3.4x the sequential
engine and 16 give 5.3x, at the price of batch-shape numerics and
memory (below). The wobble is NOT v1's "counts stable" regime: the
communal tree cascades a single flipped selection through shared
statistics. Measured G=8 vs G=1 on the GPU (noise off, tie-first,
deterministic cuDNN, 256 roots): 96.1% of roots keep identical visit
counts, their root values shifting by a median 3.3e-3 (tail to 1.5
where a neighbour's path changed); 3.9% of roots flip counts (value
shift median 0.26, max 2.1). G=16 (512 roots): 3.5% flips, same
magnitudes. So `VTREE_COMMUNAL_TRAJS > 1` is archfix's target
SEMANTICS at search-noise-floor fidelity, not bitwise.
`refresh_sweep`, `refresh_targets_vtree` and `refresh_trajs` group
their candidates (`_refresh_trajs_vtree`); `refresh_saturate` stays one
at a time (it polls the actor between trajectories by design). The arm
`effzero-repro-vtree-communal` inherits the default 1 -- exact targets,
0.59x the sequential engine's speed; the arm
`effzero-repro-vtree-communal8` (= that arm + `VTREE_COMMUNAL_TRAJS: 8`,
pinned by `tests/test_vtree_refresh.py::
test_vtree_communal8_arm_is_one_knob_from_the_exact_arm`) is the speed
configuration, with the wobble caveat above.

### Memory (audit F3)

Per node at production geometry: latent `128*6*6*4 = 18,432 B`, LSTM
state `2*512*4 = 4,096 B`, scalars ~200 B -- **~22.7 KB** (the first
draft's "~13 KB" assumed 64 latent channels). Chain rows carry unused
LSTM slots (chain nodes never hold state); that is 18% of the row and
was left for simplicity. A forest reserves `chain + 1.5 * G * W *
(sims+1) * A` rows up front, compacts IN PLACE at every release
(capacity never shrinks; only the kept rows pass through a temporary),
and grows -- to `max(needed, 1.25 x capacity)` -- only if the carry past
a window exceeds the 0.5x slack of one window's bound; that is the one
moment two tables coexist: growth allocates the new table while the
old one is still alive, a transient of up to 2.25x the steady table. `test_forest_allocates_once_and_
never_shrinks` pins "at most one growth" over a 10-window sequence.
Each trajectory's `evaluate()` latents are copied into the forest and
dropped before the next trajectory is evaluated. Measured peak
allocated (2 windows, 40-position trajectories): TRAJS=1 0.22 GB, 8
1.47 GB, 16 2.90 GB -- no double table at release any more (the
reviewed commit peaked at ~5.7 GB for TRAJS=16). Projection at
production chain length (~3000 positions x 22.7 KB = 68 MB per
trajectory on top of the window reserve): steady TRAJS=8 ~1.9 GB,
TRAJS=16 ~3.8 GB; IF a growth triggers, the transient is ~4.3 GB at
TRAJS=8 and ~8.5 GB at TRAJS=16 (the new 1.25x table alone ~4.7 GB).
Next to a live run's ~4.5 GB, TRAJS=16's growth transient exceeds the
12 GB card; on the shared GPU, 8 is the sensible group.

### Integration

`Run._vtree_engine` returns `VectorizedCommunalPUCT` when
`USE_COMMUNAL_REANALYSIS` (its fresh-root `search()` IS the v1 method),
else v1. `Run._refresh_traj_vtree` routes a trajectory through
`search_trajectories` (one shared chain tree, windows of
`COMMUNAL_WINDOW_ROOTS`, release after each) under the communal flag and
is byte-identical otherwise; application is `_apply_refresh` in both
cases, so every caller (`refresh_targets_vtree`, `refresh_saturate`,
`refresh_sweep`, `refresh_trajs`) inherits the topology switch; the
grouping callers (`refresh_targets_vtree`, `refresh_sweep`,
`refresh_trajs`) hand `VTREE_COMMUNAL_TRAJS` trajectories per forest to
`_refresh_trajs_communal`. The arm `effzero-repro-vtree-communal` is
archfix's recipe with only `REFRESH_VTREE: True` (pinned by
`tests/test_vtree_refresh.py::test_vtree_communal_arm_is_one_flag_from_
archfix`); `effzero-repro-vtree-communal8` adds only
`VTREE_COMMUNAL_TRAJS: 8` (pinned by `tests/test_vtree_refresh.py::
test_vtree_communal8_arm_is_one_knob_from_the_exact_arm`). Both are in
`tests/test_harness.py`'s augmentation allowlist.


### Main-tree additions after the merge (2026-09-14)

- `muzero.BOOTSTRAP_CACHE` (default True) toggles the target-model bootstrap value
  cache in `finish_batch_paper`; off reproduces the pre-cache builder bit-for-bit.
- Arm `effzero-repro-vtree-communal8-r12` = communal8 + `POLICY_REFRESH_COUNT` 12 +
  `BOOTSTRAP_CACHE` False: archfix's recipe at 3x refresh coverage, run sequentially.
- The scheduled-quota refresh paths (`refresh_targets`, `refresh_targets_vtree`) now
  record refresh accounting (`_note_quota_stats`) so quota rows carry refresh volume.


### Edge visits (2026-09-15): the successor root's own budget

Found answering Kai's "do the searches of roots n-1, n, n+1 affect each
other's policies?" Chain node n+1 IS node n's child for the behaviour
action (`create_tree_from_game` links `prev.children[a] = next`;
`append_chain` likewise), and `backup_simulation` / `_backup_round`
increment `visit_count` on EVERY path node, the search root included.
So while root n+1 spends its S simulations in the same window, node n's
behaviour child gains S visits that no selection at node n ever made,
and the legacy readout (counts minus one bookkeeping visit) read them
out; legacy PUCT also let them shrink that child's exploration bonus.

Measured on r12's live checkpoint (sequential engine, window 16, S=50,
`benchmarks/probes/probe_window_attrib.py`, every count attributed to
the root that backed it up, max attribution error 0.0): a root's readout
averages 167 counts -- 50 own, 65 traversals by earlier roots of the
window (70% on the behaviour action; 0 at slot 0, peaking ~90 at slot
8), 5 persisted from previous windows (18 at slot 0, ~0 past slot 8),
and 50 successor root visits at slots 0-14 (0 at slot 15, whose
successor waits in the next window). Behaviour-action mass 0.60-0.75 at
slots 0-14 vs 0.12 at slot 15; the root's OWN 50 simulations put 41% on
the behaviour action on average and 11% at slot 15 (at slots 0-14 the
behaviour child's Q is the successor's 50 search backups, at slot 15 the
one-visit Monte-Carlo anchor -- the value side of the same asymmetry,
left communal by design). archfix trained on these targets.

`COMMUNAL_EDGE_VISITS` (default False = archfix's semantics, every
archived run): nodes count `root_visits` (`rootvis` in the forest;
backups that started there; the chain seeding passes `count_root=False`),
PUCT scores on EDGE visits `visit_count - root_visits` (total, the u
denominator, the virtual-loss clamp's n) and `run_communal_mcts`'s
readout baseline adds the behaviour child's root visits; the vectorized
engine mirrors both (`_child_scores` under `use_edge`, readout). Node
values stay `value_sum / visit_count` over every backup through the
node. Parity gate holds in both settings
(`tests/test_vtree_communal.py::test_edge_visits_*`); flag off is
bit-identical to before. Arm `effzero-repro-vtree-communal8-r12-edge` =
r12 + the flag (pinned in `tests/test_vtree_refresh.py`).

Flag ON on the same checkpoint (`benchmarks/probes/probe_edge_anatomy.py`,
iter 28): the readout is exactly the edge counts (asserted per root);
behaviour-argmax targets fall from 87% to 73% -- the level of independent
per-root searches (window-1 71%, fresh-root 77%) -- while behaviour MASS
only moves 0.65 -> 0.61, because the root's own simulations now flood
the behaviour child (41% -> 55% of own visits; the +50 no longer shrinks
its exploration bonus while its Q is still the successor's 50 search
backups). Slot 15 is unchanged at 0.12 mass / 13% argmax: its successor
is unsearched, so the behaviour child carries only the one-visit
Monte-Carlo anchor -- a 16-periodic target discontinuity that the
accounting fix does not touch. Searching the successor too (window 16
plus one lookahead root, `probe_edge_followups.py`) lifts slot 15 only
to 0.19 / 19%, so an unsearched successor is NOT the main cause; the
cause is the last root's position in the concurrent round (open --
`probe_edge_slot15.py` variants). Vectorized (TRAJS=1) vs sequential on
real GPU weights, same trajectory and seeds: 96.5% of roots within
1e-12 with the flag on, 97.2% with it off (the CPU gate is bitwise; the
residue is GPU kernel nondeterminism, not the flag).
