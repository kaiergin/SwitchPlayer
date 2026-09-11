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
integration runs one). Chain nodes of tree *t* sit at ids
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

**Wave-staggered lockstep** reproduces (1) without serializing roots:
root slot *j* starts at wave *2j*; at wave *w* it takes its depth
`d = w - 2j` step. Root *j* at depth *d* needs root *j-k*'s depth-`d+k`
step (the deposit on the child it will score); that step ran at wave
`d + k + 2(j-k) = d + 2j - k < d + 2j`. Root *j*'s deposit at depth *d*
lands on a node root *j-k* reads at its depth `d+k` step... which is
wave `d + 2j - k` -- earlier -- so later roots never contaminate
earlier ones. Two roots in one wave never touch the same node: root *j*
at depth *d* and root *j'* at depth *d'* stand on the same node only if
`d + (j - j') = d'`, i.e. `w - 2j + j - j' = w - 2j'`, i.e. `j = j'`.
Waves per round: at most `2(W-1) + MAX_DEPTH`.

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
  The integration path runs one trajectory per call, matching
  `_search_traj_communal`; batching trajectories is the throughput lever
  still on the table.
* Speed at T=1 (one trajectory per forest): NO gain over the sequential
  engine. Measured on CPU with the test harness's tiny nets, 2 windows
  x 16 roots x 50 sims x MAX_DEPTH 50 with real noise: sequential
  2.23 s, vectorized 3.96 s (bitwise identical targets). Profile of the
  vectorized run: ~53% network forwards (identical batches to the
  sequential engine's), ~41% the wave loop (`_select_round`,
  ~115 small tensor ops per wave, `2(W-1)+depth` waves per round). The
  sequential engine's Python selection is cheaper than that at 16
  roots, and on a GPU every one of those ops is a launch, so T=1 on GPU
  is expected to be SLOWER than the sequential engine (unmeasured: the
  GPU was reserved for the production run). The throughput lever is
  `muzero.VTREE_COMMUNAL_TRAJS` (default 1): trajectories per forest.
  The wave loop's cost is nearly independent of how many trees it
  serves, so a group of G trajectories costs about one trajectory's
  wall time (throughput ~xG) at the price of batch-shape numerics
  (~1e-3 value wobble, visit counts empirically stable) and forest
  memory (~13 KB per stored position for chain latent + LSTM slots,
  plus `G*W*(sims+1)*A` window slots). `refresh_sweep`,
  `refresh_targets_vtree` and `refresh_trajs` group their candidates
  (`_refresh_trajs_vtree`); `refresh_saturate` stays one at a time
  (it polls the actor between trajectories by design). The arm
  `effzero-repro-vtree-communal` inherits the default 1 -- exact
  targets, no speedup; a `VTREE_COMMUNAL_TRAJS: 8`-`16` variant is the
  speed test, with the wobble caveat.
* Further wave-loop savings not taken (parity-neutral, for the audit
  round to weigh): one device sync per wave (`active.any()`) and per
  backup hop; the typed ops compute both precisions every time (a
  WEAK-only fast path would halve them where no F32 scalar is live);
  CUDA graphs / torch.compile over a fixed wave count.

### Integration

`Run._vtree_engine` returns `VectorizedCommunalPUCT` when
`USE_COMMUNAL_REANALYSIS` (its fresh-root `search()` IS the v1 method),
else v1. `Run._refresh_traj_vtree` routes a trajectory through
`search_trajectories` (one shared chain tree, windows of
`COMMUNAL_WINDOW_ROOTS`, release after each) under the communal flag and
is byte-identical otherwise; application is `_apply_refresh` in both
cases, so every caller (`refresh_targets_vtree`, `refresh_saturate`,
`refresh_sweep`, `refresh_trajs`) inherits the topology switch. The arm
`effzero-repro-vtree-communal` is archfix's recipe with only
`REFRESH_VTREE: True` (pinned by
`test_vtree_communal_arm_is_one_flag_from_archfix`).
