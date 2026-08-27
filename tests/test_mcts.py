"""Tree construction, lazy expansion, and simulation bookkeeping."""
import asyncio

import numpy as np
import pytest
import torch

import muzero
from muzero import IMAGE_RESHAPE, LATENT_CHANNELS, MCTSHelper

ACTIONS = 5
CPU = torch.device("cpu")


@pytest.fixture
def helper():
    return MCTSHelper(action_space=ACTIONS, device=CPU,
                      network_ref=None, storage_ref=None, enable_logging=False)


@pytest.fixture
def puct(monkeypatch):
    """Force the classic PUCT search for tests that pin its visit-count
    arithmetic; everything else runs whatever USE_GUMBEL_SEARCH selects."""
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)


def fake_trajectory(length=6):
    """Entries in the shape eval_obs produces: (obs, action, reward, latent,
    policy, expanded_value)."""
    h = w = muzero.LATENT_HW
    return [
        (None, i % ACTIONS, float(i * 10),
         torch.randn(1, LATENT_CHANNELS, h, w),
         np.full(ACTIONS, 1.0 / ACTIONS),
         float(i))
        for i in range(length)
    ]


def walk(root, seen=None):
    """Every node reachable from root, following children."""
    seen = seen if seen is not None else set()
    if id(root) in seen:
        return []
    seen.add(id(root))
    nodes = [root]
    for child in root.children.values():
        nodes.extend(walk(child, seen))
    return nodes


# --------------------------------------------------------------------------
# Tree construction
# --------------------------------------------------------------------------
def test_chain_is_linked_and_terminal_marked(helper):
    helper.create_tree_from_game(fake_trajectory(5))
    chain = helper.current_trajectory
    assert len(chain) == 5
    assert chain[-1].done
    assert all(n.is_gameplay for n in chain)
    # Each node links forward to its successor under the action that was taken.
    for i in range(4):
        assert chain[i].children[i % ACTIONS] is chain[i + 1]


def test_expansion_is_lazy(helper):
    """Regression: create_tree_from_game used to expand every node in the
    episode up front, allocating action_space child latents per timestep --
    roughly 5x the episode length -- before the search started."""
    helper.create_tree_from_game(fake_trajectory(20))
    assert not any(n.is_expanded for n in helper.current_trajectory)
    # Only the forward chain link exists, not a full action fan-out.
    assert all(len(n.children) <= 1 for n in helper.current_trajectory)


def test_initial_backup_is_the_monte_carlo_return(helper):
    """Values come from the discounted return with no bootstrapping, which is
    what get_value() serves as the pre-MCTS training target."""
    helper.create_tree_from_game(fake_trajectory(4))
    chain = helper.current_trajectory
    assert all(n.visit_count == 1 for n in chain)
    assert chain[-1].q_value() == pytest.approx(0.0)  # nothing follows the terminal node
    # Node i's value is the discounted sum of rewards strictly after it.
    discount, expected = 0.997, 0.0
    for node in reversed(chain[1:]):
        assert chain[chain.index(node) - 1].q_value() == pytest.approx(
            node.reward + discount * expected, rel=1e-6)
        expected = node.reward + discount * expected


# --------------------------------------------------------------------------
# Search bookkeeping
# --------------------------------------------------------------------------
def run_search(helper, index=0, sims=8, batches=4):
    return asyncio.run(helper.run_full_mcts(index, num_simulations=sims,
                                            sim_batches=batches, c_puct=1.25,
                                            discount=0.997))


def test_root_is_expanded_by_the_search(helper):
    helper.create_tree_from_game(fake_trajectory(6))
    root = helper.current_trajectory[0]
    assert not root.is_expanded
    run_search(helper)
    assert root.is_expanded


def test_every_simulation_is_counted(helper):
    """Regression: expand_selected discarded simulations whose leaf was a
    duplicate within the batch or already expanded. Those never reached
    backup_simulation, so their visits vanished and the effective budget was
    well under num_simulations."""
    helper.create_tree_from_game(fake_trajectory(8))
    root = helper.current_trajectory[0]
    before = root.visit_count          # 1, from the initial chain backup
    run_search(helper, sims=8, batches=4)
    assert root.visit_count == before + 8


@pytest.mark.parametrize("sims,batches", [(4, 4), (8, 2), (12, 4), (16, 8)])
def test_visit_count_tracks_budget_across_shapes(helper, sims, batches):
    helper.create_tree_from_game(fake_trajectory(8))
    root = helper.current_trajectory[0]
    run_search(helper, sims=sims, batches=batches)
    assert root.visit_count == 1 + (sims // batches) * batches


def test_puct_budget_is_exact_with_a_remainder(helper, puct):
    """Round-5 finding: `sims // batches` rounds dropped the remainder --
    '50 sims' at batch 8 silently ran 48. The budget must spend exactly."""
    helper.create_tree_from_game(fake_trajectory(8))
    root = helper.current_trajectory[0]
    before = root.visit_count
    run_search(helper, sims=10, batches=4)
    assert root.visit_count == before + 10


def test_virtual_loss_is_fully_released(helper):
    """Regression: backup_simulation is the only place virtual_loss is cleared,
    so every dropped simulation left a permanent penalty on the nodes it had
    traversed -- accumulating on the most-visited, i.e. most promising, ones."""
    helper.create_tree_from_game(fake_trajectory(8))
    run_search(helper, sims=8, batches=4)
    stranded = [n for n in walk(helper.current_trajectory[0]) if n.virtual_loss != 0]
    assert stranded == [], f"{len(stranded)} nodes still carrying virtual loss"


def test_search_produces_a_valid_improved_policy(helper):
    helper.create_tree_from_game(fake_trajectory(8))
    policy, value, has_policy = run_search(helper)
    assert policy.shape == (ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()
    assert isinstance(float(value), float)
    assert has_policy is True


def test_value_range_is_recorded_for_normalisation(helper):
    helper.create_tree_from_game(fake_trajectory(8))
    run_search(helper)
    stats = helper.min_max_stats
    assert stats.maximum > -float("inf") and stats.minimum < float("inf")


def test_gameplay_children_survive_expansion(helper):
    """Expanding a chain node must not overwrite its real successor with a
    dynamics-imagined one, or the trajectory is severed."""
    helper.create_tree_from_game(fake_trajectory(6))
    chain = helper.current_trajectory
    successor, action = chain[1], 0 % ACTIONS
    helper.batch_expand_nodes([chain[0]])
    assert chain[0].children[action] is successor
    assert len(chain[0].children) == ACTIONS


def test_terminal_root_yields_no_policy_target(helper):
    """The last node of an episode has no successors; searching from it must
    not raise, and the uniform placeholder it returns must be flagged as NOT a
    target -- otherwise the policy head is trained toward uniform on terminal
    observations."""
    helper.create_tree_from_game(fake_trajectory(4))
    policy, _, has_policy = run_search(helper, index=3)
    assert policy == pytest.approx(np.full(ACTIONS, 1.0 / ACTIONS))
    assert has_policy is False


def test_terminal_nodes_are_never_expanded(helper):
    helper.create_tree_from_game(fake_trajectory(4))
    terminal = helper.current_trajectory[-1]
    helper.batch_expand_nodes([terminal])
    assert terminal.children == {}
    assert not terminal.is_expanded


# --------------------------------------------------------------------------
# eval_obs
# --------------------------------------------------------------------------
def test_eval_obs_latents_are_detached(helper):
    """The OOM regression at the point it actually enters the tree."""
    traj = [(np.random.randint(0, 255, (12, *IMAGE_RESHAPE), dtype=np.uint8), 0, 0.0)
            for _ in range(5)]
    evaluated = helper.eval_obs(traj, batch_size=2)

    assert len(evaluated) == 5
    for _, _, _, latent, policy, value in evaluated:
        assert latent.grad_fn is None and not latent.requires_grad
        assert policy.shape == (ACTIONS,)
        assert np.ndim(value) == 0, "expanded_value must be a scalar, not a tensor"


def test_searching_the_same_index_twice_is_safe(helper):
    """Regression: the root's latent was released with `del node.latent`, so a
    second search on the same index raised AttributeError. Re-analysing a
    stored trajectory means exactly that, and it is on the roadmap."""
    helper.create_tree_from_game(fake_trajectory(6))
    first, _, _ = run_search(helper, index=0)
    second, _, _ = run_search(helper, index=0)   # must not raise
    assert first.sum() == pytest.approx(1.0)
    assert second.sum() == pytest.approx(1.0)


def test_root_latent_is_released_after_search(helper):
    """The memory win still has to happen -- one latent per timestep held for a
    whole episode is what the OOM work was about."""
    helper.create_tree_from_game(fake_trajectory(6))
    root = helper.current_trajectory[0]
    assert root.latent is not None
    run_search(helper, index=0)
    assert root.latent is None


def test_expansion_skips_nodes_with_released_latents(helper):
    helper.create_tree_from_game(fake_trajectory(6))
    node = helper.current_trajectory[2]
    node.latent = None
    helper.batch_expand_nodes([node])   # must not raise
    assert not node.is_expanded


# --------------------------------------------------------------------------
# Selection scoring: r + discount * V, normalisation seeding, policy targets
# --------------------------------------------------------------------------
def make_uniform_children(helper, length=4):
    """Expand the root and flatten its children to a controlled baseline:
    equal priors, no visits, identical onward values, zero edge rewards."""
    helper.create_tree_from_game(fake_trajectory(length))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    for child in root.children.values():
        child.prior = 1.0 / ACTIONS
        child.visit_count, child.value_sum, child.virtual_loss = 0, 0.0, 0.0
        child.expanded_value, child.reward, child.done = 10.0, 0.0, False
    helper.min_max_stats = muzero.MinMaxStats()
    helper.min_max_stats.update(0.0)
    helper.min_max_stats.update(100.0)
    return root


def select_once(helper):
    asyncio.run(helper.selection_task_no_expand(0, 1.25, 0.997))
    leaf, _ = helper.selection_results.get_nowait()
    return leaf


def test_selection_prefers_the_larger_edge_reward(helper):
    """Regression: selection scored children on q_value() alone -- the value
    from the child ONWARD, since backup folds the edge reward into the parent
    -- so two actions with identical futures but different immediate rewards
    were indistinguishable at the decision point. The paper's score is
    r + discount * V(child)."""
    root = make_uniform_children(helper)
    root.children[2].reward = 50.0
    assert select_once(helper) is root.children[2]


def test_unvisited_children_score_their_network_value_not_zero(helper):
    """With onward value pinned to 0 for unvisited children, every imagined
    child ranked beneath a gameplay child whose Monte Carlo return sat at the
    top of the normalisation range; expanded_value puts them on comparable
    footing before their first visit."""
    root = make_uniform_children(helper)
    root.children[1].expanded_value = 90.0
    assert select_once(helper) is root.children[1]


def test_neutral_chain_seeding_leaves_values_empty(helper):
    """seed_values=False builds the chain's structure, priors, rewards, and
    terminal flag with NO Monte-Carlo backup: the behaviour child starts
    symmetric with imagined siblings (2026-08-27 audit: a seeded chain's
    realized return vs the siblings' network estimates pulled policy
    targets toward the behaviour action early and away from it once the
    value head calibrated)."""
    helper.create_tree_from_game(fake_trajectory(5), seed_values=False)
    neutral = helper.current_trajectory
    assert all(n.visit_count == 0 and n.value_sum == 0.0 for n in neutral)
    assert neutral[-1].done
    assert neutral[0].children, "forward chain links must survive"
    # The default stays the seeded chain the product path's warmup reads.
    helper.create_tree_from_game(fake_trajectory(5))
    assert all(n.visit_count == 1 for n in helper.current_trajectory)


def test_max_depth_override_applies_at_call_time(helper, monkeypatch):
    """Regression: max_depth=MAX_DEPTH as a def-time default froze the
    import-time value, so an arm's MAX_DEPTH override was silently ignored
    (the backup_simulation-discount bug class)."""
    monkeypatch.setattr(muzero, "MAX_DEPTH", 2)
    helper.create_tree_from_game(fake_trajectory(12))
    for node in helper.current_trajectory[:-1]:
        node.is_expanded = True   # selection walks the chain until the cap
    helper.min_max_stats = muzero.MinMaxStats()
    asyncio.run(helper.selection_task_no_expand(0, 1.25, 0.997))
    _, path = helper.selection_results.get_nowait()
    # Walks max_depth nodes then appends the leaf; the frozen default (8)
    # would have produced a 9-node path.
    assert len(path) == 3


def test_virtual_loss_repels_on_negative_values(helper):
    """Regression (the Pong finding): an in-flight, not-yet-backed-up child
    used to read q_value() = 0 -- with every real value negative, the most
    optimistic number in the tree, normalising far ABOVE the seeded range --
    so every concurrent simulation in a batch descended the same root action
    and the policy target degenerated toward uniform. In-flight children now
    read their network value discounted by visits at the normalized minimum,
    so the second selection of a batch must diversify."""
    root = make_uniform_children(helper)
    for child in root.children.values():
        child.expanded_value = -10.0
    helper.min_max_stats = muzero.MinMaxStats()
    helper.min_max_stats.update(-11.0)
    helper.min_max_stats.update(-9.0)
    first = select_once(helper)
    assert first.virtual_loss == 1  # in flight: selected, no backup yet
    second = select_once(helper)
    assert second is not first


def test_communal_window_targets_and_release(helper):
    """Communal windows: every root gets a readout, the chain backup's
    bookkeeping visit -- and ONLY that visit -- is subtracted from the
    policy target, and release_span frees the span's imagined subtrees and
    latents while nodes past the window keep theirs for the next window."""
    helper.create_tree_from_game(fake_trajectory(20))
    results = asyncio.run(helper.run_communal_mcts(
        [0, 4, 8], num_simulations=8, c_puct=1.25, discount=0.997))
    assert set(results) == {0, 4, 8}
    for idx in (0, 4, 8):
        policy, value, has_policy = results[idx]
        assert has_policy
        assert policy.sum() == pytest.approx(1.0)
        root = helper.current_trajectory[idx]
        counts = np.array([
            root.children[a].visit_count
            - (1 if root.children[a].is_gameplay else 0)
            if a in root.children else 0
            for a in range(ACTIONS)], dtype=np.float64)
        assert policy == pytest.approx(counts / counts.sum())
    helper.release_span(0, 8)
    for i in range(9):
        node = helper.current_trajectory[i]
        assert node.latent is None
        assert all(c.is_gameplay for c in node.children.values())
    # Chain nodes past the span keep their latents: they are the next
    # window's reuse, and release only ever advances forward.
    assert all(helper.current_trajectory[i].latent is not None
               for i in range(9, 20))


def test_backup_records_the_composite_q_for_normalisation(helper):
    """MinMaxStats must track the same quantity selection normalises:
    r + discount * V(node), not V(node) alone."""
    root = muzero.MCTSNode(latent=None, prior=1.0, reward=0.0)
    child = muzero.MCTSNode(latent=None, prior=1.0, reward=10.0)
    stats = muzero.MinMaxStats()
    helper.backup_simulation([root, child], 5.0, discount=0.5, min_max_stats=stats)
    assert stats.maximum == pytest.approx(10.0 + 0.5 * 5.0)
    assert stats.minimum == pytest.approx(10.0 + 0.5 * 5.0)


def test_backup_excludes_the_roots_entering_reward_from_the_range(helper):
    """The search root's reward is the edge INTO the root -- outside the
    searched subtree. Folding it into MinMaxStats inflated the range right
    after big reward events and compressed every normalized advantage."""
    root = muzero.MCTSNode(latent=None, prior=1.0, reward=400.0)
    child = muzero.MCTSNode(latent=None, prior=1.0, reward=1.0)
    stats = muzero.MinMaxStats()
    helper.backup_simulation([root, child], 2.0, discount=0.5, min_max_stats=stats)
    # Only the child's in-tree composite (1 + 0.5*2 = 2) enters the range;
    # the root's 400-composite must not.
    assert stats.maximum == pytest.approx(2.0)


def test_truncated_chain_bootstraps_instead_of_faking_a_terminal(helper):
    """A capped (truncated) episode's tail is NOT terminal: marking it done
    taught V=0 at a mid-game state on every capped trajectory, and nstep_value
    refused to bootstrap past it."""
    helper.create_tree_from_game(fake_trajectory(5), terminated=False)
    tail = helper.current_trajectory[-1]
    assert not tail.done
    # The chain backup bootstraps the network's value at the truncated tail.
    assert tail.q_value() == pytest.approx(tail.expanded_value)
    # And nstep_value's horizon landing on the tail uses its network value.
    expected = sum(0.997 ** (k - 1) * helper.current_trajectory[k].reward
                   for k in range(1, 5))
    expected += 0.997 ** 4 * tail.expanded_value
    assert helper.nstep_value(0, n=4, discount=0.997) == pytest.approx(expected)


def test_normalisation_is_seeded_before_the_first_simulation(helper):
    """Regression: MinMaxStats started every search empty, normalize() was the
    identity, and round one compared the gameplay child's raw Monte Carlo
    return (order hundreds here) against exploration terms of order 0.3 --
    flooding the entire first batch of simulations down the behaviour path.

    Pins the exact COMPOSITE seeded (r + discount * onward), not merely that
    some range exists -- the suite audit showed the range-exists assertion
    passed even when seeding used V alone."""
    helper.create_tree_from_game(fake_trajectory(6))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    for a, child in root.children.items():
        child.visit_count, child.value_sum, child.virtual_loss = 0, 0.0, 0.0
        child.done = False
        child.reward = float(a)               # distinct edge rewards
        child.expanded_value = 10.0 * (a + 1)  # distinct onward values
    helper.min_max_stats = muzero.MinMaxStats()
    helper.seed_min_max_stats(root, discount=0.5)
    composites = [a + 0.5 * 10.0 * (a + 1) for a in root.children]
    assert helper.min_max_stats.maximum == pytest.approx(max(composites))
    assert helper.min_max_stats.minimum == pytest.approx(min(composites))


def test_baseline_visits_are_subtracted_exactly(helper):
    """Also pins that the policy target is the RAW visit fraction -- the
    paper's target. Temperature belongs to action selection, not targets."""
    helper.create_tree_from_game(fake_trajectory(6))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    baseline = {a: 7 for a in root.children}
    for a, child in root.children.items():
        child.visit_count = 7 + (a + 1)   # this search added a+1 visits
    policy, _, _ = helper.drop_non_gameplay_children_and_compute_improvements(
        0, baseline=baseline)
    expected = np.arange(1, ACTIONS + 1, dtype=float)
    assert policy == pytest.approx(expected / expected.sum())


def test_improved_policy_excludes_visits_inherited_from_earlier_searches(helper, puct):
    """Regression: raw visit counts fed apply_temperature, so the chain
    backup's bookkeeping visit and traffic from earlier searches -- which can
    only reach a node along the behaviour trajectory -- biased every policy
    target toward the action the agent actually took."""
    helper.create_tree_from_game(fake_trajectory(8))
    root = helper.current_trajectory[0]
    gameplay_action = 0   # fake_trajectory takes action i % ACTIONS at step i
    child = root.children[gameplay_action]
    inherited = 100
    child.value_sum *= (child.visit_count + inherited) / child.visit_count
    child.visit_count += inherited
    policy, _, _ = run_search(helper, sims=8, batches=4)
    assert policy[gameplay_action] < 0.9, (
        "policy target still dominated by visits this search never made")


# --------------------------------------------------------------------------
# Gumbel search (Danihelka et al. 2022)
# --------------------------------------------------------------------------
def test_gumbel_spends_exactly_the_budget(helper):
    helper.create_tree_from_game(fake_trajectory(10))
    root = helper.current_trajectory[0]
    before = root.visit_count
    run_search(helper, sims=32, batches=8)
    assert root.visit_count == before + 32


def test_gumbel_does_not_mutate_the_prior(helper):
    """PUCT wrote Dirichlet noise into node.policy before every search; the
    Gumbel draws live outside the node, so the stored prior must survive."""
    helper.create_tree_from_game(fake_trajectory(8))
    root = helper.current_trajectory[0]
    prior = np.array(root.policy).copy()
    run_search(helper)
    assert np.array_equal(np.array(root.policy), prior)


def test_gumbel_target_ranks_by_completed_q(helper):
    """The policy target is softmax(logits + sigma(completedQ)): with equal
    priors, a clearly better searched action must dominate -- this is the
    informativeness at small budgets that visit fractions cannot give."""
    helper.create_tree_from_game(fake_trajectory(4))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    for child in root.children.values():
        child.prior = 1.0 / ACTIONS
        child.visit_count, child.value_sum, child.virtual_loss = 4, 4 * 10.0, 0.0
        child.reward, child.done, child.expanded_value = 0.0, False, 10.0
    root.children[3].value_sum = 4 * 90.0
    root.policy = np.full(ACTIONS, 1.0 / ACTIONS)
    root.expanded_value = 10.0
    helper.min_max_stats = muzero.MinMaxStats()
    helper.min_max_stats.update(0.0)
    helper.min_max_stats.update(100.0)
    logits = np.log(np.full(ACTIONS, 1.0 / ACTIONS))
    pi = helper._gumbel_improved_policy(root, logits, 0.997, baseline={})
    assert pi.argmax() == 3
    assert pi[3] > 0.9


def test_gumbel_completes_unvisited_actions_with_the_mixed_value(helper):
    """An unvisited action is scored with v_mix, not 0: if the only visited
    action came back clearly below the node's own value estimate, the target
    must NOT concentrate on it just because it was the one searched."""
    helper.create_tree_from_game(fake_trajectory(4))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    for child in root.children.values():
        child.prior = 1.0 / ACTIONS
        child.visit_count, child.value_sum, child.virtual_loss = 0, 0.0, 0.0
        child.reward, child.done, child.expanded_value = 0.0, False, 10.0
    root.children[1].visit_count = 4
    root.children[1].value_sum = 4 * 10.0     # searched, and mediocre
    root.policy = np.full(ACTIONS, 1.0 / ACTIONS)
    root.expanded_value = 80.0                # the node itself looks better
    helper.min_max_stats = muzero.MinMaxStats()
    helper.min_max_stats.update(0.0)
    helper.min_max_stats.update(100.0)
    logits = np.log(np.full(ACTIONS, 1.0 / ACTIONS))
    pi = helper._gumbel_improved_policy(root, logits, 0.997, baseline={})
    assert pi.argmax() != 1
    # At the paper's c_scale=0.1 (sigma ~5.4) the demotion is real but soft:
    # below the uniform share, not crushed to zero as the old sigma ~54 did.
    assert pi[1] < 1.0 / ACTIONS


def test_act_search_returns_a_valid_action_and_restores_state(helper):
    """act_search is the TEST-ONLY paper acting mode: a fresh one-node tree
    per live observation. It must return a legal action and leave the
    reanalysis chain exactly as it found it."""
    helper.create_tree_from_game(fake_trajectory(6))
    chain = helper.current_trajectory
    obs = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)
    action, policy, value = asyncio.run(helper.act_search(obs, num_simulations=8))
    assert 0 <= action < ACTIONS
    assert policy.sum() == pytest.approx(1.0)
    assert isinstance(value, float)
    assert helper.current_trajectory is chain


def test_search_action_is_the_halving_survivor(helper, monkeypatch):
    """With one action's completed Q overwhelmingly best, the survivor rule
    (g + logits + sigma(qhat)) must pick it: sigma's ~50x scale dwarfs the
    Gumbel draws once the value gap is real. All actions are admitted as
    candidates so the (random) Gumbel top-m cut cannot exclude the best one
    -- with the default m=4 of 5 that exclusion is a 1-in-5 flake. The Gumbel
    draws are seeded: at the paper's c_scale=0.1 the sigma-scaled value gap is
    ~4.3 logits, decisive but not beyond a tail Gumbel draw."""
    np.random.seed(1)
    monkeypatch.setattr(muzero, "GUMBEL_MAX_CANDIDATES", ACTIONS)
    helper.create_tree_from_game(fake_trajectory(4))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    for child in root.children.values():
        child.prior = 1.0 / ACTIONS
        child.visit_count, child.value_sum, child.virtual_loss = 4, 4 * 10.0, 0.0
        child.reward, child.done, child.expanded_value = 0.0, False, 10.0
    root.children[2].value_sum = 4 * 90.0
    root.policy = np.full(ACTIONS, 1.0 / ACTIONS)
    root.expanded_value = 10.0
    asyncio.run(helper.run_full_mcts(0, num_simulations=8, sim_batches=4,
                                     discount=0.997))
    assert helper.last_search_action == 2


def test_flag_off_restores_visit_fraction_targets(helper, puct):
    """With USE_GUMBEL_SEARCH False the target must be the PUCT path's raw
    visit fractions -- quantised to multiples of 1/sims, which the Gumbel
    softmax target essentially never is."""
    helper.create_tree_from_game(fake_trajectory(8))
    policy, _, has_policy = run_search(helper, sims=8, batches=4)
    assert has_policy is True
    assert np.allclose(policy * 8, np.round(policy * 8))


# --------------------------------------------------------------------------
# n-step value targets for unsearched positions (MuZero-Reanalyze)
# --------------------------------------------------------------------------
def test_nstep_value_bootstraps_the_current_network(helper):
    """Unsearched positions get n real rewards plus the CURRENT network's
    value at the tail -- not the behaviour policy's Monte Carlo return, which
    goes stale as the policy moves."""
    helper.create_tree_from_game(fake_trajectory(8))
    n, discount = 3, 0.9
    traj = helper.current_trajectory
    expected = sum(discount ** (k - 1) * traj[k].reward for k in range(1, n + 1))
    expected += discount ** n * traj[n].expanded_value
    assert helper.nstep_value(0, n=n, discount=discount) == pytest.approx(expected)


def test_nstep_value_falls_back_to_monte_carlo_at_episode_end(helper):
    """A horizon that runs past the end consumes every remaining reward and
    bootstraps nothing, which is exactly the chain backup's MC return."""
    helper.create_tree_from_game(fake_trajectory(5))
    assert helper.nstep_value(2, n=10, discount=0.997) == pytest.approx(
        helper.get_value(2))


def test_nstep_value_does_not_bootstrap_a_terminal_state(helper):
    """A horizon landing exactly on the terminal node must use 0 for its
    onward value, not the network's estimate of the terminal observation."""
    helper.create_tree_from_game(fake_trajectory(5))
    expected = sum(0.997 ** (k - 1) * helper.current_trajectory[k].reward
                   for k in range(1, 5))
    assert helper.nstep_value(0, n=4, discount=0.997) == pytest.approx(expected)


def test_nstep_value_bootstraps_the_nearest_search_ahead(helper):
    """The paper's v is a SEARCH value. The bootstrap stretches to the nearest
    searched index at or past idx+n, and a search value at idx itself is never
    consulted -- a position's own target always opens with real rewards."""
    helper.create_tree_from_game(fake_trajectory(20))
    n, discount = 10, 0.9
    search_values = {0: 999.0, 12: 500.0}   # 0 is behind the horizon; 12 is nearest at/past 10
    traj = helper.current_trajectory
    expected = sum(discount ** (k - 1) * traj[k].reward for k in range(1, 13))
    expected += discount ** 12 * 500.0
    got = helper.nstep_value(0, n=n, discount=discount, search_values=search_values)
    assert got == pytest.approx(expected)


def test_nstep_value_uses_a_search_value_exactly_at_the_horizon(helper):
    helper.create_tree_from_game(fake_trajectory(20))
    n, discount = 10, 0.9
    traj = helper.current_trajectory
    expected = sum(discount ** (k - 1) * traj[k].reward for k in range(1, 11))
    expected += discount ** 10 * 250.0
    got = helper.nstep_value(0, n=n, discount=discount, search_values={10: 250.0})
    assert got == pytest.approx(expected)


def test_nstep_value_falls_back_to_the_network_when_no_search_lies_ahead(helper):
    """Search values behind the horizon must not change the target: the tail
    stays at idx+n on the current network's expanded_value."""
    helper.create_tree_from_game(fake_trajectory(20))
    n, discount = 10, 0.9
    with_behind = helper.nstep_value(0, n=n, discount=discount,
                                     search_values={2: 999.0})
    without = helper.nstep_value(0, n=n, discount=discount)
    assert with_behind == pytest.approx(without)


def test_truncated_tail_bootstraps_even_when_the_horizon_overruns(helper):
    """Regression: the horizon-overrun branch returned partial MC with an
    implicit V=0 without ever consulting node.done -- so a TRUNCATED (capped)
    tail still taught V=0 at a mid-game state, the exact bug the truncation
    plumbing removed one consumer upstream. The horizon now clamps to the
    trajectory and the tail's done flag decides."""
    helper.create_tree_from_game(fake_trajectory(5), terminated=False)
    traj = helper.current_trajectory
    expected = sum(0.997 ** (k - 1) * traj[k].reward for k in range(1, 5))
    expected += 0.997 ** 4 * traj[4].expanded_value
    assert helper.nstep_value(0, n=10, discount=0.997) == pytest.approx(expected)
    # The truncated tail itself gets an identity target, not zero.
    assert helper.nstep_value(4, n=10, discount=0.997) == pytest.approx(
        traj[4].expanded_value)


def test_truncated_tail_never_bootstraps_its_own_search_value(helper):
    """Round-3 regression: the horizon clamp made tail == idx reachable, and
    a SEARCHED truncated tail then received its own root value as its own
    target at full weight -- violating the never-your-own-search-value
    invariant. It must fall back to the network's expanded_value."""
    helper.create_tree_from_game(fake_trajectory(5), terminated=False)
    tail = helper.current_trajectory[4]
    got = helper.nstep_value(4, n=10, discount=0.997, search_values={4: 999.0})
    assert got == pytest.approx(tail.expanded_value)


def test_terminated_overrun_is_still_pure_monte_carlo(helper):
    """The clamp must not change terminated semantics: a horizon crossing a
    TRUE terminal consumes the remaining rewards and bootstraps nothing."""
    helper.create_tree_from_game(fake_trajectory(5))
    expected = sum(0.997 ** (k - 1) * helper.current_trajectory[k].reward
                   for k in range(1, 5))
    assert helper.nstep_value(0, n=10, discount=0.997) == pytest.approx(expected)


def test_nstep_value_ignores_a_search_value_on_a_terminal_node(helper):
    """A terminal node's onward value is 0 by definition, even if a search ran
    there and recorded something else."""
    helper.create_tree_from_game(fake_trajectory(5))
    expected = sum(0.997 ** (k - 1) * helper.current_trajectory[k].reward
                   for k in range(1, 5))
    got = helper.nstep_value(0, n=4, discount=0.997, search_values={4: 123.0})
    assert got == pytest.approx(expected)
