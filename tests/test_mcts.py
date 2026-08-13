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
                                            discount=0.997, temperature=1.0))


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
    policy, value = run_search(helper)
    assert policy.shape == (ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()
    assert isinstance(float(value), float)


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


def test_terminal_root_degrades_to_uniform(helper):
    """The last node of an episode has no successors; searching from it must not
    raise and should fall back to a uniform target."""
    helper.create_tree_from_game(fake_trajectory(4))
    policy, _ = run_search(helper, index=3)
    assert policy == pytest.approx(np.full(ACTIONS, 1.0 / ACTIONS))


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
    first, _ = run_search(helper, index=0)
    second, _ = run_search(helper, index=0)   # must not raise
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


def test_backup_records_the_composite_q_for_normalisation(helper):
    """MinMaxStats must track the same quantity selection normalises:
    r + discount * V(node), not V(node) alone."""
    node = muzero.MCTSNode(latent=None, prior=1.0, reward=10.0)
    stats = muzero.MinMaxStats()
    helper.backup_simulation([node], 5.0, discount=0.5, min_max_stats=stats)
    assert stats.maximum == pytest.approx(10.0 + 0.5 * 5.0)
    assert stats.minimum == pytest.approx(10.0 + 0.5 * 5.0)


def test_normalisation_is_seeded_before_the_first_simulation(helper):
    """Regression: MinMaxStats started every search empty, normalize() was the
    identity, and round one compared the gameplay child's raw Monte Carlo
    return (order hundreds here) against exploration terms of order 0.3 --
    flooding the entire first batch of simulations down the behaviour path."""
    helper.create_tree_from_game(fake_trajectory(6))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    helper.min_max_stats = muzero.MinMaxStats()
    helper.seed_min_max_stats(root, discount=0.997)
    assert helper.min_max_stats.maximum > helper.min_max_stats.minimum


def test_baseline_visits_are_subtracted_exactly(helper):
    helper.create_tree_from_game(fake_trajectory(6))
    root = helper.current_trajectory[0]
    helper.batch_expand_nodes([root])
    baseline = {a: 7 for a in root.children}
    for a, child in root.children.items():
        child.visit_count = 7 + (a + 1)   # this search added a+1 visits
    policy, _ = helper.drop_non_gameplay_children_and_compute_improvements(
        0, temperature=1.0, baseline=baseline)
    expected = np.arange(1, ACTIONS + 1, dtype=float)
    assert policy == pytest.approx(expected / expected.sum())


def test_improved_policy_excludes_visits_inherited_from_earlier_searches(helper):
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
    policy, _ = run_search(helper, sims=8, batches=4)
    assert policy[gameplay_action] < 0.9, (
        "policy target still dominated by visits this search never made")
