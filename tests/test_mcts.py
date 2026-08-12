"""Tree construction, lazy expansion, and simulation bookkeeping."""
import asyncio

import numpy as np
import pytest
import torch

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
    h, w = IMAGE_RESHAPE[0] // 4, IMAGE_RESHAPE[1] // 4
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
