"""Network output contracts -- shapes, and the two conventions that were broken."""
import numpy as np
import pytest
import torch

import muzero
from muzero import LATENT_CHANNELS, IMAGE_RESHAPE, NetworkManager
from networks import DynamicsNetwork, PredictionNetwork, RepresentationNetwork

ACTIONS, BINS = 5, 51
CPU = torch.device("cpu")


@pytest.fixture
def manager():
    # network_ref is only touched by setup()/update_if_needed(), so None is fine
    # for exercising the local inference paths.
    return NetworkManager(None, action_space=ACTIONS, device=CPU)


@pytest.fixture
def latent():
    rep = RepresentationNetwork(input_channels=12, latent_channels=LATENT_CHANNELS)
    with torch.no_grad():
        return rep(torch.rand(2, 12, *IMAGE_RESHAPE))


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def test_representation_downsamples_to_the_declared_latent_size(latent):
    """LATENT_HW is a constant consumers rely on; it must not drift from what
    the network actually emits."""
    assert latent.shape == (2, LATENT_CHANNELS, muzero.LATENT_HW, muzero.LATENT_HW)
    assert IMAGE_RESHAPE[0] // muzero.LATENT_HW == 14, "84 -> 6 is a 14x reduction"


def test_dynamics_preserves_latent_shape(latent):
    dyn = DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                          support_size=400, num_bins=BINS)
    with torch.no_grad():
        nxt, reward, reward_logits = dyn(latent, torch.zeros(2, dtype=torch.long))
    assert nxt.shape == latent.shape
    assert reward.shape == (2, 1)
    assert reward_logits.shape == (2, BINS)


def test_prediction_head_shapes(latent):
    pred = PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                             support_size=2000, num_bins=BINS)
    with torch.no_grad():
        policy_logits, value, value_logits = pred(latent)
    assert policy_logits.shape == (2, ACTIONS)
    assert value.shape == (2, 1)
    assert value_logits.shape == (2, BINS)


# --------------------------------------------------------------------------
# The policy head must emit logits, not probabilities
# --------------------------------------------------------------------------
def test_policy_head_returns_raw_logits(latent):
    """Regression: the head used to return softmax(logits) clamped to [1e-6, 1].

    Every caller named the result policy_logits and applied softmax or
    log_softmax to it, so acting used softmax(softmax(x)) -- at most a 2.7x
    spread between best and worst action -- and the training loss took
    log_softmax of a probability vector, which cannot produce a sharp policy.

    Zeroing policy_fc's weights makes the head output its bias exactly, so
    logits and probabilities are trivially distinguishable.
    """
    pred = PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                             support_size=2000, num_bins=BINS)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    with torch.no_grad():
        pred.policy_fc.weight.zero_()
        pred.policy_fc.bias.copy_(bias)
        policy_logits, _, _ = pred(latent)

    assert policy_logits[0] == pytest.approx(bias.tolist())
    assert policy_logits.sum(dim=1)[0] != pytest.approx(1.0)  # not a distribution
    assert torch.softmax(policy_logits, dim=1).sum(dim=1)[0] == pytest.approx(1.0)


def test_softmaxing_the_head_output_gives_a_usable_distribution(manager):
    """np.random.choice rejects vectors that do not sum to 1; the old clamp
    broke normalisation, which could make acting raise outright."""
    policy, value = manager.initial_inference(np.random.randint(0, 255, (12, *IMAGE_RESHAPE), dtype=np.uint8))
    assert policy.shape == (ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    np.random.choice(np.arange(ACTIONS), p=policy)  # must not raise
    assert isinstance(value, float)


# --------------------------------------------------------------------------
# Inference must not retain the autograd graph
# --------------------------------------------------------------------------
def test_batch_inference_returns_detached_latents(manager):
    """Regression, and the cause of the out-of-memory failures.

    Every gameplay node stores one of these latents for the whole episode. When
    they carried grad_fn, the entire forward graph -- all ResNet activations
    for the batch -- stayed resident with them: multiple GB per episode.
    """
    obs = np.random.randint(0, 255, (4, 12, *IMAGE_RESHAPE), dtype=np.uint8)
    latents, policies, values = manager.batch_inference(obs)

    assert latents.grad_fn is None, "latents still carry the forward graph"
    assert not latents.requires_grad
    assert latents.shape[0] == 4


def test_batch_inference_value_is_scalar_per_row(manager):
    """expanded_value has to be a scalar. It used to be a (1,)-shaped tensor for
    gameplay nodes and a float elsewhere, so value_sum silently became a
    tensor mid-backup."""
    obs = np.random.randint(0, 255, (3, 12, *IMAGE_RESHAPE), dtype=np.uint8)
    _, policies, values = manager.batch_inference(obs)

    assert isinstance(values, np.ndarray) and values.shape == (3,)
    assert np.isscalar(values[0]) or values[0].ndim == 0
    assert isinstance(policies, np.ndarray) and policies.shape == (3, ACTIONS)
    assert policies.sum(axis=1) == pytest.approx(np.ones(3))


def test_initial_inference_does_not_leak_graph(manager):
    """no_grad has to be inside the method: Ray actors run in their own process,
    so a `with torch.no_grad()` at the call site does nothing."""
    obs = np.random.randint(0, 255, (12, *IMAGE_RESHAPE), dtype=np.uint8)
    before = torch.is_grad_enabled()
    policy, value = manager.initial_inference(obs)
    assert torch.is_grad_enabled() == before
    assert isinstance(value, float)


# --------------------------------------------------------------------------
# BatchNorm mode: inference consumers eval, training copies train
# --------------------------------------------------------------------------
def test_inference_consumers_run_in_eval_mode(manager):
    """Regression: no code path ever called .eval(), so acting normalised each
    batch-1 frame by its own spatial statistics and search batches both used
    and polluted the running stats. The function that acted was not the
    function training optimised."""
    assert not manager.representation.training
    assert not manager.prediction.training
    assert not manager.dynamics.training
    assert not manager.value_prefix.training


def test_eval_mode_survives_a_weight_pull(manager):
    """load_state_dict refreshes weights and BN running stats but must not
    flip the module back into train mode."""
    fresh = NetworkManager(None, action_space=ACTIONS, device=CPU)
    manager.representation.load_state_dict(fresh.representation.state_dict())
    assert not manager.representation.training


def test_network_helper_serves_in_eval_mode(tmp_path):
    from muzero import NetworkHelper
    helper = NetworkHelper(action_space=ACTIONS, device=CPU,
                           checkpoint_path=str(tmp_path / "missing.pth"))
    for net in (helper.representation, helper.dynamics,
                helper.prediction, helper.value_prefix):
        assert not net.training


def test_training_copies_stay_in_train_mode():
    """TrainingHelper's networks are the one place BatchNorm should be
    accumulating running statistics."""
    from muzero import TrainingHelper
    trainer = TrainingHelper(action_space=ACTIONS, device=CPU,
                             network_ref=None, storage_ref=None)
    for net in (trainer.representation, trainer.dynamics,
                trainer.prediction, trainer.value_prefix):
        assert net.training


# --------------------------------------------------------------------------
# Hidden-state min-max scaling (MuZero appendix G)
# --------------------------------------------------------------------------
def test_representation_latent_is_scaled_to_unit_range(latent):
    """Appendix G: the emitted hidden state is min-max scaled per sample.
    Without it the latent scale drifts with unroll depth and search depth --
    the instability the paper says this scaling exists to prevent."""
    flat = latent.flatten(start_dim=1)
    assert flat.min(dim=1).values.max().item() == pytest.approx(0.0, abs=1e-4)
    assert flat.max(dim=1).values.min().item() == pytest.approx(1.0, abs=1e-4)


def test_dynamics_latent_is_scaled_to_unit_range(latent):
    dyn = DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                          support_size=400, num_bins=BINS)
    with torch.no_grad():
        nxt, _, _ = dyn(latent, torch.zeros(2, dtype=torch.long))
    flat = nxt.flatten(start_dim=1)
    assert flat.min(dim=1).values.max().item() == pytest.approx(0.0, abs=1e-4)
    assert flat.max(dim=1).values.min().item() == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# SimSiam projector (EfficientZero 4.1)
# --------------------------------------------------------------------------
def test_projector_branches_share_output_shape(latent):
    from networks import LatentProjector
    proj = LatentProjector(latent[0].numel())
    target = proj(latent, with_prediction=False)
    online = proj(latent, with_prediction=True)
    assert target.shape == online.shape == (2, 1024)


def test_checkpoint_roundtrips_the_projector(tmp_path):
    from muzero import NetworkHelper
    path = str(tmp_path / "ck.pth")
    helper = NetworkHelper(action_space=ACTIONS, device=CPU, checkpoint_path=path)
    helper.save_checkpoint(5, checkpoint_path=path)

    reloaded = NetworkHelper(action_space=ACTIONS, device=CPU, checkpoint_path=path)
    a = helper.projector.state_dict()
    b = reloaded.projector.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoint_roundtrips_the_prefix_head(tmp_path):
    """Regression: value_prefix_state_dict was SAVED but never LOADED, so
    every restart resumed with a fresh prefix head and optimizer while
    training_steps claimed otherwise."""
    from muzero import NetworkHelper
    path = str(tmp_path / "ck.pth")
    helper = NetworkHelper(action_space=ACTIONS, device=CPU, checkpoint_path=path)
    helper.save_checkpoint(5, checkpoint_path=path)

    reloaded = NetworkHelper(action_space=ACTIONS, device=CPU, checkpoint_path=path)
    a = helper.value_prefix.state_dict()
    b = reloaded.value_prefix.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_pre_projector_checkpoint_still_loads(tmp_path):
    """Checkpoints written before SimSiam landed have no projector entry; they
    must load with a fresh projector rather than raising KeyError -- the
    projector is training-only and carries no policy."""
    from muzero import NetworkHelper
    path = str(tmp_path / "ck.pth")
    NetworkHelper(action_space=ACTIONS, device=CPU,
                  checkpoint_path=path).save_checkpoint(5, checkpoint_path=path)
    ck = torch.load(path)
    del ck["projector_state_dict"]
    del ck["proj_optimizer_state_dict"]
    torch.save(ck, path)
    NetworkHelper(action_space=ACTIONS, device=CPU, checkpoint_path=path)  # must not raise


# --------------------------------------------------------------------------
# The decode transform flag
# --------------------------------------------------------------------------
def test_decode_honours_the_transform_flag(latent):
    """The first no-transform ablation arm was mislabeled because this flag
    did not exist: it changed the support sizes but h() still applied on both
    sides of the codec. transform=False must decode the raw expectation;
    transform=True is h_inverse of exactly that expectation."""
    from scalar_categorical import h_inverse
    dyn = DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                          support_size=400, num_bins=BINS, transform=False)
    with torch.no_grad():
        _, linear, _ = dyn(latent, torch.zeros(2, dtype=torch.long))
        dyn.transform = True
        _, transformed, _ = dyn(latent, torch.zeros(2, dtype=torch.long))
    assert torch.allclose(h_inverse(linear), transformed, rtol=1e-4, atol=1e-3)
    assert not torch.allclose(linear, transformed)
