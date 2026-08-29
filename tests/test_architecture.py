"""Audit findings 5-7: dynamics residual skip, representation norms, and
the size/encoding knobs."""
import numpy as np
import pytest
import torch

import muzero
from networks import (DynamicsNetwork, PredictionNetwork,
                      RepresentationNetwork, renormalize)


def test_dynamics_skip_passes_the_latent_through_a_zeroed_delta():
    """The residual contract: with conv1 and every block's convs zeroed, the
    dynamics output is exactly renormalize(latent) -- identity plus a learned
    delta, where the delta is zero. Pre-fix, a zeroed conv1 destroyed the
    state entirely (output was renormalized noise from zero activations)."""
    net = DynamicsNetwork(latent_channels=16, action_space=4,
                          pool=(6, 6), blocks=2)
    net.eval()
    with torch.no_grad():
        net.conv1.weight.zero_()
        for block in net.blocks:
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
    latent = torch.rand(3, 16, 6, 6)
    action = torch.tensor([0, 1, 3])
    with torch.no_grad():
        next_latent, _, _ = net(latent, action)
    assert torch.allclose(next_latent, renormalize(latent), atol=1e-5)


@pytest.mark.parametrize("encoding,extra", [("embed", 16), ("plane", 1),
                                            ("onehot", 4)])
def test_action_encodings_shape_conv1_input(encoding, extra):
    net = DynamicsNetwork(latent_channels=16, action_space=4,
                          action_encoding=encoding)
    assert net.conv1.in_channels == 16 + extra
    latent = torch.rand(2, 16, 6, 6)
    with torch.no_grad():
        net.eval()
        out, reward, _ = net(latent, torch.tensor([1, 3]))
    assert out.shape == latent.shape


def test_onehot_and_plane_encode_the_chosen_action():
    net = DynamicsNetwork(latent_channels=8, action_space=4,
                          action_encoding="onehot")
    planes = net._encode_action(torch.tensor([2]), 1, 6, 6, torch.device("cpu"))
    assert planes.shape == (1, 4, 6, 6)
    assert planes[0, 2].min() == 1.0 and planes.sum() == 36.0
    net = DynamicsNetwork(latent_channels=8, action_space=4,
                          action_encoding="plane")
    plane = net._encode_action(torch.tensor([3]), 1, 6, 6, torch.device("cpu"))
    assert plane.shape == (1, 1, 6, 6)
    assert torch.allclose(plane, torch.full((1, 1, 6, 6), 3 / 4))


def test_representation_norms_every_strided_conv():
    net = RepresentationNetwork(input_channels=12, latent_channels=32,
                                trunk_blocks=1)
    for i in (1, 2, 3, 4):
        conv = getattr(net, f"conv{i}")
        assert conv.bias is None, f"conv{i} kept its bias despite BN"
        assert isinstance(getattr(net, f"bn{i}"), torch.nn.BatchNorm2d)
    assert len(net.trunk) == 1
    with torch.no_grad():
        net.eval()
        out = net(torch.rand(2, 12, 84, 84))
    assert out.shape == (2, 32, 6, 6)
    assert out.min() >= 0 and out.max() <= 1.0 + 1e-5


def test_prediction_head_knobs_and_zero_init_survive_fc_hidden():
    """The EffZero-dims head (16ch + [32] hidden) must still decode value 0
    and a uniform policy at init -- zero-init moves to the FINAL layer."""
    net = PredictionNetwork(latent_channels=16, action_space=5,
                            support_size=10, num_bins=21,
                            head_channels=16, fc_hidden=32)
    assert net.policy_hidden.out_features == 32
    net.eval()
    with torch.no_grad():
        logits, value, _ = net(torch.rand(2, 16, 6, 6))
    # Logits are EXACTLY zero; the decoded scalar carries float32 residue
    # from the uniform expectation through h_inverse (~1e-5).
    assert torch.allclose(logits, torch.zeros_like(logits), atol=0)
    assert torch.allclose(value, torch.zeros(2, 1), atol=1e-3)


def test_smallnet_arm_overrides_reach_the_builders(monkeypatch):
    """The smallnet arm's knobs must actually shape the nets -- guards the
    from-import capture bug (LATENT_CHANNELS bound at import time in the
    harness builder would silently ignore the 64-channel override)."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent / "experiments"))
    import ablation
    spec = ablation.ARMS["effzero-repro-smallnet"]
    for key, value in spec.items():
        if not key.startswith("_"):
            monkeypatch.setattr(muzero, key, value, raising=True)
    saved = ablation.ACTIONS
    try:
        ablation.ACTIONS = 6
        nets = ablation.build_networks(spec)
    finally:
        ablation.ACTIONS = saved
    assert nets["representation"].conv4.out_channels == 64
    assert len(nets["dynamics"].blocks) == 1
    assert nets["dynamics"].action_encoding == "plane"
    assert nets["prediction"].head_channels == 16
