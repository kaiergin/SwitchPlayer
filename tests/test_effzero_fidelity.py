"""The EfficientZero-fidelity bundle (2026-09-24 audit): flag-gated ports of
the verified deviations from EfficientZero's released code, and the
effzero-repro-refresh5x-communal-edge-faithful arm that turns them all on.

Every flag defaults to the repo's original behaviour; these tests pin both
sides and the exact gradient arithmetic each port claims."""
import asyncio
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_reanalysis import ACTIONS, build_trainer, fake_batch

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation

FAITHFUL = "effzero-repro-refresh5x-communal-edge-faithful"
FLAGS = {"AUG_INDEPENDENT_TARGETS", "AUG_CLAMP", "OPTIMIZER", "LEARNING_RATE",
         "LR_WARMUP_STEPS", "EFFZERO_LOSS_SCALING", "CONSISTENCY_MASK_PADDING",
         "LATENT_RENORMALIZE", "SKIP_NONFINITE_GRAD_STEPS"}


# --------------------------------------------------------------------------
# The arm
# --------------------------------------------------------------------------
def test_faithful_arm_is_edge_plus_exactly_the_fidelity_flags():
    edge = ablation.ARMS["effzero-repro-refresh5x-communal-edge"]
    arm = ablation.ARMS[FAITHFUL]
    delta = {k for k in set(edge) | set(arm)
             if k != "_doc" and edge.get(k, "<absent>") != arm.get(k, "<absent>")}
    assert delta == FLAGS
    assert arm["OPTIMIZER"] == "sgd" and arm["LEARNING_RATE"] == 0.2
    assert arm["LR_WARMUP_STEPS"] == 1000
    assert arm["AUG_INDEPENDENT_TARGETS"] is True and arm["AUG_CLAMP"] is False
    assert arm["EFFZERO_LOSS_SCALING"] is True
    assert arm["CONSISTENCY_MASK_PADDING"] is True
    assert arm["LATENT_RENORMALIZE"] is False
    assert arm["SKIP_NONFINITE_GRAD_STEPS"] is True
    # Unchanged EfficientZero constants the port relies on.
    assert arm["USE_AUGMENTATION"] is True
    assert muzero.WEIGHT_DECAY == 1e-4 and muzero.MAX_GRAD_NORM == 5.0


def test_every_faithful_override_names_a_real_module_constant():
    """arm_config setattr()s blindly: a misspelt key would create a new
    attribute nothing reads, and the run would silently be the edge arm."""
    for key in ablation.ARMS[FAITHFUL]:
        if not key.startswith("_"):
            assert hasattr(muzero, key), key


def test_defaults_are_the_original_behaviour():
    assert muzero.AUG_INDEPENDENT_TARGETS is False
    assert muzero.AUG_CLAMP is True
    assert muzero.OPTIMIZER == "adamw"
    assert muzero.LR_WARMUP_STEPS == 0
    assert muzero.EFFZERO_LOSS_SCALING is False
    assert muzero.CONSISTENCY_MASK_PADDING is False
    assert muzero.LATENT_RENORMALIZE is True
    assert muzero.SKIP_NONFINITE_GRAD_STEPS is False


# --------------------------------------------------------------------------
# Optimizer and warmup
# --------------------------------------------------------------------------
def test_make_optimizer_builds_effzero_sgd(monkeypatch):
    p = [torch.nn.Parameter(torch.zeros(3))]
    assert isinstance(muzero.make_optimizer(p), torch.optim.AdamW)
    monkeypatch.setattr(muzero, "OPTIMIZER", "sgd")
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)
    opt = muzero.make_optimizer(p)
    assert type(opt) is torch.optim.SGD
    g = opt.param_groups[0]
    assert g["lr"] == 0.2 and g["momentum"] == 0.9 and g["weight_decay"] == 1e-4
    monkeypatch.setattr(muzero, "OPTIMIZER", "adam")
    with pytest.raises(ValueError):
        muzero.make_optimizer(p)


def test_lr_warmup_is_linear_then_hands_off(monkeypatch):
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.2)
    lr = lambda: opt.param_groups[0]["lr"]
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 0)
    muzero.apply_lr_warmup([opt], 0)
    assert lr() == 0.2                                  # disabled: untouched
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 1000)
    for step, want in ((0, 0.0), (250, 0.05), (500, 0.1), (1000, 0.2)):
        muzero.apply_lr_warmup([opt], step)
        assert lr() == pytest.approx(want)
    opt.param_groups[0]["lr"] = 0.02                    # e.g. a later decay
    muzero.apply_lr_warmup([opt], 1001)
    assert lr() == 0.02


def test_train_step_applies_the_warmup(monkeypatch):
    monkeypatch.setattr(muzero, "OPTIMIZER", "sgd")
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 1000)
    torch.manual_seed(0)
    t = build_trainer()
    t.training_steps = 250
    np.random.seed(0)
    mask = np.ones((2, muzero.UNROLL_STEPS + 1), dtype=bool)
    asyncio.run(t.train_step(fake_batch(2, muzero.UNROLL_STEPS, mask)))
    for opt in (t.rep_optimizer, t.dyn_optimizer, t.pred_optimizer,
                t.prefix_optimizer, t.proj_optimizer):
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# Gradient placement and the consistency mask
# --------------------------------------------------------------------------
def _grads(monkeypatch, weights, flags, mask=None, extra=None, live_heads=False):
    """One train_step from identical weights and batch; returns per-net grads
    and the loss. Augmentation off and clipping out of reach, so gradients
    differ only by what the flags do."""
    monkeypatch.setattr(muzero, "USE_AUGMENTATION", False)
    monkeypatch.setattr(muzero, "MAX_GRAD_NORM", 1e12)
    for k, v in {**weights, **flags}.items():
        monkeypatch.setattr(muzero, k, v)
    torch.manual_seed(0)
    t = build_trainer()
    if live_heads:
        # The heads are zero-initialised, so on the first step they pass no
        # gradient back into the trunk; wake them so trunk gradients exist.
        with torch.no_grad():
            for q in t.prediction.parameters():
                if q.abs().sum() == 0:
                    q.normal_(0.0, 0.05)
    np.random.seed(0)
    K = muzero.UNROLL_STEPS
    if mask is None:
        mask = np.ones((4, K + 1), dtype=bool)
    batch = fake_batch(4, K, mask)
    batch.update(extra or {})
    loss = asyncio.run(t.train_step(batch))
    out = {}
    for name in ("representation", "dynamics", "prediction", "projector"):
        out[name] = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                     for p in getattr(t, name).parameters()]
    return out, loss


ONLY_CONSISTENCY = {"POLICY_LOSS_WEIGHT": 0.0, "VALUE_LOSS_WEIGHT": 0.0,
                    "PREFIX_LOSS_WEIGHT": 0.0, "REWARD_LOSS_WEIGHT": 0.0,
                    "CONSISTENCY_LOSS_WEIGHT": 1.0}
ONLY_POLICY = {"POLICY_LOSS_WEIGHT": 1.0, "VALUE_LOSS_WEIGHT": 0.0,
               "PREFIX_LOSS_WEIGHT": 0.0, "REWARD_LOSS_WEIGHT": 0.0,
               "CONSISTENCY_LOSS_WEIGHT": 0.0}


def _close(a, b, factor=1.0):
    """Equal up to float reassociation, measured against each tensor's scale
    (an element-wise rtol trips on entries that are ~0 in both)."""
    return all(float((x * factor - y).abs().max())
               <= 1e-5 * max(float((x * factor).abs().max()), 1e-12)
               for x, y in zip(a, b))


def _nonzero(g):
    return sum(float(x.abs().sum()) for x in g) > 1e-4


def test_effzero_hook_halves_every_unrolled_gradient_into_the_trunk(monkeypatch):
    """Consistency only. Theirs: a step-k loss reaches the representation at
    0.5^k/K; ours 0.5^(k-1)/K. Because the hook halves the whole gradient
    into each dynamics output, EVERY trunk gradient is exactly half, while
    the projector's own gradient (read straight off the loss) is unchanged."""
    off, loss_off = _grads(monkeypatch, ONLY_CONSISTENCY, {"EFFZERO_LOSS_SCALING": False})
    on, loss_on = _grads(monkeypatch, ONLY_CONSISTENCY, {"EFFZERO_LOSS_SCALING": True})
    assert loss_on == pytest.approx(loss_off)           # forward untouched
    assert _nonzero(off["representation"]) and _nonzero(off["dynamics"])
    assert _close(off["representation"], on["representation"], 0.5)
    assert _close(off["dynamics"], on["dynamics"], 0.5)
    assert _close(off["projector"], on["projector"])


def test_effzero_scaling_puts_step_zero_at_one_over_k(monkeypatch):
    """Policy only, and only step 0 carries a policy target: the whole
    gradient is step 0's, which theirs scales by 1/K and ours does not."""
    K = muzero.UNROLL_STEPS
    mask = np.zeros((4, K + 1), dtype=bool)
    mask[:, 0] = True
    # One-hot targets: fake_batch's uniform target against the zero-init
    # policy head has an exactly-zero gradient (only float noise remains).
    onehot = np.zeros((4, K + 1, ACTIONS))
    onehot[..., 1] = 1.0
    extra = {"target_policy": onehot}
    off, _ = _grads(monkeypatch, ONLY_POLICY, {"EFFZERO_LOSS_SCALING": False}, mask, extra,
                    live_heads=True)
    on, _ = _grads(monkeypatch, ONLY_POLICY, {"EFFZERO_LOSS_SCALING": True}, mask, extra,
                   live_heads=True)
    assert _nonzero(off["prediction"]) and _nonzero(off["representation"])
    assert _close(off["prediction"], on["prediction"], 1.0 / K)
    assert _close(off["representation"], on["representation"], 1.0 / K)


def test_consistency_mask_zeroes_padded_positions(monkeypatch):
    K = muzero.UNROLL_STEPS
    none_real = np.zeros((4, K + 1), dtype=bool)
    none_real[:, 0] = True
    all_real = np.ones((4, K + 1), dtype=bool)
    base, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": False})
    same, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True},
                     extra={"real_mask": all_real})
    dead, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True},
                     extra={"real_mask": none_real})
    assert _nonzero(base["projector"])
    assert _close(base["projector"], same["projector"])
    assert _close(base["representation"], same["representation"])
    assert not _nonzero(dead["projector"]) and not _nonzero(dead["representation"])
    # Row-level: masking only sample 0's positions must change the gradient.
    part = all_real.copy()
    part[0, 1:] = False
    partial, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True},
                        extra={"real_mask": part})
    assert not _close(base["projector"], partial["projector"])


def test_consistency_mask_refuses_a_batch_without_real_mask(monkeypatch):
    with pytest.raises(KeyError):
        _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True})


def test_paper_batch_real_mask_marks_positions_inside_the_trajectory(monkeypatch):
    from tests.test_paper_mode import ACTIONS as PA, fake_paper_traj
    monkeypatch.setattr(muzero, "PER_ENABLED", False)
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", False)
    t = build_trainer()
    t.training_steps = 0
    traj = fake_paper_traj([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 0.0])
    np.random.seed(0)
    batch = t.prepare_batch_paper([traj], 16, 3, PA)
    assert batch["real_mask"].shape == (16, 4) and batch["real_mask"].dtype == bool
    starts = set()
    for b, ref in enumerate(batch["sample_refs"]):
        start = next(i for i, x in enumerate(traj) if x is ref)
        starts.add(start)
        assert batch["real_mask"][b].tolist() == [start + s < 4 for s in range(4)]
    assert len(starts) > 1                               # several windows seen


# --------------------------------------------------------------------------
# Latent renormalisation and augmentation clamp
# --------------------------------------------------------------------------
def _minmax(x):
    f = x.flatten(1)
    return f.min(1).values, f.max(1).values


def test_latent_renormalize_flag_reaches_both_factories(monkeypatch):
    torch.manual_seed(0)
    x = torch.rand(4, 12, *muzero.IMAGE_RESHAPE)
    a = torch.randint(0, ACTIONS, (4,))
    for flag in (True, False):
        monkeypatch.setattr(muzero, "LATENT_RENORMALIZE", flag)
        rep = muzero.build_representation().eval()
        dyn = muzero.build_dynamics(ACTIONS).eval()
        assert rep.renormalize_output is flag and dyn.renormalize_output is flag
        with torch.no_grad():
            lat = rep(x)
            nxt, _, _ = dyn(lat, a)
        for z in (lat, nxt):
            lo, hi = _minmax(z)
            unit = torch.allclose(lo, torch.zeros(4), atol=1e-4) and \
                torch.allclose(hi, torch.ones(4), atol=1e-3)
            assert unit is flag


def test_target_model_mirrors_the_live_renormalize_setting(monkeypatch):
    monkeypatch.setattr(muzero, "LATENT_RENORMALIZE", False)
    t = build_trainer()
    t.representation = muzero.build_representation()
    t.dynamics = muzero.build_dynamics(ACTIONS, support_size=muzero.REWARD_SUPPORT,
                                       num_bins=muzero.REWARD_BINS)
    # Flip the module flag back: the mirror must follow the LIVE nets, not
    # the constant (the same rule as the rest of the target geometry).
    monkeypatch.setattr(muzero, "LATENT_RENORMALIZE", True)
    t._ensure_target_model()
    assert t.target_representation.renormalize_output is False
    assert t.target_dynamics.renormalize_output is False


def test_init_weights_refuses_a_renormalize_mismatch(tmp_path, monkeypatch):
    """The flag is not in the state_dict, so a strict load cannot catch a
    file trained under the other setting; load_init_weights must."""
    from tests.test_checkpoint_resume import make_run
    run = make_run()                                     # renormalize on
    nets = {n: getattr(run.trainer, n).state_dict()
            for n in ("representation", "dynamics", "prediction", "value_prefix")}
    faithful = tmp_path / "faithful.pth"
    torch.save({"arm": FAITHFUL, "spec": ablation.ARMS[FAITHFUL],
                "training_steps": 7, **nets}, faithful)
    with pytest.raises(SystemExit, match="LATENT_RENORMALIZE"):
        ablation.load_init_weights(run.trainer, faithful)
    legacy = tmp_path / "legacy.pth"                     # no spec: renorm on
    torch.save({"arm": "x", "training_steps": 7, **nets}, legacy)
    assert ablation.load_init_weights(run.trainer, legacy) == 7
    run.trainer.representation.renormalize_output = False
    with pytest.raises(SystemExit, match="LATENT_RENORMALIZE"):
        ablation.load_init_weights(run.trainer, legacy)
    assert ablation.load_init_weights(run.trainer, faithful) == 7


def test_warmup_governs_the_step_it_runs_in(monkeypatch):
    """At step 0 the warmed lr is 0, so an SGD step must leave every weight
    untouched -- which fails if the warmup ran after optimizer.step()."""
    monkeypatch.setattr(muzero, "OPTIMIZER", "sgd")
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)
    monkeypatch.setattr(muzero, "LR_WARMUP_STEPS", 1000)
    torch.manual_seed(0)
    t = build_trainer()
    t.training_steps = 0
    before = [p.detach().clone() for p in t.representation.parameters()]
    np.random.seed(0)
    mask = np.ones((2, muzero.UNROLL_STEPS + 1), dtype=bool)
    asyncio.run(t.train_step(fake_batch(2, muzero.UNROLL_STEPS, mask)))
    assert all(torch.equal(a, b) for a, b in zip(before, t.representation.parameters()))


def test_consistency_mask_composes_with_per_weights(monkeypatch):
    """Production batches carry is_weights: masking sample 0's unrolled
    positions must equal zeroing its importance weight for the consistency
    term (the is_weights * real_mask branch)."""
    K = muzero.UNROLL_STEPS
    w = np.array([0.7, 1.0, 0.4, 0.9], dtype=np.float32)
    part = np.ones((4, K + 1), dtype=bool)
    part[0, 1:] = False
    w0 = w.copy()
    w0[0] = 0.0
    masked, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True},
                       extra={"real_mask": part, "is_weights": w})
    zeroed, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True},
                       extra={"real_mask": np.ones((4, K + 1), dtype=bool),
                              "is_weights": w0})
    unmasked, _ = _grads(monkeypatch, ONLY_CONSISTENCY, {"CONSISTENCY_MASK_PADDING": True},
                         extra={"real_mask": np.ones((4, K + 1), dtype=bool),
                                "is_weights": w})
    assert _nonzero(masked["projector"])
    assert _close(zeroed["projector"], masked["projector"])
    assert _close(zeroed["representation"], masked["representation"])
    assert not _close(unmasked["projector"], masked["projector"])


def test_nonfinite_gradient_steps_are_skipped(monkeypatch):
    """A NaN gradient on one weight makes the global norm NaN, and clipping
    then scales EVERY gradient by NaN. With the guard the weights,
    SGD momentum and PER priorities are untouched; without it the weights
    are poisoned (the counterfactual the guard exists for)."""
    monkeypatch.setattr(muzero, "OPTIMIZER", "sgd")
    monkeypatch.setattr(muzero, "LEARNING_RATE", 0.2)

    def one_step(guard):
        monkeypatch.setattr(muzero, "SKIP_NONFINITE_GRAD_STEPS", guard)
        torch.manual_seed(0)
        t = build_trainer()
        t.representation.conv1.weight.register_hook(lambda g: g * float("nan"))
        before = [p.detach().clone() for p in t.representation.parameters()]
        np.random.seed(0)
        mask = np.ones((2, muzero.UNROLL_STEPS + 1), dtype=bool)
        batch = fake_batch(2, muzero.UNROLL_STEPS, mask)
        refs = [{"priority": 1.0}, {"priority": 1.0}]
        batch["sample_refs"] = refs
        asyncio.run(t.train_step(batch))
        after = list(t.representation.parameters())
        return t, before, after, refs

    t, before, after, refs = one_step(True)
    assert t.nonfinite_skips == 1
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert not t.rep_optimizer.state                    # no momentum buffers
    assert [r["priority"] for r in refs] == [1.0, 1.0]
    _, _, after, _ = one_step(False)
    assert any(not torch.isfinite(p).all() for p in after)


def test_aug_clamp_flag(monkeypatch):
    obs = torch.ones(32, 2, 1, 6, 6)
    torch.manual_seed(0)
    clamped = muzero.augment_observations(obs, shift=0, intensity=0.05)
    assert clamped.max().item() <= 1.0
    monkeypatch.setattr(muzero, "AUG_CLAMP", False)
    torch.manual_seed(0)
    raw = muzero.augment_observations(obs, shift=0, intensity=0.05)
    assert raw.max().item() > 1.0
    assert torch.equal(raw.clamp(0.0, 1.0), clamped)
