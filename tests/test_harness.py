"""Static pins on experiments/ablation.py.

The harness produces every conclusion this repo draws and had no tests at all
(suite audit, 2026-08-17). The shipped --frameskip bug -- a name assigned in
main() but missing from its `global` declaration, so the flag silently never
reached the env while the banner claimed it did -- is the exact class the AST
test below pins forever.
"""
import ast
import pathlib
import sys

import pytest

EXPERIMENTS = pathlib.Path(__file__).resolve().parent.parent / "experiments"
ABLATION = EXPERIMENTS / "ablation.py"


def _assigned_names(node):
    """Every Name that `node` (a statement or function body) assigns,
    including tuple/starred unpacking, augmented, and annotated assignments
    -- the round-4 audit showed the plain-Assign-only version let
    `A, B = ...` shadow module globals invisibly."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        else:
            continue
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def test_every_module_global_assigned_in_main_is_declared():
    """Any module-level name that main() assigns MUST be in its global
    declaration, or the assignment creates a silent local shadow."""
    tree = ast.parse(ABLATION.read_text(encoding="utf-8"))
    module_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            module_names |= _assigned_names(node)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    declared = {name for n in ast.walk(main)
                if isinstance(n, ast.Global) for name in n.names}
    shadowed = (_assigned_names(main) & module_names) - declared
    assert not shadowed, (
        f"main() assigns module-level names without declaring them global: "
        f"{sorted(shadowed)} -- the --frameskip silent-no-op bug class")


@pytest.fixture(scope="module")
def ablation():
    sys.path.insert(0, str(EXPERIMENTS))
    import ablation as mod
    return mod


def test_legacy_arms_pin_augmentation_off(ablation):
    """Arms defined before augmentation existed must keep their archived
    meaning: everything except the kit arms runs without augmentation."""
    for name, spec in ablation.ARMS.items():
        assert "USE_AUGMENTATION" in spec, f"{name} has no augmentation pin"
        if name not in ("pong-kit", "pong-kit2", "effzero-repro"):
            assert spec["USE_AUGMENTATION"] is False, name


def test_every_arm_override_targets_a_real_muzero_global(ablation):
    """An arm key with no muzero counterpart would be applied via setattr and
    silently read by nothing."""
    import muzero
    for name, spec in ablation.ARMS.items():
        for key in spec:
            if key.startswith("_"):
                continue
            assert hasattr(muzero, key), (
                f"arm {name!r} overrides {key!r}, which muzero does not define")


def test_paper_targets_arms_also_search_act(ablation):
    """PAPER_TARGETS consumes 5-field search records; an arm setting it
    without USE_SEARCH_ACTING would crash at its first reanalyse."""
    for name, spec in ablation.ARMS.items():
        if spec.get("PAPER_TARGETS"):
            assert spec.get("USE_SEARCH_ACTING"), name


def test_paper_targets_rejects_recordless_trajectories(ablation):
    """The guard for the flag-mix hazard: a clear error instead of a
    five-way unpack ValueError deep in the loop."""
    import types

    import numpy as np
    run = object.__new__(ablation.Run)
    run.trainer = types.SimpleNamespace(training_steps=0)
    with pytest.raises(ValueError, match="USE_SEARCH_ACTING"):
        run.paper_targets([(np.zeros(1), 0, 0.0)], True)


def test_arm_config_restores_after_an_exception(ablation):
    import muzero
    before = muzero.DISCOUNT
    with pytest.raises(RuntimeError):
        with ablation.arm_config("horizon-0.9"):
            assert muzero.DISCOUNT == 0.9
            raise RuntimeError("mid-run failure")
    assert muzero.DISCOUNT == before


def test_greedy_eval_searches_only_in_paper_mode(ablation, monkeypatch):
    """Round-9 finding: the paper-mode commit made eval search unconditional
    for every search-acting arm, silently inverting pong-act-search's _doc
    ("greedy eval still never searches") against its own archived rows.
    Greedy eval searches ONLY under PAPER_TARGETS (the papers evaluate with
    search); non-paper search-acting arms measure the deployable raw-policy
    agent."""
    import types

    import numpy as np
    import muzero
    calls = []

    async def fake_search(obs, greedy=False, **kw):
        calls.append(greedy)
        return 0, np.ones(5) / 5, 0.0

    class OneStepEnv:
        def reset(self, **kw):
            return np.zeros((210, 160, 3), dtype=np.uint8), {}

        def step(self, action):
            return (np.zeros((210, 160, 3), dtype=np.uint8),
                    0.0, True, False, {})

    run = object.__new__(ablation.Run)
    run.env = OneStepEnv()
    run.env_steps = 0
    run.searcher = types.SimpleNamespace(
        act_search=fake_search,
        network_manager=types.SimpleNamespace(
            initial_inference=lambda s: (np.ones(5) / 5, 0.0)))
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    monkeypatch.setattr(muzero, "PAPER_TARGETS", False)
    run.play_episode(greedy=True)
    assert calls == [], "non-paper greedy eval ran a search"
    run.play_episode(greedy=False)
    assert calls == [False], "self-play must still search"
    monkeypatch.setattr(muzero, "PAPER_TARGETS", True)
    run.play_episode(greedy=True)
    # Exact sequence, not membership: a gate that searched greedy eval but
    # dropped self-play search would also show True in a looser assert.
    assert calls == [False, True]
