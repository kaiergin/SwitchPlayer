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
        if name not in ("pong-kit", "pong-kit2"):
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


def test_arm_config_restores_after_an_exception(ablation):
    import muzero
    before = muzero.DISCOUNT
    with pytest.raises(RuntimeError):
        with ablation.arm_config("horizon-0.9"):
            assert muzero.DISCOUNT == 0.9
            raise RuntimeError("mid-run failure")
    assert muzero.DISCOUNT == before
