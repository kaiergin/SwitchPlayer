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
        if name not in ("pong-kit", "pong-kit2", "effzero-repro",
                        "effzero-repro-refresh5x",
                        "effzero-repro-refresh5x-communal",
                        "effzero-repro-smallnet",
                        "effzero-repro-smallbuf20",
                        "effzero-repro-communal-entropyrefresh",
                        "effzero-repro-vtree-refresh",
                        "effzero-repro-vtree-communal",
                        "effzero-repro-vtree-communal8",
                        "effzero-repro-vtree-saturate",
                        "effzero-repro-vtree-sat-stale",
                        "effzero-repro-vtree-sat-guard",
                        "effzero-repro-async-fresh",
                        "effzero-repro-async-throttle",
                        "effzero-repro-async-keepnew",
                        "effzero-repro-vtree-communal8-r12",
                        "effzero-repro-vtree-communal8-r12-edge",
                        "effzero-repro-refresh5x-communal-edge",
                        "effzero-repro-refresh5x-communal-edge-augfix",
                        "effzero-repro-refresh5x-communal-edge-faithful",
                        "effzero-repro-faithful-prioract",
                        "effzero-repro-faithful-prioract-sve",
                        "effzero-repro-faithful-searchact",
                        "effzero-repro-faithful-searchact-noredo",
                        "effzero-repro-faithful-seqvtree",
                        "effzero-repro-faithful-prioract-noredo",
                        "effzero-repro-faithful-prioract-noredo-temp05",
                        "effzero-repro-faithful-prioract-noredo-seqref",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal",
                        "effzero-repro-faithful-searchact8-birth",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-sve",
                        "effzero-repro-faithful-gumbelact8-birth",
                        "effzero-repro-faithful-gumbelact8-birth-mspacman",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-tokens",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-rgrow",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow-student-distill",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow-student-fresh",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen0",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen1-distill",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-gen1-fresh",
                        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-rgrow-pbn"):
            assert spec["USE_AUGMENTATION"] is False, name


def test_checkpoint_reader_handles_both_formats(ablation, tmp_path):
    """_read_checkpoint must load the SPCHKPT2 streamed container AND every
    torch.save file (old checkpoints, completion markers, --init-weights)."""
    import pickle
    import torch

    old = tmp_path / "old.ckpt"
    torch.save({"completed": True, "arm": "x"}, old)
    assert ablation._read_checkpoint(old)["completed"] is True

    new = tmp_path / "new.ckpt"
    with open(new, "wb") as f:
        f.write(ablation.CKPT_STREAM_MAGIC)
        pickle.dump({"arm": "x", "replay_trajs": 2}, f, protocol=5)
        pickle.dump([{"obs": 1}], f, protocol=5)
        pickle.dump([{"obs": 2}], f, protocol=5)
    state = ablation._read_checkpoint(new)
    assert state["arm"] == "x"
    assert [t[0]["obs"] for t in state["replay"]] == [1, 2]
    assert "replay_trajs" not in state


def test_cpu_state_moves_cuda_tensors(ablation):
    """The streamed checkpoint has no torch.load map_location, so _cpu_state
    must strip devices; the rest of the suite runs CPU-only where it is an
    identity, making this the one real exercise of the CUDA branch."""
    import torch
    if not torch.cuda.is_available():
        import pytest as _pytest
        _pytest.skip("no CUDA device")
    src = {"w": torch.ones(3, device="cuda"),
           "nested": [{"b": torch.zeros(2, device="cuda")}, 7],
           "t": (torch.full((2,), 5.0, device="cuda"),)}
    out = ablation._cpu_state(src)
    assert out["w"].device.type == "cpu" and out["w"].sum().item() == 3.0
    assert out["nested"][0]["b"].device.type == "cpu"
    assert out["nested"][1] == 7
    assert out["t"][0].device.type == "cpu" and out["t"][0][0].item() == 5.0
    # Originals untouched: the live nets must stay on their device.
    assert src["w"].device.type == "cuda"


def test_replay_eviction_caps_trajectories_and_transitions(ablation, monkeypatch):
    """evict_replay enforces both caps FIFO on whole trajectories, keeping
    the newest even when it alone exceeds the transition cap."""
    import muzero

    def traj(n):
        return [{"obs": None}] * n

    # Trajectory cap alone.
    monkeypatch.setattr(muzero, "MAX_REPLAY", 3)
    monkeypatch.setattr(muzero, "MAX_REPLAY_TRANSITIONS", 0)
    replay = [traj(10), traj(10), traj(10), traj(10)]
    ablation.evict_replay(replay)
    assert len(replay) == 3
    # Transition cap binds before the trajectory cap.
    monkeypatch.setattr(muzero, "MAX_REPLAY", 85)
    monkeypatch.setattr(muzero, "MAX_REPLAY_TRANSITIONS", 25)
    first, rest = traj(10), [traj(10), traj(10)]
    replay = [first] + rest
    ablation.evict_replay(replay)
    assert replay == rest, "oldest trajectory must be evicted first"
    # The newest survives even when it alone exceeds the cap.
    big = traj(100)
    replay = [traj(10), big]
    ablation.evict_replay(replay)
    assert replay == [big]
    # 0 disables the transition cap.
    monkeypatch.setattr(muzero, "MAX_REPLAY_TRANSITIONS", 0)
    replay = [traj(50), traj(50)]
    ablation.evict_replay(replay)
    assert len(replay) == 2


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
    without USE_SEARCH_ACTING would crash at its first reanalyse -- unless
    BIRTH_SEARCH supplies the records (policy-only acting), in which case
    it must NOT also search-act (birth_search_gate refuses the mix) --
    except under SEARCH_ACT_BIRTH_TARGETS (2026-10-09), where the search
    only picks the action and birth search still makes every target."""
    for name, spec in ablation.ARMS.items():
        if spec.get("PAPER_TARGETS"):
            if spec.get("BIRTH_SEARCH"):
                assert (not spec.get("USE_SEARCH_ACTING")
                        or spec.get("SEARCH_ACT_BIRTH_TARGETS")), name
            else:
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

    # Truncated GREEDY episodes must not burn a tail search either --
    # evaluate() discards the trajectory (round-10 finding, unpinned until
    # round 11). A never-terminating env + cap=1 forces truncation; exactly
    # one per-step search, no second (tail) call.
    class NeverEndsEnv(OneStepEnv):
        def step(self, action):
            return (np.zeros((210, 160, 3), dtype=np.uint8),
                    0.0, False, False, {})

    calls.clear()
    run.env = NeverEndsEnv()
    run.play_episode(greedy=True, cap=1)
    assert calls == [True], "truncated greedy eval burned a tail search"


def test_eval_summary_flags_capped_partial_scores(ablation):
    """Greedy eval in deterministic Pong can lock into a perpetual rally
    that only EPISODE_CAP ends; its return is a partial score. The row
    must carry the mean length and the capped count so such reads are
    visible, alongside the unchanged return_mean/return_max."""
    rows = ablation.eval_summary([(-21.0, 760, True),
                                  (-3.0, 3000, False),   # capped mid-game
                                  (+4.0, 2100, True)])
    assert rows["return_mean"] == pytest.approx((-21.0 - 3.0 + 4.0) / 3)
    assert rows["return_max"] == 4.0
    assert rows["eval_len_mean"] == pytest.approx((760 + 3000 + 2100) / 3)
    assert rows["eval_capped"] == 1
    clean = ablation.eval_summary([(-21.0, 760, True)])
    assert clean["eval_capped"] == 0 and clean["eval_len_mean"] == 760.0
