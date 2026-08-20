"""Checkpoint/resume for the ablation harness: a killed run must come back
with its full state -- nets, optimizers, the target-model double buffer,
replay buffer, counters, and RNG streams."""
import random
import sys

import numpy as np
import pytest
import torch

import muzero
from tests.test_paper_mode import ACTIONS, make_trainer
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation

CPU = torch.device("cpu")

META = {"arm": "effzero-repro", "seed": 0, "env": "Pong-v4", "iterations": 60,
        "train_steps_per_iter": 2000, "episodes_per_iter": 1,
        "iteration": 20, "training_steps": 40000, "elapsed_s": 1234.5}
EXPECT = {k: META[k] for k in ("arm", "seed", "env", "iterations",
                               "train_steps_per_iter", "episodes_per_iter")}


def make_run():
    run = object.__new__(ablation.Run)
    run.trainer = make_trainer()
    run.replay = []
    run.env_steps = 0
    return run


def state_dicts_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return (sa.keys() == sb.keys()
            and all(torch.equal(sa[k], sb[k]) for k in sa))


def test_checkpoint_round_trips_every_piece_of_state(tmp_path):
    np.random.seed(3)
    a = make_run()
    a.trainer.training_steps = META["training_steps"]
    a.env_steps = 12345
    a.replay = [obs_traj([1.0] * 4, [2.0] * 4)]
    # Non-trivial optimizer moments on every net.
    for net_name, opt_name in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        net, opt = getattr(a.trainer, net_name), getattr(a.trainer, opt_name)
        loss = sum((p ** 2).sum() for p in net.parameters())
        opt.zero_grad(); loss.backward(); opt.step()
    # A target model whose state genuinely differs from the online nets.
    a.trainer._ensure_target_model()
    with torch.no_grad():
        torch.manual_seed(5)
        for p in a.trainer.prediction.parameters():
            p.add_(torch.randn_like(p))
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    # RNG draws AFTER the save: the restored streams must reproduce these.
    want_np = np.random.rand(3)
    want_py = random.random()
    want_torch = torch.rand(2)

    b = make_run()
    loaded = ablation.load_run_checkpoint(b, path, EXPECT)
    assert loaded == {"iteration": 20, "elapsed_s": 1234.5,
                      "init_weights": None}
    assert b.trainer.training_steps == META["training_steps"]
    assert b.env_steps == 12345
    for net_name in ablation.NET_NAMES:
        assert state_dicts_equal(getattr(a.trainer, net_name),
                                 getattr(b.trainer, net_name)), net_name
    for net_name in ablation.NET_NAMES[:4]:
        assert state_dicts_equal(getattr(a.trainer, f"target_{net_name}"),
                                 getattr(b.trainer, f"target_{net_name}"))
    # The double buffer itself: snapshot tensors and the sync clock. The
    # online prediction was mutated after the sync, so target != online here
    # -- a load that silently re-synced would fail the target comparison.
    assert b.trainer._target_synced_at == a.trainer._target_synced_at
    for da, db in zip(a.trainer._target_snapshot, b.trainer._target_snapshot):
        assert da.keys() == db.keys()
        assert all(torch.equal(da[k].cpu(), db[k].cpu()) for k in da)
    assert not state_dicts_equal(b.trainer.prediction,
                                 b.trainer.target_prediction)
    # Optimizer moments.
    for opt_name in ablation.OPT_NAMES:
        oa = getattr(a.trainer, opt_name).state_dict()
        ob = getattr(b.trainer, opt_name).state_dict()
        assert len(oa["state"]) == len(ob["state"]) > 0, opt_name
        for k in oa["state"]:
            assert torch.equal(oa["state"][k]["exp_avg"].cpu(),
                               ob["state"][k]["exp_avg"].cpu())
    # Replay byte-exact.
    assert len(b.replay) == 1 and len(b.replay[0]) == len(a.replay[0])
    for ta, tb in zip(a.replay[0], b.replay[0]):
        assert np.array_equal(ta["obs"], tb["obs"])
        assert ta["reward"] == tb["reward"]
    # RNG streams continue exactly where the save left them.
    assert np.allclose(np.random.rand(3), want_np)
    assert random.random() == want_py
    assert torch.equal(torch.rand(2), want_torch)


def test_resume_refuses_a_mismatched_recipe(tmp_path):
    np.random.seed(4)
    a = make_run()
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = make_run()
    # match covers the advice too (round-9: the mismatch site was the one
    # advice string with no wording pin -- reverting it stayed green).
    with pytest.raises(ValueError, match="iterations") as e:
        ablation.load_run_checkpoint(b, path, {**EXPECT, "iterations": 90})
    assert "THIS command" in str(e.value)


def test_completed_marker_skips_the_run(tmp_path):
    path = tmp_path / "resume.ckpt"
    ablation.mark_run_completed(path, EXPECT)
    assert path.stat().st_size < 100_000   # marker, not a buffer dump
    b = make_run()
    assert ablation.load_run_checkpoint(b, path, EXPECT) is None


def test_completed_marker_with_wrong_recipe_raises(tmp_path):
    """Audit finding: the completed check ran BEFORE recipe validation, so a
    finished 60-iter marker silently skipped a NEW 90-iter --resume
    invocation instead of erroring."""
    path = tmp_path / "resume.ckpt"
    ablation.mark_run_completed(path, EXPECT)
    b = make_run()
    with pytest.raises(ValueError, match="iterations"):
        ablation.load_run_checkpoint(b, path, {**EXPECT, "iterations": 90})


def test_resume_prunes_stale_timeline_rows(tmp_path):
    """Audit finding (HIGH): rows written after the last checkpoint belong
    to the abandoned timeline; resume must drop them or plot() averages
    both timelines as if they were extra seeds."""
    out = tmp_path / "ablation.jsonl"
    rows = [{"arm": "effzero-repro", "seed": 0, "iteration": it, "return_mean": it}
            for it in (10, 20, 30)]
    rows.append({"arm": "effzero-repro", "seed": 1, "iteration": 30,
                 "return_mean": 99})   # other seed: must survive
    with open(out, "w") as fh:
        for r in rows:
            fh.write(__import__("json").dumps(r) + "\n")
    pruned = ablation.prune_stale_rows(out, "effzero-repro", 0, start_it=20)
    assert pruned == 1
    kept = [__import__("json").loads(l) for l in open(out)]
    assert [(r["seed"], r["iteration"]) for r in kept] == [(0, 10), (0, 20), (1, 30)]
    # Idempotent when nothing is stale.
    assert ablation.prune_stale_rows(out, "effzero-repro", 0, start_it=20) == 0


def write_rows(out, rows):
    import json
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_fresh_start_refuses_an_existing_checkpoint(tmp_path):
    """Audit finding: a fresh run at the same path clobbered a killed run's
    resumable state at its first save; its completion marker then satisfied
    the original run's later --resume silently."""
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    ablation.refuse_conflicting_state(ck, out, "a", 0, False, False)  # clean: fine
    ck.write_bytes(b"x")
    ablation.refuse_conflicting_state(ck, out, "a", 0, True, False)   # resuming: fine
    with pytest.raises(SystemExit, match="--resume"):
        ablation.refuse_conflicting_state(ck, out, "a", 0, False, False)


def test_fresh_start_refuses_when_rows_exist_without_a_checkpoint(tmp_path):
    """Audit round-2 HIGH: every no-checkpoint path (pre-checkpointing runs,
    kills before the first save, --checkpoint-every 0) could append a second
    timeline over existing rows. Protection must key on the ROWS, not just
    the checkpoint."""
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    with pytest.raises(SystemExit, match="--overwrite"):        # fresh
        ablation.refuse_conflicting_state(ck, out, "a", 0, False, False)
    with pytest.raises(SystemExit, match="--overwrite"):        # resume, no ck
        ablation.refuse_conflicting_state(ck, out, "a", 0, True, False)
    # Other (arm, seed) pairs are unaffected.
    ablation.refuse_conflicting_state(ck, out, "a", 1, False, False)
    ablation.refuse_conflicting_state(ck, out, "b", 0, False, False)


def test_overwrite_drops_rows_and_checkpoint_and_proceeds(tmp_path):
    import json
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    ck.write_bytes(b"x")
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10},
                     {"arm": "a", "seed": 1, "iteration": 10}])
    ablation.refuse_conflicting_state(ck, out, "a", 0, False, True)
    assert not ck.exists()
    kept = [json.loads(l) for l in open(out)]
    assert kept == [{"arm": "a", "seed": 1, "iteration": 10}]


def test_torn_final_line_is_dropped_but_midfile_corruption_raises(tmp_path):
    """Audit round-2 MEDIUM: a kill mid-append leaves a partial final line;
    prune (and plot) must drop it rather than crash -- but corruption
    mid-file is not a kill artifact and must still raise."""
    import json
    out = tmp_path / "ablation.jsonl"
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    with open(out, "a") as fh:
        fh.write('{"arm": "a", "se')          # torn append
    assert ablation.prune_stale_rows(out, "a", 0, start_it=99) == 1
    assert [json.loads(l) for l in open(out)] == [
        {"arm": "a", "seed": 0, "iteration": 10}]
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    with open(out, "r+") as fh:
        content = fh.read()
    with open(out, "w") as fh:
        fh.write('{"torn": tr\n' + content)   # corrupt FIRST line
    with pytest.raises(Exception):
        ablation.prune_stale_rows(out, "a", 0, start_it=99)


def test_completed_pair_with_rows_is_not_refused_on_resume(tmp_path):
    """Round-3 flagship ordering pin: (marker exists AND rows exist AND
    --resume) must proceed to the marker skip, NOT refuse on the rows. A
    'simplification' checking rows before ck.exists() would refuse every
    completed pair in a multi-seed sweep."""
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    ablation.mark_run_completed(ck, EXPECT)
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    ablation.refuse_conflicting_state(ck, out, "a", 0, True, False)  # no raise
    b = make_run()
    assert ablation.load_run_checkpoint(b, ck, EXPECT) is None       # skip


def test_torn_tail_is_healed_before_any_append(tmp_path):
    """Round-3 HIGH: read-side tolerance without write-side healing let the
    next append concatenate onto the torn fragment, creating MID-FILE
    corruption that crashed every reader -- including --overwrite, the
    advertised recovery."""
    import json
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    with open(out, "a") as fh:
        fh.write('{"arm": "a", "se')                    # torn, no newline
    # Any run path heals before appending: pair (b, 0) starts fresh here.
    ablation.refuse_conflicting_state(ck, out, "b", 0, False, False)
    content = open(out).read()
    assert content.endswith("\n")
    assert [json.loads(l) for l in open(out)] == [
        {"arm": "a", "seed": 0, "iteration": 10}]
    # And --overwrite on a torn file must not crash (it prunes tolerantly).
    with open(out, "a") as fh:
        fh.write('{"arm": "a", "se')
    ablation.refuse_conflicting_state(ck, out, "a", 0, False, True)
    assert open(out).read() == ""


def test_overwrite_unlinks_the_pair_weights(tmp_path, monkeypatch):
    """Round-3 LOW: a redo dying mid-run must not leave the OLD run's
    completed weights beside the NEW run's partial rows."""
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "-pong-skip4")
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    w = tmp_path / "weights-pong-skip4-a-seed0.pth"
    w.write_bytes(b"old")
    other = tmp_path / "weights-pong-skip4-a-seed1.pth"
    other.write_bytes(b"keep")
    ablation.refuse_conflicting_state(ck, out, "a", 0, False, True)
    assert not w.exists()
    assert other.exists()


def test_newline_terminated_torn_line_is_also_healed(tmp_path):
    """Round-4 finding: heal keyed on a missing final newline, but readers
    tolerate any undecodable FINAL line -- so '{"torn\\n' passed heal,
    passed refusal, and the next append created mid-file corruption."""
    import json
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    with open(out, "a") as fh:
        fh.write('{"torn\n')                  # undecodable, newline-terminated
    ablation.refuse_conflicting_state(ck, out, "b", 0, False, False)
    with open(out, "a") as fh:                # the append heal must protect
        fh.write(json.dumps({"arm": "b", "seed": 0, "iteration": 1}) + "\n")
    rows, torn = ablation.read_rows_tolerant(out)   # must NOT raise
    assert not torn
    assert [r["arm"] for r in rows] == ["a", "b"]


def test_completed_marker_semantics(tmp_path):
    """Marker + live rows = finished pair, skip; marker + zero rows = an
    archived era's leftover, drop it and run fresh."""
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    ablation.mark_run_completed(ck, EXPECT)
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    assert ablation.handle_completed_marker(ck, out, "a", 0) == "skip"
    assert ck.exists()
    write_rows(out, [])                       # rows archived away
    assert ablation.handle_completed_marker(ck, out, "a", 0) == "fresh"
    assert not ck.exists()


def test_resume_and_overwrite_together_is_a_parser_error(monkeypatch, capsys):
    """Round-4 finding companion: a hedged '--resume --overwrite' must die
    at the parser, not silently destroy resumable state."""
    monkeypatch.setattr(sys, "argv",
                        ["ablation.py", "--resume", "--overwrite"])
    with pytest.raises(SystemExit) as e:
        ablation.main()
    assert e.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_seed_list_selects_a_single_pair():
    """Round-4 finding: there was NO invocation that overwrote seed 3
    without destroying seeds 0-2; --seed provides one."""
    assert ablation.seed_list(None, 3) == [0, 1, 2]
    assert ablation.seed_list(3, 5) == [3]
    assert ablation.seed_list(0, 5) == [0]


def test_seed_flag_wires_through_main(tmp_path, monkeypatch, capsys):
    """Round-5 finding: seed_list was pinned but main()'s use of it was not
    -- reverting the loop to range(args.seeds) left all tests green while
    --seed silently did nothing (and the advice strings claimed otherwise).
    Also pins the banner printing the EFFECTIVE seed list."""
    calls = []
    monkeypatch.setattr(ablation, "run_one",
                        lambda arm, seed, *a, **k: calls.append((arm, seed)))
    monkeypatch.setattr(ablation, "plot", lambda *a, **k: None)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    for g in ("ENV_NAME", "ACTIONS", "EPISODE_CAP", "TAG", "FRAMESKIP"):
        monkeypatch.setattr(ablation, g, getattr(ablation, g))
    monkeypatch.setattr(sys, "argv",
                        ["ablation.py", "--arms", "per-step", "--seed", "3"])
    ablation.main()
    assert calls == [("per-step", 3)]
    assert "seeds [3]" in capsys.readouterr().out


def test_stale_marker_call_site_reruns_fresh(tmp_path, monkeypatch, capsys):
    """Round-5 finding companion: handle_completed_marker was pinned but its
    run_one call site was not -- an unconditional skip-on-marker regression
    stayed green. iterations=0 keeps the rerun's loop empty."""
    import types
    monkeypatch.setattr(ablation, "Run", lambda arm, seed, spec:
                        types.SimpleNamespace(env=types.SimpleNamespace(
                            close=lambda: None)))
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "EPISODE_CAP", 600)
    out = tmp_path / "ablation.jsonl"
    ck = ablation.resume_path("per-step", 0)
    ablation.mark_run_completed(
        ck, ablation.run_expect("per-step", 0, 0, 1, 1, 1))
    write_rows(out, [{"arm": "per-step", "seed": 0, "iteration": 1}])
    ablation.run_one("per-step", 0, 0, 1, 1, 1, out,
                     checkpoint_every=1, resume=True)
    assert "already completed, skipping" in capsys.readouterr().out
    write_rows(out, [])                       # rows archived away
    ablation.run_one("per-step", 0, 0, 1, 1, 1, out,
                     checkpoint_every=1, resume=True)
    assert "archived era; removed, starting fresh" in capsys.readouterr().out
    # Round-6 finding: capsys asserts alone were regression-blind -- an
    # early return AFTER the print passed them. The fall-through is pinned
    # by the completion marker the run must RECREATE at its end
    # (handle_completed_marker unlinked it; only mark_run_completed at the
    # loop's end brings it back).
    assert ck.exists(), "fresh rerun never reached completion"


def test_env_tag_fragment_is_filename_safe():
    """Round-7 finding: ALE/Pong-v5 put a path separator in TAG, so every
    output resolved into a nonexistent subdirectory and the run died at
    its first write after eval_every iterations of training."""
    assert ablation.env_tag_fragment("ALE/Pong-v5") == "alepong"
    assert ablation.env_tag_fragment("Pong-v4") == "pong"
    assert "/" not in ablation.env_tag_fragment("ALE/MsPacman-v5")
    assert "\\" not in ablation.env_tag_fragment("ALE\\Weird-v0")


def test_ale_namespaced_env_wires_a_safe_tag_through_main(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ablation, "run_one", lambda *a, **k: None)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    for g in ("ENV_NAME", "ACTIONS", "EPISODE_CAP", "TAG", "FRAMESKIP"):
        monkeypatch.setattr(ablation, g, getattr(ablation, g))
    monkeypatch.setattr(sys, "argv",
                        ["ablation.py", "--env", "ALE/Pong-v5", "--actions",
                         "6", "--plot-only"])
    ablation.main()                      # must not raise
    assert "/" not in ablation.TAG and "\\" not in ablation.TAG
    # plot() on the (missing) tagged file degrades politely (round-7 LOW).
    assert "no results to plot" in capsys.readouterr().out


def test_fresh_start_advice_names_this_command(tmp_path):
    """Round-7 finding: round 6 fixed the mismatch site's advice but left
    both fresh-start refusals telling the user to rerun their ORIGINAL
    command -- burning hours retraining an abandoned experiment. All
    conflict state was found under THIS command's flags by construction."""
    ck, out = tmp_path / "resume.ckpt", tmp_path / "ablation.jsonl"
    ck.write_bytes(b"x")
    with pytest.raises(SystemExit, match="THIS command"):
        ablation.refuse_conflicting_state(ck, out, "a", 0, False, False)
    ck.unlink()
    write_rows(out, [{"arm": "a", "seed": 0, "iteration": 10}])
    with pytest.raises(SystemExit, match="THIS command"):
        ablation.refuse_conflicting_state(ck, out, "a", 0, False, False)
    # Round-8 finding: the resume-with-rows-but-no-checkpoint branch was
    # the fourth sibling, still advising the ORIGINAL command.
    with pytest.raises(SystemExit, match="THIS command"):
        ablation.refuse_conflicting_state(ck, out, "a", 0, True, False)


def test_plot_warns_on_mixed_env_histories(tmp_path, capsys):
    out = tmp_path / "ablation.jsonl"
    write_rows(out, [
        {"arm": "a", "seed": 0, "iteration": 1, "return_mean": 1.0,
         "env": "Pong-v4", "frameskip": 4},
        {"arm": "a", "seed": 1, "iteration": 1, "return_mean": 2.0,
         "env": "Pong-v5", "frameskip": 4},
    ])
    ablation.plot(out, tmp_path)
    assert "mixes runs from" in capsys.readouterr().out
    # Round-8 finding: pre-provenance rows (no env/frameskip fields) were
    # guarded OUT of the mixed set, so a legacy/modern straddle -- exactly
    # the rows the warning was retrofitted for -- stayed silent.
    write_rows(out, [
        {"arm": "a", "seed": 0, "iteration": 1, "return_mean": 1.0,
         "env": "Pong-v4", "frameskip": 4},
        {"arm": "a", "seed": 1, "iteration": 1, "return_mean": 2.0},
    ])
    ablation.plot(out, tmp_path)
    assert "mixes runs from" in capsys.readouterr().out


def test_eval_every_zero_does_not_divide(tmp_path, monkeypatch, capsys):
    """Round-8 finding: the --eval-every 0 guard was the one round-7 fix
    with no pinning test -- reverting it left the suite green. This drives
    one real loop iteration through the guard (stubbed Run)."""
    import types

    # Structural pin FIRST (before Run is monkeypatched away): the stub
    # assumes this signature; drift must trip loudly here rather than
    # leave the stub permanently green.
    import inspect
    assert list(inspect.signature(ablation.Run.play_episode).parameters) \
        == ["self", "greedy", "cap"]

    def stub_run(arm, seed, spec):
        net = lambda: types.SimpleNamespace(state_dict=lambda: {})
        return types.SimpleNamespace(
            env=types.SimpleNamespace(close=lambda: None),
            searcher=types.SimpleNamespace(
                network_manager=types.SimpleNamespace(training_steps=0)),
            trainer=types.SimpleNamespace(
                training_steps=0, representation=net(), dynamics=net(),
                prediction=net(), value_prefix=net()),
            replay=[], env_steps=0,
            play_episode=lambda greedy=False, cap=None: ([], 0.0, 0.0, True),
            reanalyse=lambda traj, terminated=True: traj,
            set_train_mode=lambda t: None,
            evaluate=lambda: {"return_mean": 0.0, "return_max": 0.0,
                              "prefix": 0.0, "control": 0.0, "per_step": 0.0})

    monkeypatch.setattr(ablation, "Run", stub_run)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "EPISODE_CAP", 600)
    out = tmp_path / "ablation.jsonl"
    ablation.run_one("per-step", 0, 1, 1, 0, 0, out,
                     checkpoint_every=0, resume=False)
    assert "iter   1" in capsys.readouterr().out


def test_wrapper_arm_refuses_without_frameskip(tmp_path, monkeypatch):
    """Round-10 finding: the wrapper-arm refusal had no pinning test. Both
    layers: run_one's per-pair check, and main()'s pre-flight (which stops
    a mixed sweep BEFORE non-wrapper arms burn hours into the untagged
    history)."""
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "EPISODE_CAP", 3000)
    monkeypatch.setattr(ablation, "FRAMESKIP", None)
    with pytest.raises(SystemExit, match="skip4"):
        ablation.run_one("effzero-repro", 0, 1, 1, 0, 1,
                         tmp_path / "ablation.jsonl",
                         checkpoint_every=0, resume=False)
    # main()'s pre-flight dies at the parser, before anything runs.
    monkeypatch.setattr(sys, "argv",
                        ["ablation.py", "--arms", "effzero-repro",
                         "--env", "Pong-v4"])
    with pytest.raises(SystemExit) as e:
        ablation.main()
    assert e.value.code == 2


def test_init_weights_warm_start_round_trips(tmp_path):
    """--init-weights loads the four function nets from a completed run's
    weights file (strict), returns its training_steps, and leaves
    everything else fresh."""
    a = make_run()
    with torch.no_grad():
        torch.manual_seed(9)
        for net_name in ("representation", "dynamics", "prediction",
                         "value_prefix"):
            for p in getattr(a.trainer, net_name).parameters():
                p.add_(torch.randn_like(p))
    w = tmp_path / "weights.pth"
    torch.save({"arm": "x", "seed": 0, "training_steps": 12345,
                **{n: getattr(a.trainer, n).state_dict()
                   for n in ("representation", "dynamics", "prediction",
                             "value_prefix")}}, w)
    b = make_run()
    assert not state_dicts_equal(a.trainer.prediction, b.trainer.prediction)
    assert ablation.load_init_weights(b.trainer, w) == 12345
    for net_name in ("representation", "dynamics", "prediction",
                     "value_prefix"):
        assert state_dicts_equal(getattr(a.trainer, net_name),
                                 getattr(b.trainer, net_name)), net_name
    # A non-weights file fails loudly, naming the missing net.
    bad = tmp_path / "bad.pth"
    torch.save({"representation": a.trainer.representation.state_dict()}, bad)
    with pytest.raises(SystemExit, match="dynamics"):
        ablation.load_init_weights(b.trainer, bad)


def test_lineage_flows_through_the_checkpoint(tmp_path):
    """Audit finding: a paused warm-started run resumed WITHOUT
    --init-weights (the parser forbids it) stamped its later rows as
    from-scratch. The checkpoint meta carries the lineage instead."""
    np.random.seed(21)
    a = make_run()
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, {**META, "init_weights": "w.pth"})
    b = make_run()
    loaded = ablation.load_run_checkpoint(b, path, EXPECT)
    assert loaded["init_weights"] == "w.pth"


def test_overwrite_refuses_the_pairs_own_weights_as_init(tmp_path, monkeypatch):
    """Audit finding: --overwrite unlinks the pair's weights BEFORE the init
    load reads them -- the natural continue-in-place command destroyed the
    only copy of the trained nets."""
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "-pong-skip4")
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "EPISODE_CAP", 3000)
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    w = tmp_path / "weights-pong-skip4-effzero-repro-refresh5x-seed0.pth"
    w.write_bytes(b"x")
    with pytest.raises(SystemExit, match="Copy"):
        ablation.run_one("effzero-repro-refresh5x", 0, 1, 1, 0, 1,
                         tmp_path / "ablation.jsonl", checkpoint_every=0,
                         resume=False, overwrite=True, init_weights=str(w))
    assert w.exists(), "refusal must happen BEFORE the unlink"


def test_plot_warns_on_mixed_lineages(tmp_path, capsys):
    """Audit finding: warm-started and from-scratch rows of the same env
    averaged into one curve with no warning."""
    out = tmp_path / "ablation.jsonl"
    write_rows(out, [
        {"arm": "a", "seed": 0, "iteration": 1, "return_mean": 1.0,
         "env": "Pong-v4", "frameskip": 4, "init_weights": "w.pth"},
        {"arm": "a", "seed": 1, "iteration": 1, "return_mean": 2.0,
         "env": "Pong-v4", "frameskip": 4},
    ])
    ablation.plot(out, tmp_path)
    assert "mixes runs from" in capsys.readouterr().out


def test_resume_and_init_weights_is_a_parser_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["ablation.py", "--resume", "--init-weights", "x.pth"])
    with pytest.raises(SystemExit) as e:
        ablation.main()
    assert e.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_run_expect_covers_the_full_protocol(monkeypatch):
    """Audit round-2 LOW: episode_cap/eval_every joined the validated recipe
    but nothing pinned it -- reverting that hunk left the suite green."""
    monkeypatch.setattr(ablation, "EPISODE_CAP", 3000)
    e = ablation.run_expect("effzero-repro", 0, 90, 2000, 1, 10)
    assert e["episode_cap"] == 3000
    assert e["eval_every"] == 10
    assert set(e) == {"arm", "seed", "env", "iterations",
                      "train_steps_per_iter", "episodes_per_iter",
                      "episode_cap", "eval_every"}


def test_failed_save_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    np.random.seed(6)
    a = make_run()
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ablation.torch, "save", boom)
    with pytest.raises(RuntimeError):
        ablation.save_run_checkpoint(a, path, {**META, "iteration": 30})
    monkeypatch.undo()
    b = make_run()
    loaded = ablation.load_run_checkpoint(b, path, EXPECT)
    assert loaded["iteration"] == 20, "failed save clobbered the checkpoint"
