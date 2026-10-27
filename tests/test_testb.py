"""Test B (Kai, 2026-10-27; CURRICULUM.md "Next experiment (pre-registered)
-- Test B: a fast lineage from scratch"): five generations F0-F4, each
ending at its env-step stop, its within-run schedules following its
env-step progress, its own play logged at raw-step resolution for the next
generation's distillation set, the discount 0.97 -> 0.997 per swap. Pins
the arms, the env-keyed schedules (off: bit for bit), the stop gate, the
own-play log and its reader, the actor's and the pipeline's logging, the
loop's stop, student_arm's / student_gate's discount, the prep's discount
rule, value check and log reader, master_spec's discount -- and, slow, a
prep from a real logged F0-like master and F1's offline phase on it. CPU
only."""
import inspect
import json
import pathlib
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import distill
import muzero
import ray_actors
from tests.test_gen1 import _boosted, _pin
from tests.test_prioract2 import restore_rngs  # noqa: F401

TK_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-rgrow")
G0_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
          "tokens-gen0")
DS2_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
           "tokens-gen1-distill")
FAST = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-"
        "tokens-fast")
F = [f"{FAST}-f{k}" for k in range(5)]
ENV_ID = "MsPacmanNoFrameskip-v4"
TK_VOCAB = ablation.ARMS[TK_ARM]["TOKEN_VOCAB"]
DROP = ((0, 8), (0, 32), (5, 16), (1, 16))
F1_VOCAB = tuple(t for t in TK_VOCAB if t not in DROP)


@pytest.fixture
def arms(monkeypatch):
    """ARMS restored after a test applies --token-vocab."""
    for arm in F + [G0_ARM, DS2_ARM, TK_ARM]:
        monkeypatch.setitem(ablation.ARMS, arm, ablation.ARMS[arm])


@pytest.fixture
def globals_kept(monkeypatch):
    for k in ("TEMPERATURE_TOTAL_STEPS", "OFFPOLICY_DECAY_STEPS",
              "SCHEDULE_ENV_TARGET", "SCHEDULE_ENV_STEPS",
              "SCHEDULE_OFFSET_STEPS", "OWN_PLAY_LOG", "DISCOUNT"):
        monkeypatch.setattr(muzero, k, getattr(muzero, k))


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def test_f0_is_g0_with_the_fix_bbfs_discount_and_the_env_stop():
    g0, f0 = ablation.ARMS[G0_ARM], ablation.ARMS[F[0]]
    d = {k for k in set(g0) | set(f0) if g0.get(k) != f0.get(k)}
    assert d == {"PREFIX_BN_PER_STEP", "DISCOUNT", "OWN_PLAY_LOG",
                 "_env_stop", "_nominal_iterations", "_doc"}
    assert f0["DISCOUNT"] == 0.97 and f0["PREFIX_BN_PER_STEP"] is True
    assert f0["OWN_PLAY_LOG"] is True and f0["_env_stop"] is True
    assert f0["_nominal_iterations"] == 30
    # the sign clip, TK's 36 tokens, TK's schedules
    assert f0["REWARD_GROWTH_START"] == f0["REWARD_GROWTH_CEILING"] == 1.0
    assert f0["TOKEN_VOCAB"] == TK_VOCAB
    assert f0["PRIOR_ACT_STAGED_TEMP"] is True and f0["TOKEN_LONG_BIAS"] == 0.5
    assert "_student" not in f0
    # the arms' 0.997 per FRAME was 0.988 per base step; F0 starts below
    assert ablation.ARMS[G0_ARM]["DISCOUNT"] == pytest.approx(0.997 ** 4)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_each_later_generation_is_the_previous_ones_distilled_student(k):
    f0, fk = ablation.ARMS[F[0]], ablation.ARMS[F[k]]
    d = {key for key in set(f0) | set(fk) if f0.get(key) != fk.get(key)}
    assert d == {"SCHEDULE_OFFSET_STEPS", "_student", "_distill_offline_steps",
                 "_distill_snapshot_every", "_ceiling_from_prep",
                 "_discount_from_prep", "_master_arm", "_doc"}
    assert fk["_master_arm"] == F[k - 1]
    # (None, as F0's absent key: the master's final iteration is the
    # prep's, pinned by master_checks)
    assert "_master_iterations" in fk and fk["_master_iterations"] is None
    assert (fk["SCHEDULE_OFFSET_STEPS"], fk["_distill_offline_steps"],
            fk["_distill_snapshot_every"]) == (20000, 20000, 4000)
    assert fk["_student"] == "distill"
    assert fk["_ceiling_from_prep"] is True
    assert fk["_discount_from_prep"] is True


# --------------------------------------------------------------------------
# the env-keyed schedules
# --------------------------------------------------------------------------
def test_off_the_schedules_read_the_training_steps_bit_for_bit(globals_kept):
    muzero.TEMPERATURE_TOTAL_STEPS = 60000
    muzero.SCHEDULE_ENV_TARGET = None
    muzero.SCHEDULE_ENV_STEPS = 12345
    for s in (0, 1, 29999, 45000, 70000):
        assert muzero.schedule_steps(s) == s


@pytest.mark.parametrize("env, frac", [(0, 0.0), (5000, 0.25),
                                       (10000, 0.5), (15000, 0.75),
                                       (20000, 1.0), (26000, 1.0)])
def test_on_they_read_the_env_step_progress(globals_kept, env, frac):
    muzero.TEMPERATURE_TOTAL_STEPS = 60000
    muzero.SCHEDULE_ENV_TARGET = 20000
    muzero.SCHEDULE_ENV_STEPS = env
    muzero.SCHEDULE_OFFSET_STEPS = 20000          # a student: inert here
    for s in (0, 20000, 123456):                  # the steps are not read
        assert muzero.schedule_steps(s) == int(60000 * frac)


def test_the_staged_schedules_follow_the_env_steps(globals_kept, arms):
    with ablation.arm_config(F[0]), ablation._run_length_globals(30, 2000):
        muzero.SCHEDULE_ENV_TARGET = 20000
        seen = {}
        for env in range(0, 22001, 1000):
            muzero.SCHEDULE_ENV_STEPS = env
            seen[env] = (muzero.prior_acting_temperature(0),
                         muzero.token_long_bias(0), muzero.per_beta(0))
    # temperature 0.5 / 0.25 / 0.125 at 50% / 75% of the 20k env steps
    assert {seen[e][0] for e in range(0, 10000, 1000)} == {0.5}
    assert {seen[e][0] for e in range(10000, 15000, 1000)} == {0.25}
    assert {seen[e][0] for e in range(15000, 22001, 1000)} == {0.125}
    # the long-token bias falls to 0 by 25% (5k env steps)
    assert seen[0][1] == 0.5 and seen[5000][1] == 0.0
    assert 0 < seen[2000][1] < 0.5
    # PER beta 0.4 -> 1 over the generation's env steps
    assert seen[0][2] == pytest.approx(0.4)
    assert seen[10000][2] == pytest.approx(0.7)
    assert seen[20000][2] == pytest.approx(1.0) == seen[22000][2]


def test_env_schedule_sets_and_restores(globals_kept):
    muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS = None, 7
    with ablation.env_schedule(None):
        assert (muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS) == \
            (None, 7)
    with pytest.raises(RuntimeError):
        with ablation.env_schedule(20000):
            assert (muzero.SCHEDULE_ENV_TARGET,
                    muzero.SCHEDULE_ENV_STEPS) == (20000, 0)
            raise RuntimeError
    assert (muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS) == (None, 7)


def test_the_planned_run_is_the_nominal_length():
    src = inspect.getsource(ablation.run_one)
    assert ('planned = ARMS[arm].get("_nominal_iterations") or iterations'
            in src)
    assert "_run_length_globals(planned, train_steps)" in src
    assert "env_schedule(stop_env_steps)" in src


# --------------------------------------------------------------------------
# the stop gate and the loop's stop
# --------------------------------------------------------------------------
def _gate(arm, stop, ray=True, cpu=True, iterations=60):
    with ablation.arm_config(arm) as spec:
        ablation.env_stop_gate(spec, stop, ray, cpu, iterations)


@pytest.mark.parametrize("stop", [None, 0, -5, True, 20000.0, "20000"])
def test_an_env_stopped_arm_needs_a_positive_stop(stop):
    with pytest.raises(SystemExit, match="--stop-env-steps"):
        _gate(F[0], stop)


def test_the_gate(monkeypatch, arms):
    _gate(F[0], 20000)
    _gate(F[3], 19123)
    with pytest.raises(SystemExit, match="env-stopped arms only"):
        _gate(G0_ARM, 20000)
    _gate(G0_ARM, None)                               # untouched
    with pytest.raises(SystemExit, match="within the --iterations cap"):
        _gate(F[0], 20000, iterations=29)
    with pytest.raises(SystemExit, match="--ray --cpu-actor"):
        _gate(F[0], 20000, cpu=False)
    monkeypatch.setitem(ablation.ARMS, F[1], {**ablation.ARMS[F[1]],
                                              "ASYNC_LOOP": True})
    with pytest.raises(SystemExit, match="ASYNC_LOOP"):
        _gate(F[1], 20000)


def test_the_gate_runs_at_startup_and_the_stop_is_the_recipe():
    src = inspect.getsource(ablation.run_one)
    assert src.index("student_gate(spec, distill_set, snapshot_weights)") < \
        src.index("env_stop_gate(spec, stop_env_steps, use_ray, cpu_actor,")
    assert src.index("env_stop_gate(") < src.index(
        "refuse_conflicting_state(")
    assert 'expect["stop_env_steps"] = int(stop_env_steps)' in src


def test_the_loop_stops_at_the_iteration_that_passes_the_stop():
    """The last iteration: `it == iterations` or the env steps past the
    stop, computed AFTER the iteration's collect; nothing is requested
    after it; it saves the weights, snapshots, writes its row (drained),
    skips the checkpoint and breaks."""
    src = inspect.getsource(ablation.run_one)
    loop = src[src.index("        for it in range(start_it + 1, "
                         "iterations + 1):"):]
    assert "muzero.SCHEDULE_ENV_STEPS = run.env_steps" in loop
    collect = loop.index("                    pipeline.collect()\n"
                         "            # Test B")
    last = loop.index("            last = it == iterations or bool(\n"
                      "                stop_env_steps and run.env_steps "
                      ">= stop_env_steps)")
    assert collect < last
    after = loop[last:]
    assert after.index("if not last:") < after.index("pipeline.sync()") < \
        after.index("pipeline.request(it + 1)")
    assert "            if last:\n                # Save the trained" in after
    assert "if (eval_every and it % eval_every == 0) or last:" in after
    assert "if last:\n                        pipeline.drain_eval()" in after
    assert ("if checkpoint_every and it % checkpoint_every == 0 and not "
            "last:") in after
    body = after[:after.index("        # Also under --resume with "
                              "checkpointing off")]
    assert body.rstrip().endswith("            if last:\n"
                                  "                # Test B: an env-stopped "
                                  "generation ends here (its final\n"
                                  "                # weights, snapshot and "
                                  "row written above)\n"
                                  "                break")
    for old in ("it == iterations:", "it != iterations", "it < iterations"):
        assert old not in body, old


def test_the_cli_carries_the_stop():
    src = inspect.getsource(ablation.main)
    assert 'ap.add_argument("--stop-env-steps", type=int' in src
    assert "stop_env_steps=args.stop_env_steps" in src


# --------------------------------------------------------------------------
# the own-play log
# --------------------------------------------------------------------------
def _life(tag, steps=3):
    stack = np.full((4, 2, 2), tag, dtype=np.uint8)
    return ([(stack, 1, (float(tag),))] * steps,
            {"inner": [[np.full((2, 2), tag, dtype=np.uint8)]] * steps,
             "real_done": tag % 2 == 0, "temperature": 0.5})


def test_own_play_log_writes_one_file_per_iteration(tmp_path):
    log = ablation.OwnPlayLog(tmp_path / "op")
    for it, tags in ((1, (1, 2)), (2, (3,)), (3, (4, 5, 6))):
        for t in tags:
            traj, extra = _life(t)
            log.write(it, traj, extra, terminated=True, steps=len(traj),
                      env_steps=100 * t)
    lives = ablation.read_own_play(tmp_path / "op", 3)
    assert [int(l["traj"][0][0][0, 0, 0]) for l in lives] == [1, 2, 3, 4, 5,
                                                               6]
    assert [l["iteration"] for l in lives] == [1, 1, 2, 3, 3, 3]
    assert lives[1]["real_done"] and not lives[0]["real_done"]
    assert lives[5]["env_steps"] == 600 and lives[5]["temperature"] == 0.5
    # the reader stops at the master's final iteration
    assert len(ablation.read_own_play(tmp_path / "op", 2)) == 3
    with pytest.raises(SystemExit, match="iteration 4 is missing"):
        ablation.read_own_play(tmp_path / "op", 4)


def test_a_resumed_process_truncates_the_iterations_it_replays(tmp_path):
    dead = ablation.OwnPlayLog(tmp_path / "op")
    for it in (1, 2, 3):
        dead.write(it, *_life(it), terminated=True, steps=3, env_steps=it)
    dead.write(3, *_life(9), terminated=True, steps=3, env_steps=9)
    # a resume from the checkpoint after iteration 2 replays iteration 3
    alive = ablation.OwnPlayLog(tmp_path / "op")
    alive.write(3, *_life(7), terminated=True, steps=3, env_steps=7)
    alive.write(3, *_life(8), terminated=True, steps=3, env_steps=8)
    lives = ablation.read_own_play(tmp_path / "op", 3)
    assert [int(l["traj"][0][0][0, 0, 0]) for l in lives] == [1, 2, 7, 8]


def test_the_log_folder_is_the_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "-x")
    assert ablation.own_play_dir(F[0], 0) == (
        tmp_path / f"own-play-x-{F[0]}-seed0")
    src = inspect.getsource(ablation.run_one)
    assert ("if muzero.OWN_PLAY_LOG:\n            run.own_play = "
            "OwnPlayLog(own_play_dir(arm, seed))") in src


def _pipeline(results, run, it=4):
    pipe = object.__new__(ablation.RayPipeline)
    pipe.pending = (["f"], 0)
    pipe.ray = SimpleNamespace(get=lambda futures: results)
    pipe._birth_states = None
    pipe._birth_acting = None
    pipe._request_it = it
    pipe.run = run
    return pipe


def test_collect_logs_each_life_before_admission(tmp_path, globals_kept):
    log = ablation.OwnPlayLog(tmp_path / "op")
    run = SimpleNamespace(env_steps=1000, own_play=log, replay=[],
                          reanalyse=lambda traj, **k: list(traj))
    a, ea = _life(2)
    b, eb = _life(3)
    pipe = _pipeline([(a, True, 3, ea), (b, False, 3, eb)], run)
    pipe.collect(admit=lambda steps: False)       # dropped, still logged
    assert run.env_steps == 1006 and muzero.SCHEDULE_ENV_STEPS == 1006
    with open(log.path(4), "rb") as f:
        got = [pickle.load(f), pickle.load(f)]
    assert [g["env_steps"] for g in got] == [1003, 1006]
    assert [g["terminated"] for g in got] == [True, False]
    assert [g["iteration"] for g in got] == [4, 4]
    assert run.replay == []


def test_collect_without_the_log_is_unchanged(globals_kept):
    run = SimpleNamespace(env_steps=10, own_play=None, replay=[],
                          reanalyse=lambda traj, **k: list(traj))
    a, _ = _life(1)
    _pipeline([(a, True, 3)], run).collect()
    assert run.env_steps == 13 and len(run.replay) == 1


def test_sync_ships_the_env_steps_only_when_keyed(globals_kept):
    calls = []
    actor = SimpleNamespace(set_weights=SimpleNamespace(
        remote=lambda *a: calls.append(a)))
    pipe = object.__new__(ablation.RayPipeline)
    pipe.server, pipe.actor = None, actor
    pipe.ray = SimpleNamespace(get=lambda x: x)
    pipe.run = SimpleNamespace(env_steps=4321,
                               trainer=SimpleNamespace(training_steps=77))
    pipe._acting_states = lambda: {"s": 1}
    muzero.SCHEDULE_ENV_TARGET = None
    pipe.sync()
    muzero.SCHEDULE_ENV_TARGET = 20000
    pipe.sync()
    assert calls == [({"s": 1}, 77), ({"s": 1}, 77, 4321)]


def test_the_actor_ships_its_target_and_takes_the_count():
    src = inspect.getsource(ablation.RayPipeline._make_actor)
    assert '"SCHEDULE_ENV_TARGET": muzero.SCHEDULE_ENV_TARGET' in src
    actor = object.__new__(ray_actors.SelfPlayActor)
    actor.mz = SimpleNamespace(SCHEDULE_ENV_STEPS=0)
    actor.searcher = SimpleNamespace(network_manager=SimpleNamespace(
        training_steps=0))
    with pytest.raises(AttributeError):
        # (no nets on this shell: past the count, the load fails)
        actor.set_weights({"representation": {}}, 5, 999)
    assert actor.mz.SCHEDULE_ENV_STEPS == 999
    actor.set_weights({}, 6)
    assert actor.mz.SCHEDULE_ENV_STEPS == 999      # None leaves it


# --------------------------------------------------------------------------
# student_arm / student_gate: the discount
# --------------------------------------------------------------------------
def _doc_file(tmp_path, d=0.985, c=11.25, master_arm=F[0],
              master_iterations=31, set_path=None, **extra):
    doc = {"vocab": [list(t) for t in F1_VOCAB], "student_ceiling": c,
           "master_arm": master_arm, "master_iterations": master_iterations,
           **extra}
    if d is not None:
        doc["student_discount"] = d
    if set_path is not None:
        doc["set_sha256"] = ablation.file_sha256(set_path)
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(doc))
    return p


def test_student_arm_applies_the_preps_discount(tmp_path, arms):
    ablation.student_arm(F[1], _doc_file(tmp_path, d=0.985))
    spec = ablation.ARMS[F[1]]
    assert spec["DISCOUNT"] == spec["_student_discount"] == 0.985
    assert spec["REWARD_GROWTH_CEILING"] == 11.25
    with ablation.arm_config(F[1]):
        assert muzero.DISCOUNT == 0.985
        assert muzero.token_edge_discounts()[0] == 0.985


@pytest.mark.parametrize("bad", [None, True, 0, 1, 1.2, float("nan"),
                                 0.998, "0.985"])
def test_student_arm_refuses_a_bad_discount(bad, tmp_path, arms):
    with pytest.raises(SystemExit, match="student_discount"):
        ablation.student_arm(F[1], _doc_file(tmp_path, d=bad))


def test_any_master_iteration_but_the_master_arm_is_checked(tmp_path, arms):
    ablation.student_arm(F[1], _doc_file(tmp_path, master_iterations=58))
    with pytest.raises(SystemExit, match="master_arm"):
        ablation.student_arm(F[1], _doc_file(tmp_path, master_arm=F[2]))
    # generation 1 / 2 students still pin their master's 90 iterations
    with pytest.raises(SystemExit, match="master_iterations"):
        ablation.student_arm(DS2_ARM, _doc_file(
            tmp_path, d=None, master_arm=G0_ARM, master_iterations=58))


def test_the_gate_checks_the_discount(tmp_path, arms):
    header = {"format": distill.FORMAT,
              "vocab": [list(t) for t in F1_VOCAB],
              "master_vocab": [list(t) for t in TK_VOCAB],
              "master_snapshot": "m.pth", "master_arm": F[0],
              "student_ceiling": 11.25, "student_discount": 0.985}
    good = tmp_path / "set.pkl"
    distill.write_set(good, header, [])
    other = tmp_path / "other.pkl"
    distill.write_set(other, {**header, "student_discount": 0.9925}, [])
    static = ablation.ARMS[F[1]]

    def gate(dset, mutate=None):
        ablation.ARMS[F[1]] = static
        ablation.student_arm(F[1], _doc_file(tmp_path, set_path=dset))
        with ablation.arm_config(F[1]) as spec:
            if mutate:
                mutate()
            ablation.student_gate(spec, dset, True)

    gate(good)
    with pytest.raises(SystemExit, match="its student_discount"):
        gate(other)

    def moved():
        muzero.DISCOUNT = 0.97
    with pytest.raises(SystemExit, match="prep's student_discount"):
        gate(good, moved)


def test_the_resume_recipe_carries_the_discount(tmp_path, arms):
    ablation.student_arm(F[1], _doc_file(tmp_path))
    with ablation.arm_config(F[1]) as spec, ablation.token_actions():
        out = ablation.student_expect(spec, None)
    assert out["discount"] == 0.985 and out["reward_ceiling"] == 11.25


# --------------------------------------------------------------------------
# the prep: the discount rule, the value check, the log
# --------------------------------------------------------------------------
def test_the_discount_rule_halves_the_gap_and_caps():
    g, seen = 0.97, []
    for _ in range(5):
        seen.append(g)
        g = distill.student_discount_rule(g)
    assert seen == pytest.approx([0.97, 0.985, 0.9925, 0.99625, 0.997])
    assert distill.student_discount_rule(0.997) == 0.997
    assert distill.student_discount_rule(0.988053892081) == pytest.approx(
        0.994026946)


def test_discounted_return_max():
    life = {"traj": [(None, 0, ()), (None, 1, (100.0, 0.0)),
                     (None, 1, (0.0, 2000.0))]}
    # units: 10, 0, 0, 200 -> clipped at 5: 5, 0, 0, 5
    got = distill.discounted_return_max([life], 10.0, 5.0, 0.5)
    assert got == pytest.approx(5 + 0.5 ** 3 * 5)
    neg = {"traj": [(None, 0, (-300.0,))]}
    assert distill.discounted_return_max([neg], 10.0, 5.0, 0.9) == 5.0


def test_the_master_spec_excuses_the_discount_only_where_the_prep_sets_it():
    f1 = {**ablation.ARMS[F[1]], "TOKEN_VOCAB": F1_VOCAB,
          "_token_vocab_sha256": "a" * 64, "_vocab_set_sha256": "b" * 64,
          "REWARD_GROWTH_START": 11.25, "REWARD_GROWTH_CEILING": 11.25,
          "_student_ceiling": 11.25, "DISCOUNT": 0.985,
          "_student_discount": 0.985}
    assert distill.master_spec(F[1], {"spec": f1}, "m") == f1
    with pytest.raises(SystemExit, match="_student_discount"):
        distill.master_spec(F[1], {"spec": {**f1, "DISCOUNT": 0.99}}, "m")
    # DS2 (no prep discount): DISCOUNT must stay its arm's
    ds2 = {**ablation.ARMS[DS2_ARM], "TOKEN_VOCAB": F1_VOCAB,
           "_token_vocab_sha256": "a" * 64, "_vocab_set_sha256": "b" * 64,
           "REWARD_GROWTH_START": 11.25, "REWARD_GROWTH_CEILING": 11.25,
           "_student_ceiling": 11.25}
    distill.master_spec(DS2_ARM, {"spec": ds2}, "m")
    with pytest.raises(SystemExit, match="beyond the student keys"):
        distill.master_spec(DS2_ARM, {"spec": {**ds2, "DISCOUNT": 0.99}}, "m")
    with pytest.raises(SystemExit, match="beyond the student keys"):
        distill.master_spec(DS2_ARM, {"spec": {**ds2,
                                                "_student_discount": 0.99}},
                            "m")


# --------------------------------------------------------------------------
# real Ms. Pac-Man (CPU): an F0-like master's own play logged as the actor
# logs it, its prep from the log, F1's offline phase on it
# --------------------------------------------------------------------------
MASTER_ARM = "test-fast-master"


@pytest.fixture
def logged(monkeypatch, tmp_path, restore_rngs, arms):  # noqa: F811
    """An F0-like master (random nets, a 4-sim search for speed, ending
    after 2 iterations) whose own play -- a few lives with rewards past one
    unit (_boosted) -- is logged as SelfPlayActor.play logs it, then
    prepped from the log."""
    _pin(monkeypatch)
    for k in ("SCHEDULE_ENV_TARGET", "SCHEDULE_ENV_STEPS", "OWN_PLAY_LOG"):
        monkeypatch.setattr(muzero, k, getattr(muzero, k))
    static = {**ablation.ARMS[F[0]], "ACT_SEARCH_SIMS": 4}
    monkeypatch.setitem(ablation.ARMS, MASTER_ARM, static)
    monkeypatch.setitem(ablation.ARMS, F[1], {
        **ablation.ARMS[F[1]], "_master_arm": MASTER_ARM})
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = ENV_ID, 4, 9
    log = ablation.OwnPlayLog(tmp_path / "own-play")
    with ablation.arm_config(MASTER_ARM) as spec, \
            ablation._run_length_globals(2, 2000), ablation.token_actions():
        torch.manual_seed(5)
        run = ablation.Run(MASTER_ARM, 0, spec)
        try:
            n = 0
            for it in (1, 2):
                for _ in range(2):
                    ablation.INNER_FRAMES = []
                    try:
                        traj, _, steps, term = run.play_episode(greedy=False)
                        inner = ablation.INNER_FRAMES
                    finally:
                        ablation.INNER_FRAMES = None
                    n += steps
                    log.write(it, traj, {"inner": inner,
                                         "real_done": bool(
                                             run.env.was_real_done),
                                         "temperature": 0.5},
                              term, steps, n)
            nets = {k: getattr(run.trainer, k) for k in ablation.NET_NAMES}
            run.trainer.training_steps = 4000
            blob = {"arm": MASTER_ARM, "seed": 0, "spec": dict(spec),
                    "training_steps": 4000, "iteration": 2,
                    **{k: v.state_dict() for k, v in nets.items()},
                    "acting_states": {k: nets[k].state_dict()
                                      for k in ("representation",
                                                "prediction")}}
        finally:
            run.env.close()
    # rewards past one unit, as Ms. Pac-Man's power pellets and ghosts pay
    lives = ablation.read_own_play(tmp_path / "own-play", 2)
    boosted = _boosted(lambda run, b: ([{"traj": l["traj"]} for l in lives],
                                       [], 0))(None, 0)[0]
    for p in sorted((tmp_path / "own-play").glob("iter-*.pkl")):
        p.unlink()
    log2 = ablation.OwnPlayLog(tmp_path / "own-play")
    for l, b in zip(lives, boosted):
        log2.write(l["iteration"], b["traj"],
                   {"inner": l["inner"], "real_done": l["real_done"],
                    "temperature": l["temperature"]},
                   l["terminated"], l["steps"], l["env_steps"])
    master = tmp_path / "weights-iter002.pth"
    torch.save(blob, master)
    out = tmp_path / "prep"
    logs = []
    header = distill.prep(master, 2, out, log=logs.append,
                          from_log=tmp_path / "own-play")
    return header, out, logs, lives


@pytest.mark.slow
def test_prep_from_an_own_play_log(logged):
    header, out, logs, lives = logged
    h, recs = distill.read_set(out / "set.pkl")
    assert h["source"] == "own-play log" and h["log_iterations"] == 2
    assert h["log_lives"] == len(lives) == len(recs)
    assert h["base_steps"] == sum(l["steps"] for l in lives)
    assert h["log_temperatures"] == {"0.5": len(lives)}
    assert h["master_ceiling"] == 1.0 and h["student_ceiling"] > 1.0
    assert h["master_discount"] == 0.97
    assert h["student_discount"] == pytest.approx(0.985)
    logged_lives = distill.log_lives(pathlib.Path(h["log_dir"]), 2)[0]
    m = distill.reward_magnitudes(logged_lives, 10.0)
    assert h["student_ceiling"] == distill.halved_ceiling(m, 1.0)
    assert h["value_max_student"] == pytest.approx(
        distill.discounted_return_max(logged_lives, 10.0,
                                      h["student_ceiling"], 0.985),
        abs=1e-3)
    assert 0 < h["value_max_student"] <= h["value_ceiling"] / 2
    doc = json.loads((out / "vocab.json").read_text())
    assert doc["student_discount"] == h["student_discount"]
    assert any("own-play log" in line for line in logs)
    assert any("student discount: 0.97 -> 0.985" in line for line in logs)


@pytest.mark.slow
def test_f1_trains_offline_at_the_preps_discount(logged, monkeypatch,
                                                tmp_path):
    header, out, _, _ = logged
    monkeypatch.setitem(ablation.ARMS, F[1], {
        **ablation.ARMS[F[1]], "_distill_offline_steps": 4,
        "_distill_snapshot_every": 2, "SCHEDULE_OFFSET_STEPS": 4})
    static = ablation.ARMS[F[1]]
    ablation.student_arm(F[1], out / "vocab.json")
    results = tmp_path / "f1"
    results.mkdir()
    ablation.RESULTS, ablation.TAG = results, "-test"
    _, lives = distill.read_set(out / "set.pkl")
    with ablation.arm_config(F[1]) as spec, \
            ablation._run_length_globals(30, 3), ablation.token_actions(), \
            ablation.env_schedule(20000):
        muzero.TRAINING_BATCHES = 8
        ablation.student_gate(spec, out / "set.pkl", True)
        ablation.env_stop_gate(spec, 20000, True, True, 60)
        assert muzero.DISCOUNT == header["student_discount"]
        run = ablation.Run(F[1], 0, spec)
        try:
            ablation.distill_prefill(run, lives)
            ablation.distill_offline(run, F[1], 0, spec, True)
            assert run.trainer.training_steps == 4
            # the offline phase played no env step: every schedule at 0
            assert muzero.SCHEDULE_ENV_STEPS == 0
            assert muzero.prior_acting_temperature(4) == 0.5
        finally:
            run.env.close()
    row0 = torch.load(results / f"snapshots-test-{F[1]}-seed0" /
                      "weights-iter000.pth", weights_only=False)
    assert row0["spec"]["DISCOUNT"] == header["student_discount"]
    assert row0["spec"]["_student_discount"] == header["student_discount"]
    # F2's prep could take it as its student master (its process sees the
    # static arm, not student_arm's copy)
    ablation.ARMS[F[1]] = static
    got = distill.master_spec(F[1], row0, "row0")
    assert got["DISCOUNT"] == header["student_discount"]
