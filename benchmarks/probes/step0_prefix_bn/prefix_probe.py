"""Direct test of the value-prefix head on the probe's real states.
usage: prefix_probe.py 45 50 55 60 75
Per snapshot: step-1 prefix (fresh LSTM state) after the PLAYED token, vs the
true token reward; step-2 prefix (second token = prior argmax at depth 1);
eval-mode BN (what the search uses) vs train-mode BN batch statistics on
the same 480-row batch (what each per-step training call sees), the prefix
head alone switched (a deep copy; the loaded weights are never written)."""
import copy
import sys
import numpy as np
import torch
import probe_qspread as P
import muzero
import ablation


def run_iter(run, it, obs, acts, r_true):
    P.load(run, P.snap(it))
    rep, dyn, pred, vp = (run.trainer.representation, run.trainer.dynamics,
                          run.trainer.prediction, run.trainer.value_prefix)
    out = {}
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32) / 255.0
        l0 = rep(x)
        a = torch.tensor(acts, dtype=torch.long)
        l1, rew1, _ = dyn(l0, a)
        lg1, v1, _ = pred(l1)
        a2 = lg1.argmax(1)
        l2, _, _ = dyn(l1, a2)
        l3, _, _ = dyn(l2, pred(l2)[0].argmax(1))
        for mode in ("eval", "trainBN"):
            net = vp if mode == "eval" else copy.deepcopy(vp).train()
            st = net.initial_state(len(obs), "cpu")
            p1, _, st1 = net(l1, st)
            p2, _, st2 = net(l2, st1)
            p3, _, _ = net(l3, st2)
            p1, p2, p3 = p1.squeeze(-1).numpy(), p2.squeeze(-1).numpy(), p3.squeeze(-1).numpy()
            out[mode] = dict(p1_mean=p1.mean(), p1_absmean=np.abs(p1).mean(),
                             p1_mae=np.abs(p1 - r_true).mean(),
                             p2_mean=p2.mean(), edge2_mean=(p2 - p1).mean(),
                             p3_mean=p3.mean(), edge3_mean=(p3 - p2).mean())
        # step 1 over ALL 14 tokens (the root expansion), eval mode
        A = ablation.ACTIONS
        lat = l0.repeat_interleave(A, 0)
        acts_all = torch.arange(A).repeat(len(obs))
        n1, _, _ = dyn(lat, acts_all)
        pall, _, _ = vp(n1, vp.initial_state(len(lat), "cpu"))
        pall = pall.view(len(obs), A).numpy()
        out["all_tokens_p1"] = dict(mean=pall.mean(), per_state_range=np.ptp(pall, 1).mean(),
                                    per_token_mean=np.round(pall.mean(0), 2))
        # norm_out running stats vs the step-1 / step-2 batch statistics
        h = {}
        def grab(name):
            def f(m, i, o):
                h[name] = i[0].detach()
            return f
        hk = vp.norm_out.register_forward_hook(grab("pre_norm_out"))
        st = vp.initial_state(len(obs), "cpu")
        _, _, st1 = vp(l1, st); s1 = h["pre_norm_out"]
        vp(l2, st1); s2 = h["pre_norm_out"]
        hk.remove()
        rm, rv = vp.norm_out.running_mean, vp.norm_out.running_var
        def z(s):  # how far the step's batch mean sits from the running mean, in running sd
            return float(((s.mean(0) - rm).abs() / rv.sqrt().clamp(min=1e-5)).mean())
        out["norm_out_shift_sd"] = dict(step1=z(s1), step2=z(s2),
                                        var_ratio_step1=float((s1.var(0) / rv.clamp(min=1e-8)).mean()),
                                        var_ratio_step2=float((s2.var(0) / rv.clamp(min=1e-8)).mean()))
        out["reward_head_absmax"] = float(rew1.abs().max())
        out["r_true_mean"] = float(r_true.mean())
    return out


def main(argv):
    P.make_run()
    with ablation.arm_config(P.ARM), ablation._run_length_globals(P.ITERS, P.TRAIN_STEPS), \
            ablation.token_actions():
        st = np.load(P.HERE / "states.npz")
        obs, meta = st["obs"], st["meta"]
        acts = meta[:, 3].astype(int)
        r_true = meta[:, 4] / muzero.REWARD_UNIT   # ceiling 160 never binds here
        run = ablation.Run(P.ARM, 0, ablation.ARMS[P.ARM])
        for it in [int(a) for a in argv]:
            o = run_iter(run, it, obs, acts, r_true)
            print(f"== iter {it}")
            for k, v in o.items():
                if isinstance(v, dict):
                    print(f"  {k}: " + ", ".join(
                        f"{kk} {vv:.3f}" if np.isscalar(vv) else f"{kk} {vv}" for kk, vv in v.items()))
                else:
                    print(f"  {k}: {v:.4f}")
            sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
