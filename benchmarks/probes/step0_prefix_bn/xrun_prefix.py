"""Cross-run step-1 value-prefix BN probe (2026-10-24). Read-only.

usage: python xrun_prefix.py TREE LABEL SNAPSHOT [SNAPSHOT ...]
  env XR_SEARCH_N (default 48): states for the birth-search comparison
  (0 = skip).

Per snapshot, on the fixed 480-state set (states.npz):
  * step-1 prefix from a fresh LSTM state after the snapshot's OWN prior-
    argmax action/token: eval-mode BN (what search uses) vs train-mode BN
    batch statistics (a deep copy; what each per-step training call sees);
    step 2 after the depth-1 prior argmax; the root expansion's step-1
    prefix over ALL actions (eval) on 120 states;
  * norm_out (post-LSTM BatchNorm) batch-mean offset from its running mean,
    in running SDs, at step 1 and step 2;
  * birth-search target entropy / ln A on XR_SEARCH_N states (root noise,
    ACT_SEARCH_SIMS, the tree's OWN VectorizedPUCT) with the eval nets vs
    with the STEP-1 prefix BN statistics recalibrated on this state set
    (reset rows only; every other row the engine's own forward).
The arm config is the snapshot's own spec (what the run trained with); code
from TREE (sys.path; no bytecode written)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.dont_write_bytecode = True
import contextlib
import copy
import json
import math
import pathlib
import time

TREE = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(TREE)); sys.path.insert(0, str(TREE / "experiments"))
import numpy as np
import torch
torch.set_num_threads(2)
import muzero
import muzero_vtree
import ablation
assert not torch.cuda.is_available()
assert pathlib.Path(muzero.__file__).resolve().parent == TREE.resolve(), muzero.__file__

HERE = pathlib.Path(__file__).resolve().parent
SEARCH_N = int(os.environ.get("XR_SEARCH_N", "48"))
ablation.ENV_NAME, ablation.FRAMESKIP, ablation.EPISODE_CAP, ablation.ACTIONS = \
    "MsPacmanNoFrameskip-v4", 4, 3000, 9


def tupled(v):
    return tuple(tuple(t) for t in v) if v is not None else None


class Step1Recal(muzero_vtree.VectorizedPUCT):
    """The tree's own engine; only the prefixes of RESET rows (fresh LSTM
    state: the root's children and every horizon reset) are recomputed by
    a value-prefix copy whose BN statistics were re-estimated at step 1."""
    step1_net = None

    def _expand_forward(self, parent_latent, h, c, reset):
        out = super()._expand_forward(parent_latent, h, c, reset)
        if not (self.use_prefix and self.step1_net is not None and bool(reset.any())):
            return out
        nxt, rew, pol, val, prefixes, nh, nc = out
        B, A = nxt.shape[0], nxt.shape[1]
        m = reset.repeat_interleave(A)
        flat = nxt.reshape(B * A, *nxt.shape[2:])
        hh = h.repeat_interleave(A, dim=1).contiguous()
        cc = c.repeat_interleave(A, dim=1).contiguous()
        with torch.no_grad():
            p, _, (h2, c2) = self.step1_net(flat[m], (hh[:, m], cc[:, m]))
        prefixes = prefixes.clone().reshape(-1)
        prefixes[m] = p.squeeze(-1).double()
        L, H = nh.shape[0], nh.shape[-1]
        nh = nh.clone().reshape(L, B * A, H); nc = nc.clone().reshape(L, B * A, H)
        nh[:, m] = h2; nc[:, m] = c2
        return (nxt, rew, pol, val, prefixes.view(B, A),
                nh.view(L, B, A, H), nc.view(L, B, A, H))


def recal_copy(vp, lat, state):
    net = copy.deepcopy(vp)
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None
    net.train()
    with torch.no_grad():
        net(lat, state)
    net.eval()
    return net


def H(p):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(-1)


def probe(label, path, obs):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    spec = dict(d["spec"])
    if spec.get("TOKEN_VOCAB") is not None:
        spec["TOKEN_VOCAB"] = tupled(spec["TOKEN_VOCAB"])
    name = "__snapshot__"
    ablation.ARMS[name] = spec
    missing = [k for k in spec if not k.startswith("_") and not hasattr(muzero, k)]
    for k in missing:
        spec.pop(k)
    tok_ctx = getattr(ablation, "token_actions", None)
    with ablation.arm_config(name), ablation._run_length_globals(90, 2000), \
            (tok_ctx() if tok_ctx else contextlib.nullcontext()):
        A = ablation.ACTIONS
        run = ablation.Run(name, 0, spec)
        for n in ablation.NET_NAMES:
            getattr(run.trainer, n).load_state_dict(d[n])
        renorm = spec.get("LATENT_RENORMALIZE", True)
        run.trainer.representation.renormalize_output = renorm
        run.trainer.dynamics.renormalize_output = renorm
        run.set_train_mode(False)
        rep, dyn, pred, vp = (run.trainer.representation, run.trainer.dynamics,
                              run.trainer.prediction, run.trainer.value_prefix)
        rec = {"label": label, "snapshot": str(path), "arm": d["arm"],
               "iteration": d.get("iteration"), "training_steps": d.get("training_steps"),
               "A": A, "missing_spec_keys": missing,
               "prefix_in_tree": bool(muzero.USE_VALUE_PREFIX_IN_TREE)}
        if getattr(muzero, "REWARD_GROWTH", False):
            rec["ceiling_at_steps"] = float(muzero.reward_ceiling(d.get("training_steps", 0)))
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32) / 255.0
            l0 = rep(x)
            lg0, v0, _ = pred(l0)
            pr = torch.softmax(lg0, 1).numpy()
            rec["prior_H/lnA"] = float(H(pr).mean() / math.log(A))
            rec["prior_maxp"] = float(pr.max(1).mean())
            rec["root_value_mean"] = float(v0.mean())
            a1 = lg0.argmax(1)
            l1, _, _ = dyn(l0, a1)
            a2 = pred(l1)[0].argmax(1)
            l2, _, _ = dyn(l1, a2)
            for mode in ("eval", "trainBN"):
                net = vp if mode == "eval" else copy.deepcopy(vp).train()
                s0 = net.initial_state(len(obs), "cpu")
                p1, _, s1 = net(l1, s0)
                p2, _, _ = net(l2, s1)
                p1, p2 = p1.squeeze(-1).numpy(), p2.squeeze(-1).numpy()
                rec[f"p1_mean_{mode}"] = float(p1.mean())
                rec[f"p1_abs_{mode}"] = float(np.abs(p1).mean())
                rec[f"p2_mean_{mode}"] = float(p2.mean())
                rec[f"edge2_mean_{mode}"] = float((p2 - p1).mean())
            # root expansion: all actions, eval, 120 states
            sub = l0[::4]
            lat = sub.repeat_interleave(A, 0)
            n1, _, _ = dyn(lat, torch.arange(A).repeat(len(sub)))
            pall, _, _ = vp(n1, vp.initial_state(len(lat), "cpu"))
            rec["p1_allactions_abs_eval"] = float(pall.abs().mean())
            # norm_out offsets
            h = {}
            hk = vp.norm_out.register_forward_hook(lambda m, i, o: h.__setitem__("x", i[0].detach()))
            s0 = vp.initial_state(len(obs), "cpu")
            _, _, s1 = vp(l1, s0); x1 = h["x"]
            vp(l2, s1); x2 = h["x"]
            hk.remove()
            rm, rv = vp.norm_out.running_mean, vp.norm_out.running_var
            sd = rv.sqrt().clamp(min=1e-5)
            rec["normout_offset_sd_step1"] = float(((x1.mean(0) - rm).abs() / sd).mean())
            rec["normout_offset_sd_step2"] = float(((x2.mean(0) - rm).abs() / sd).mean())
            step1 = recal_copy(vp, l1, vp.initial_state(len(obs), "cpu"))
        if SEARCH_N and muzero.USE_VALUE_PREFIX_IN_TREE:
            idx = np.linspace(0, len(obs) - 1, SEARCH_N).round().astype(int)
            nets = {"representation": rep, "dynamics": dyn, "prediction": pred,
                    "value_prefix": vp}
            eng = Step1Recal(nets, A, "cpu")
            batch = 16 if A > 20 else 24
            t0 = time.time()
            for variant in ("eval", "step1recal"):
                eng.step1_net = step1 if variant == "step1recal" else None
                Hs, nvis, rq = [], [], []
                for b0 in range(0, len(idx), batch):
                    torch.manual_seed(777 + b0)
                    p, v, cnt = eng.search(obs[idx[b0:b0 + batch]],
                                           num_simulations=muzero.ACT_SEARCH_SIMS,
                                           c_puct=muzero.C_PUCT,
                                           discount=muzero.DISCOUNT, root_noise=True)
                    Hs.append(H(p)); nvis.append((cnt > 0).sum(1)); rq.append(v)
                rec[f"search_H/lnA_{variant}"] = float(np.concatenate(Hs).mean() / math.log(A))
                rec[f"search_children_visited_{variant}"] = float(np.concatenate(nvis).mean())
                rec[f"search_rootq_{variant}"] = float(np.concatenate(rq).mean())
            rec["search_n"] = int(len(idx)); rec["search_s"] = round(time.time() - t0)
    del ablation.ARMS[name]
    return rec


def main(argv):
    label = argv[2]
    obs = np.load(HERE / "states.npz")["obs"]
    for p in argv[3:]:
        t0 = time.time()
        rec = probe(label, pathlib.Path(p), obs)
        rec["secs"] = round(time.time() - t0)
        print(json.dumps(rec), flush=True)
        with open(HERE / "xrun_results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(sys.argv)
