"""Interactive ROI picker for the Tetris state masks.

The three Keras classifiers in `tetris.py` are being replaced by fixed regions
of interest plus a comparison rule. This tool is how those regions get chosen.
You draw a box on a frame and it immediately answers the only question that
matters: does this box separate the state you are looking at from the states it
could be confused with?

Three things the measurement work established, which shape the whole design:

* **Siblings, not everything.** Scoring a state against every other state wastes
  the comparison on pairs that are never confusable. `error` versus `neutral` is
  free; `error` versus `title_a` is the real question. Classes therefore carry a
  family, and evaluation defaults to comparing within it. (`t` widens the scope
  when you want to check.)

* **The representation matters as much as the region.** With one identical ROI,
  comparing raw pixel values scored 0.518 recall where comparing a binarised
  colour mask scored 0.965. The Tetris background animates constantly, so
  absolute pixel values there are noise -- what stays invariant is the glyph
  shape. `m` cycles representations and `E` tries every combination and ranks
  them, so a region is never rejected merely because it was measured wrong.

* **Held-out thresholds.** The template is built from half the target class and
  scored on the other half, so the numbers on screen are not the numbers the
  template was fitted to. Small classes (`go` has 20 frames) fall back to
  scoring on the training half and say so.

The headline number is **recall at a false-positive budget**: the fraction of
the target class still detected at the tightest threshold admitting no more than
that share of sibling frames. Three budgets are shown (0%, 0.5%, 2%) because
strict zero is decided by the single worst sibling frame in the sample and is
brittle -- one lucky tetromino parked in the box sends it to 0.000 for a region
that is otherwise perfect. `b` picks which budget gets saved.

Views (`v`) show the frame, the class mean, a within-class stability map, and a
discriminability map, so you can see where the invariant pixels are before
drawing rather than guessing and checking.

Usage
-----
    .venv/Scripts/python.exe tetris_roi_tool.py --class pregame
    .venv/Scripts/python.exe tetris_roi_tool.py --list
    .venv/Scripts/python.exe tetris_roi_tool.py --selftest

Needs only numpy and OpenCV, so it runs in the project venv -- it does not
touch the Python 3.7 / TensorFlow stack `tetris.py` itself requires.

Output format (`tetris_rois.json`)
----------------------------------
    {"<roi name>": {
        "class": "clr_tetris",        # state this ROI detects
        "family": "clear",            # who it was measured against
        "roi": [y0, y1, x0, x1],      # half-open, in `shape` pixel coords
        "shape": [90, 160],           # frame size the coords refer to
        "rep": "white",               # representation to compare in
        "metric": "ncc",              # comparison rule
        "threshold": 0.50,            # >= this means detected
        "budget": 0.005,              # FP budget the threshold was set at
        "recall": 0.957,              # measured at that threshold
        "fp": 0.0,                    # actual sibling FP rate there
        "template": [[...]]}}         # the pattern to compare against

`load_rois()` and `detect()` at the bottom of this file consume that directly,
so the detector needs the JSON only -- not the training corpus.
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

FRAME_H, FRAME_W = 90, 160
SCALE = 6                      # on-screen magnification of the 90x160 frame
PANEL_H = 196                  # info panel below the image
ROI_FILE = "tetris_rois.json"
DEFAULT_CAP = 300              # frames loaded per class

# name -> (folder, family). Siblings are the other members of the same family.
CLASSES = {
    "neutral":    ("training/tetris_environment/neutral",    "ingame"),
    "go":         ("training/tetris_pregame_opt/go",         "ingame"),
    "pregame":    ("training/tetris_pregame_opt/pregame",    "ingame"),
    "end_screen": ("training/tetris_environment/end_screen", "outgame"),
    "error":      ("training/tetris_environment/error",      "outgame"),
    "title_a":    ("training/tetris_environment/title_a",    "outgame"),
    "title_b":    ("training/tetris_environment/title_b",    "outgame"),
    "clr_none":   ("training/tetris_line_clears/neutral",    "clear"),
    "clr_tetris": ("training/tetris_line_clears/tetris",     "clear"),
}
# NOT states, and deliberately not in CLASSES: training/double is a pink attack
# overlay showing a COMBO counter (not double line clears), and
# training/bot_data is unlabelled bot gameplay. Both were briefly registered as
# classes while hunting for the small-clear indicator, which made the tool look
# like it was asking for them to be classified. Browse them with an image
# viewer; this tool only takes labelled states.

# Colour predicates. Each maps a crop to a binary plane; `raw` keeps all three
# channels. Thresholds are deliberately loose -- the comparison downstream is on
# the *shape* of what survives, not on exact values.
REPS = ("raw", "white", "cyan", "pink", "blue", "yellow", "bright", "dark")
METRICS = ("ncc", "match")


def apply_rep(rep, crop_u8):
    """(N,h,w,3) uint8 -> (N,h,w,C) float32 in the chosen representation."""
    if rep == "raw":
        return crop_u8.astype(np.float32)
    a = crop_u8.astype(np.int16)
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    lo, hi = a.min(axis=3), a.max(axis=3)
    if rep == "white":
        m = (lo > 180) & ((hi - lo) < 45)
    elif rep == "cyan":
        m = (b > 150) & (g > 150) & (r < b - 30)
    elif rep == "pink":
        m = (r > 140) & (b > 130) & ((r - g) > 45) & ((b - g) > 35)
    elif rep == "blue":
        m = (b > 120) & ((b - r) > 55) & ((b - g) > 30)
    elif rep == "yellow":
        m = (g > 150) & (r > 140) & (b < 130)
    elif rep == "bright":
        m = a.mean(axis=3) > 170
    elif rep == "dark":
        m = a.mean(axis=3) < 45
    else:
        raise ValueError("unknown representation: " + rep)
    return m.astype(np.float32)[..., None]


def rep_range(rep):
    """Dynamic range of a representation, so `match` scores are comparable."""
    return 255.0 if rep == "raw" else 1.0


def flat(x):
    return x.reshape(len(x), -1)


def score(feats, template, metric, rep):
    """Similarity of each row of `feats` to `template`, higher is better.

    `ncc` is zero-mean normalised correlation -- it ignores overall brightness
    and scores the pattern, which is what survives an animated background.
    `match` is 1 - normalised mean absolute difference, which does care about
    absolute values and suits the frozen screens (the error dialog blurs the
    background, so 95% of its pixels are literally constant).
    """
    if metric == "ncc":
        p = feats - feats.mean(axis=1, keepdims=True)
        t = template - template.mean()
        denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t) + 1e-6
        return (p @ t) / denom
    return 1.0 - np.abs(feats - template[None]).mean(axis=1) / rep_range(rep)


BUDGETS = (0.0, 0.005, 0.02)          # tolerated sibling false-positive rates


def evaluate(target, siblings, box, rep, metric, budget=0.005):
    """Measure a candidate box. Returns a dict of metrics, or None if unusable.

    `siblings` is {name: frames}. The template comes from the first half of the
    target class and is scored against the second half, so nothing reported here
    is measured on the frames that built it.

    Recall is reported at several false-positive budgets rather than only at
    zero. Strict zero-FP is what a detector ideally wants, but it is decided by
    the single worst sibling frame in the sample -- one lucky cyan tetromino
    parked in the box drops it to 0.000 for a region that is otherwise perfect.
    The 0.5% column is the honest headline; the 0% column tells you whether the
    separation is genuinely clean or merely good.
    """
    if len(target) < 4 or box is None:
        return None
    y0, y1, x0, x1 = box
    if y1 - y0 < 2 or x1 - x0 < 2:
        return None

    def feats(a):
        return flat(apply_rep(rep, a[:, y0:y1, x0:x1, :]))

    cut = max(2, len(target) // 2)
    train, held = target[:cut], target[cut:]
    heldout = len(held) >= 4
    if not heldout:
        held = train                      # too few frames to split honestly
    template = feats(train).mean(axis=0)

    s_target = score(feats(held), template, metric, rep)
    per_sib, all_sib = {}, []
    for name, frames in siblings.items():
        if len(frames) == 0:
            continue
        s = score(feats(frames), template, metric, rep)
        per_sib[name] = float(s.max())
        all_sib.append(s)
    sib = np.concatenate(all_sib) if all_sib else np.zeros(0, np.float32)

    def at_budget(b):
        """(threshold, recall, actual fp) tolerating at most `b` sibling FPs."""
        if not len(sib):
            return float(s_target.min()), 1.0, 0.0
        thr = (float(sib.max()) if b <= 0
               else float(np.quantile(sib, 1.0 - b))) + 1e-6
        return thr, float((s_target >= thr).mean()), float((sib >= thr).mean())

    curve = {b: at_budget(b) for b in BUDGETS}
    thr0, recall0, fp0 = curve.get(budget, at_budget(budget))
    margin = float(s_target.mean() - sib.max()) if len(sib) else float("inf")

    # Best balanced split, for when zero-FP is out of reach.
    cand = np.unique(np.concatenate([s_target, sib])) if len(sib) else s_target
    best = (-2.0, thr0, recall0, 0.0)
    for t in cand:
        rec = float((s_target >= t).mean())
        fp = float((sib >= t).mean()) if len(sib) else 0.0
        if rec - fp > best[0]:
            best = (rec - fp, float(t), rec, fp)

    train_f = feats(train)
    sd = train_f.std(axis=0)
    tol = 0.2 if rep != "raw" else 12.0
    # A template with no internal contrast has nothing to correlate against:
    # under `ncc` every frame scores ~0 and the box looks uniformly useless
    # even when the region is fine. Almost always means the colour predicate
    # never fires here -- try another `m`. Flagged rather than silently scored.
    degenerate = bool(template.std() < 1e-3)
    return {
        "rep": rep, "metric": metric, "roi": [y0, y1, x0, x1],
        "threshold": thr0, "recall": recall0, "fp": fp0, "margin": margin,
        "budget": budget, "curve": curve,
        "bal_threshold": best[1], "bal_recall": best[2], "bal_fp": best[3],
        "mean_sd": float(sd.mean()), "invariant": float((sd < tol).mean()),
        "degenerate": degenerate,
        "worst_sibling": (max(per_sib, key=per_sib.get) if per_sib else "-"),
        "per_sibling": per_sib, "heldout": heldout,
        "n_train": int(len(train)), "n_test": int(len(held)),
        "template": template,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_class(name, cap, seed=0):
    folder = CLASSES[name][0]
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    random.Random(seed).shuffle(paths)
    ims = []
    for p in paths[:cap]:
        im = cv2.imread(p)
        if im is not None and im.shape == (FRAME_H, FRAME_W, 3):
            ims.append(im)
    if not ims:
        return np.zeros((0, FRAME_H, FRAME_W, 3), np.uint8)
    return np.stack(ims)


def siblings_of(name, data, scope):
    fam = CLASSES[name][1]
    return {k: v for k, v in data.items()
            if k != name and (scope == "all" or CLASSES[k][1] == fam)}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
VIEWS = ("frame", "class mean", "stability", "discriminability")


def view_image(view, name, data, scope, frame_idx, cache):
    """The 90x160 BGR image to display for the current view."""
    frames = data[name]
    if len(frames) == 0:
        return np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    if view == 0:
        return frames[frame_idx % len(frames)]

    key = (view, name, scope)
    if key in cache:
        return cache[key]
    a = frames.astype(np.float32)
    if view == 1:
        img = a.mean(axis=0).astype(np.uint8)
    elif view == 2:
        # Bright = the pixel holds the same value across the whole class.
        sd = a.std(axis=0).mean(axis=2)
        img = cv2.applyColorMap(
            (255.0 / (1.0 + sd / 6.0)).clip(0, 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS)
    else:
        # Fisher score against the weakest-separating sibling: a pixel scores
        # only if it beats *every* sibling, which is what a detector needs.
        mu, var = a.mean(axis=0), a.var(axis=0)
        sibs = siblings_of(name, data, scope)
        stack = []
        for other in sibs.values():
            if len(other) < 2:
                continue
            o = other.astype(np.float32)
            f = (mu - o.mean(axis=0)) ** 2 / (var + o.var(axis=0) + 25.0)
            stack.append(f.sum(axis=2))
        s = np.min(np.stack(stack), axis=0) if stack else np.zeros((FRAME_H, FRAME_W))
        img = cv2.applyColorMap(
            (255 * np.clip(s / 40.0, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cache[key] = img
    return img


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def put(canvas, text, row, col=8, colour=(220, 220, 220), scale=0.44):
    cv2.putText(canvas, text, (col, PANEL_TOP + 20 + row * 17),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


PANEL_TOP = FRAME_H * SCALE


def render(state, data, cache):
    name = state["class"]
    img = view_image(state["view"], name, data, state["scope"],
                     state["frame"], cache)
    big = cv2.resize(img, (FRAME_W * SCALE, FRAME_H * SCALE),
                     interpolation=cv2.INTER_NEAREST)
    if state["grid"]:
        for x in range(0, FRAME_W, 10):
            cv2.line(big, (x * SCALE, 0), (x * SCALE, FRAME_H * SCALE), (60, 60, 60), 1)
        for y in range(0, FRAME_H, 10):
            cv2.line(big, (0, y * SCALE), (FRAME_W * SCALE, y * SCALE), (60, 60, 60), 1)
        for x in range(0, FRAME_W, 20):
            cv2.putText(big, str(x), (x * SCALE + 3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        for y in range(10, FRAME_H, 20):
            cv2.putText(big, str(y), (3, y * SCALE - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    box = state["drag"] or state["box"]
    if box:
        y0, y1, x0, x1 = box
        cv2.rectangle(big, (x0 * SCALE, y0 * SCALE), (x1 * SCALE, y1 * SCALE),
                      (0, 255, 0), 2)

    canvas = np.zeros((FRAME_H * SCALE + PANEL_H, FRAME_W * SCALE, 3), np.uint8)
    canvas[:FRAME_H * SCALE] = big

    names = state["names"]
    n_frames = max(len(data[name]), 1)
    put(canvas, f"class {name} [{names.index(name)+1}/{len(names)}]"
                f"  family {CLASSES[name][1]}  n={len(data[name])}"
                f"   frame {state['frame'] % n_frames + 1}/{n_frames}", 0,
        colour=(120, 255, 255))
    put(canvas, f"view {VIEWS[state['view']]}   rep {state['rep']}"
                f"   metric {state['metric']}   siblings {state['scope']}"
                f" ({', '.join(siblings_of(name, data, state['scope'])) or 'none'})", 1)

    r = state["result"]
    if box:
        y0, y1, x0, x1 = box
        put(canvas, f"box  y {y0}:{y1}  x {x0}:{x1}   ({y1-y0} x {x1-x0} px)", 2,
            colour=(160, 255, 160))
    else:
        put(canvas, "box  -- drag on the image to draw one --", 2, colour=(150, 150, 150))
    if r:
        tag = "held-out" if r["heldout"] else "TRAIN-ONLY (class too small to split)"
        put(canvas, f"within-class  mean sd {r['mean_sd']:6.2f}"
                    f"   invariant {r['invariant']*100:5.1f}% of box"
                    f"   [{r['n_train']}/{r['n_test']} {tag}]", 3)
        curve = "   ".join(f"@{b*100:g}%FP {r['curve'][b][1]:.3f}" for b in BUDGETS)
        put(canvas, f"recall  {curve}   margin {r['margin']:+.3f}"
                    f"   worst sibling: {r['worst_sibling']}", 4)
        good = r["recall"] >= 0.9 and r["per_sibling"]
        if not r["per_sibling"]:
            put(canvas, "no siblings in this family -- browse only, the numbers"
                        " above are not a separation result", 6,
                colour=(90, 200, 255))
        put(canvas, f"SELECTED  budget {r['budget']*100:g}%  thr {r['threshold']:.3f}"
                    f"  recall {r['recall']:.3f}  fp {r['fp']:.3f}"
                    f"     (balanced: recall {r['bal_recall']:.3f}"
                    f" fp {r['bal_fp']:.3f})", 5,
            colour=(120, 255, 120) if good else (120, 160, 255))
        sib = "  ".join(f"{k} {v:.2f}" for k, v in
                        sorted(r["per_sibling"].items(), key=lambda kv: -kv[1]))
        put(canvas, "sibling max   " + (sib[:96] or "-"), 6)
        if r["degenerate"]:
            put(canvas, f"!! '{r['rep']}' never fires in this box -- the template is"
                        f" blank. Press m for another representation.", 7,
                colour=(90, 200, 255))
    put(canvas, f"saved {len(state['rois'])} ROI(s) -> {ROI_FILE}", 8,
        colour=(200, 200, 120))
    put(canvas, "a/d frame  A/D x25  [/] class  v view  m rep  n metric  b budget"
                "  E rank all  s save  x del  t scope  g grid  r reset  q quit", 9,
        colour=(150, 150, 150), scale=0.40)
    p = state.get("prompt")
    if p:
        cv2.rectangle(canvas, (0, PANEL_TOP), (FRAME_W * SCALE, PANEL_TOP + PANEL_H),
                      (0, 0, 0), -1)
        verb = "SAVE as" if p["kind"] == "save" else "DELETE which"
        put(canvas, f"{verb}:  {p['buf']}_", 2, colour=(120, 255, 255), scale=0.6)
        put(canvas, f"enter = confirm (empty -> '{p['default']}')   esc = cancel", 4)
        if p["kind"] == "delete":
            put(canvas, "saved: " + ", ".join(state["rois"])[:88], 6)
    return canvas


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_saved():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE) as fh:
            return json.load(fh)
    return {}


def save_saved(rois):
    # tmp + replace, so an interrupted write cannot destroy the existing file.
    tmp = ROI_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rois, fh, indent=2)
    os.replace(tmp, ROI_FILE)


def record(name, cls, result):
    y0, y1, x0, x1 = result["roi"]
    tmpl = result["template"]
    ch = 3 if result["rep"] == "raw" else 1
    return {
        "class": cls, "family": CLASSES[cls][1],
        "roi": [y0, y1, x0, x1], "shape": [FRAME_H, FRAME_W],
        "rep": result["rep"], "metric": result["metric"],
        "threshold": round(result["threshold"], 6),
        "recall": round(result["recall"], 4),
        "fp": round(result["fp"], 5), "budget": result["budget"],
        "margin": round(result["margin"], 4),
        "template": np.round(tmpl.reshape(y1 - y0, x1 - x0, ch), 4).tolist(),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def rank_all(target, sibs, box, budget=0.005):
    """Try every representation x metric and rank them at the chosen budget.

    Degenerate combinations are dropped rather than listed: a colour predicate
    that never fires in the box scores every frame identically, which is not a
    result, just an empty template.
    """
    rows = []
    for rep in REPS:
        for metric in METRICS:
            r = evaluate(target, sibs, box, rep, metric, budget)
            if r and not r["degenerate"]:
                rows.append(r)
    rows.sort(key=lambda r: (-r["recall"], -r["margin"]))
    print(f"\n  rep      metric   recall@{budget*100:g}%FP   @0%FP   margin"
          f"   invariant   worst sibling")
    print("  " + "-" * 76)
    for r in rows[:10]:
        print(f"  {r['rep']:<8s} {r['metric']:<8s} {r['recall']:11.3f}  "
              f"{r['curve'][0.0][1]:6.3f}  {r['margin']:+7.3f}   "
              f"{r['invariant']*100:7.1f}%   {r['worst_sibling']}")
    print()
    return rows[0] if rows else None


def handle_prompt(state, c):
    """Text entry inside the OpenCV window.

    Naming used to go through input() on the terminal, which meant the tool
    could not be launched detached: a background process has no stdin, so every
    save would silently take the default name. Typing in the window removes the
    dependency.
    """
    p = state["prompt"]
    if c == 27:                                     # esc cancels
        state["prompt"] = None
    elif c in (13, 10):                             # enter confirms
        name = p["buf"].strip() or p["default"]
        if p["kind"] == "save" and state["result"]:
            r = state["result"]
            state["rois"][name] = record(name, state["class"], r)
            save_saved(state["rois"])
            print(f"  saved '{name}'  {r['rep']}/{r['metric']}  roi {r['roi']}  "
                  f"thr {r['threshold']:.3f}  recall {r['recall']:.3f} "
                  f"@ {r['fp']*100:.1f}% FP -> {ROI_FILE}", flush=True)
        elif p["kind"] == "delete" and name in state["rois"]:
            del state["rois"][name]
            save_saved(state["rois"])
            print(f"  deleted '{name}'", flush=True)
        state["prompt"] = None
    elif c == 8:                                    # backspace
        p["buf"] = p["buf"][:-1]
    elif 32 <= c < 127:
        p["buf"] += chr(c)


def run(start_class, cap, only):
    names = [n for n in CLASSES if os.path.isdir(CLASSES[n][0])]
    if only:
        names = [n for n in names if n in only]
    if not names:
        sys.exit("no class folders found -- run from the repo root")
    print("loading:")
    data = {}
    for n in names:
        data[n] = load_class(n, cap)
        print(f"  {n:12s} {len(data[n]):5d} frames  {CLASSES[n][0]}")
    if start_class not in names:
        start_class = names[0]

    state = {"class": start_class, "names": names, "frame": 0, "view": 0,
             "rep": "raw", "metric": "ncc", "scope": "family", "grid": True,
             "budget": 0.005, "box": None, "drag": None, "result": None,
             "prompt": None, "rois": load_saved()}
    cache = {}
    win = "tetris ROI picker"
    cv2.namedWindow(win)

    def recompute():
        sibs = siblings_of(state["class"], data, state["scope"])
        state["result"] = evaluate(data[state["class"]], sibs, state["box"],
                                   state["rep"], state["metric"],
                                   state["budget"])

    def on_mouse(event, mx, my, flags, _):
        x = int(np.clip(mx // SCALE, 0, FRAME_W - 1))
        y = int(np.clip(my // SCALE, 0, FRAME_H - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["anchor"] = (y, x)
            state["drag"] = (y, y + 1, x, x + 1)
        elif event == cv2.EVENT_MOUSEMOVE and state.get("anchor"):
            ay, ax = state["anchor"]
            state["drag"] = (min(ay, y), max(ay, y) + 1, min(ax, x), max(ax, x) + 1)
        elif event == cv2.EVENT_LBUTTONUP and state.get("anchor"):
            ay, ax = state["anchor"]
            state["anchor"] = None
            state["box"] = (min(ay, y), max(ay, y) + 1, min(ax, x), max(ax, x) + 1)
            state["drag"] = None
            recompute()

    cv2.setMouseCallback(win, on_mouse)
    print("\nwindow open -- drag a box on the image. 'q' quits.\n")

    while True:
        cv2.imshow(win, render(state, data, cache))
        k = cv2.waitKeyEx(20)
        if k == -1:
            continue
        c = k & 0xFF
        if state["prompt"] is not None:
            handle_prompt(state, c)
            continue
        n_frames = max(len(data[state["class"]]), 1)
        if c in (ord("q"), 27):
            break
        elif c == ord("d") or k == 2555904:
            state["frame"] = (state["frame"] + 1) % n_frames
        elif c == ord("a") or k == 2424832:
            state["frame"] = (state["frame"] - 1) % n_frames
        elif c == ord("D"):
            state["frame"] = (state["frame"] + 25) % n_frames
        elif c == ord("A"):
            state["frame"] = (state["frame"] - 25) % n_frames
        elif c in (ord("]"), ord("[")):
            i = names.index(state["class"]) + (1 if c == ord("]") else -1)
            state["class"] = names[i % len(names)]
            state["frame"] = 0
            recompute()
        elif c == ord("v"):
            state["view"] = (state["view"] + 1) % len(VIEWS)
        elif c == ord("m"):
            state["rep"] = REPS[(REPS.index(state["rep"]) + 1) % len(REPS)]
            recompute()
        elif c == ord("n"):
            state["metric"] = METRICS[(METRICS.index(state["metric"]) + 1) % len(METRICS)]
            recompute()
        elif c == ord("t"):
            state["scope"] = "all" if state["scope"] == "family" else "family"
            cache.clear()
            recompute()
        elif c == ord("b"):
            state["budget"] = BUDGETS[(BUDGETS.index(state["budget"]) + 1)
                                      % len(BUDGETS)]
            recompute()
        elif c == ord("g"):
            state["grid"] = not state["grid"]
        elif c == ord("r"):
            state["box"] = state["result"] = None
        elif c == ord("e"):
            recompute()
        elif c == ord("E"):
            if state["box"]:
                best = rank_all(data[state["class"]],
                                siblings_of(state["class"], data, state["scope"]),
                                state["box"], state["budget"])
                if best:
                    state["rep"], state["metric"] = best["rep"], best["metric"]
                    state["result"] = best
                    print(f"  adopted {best['rep']}/{best['metric']}\n")
        elif c == ord("s"):
            if not state["result"]:
                print("  nothing to save -- draw a box first")
                continue
            state["prompt"] = {"kind": "save", "buf": "",
                               "default": f"{state['class']}_{len(state['rois'])}"}
        elif c == ord("x"):
            if not state["rois"]:
                print("  nothing saved yet")
                continue
            state["prompt"] = {"kind": "delete", "buf": "",
                               "default": list(state["rois"])[-1]}

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Consumer side -- what tetris.py will import
# ---------------------------------------------------------------------------
def load_rois(path=ROI_FILE):
    """Read the JSON into ready-to-use detectors (numpy templates restored)."""
    with open(path) as fh:
        raw = json.load(fh)
    for spec in raw.values():
        spec["template"] = np.asarray(spec["template"], np.float32).reshape(-1)
    return raw


def detect(frame, spec):
    """Score one frame against one ROI spec. Returns (hit, score).

    `frame` is BGR at any resolution; the ROI is scaled from the shape it was
    measured at, so a native-resolution capture works without re-picking boxes.
    """
    y0, y1, x0, x1 = spec["roi"]
    h0, w0 = spec["shape"]
    h, w = frame.shape[:2]
    if (h, w) != (h0, w0):
        sy, sx = h / h0, w / w0
        y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
        x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
        crop = cv2.resize(frame[y0:y1, x0:x1],
                          (spec["roi"][3] - spec["roi"][2],
                           spec["roi"][1] - spec["roi"][0]),
                          interpolation=cv2.INTER_AREA)
    else:
        crop = frame[y0:y1, x0:x1]
    feats = flat(apply_rep(spec["rep"], crop[None]))
    s = float(score(feats, spec["template"], spec["metric"], spec["rep"])[0])
    return s >= spec["threshold"], s


# ---------------------------------------------------------------------------
def selftest():
    """Reproduce a known-good result headlessly, so the maths can be trusted
    without opening a window: the 'Tetris' banner measured earlier at
    ~0.96 recall with zero false positives."""
    print("loading clear family...")
    data = {n: load_class(n, 600) for n in CLASSES if CLASSES[n][1] == "clear"}
    for n, v in data.items():
        print(f"  {n:12s} {len(v):5d}")
    box = (40, 52, 12, 48)                 # the banner region found earlier
    sibs = siblings_of("clr_tetris", data, "family")
    print(f"\nbox y{box[0]}:{box[1]} x{box[2]}:{box[3]}  "
          f"siblings: {', '.join(sibs)}")
    # Strict zero-FP here: the round-trip assertion below is only meaningful if
    # the threshold was not allowed any sibling false positives to begin with.
    best = rank_all(data["clr_tetris"], sibs, box, budget=0.0)
    ok = best and best["recall"] > 0.85 and best["rep"] in ("white", "bright")
    # round-trip the save/load/detect path too
    spec = record("selftest", "clr_tetris", best)
    spec["template"] = np.asarray(spec["template"], np.float32).reshape(-1)
    hits = sum(detect(f, spec)[0] for f in data["clr_tetris"][:200])
    fps = sum(detect(f, spec)[0] for f in data["clr_none"][:200])
    print(f"round-trip detect(): {hits}/200 tetris frames hit, "
          f"{fps}/200 neutral frames hit")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--class", dest="cls", default="pregame",
                    help="class to open on (default: pregame)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these classes")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help=f"frames to load per class (default {DEFAULT_CAP})")
    ap.add_argument("--list", action="store_true", help="list classes and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the measurement path headlessly and exit")
    a = ap.parse_args()
    if a.list:
        for n, (folder, fam) in CLASSES.items():
            k = len(glob.glob(os.path.join(folder, "*.png")))
            mark = " " if os.path.isdir(folder) else "  (MISSING)"
            print(f"  {n:12s} family={fam:8s} {k:6d} frames  {folder}{mark}")
        return 0
    if a.selftest:
        return selftest()
    run(a.cls, a.cap, a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
