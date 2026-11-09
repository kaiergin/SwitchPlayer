"""Mask-based state and reward detection for Tetris 99.

Replaces the three Keras classifiers in `tetris.py` (`model_environment`,
`model_reward`, `model_pregame`) with fixed regions of interest, colour
predicates and template correlation. No TensorFlow, no training, no weights --
every threshold here was measured against the labelled corpus and is
reproducible with `--eval`.

Structure follows what the corpus actually supports, which is a decision tree,
not a flat classifier:

    chrome present? ---- no ---> which out-of-game screen?
        |                        end_screen / error / title_a / title_b
        yes
        |
        +-- state machine: PREGAME --(GO glyph)--> PLAYING
                                                     |
                                                     +-- reward detection

`chrome` is the set of pixels every in-game state agrees on and every
out-of-game state contradicts -- found by asking what the in-game states have
in COMMON rather than what distinguishes each one, which is what makes it a
single test instead of four.

Reward, in the priority order the clear semantics require:

    1. "Tetris" glyph   (white text, ROI y40:52 x12:48)     -> REWARD_TETRIS
    2. garbage counter  (the in-matrix numeral: how many    -> REWARD_GARBAGE
                         lines the clear SENT. x74:84, y
                         slides with the stack, so the ROI
                         is a vertical strip searched by
                         connected component, not a box)
    3. otherwise                                            -> 0

Garbage sent is the score, deliberately: the object of the game is to attack,
so a lone single clear sends nothing and is worth nothing, while a single
inside a combo does send and does pay. The event is named `garbage` rather
than `clear` because it is not one-per-line-cleared.

Both are EDGE-TRIGGERED. The banner sits on screen for a second or more, and
the old `rewarded_steps` window paid +1 on every frame it was visible -- so a
single Tetris was worth anywhere from 1 to 7 depending on how the hard-drop
counter happened to line up. Here a clear pays exactly once.

Measured on held-out halves of the labelled corpus (`--eval` reproduces):

    chrome / in-game vs out      recall 0.998   reject 1.000
    out-of-game screen ID        0.998 / 1.000 / 1.000 / 1.000
    GO glyph                     10/10 (the whole corpus; GO is ~1s of frames)
    "Tetris" glyph               recall 0.970 at zero false positives

Channel order: the corpus PNGs read as BGR under cv2.imread, verified against
both a cyan reference (the GO glyph) and a yellow one (the playfield rail), so
templates apply directly to live `VideoCapture` frames with no conversion.

Usage
-----
    python tetris_masks.py --build     # regenerate templates from the corpus
    python tetris_masks.py --eval      # measure against the labelled corpus
    python tetris_masks.py --demo      # step through frames with overlays

    from tetris_masks import TetrisMasks, TetrisState
    masks = TetrisMasks()
    sm = TetrisState(masks)
    out = sm.step(frame_from_capture_card)   # dict(state, reward, done, event)
"""
import argparse
import glob
import os
import random
import sys
import time
from collections import deque

import cv2
import numpy as np

FRAME_H, FRAME_W = 90, 160          # reference frame all geometry refers to
TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tetris_masks.npz")

# --- geometry, in reference-frame pixels ----------------------------------
RAIL_L, RAIL_R = 60, 99             # playfield side rails
PLAYFIELD = (8, 84, 61, 99)         # matrix interior (y0, y1, x0, x1)
ROI_TETRIS = (40, 52, 12, 48)       # "Tetris" banner, left of the matrix
ROI_GO = (40, 50, 70, 92)           # "GO" glyph, centred in the matrix
ROI_OUT = {                         # one discriminative ROI per out-of-game screen
    "end_screen": (4, 12, 91, 105),
    "error":      (80, 88, 3, 17),
    "title_a":    (79, 87, 39, 53),
    "title_b":    (20, 28, 97, 111),
}
# The clear numeral's column is fixed but its ROW tracks the stack height, so
# this is a search strip, not a box. Sizes bound a digit glyph: a line-clear
# flash spans the matrix and is rejected on width.
# Bottom extended 78 -> 86: the counter is drawn where the clear happened, so
# a low clear puts it near the matrix floor, outside the original strip.
NUM_STRIP = (20, 86, 72, 86)
# The width floor is 1, not 3, because at 90x160 the digit "1" is literally a
# single pixel wide (w=1 h=4 area=4, measured). A floor of 3 rejected it on
# both width and area -- which meant the detector was blind to the most common
# value the counter takes. Loosening cost nothing measurable: clr_none stayed
# at 0.001 while clr_tetris went 0.203 -> 0.278 and bot_data 0.158 -> 0.273.
# Live glyphs measured 5x8 to 7x8 -- larger than the corpus 4x4/4x5, because
# the current UI renders them bigger. Bounds cover both.
NUM_W = (1, 9)
# Height floor 4, not 3. Measured on a live match: every false positive was
# exactly 3px tall -- the white text inside the blue "Get Ready!"/"Matching..."
# bar, which fired 454 times -- while every genuine numeral was 4px or more
# (4x4, 5x8, 6x8, 7x8, 2x8). One pixel of height separates them cleanly.
NUM_H = (4, 10)
NUM_AREA = 3
NUM_X = (74, 84)
# Discriminators that separate a rendered digit from the board behind it.
# Measured on 26 live reward events (3 of which were genuine digits):
#     size bounds only          23/23 false positives
#     + perimeter darkness       3/23
#     + colour                   0/23, keeping the clear white digits
GLYPH_RING_DARK = 0.6       # fraction of the surrounding ring that must be dark
GLYPH_MIN_CHANNEL = 150     # digits are bright in EVERY channel
GLYPH_GREEN_BIAS = 20       # reject green-dominant blobs (the KO badges)
# Shape templates for the UI digit font, harvested from the pregame player
# count (same font, plain dark background, hundreds of clean samples) plus one
# confirmed in-game '4'. A candidate must LOOK LIKE A DIGIT, not merely be a
# bright compact blob -- the property the earlier threshold cascade never
# tested. Normalised cross-correlation, exactly as the "Tetris" banner is
# matched. Incomplete: 0/1/3/4/7 so far, so it is a gate ALONGSIDE the shape
# checks rather than a replacement, and it improves as more digits are seen.
DIGIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "tetris_digits.npy")
DIGIT_SIZE = (20, 28)       # (w, h) each candidate is normalised to
DIGIT_NCC = 0.78
DIGIT_LABEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tetris_digit_labels.npy")

# --- what the agent sees --------------------------------------------------
# Just the player's own board: hold box, matrix, next queue. Measured on live
# gameplay by finding the columns that stay dark across frames --
#
#     hold box    x  46..56   y  5..20
#     matrix      x  59..97   y  5..88
#     next queue  x 100..112  y  5..55
#
# -- which are contiguous, so one rectangle covers all three and keeps their
# spatial relationship intact. Everything outside it is the 98 opponent
# boards: a large, busy, constantly-changing fraction of the frame that the
# agent cannot act on.
PLAY_AREA = (3, 89, 45, 114)        # (y0, y1, x0, x1) in reference coords
OBS_SIZE = (84, 84)                 # matches IMAGE_RESHAPE in muzero.py

# --- board decoding (DIAGNOSTIC ONLY) -------------------------------------
# Occupancy is decoded for the live monitor's overlay. It is deliberately NOT
# the reward signal.
#
# The in-matrix numeral is the GARBAGE-SENT counter, and garbage sent is
# exactly what the reward is meant to measure: the object of Tetris 99 is to
# attack, and a lone single line clear sends nothing, so it is net neutral and
# should score nothing. A single that is part of a combo DOES send garbage and
# does score -- which falls out of using the counter, for free.
#
# This is why measuring the numeral against "every row that vanished" looked
# like 1/11 recall. Those 11 included net-neutral singles the detector is
# supposed to ignore. It was right; the yardstick was wrong.
BOARD = (8, 88, 61, 98)             # matrix interior (y0, y1, x0, x1)
BOARD_ROWS, BOARD_COLS = 20, 10
BOARD_OCCUPIED = 60                 # cell is filled if its centre exceeds this
# Cells that must vanish to count as a clear. One row is 10 cells; 8 allows
# for a cell or two misread at the edges without admitting the churn of a
# piece locking (which ADDS cells) or a single misdecoded cell. Swept against
# ground truth: 8 -> 7/11, 7 -> 10/11 with 2 spurious, 5 -> 11/11 with 4.
CLEAR_DROP = 6
# Averaging window either side of the drop, in SECONDS not frames. A frame
# count means different things at different rates: 3 frames spans 0.19s when
# stepping every 2nd frame of a 74fps recording but only 0.09s live at full
# rate, so the live detector smoothed half as much and fired on the churn of a
# falling piece. Same failure as the frame-based re-arm.
CLEAR_WINDOW_S = 0.10

# --- structural in-game test ----------------------------------------------
# The matrix is a DARK rectangle at a fixed position, flanked by the busy
# opponent grids. That is a geometric relationship, so it survives a reskin --
# unlike matching the rails' absolute colour, which is what the original
# `chrome` test did and which broke the moment the game's palette changed.
# Measured on the 2020 corpus AND a 2026 live capture: in-game recall 0.910,
# out-of-game rejection 1.000 on all four screen classes, live menus 0.000.
IG_INSIDE = (20, 78, 63, 97)        # matrix interior
IG_LEFT = (20, 78, 8, 45)           # background left of the matrix
IG_RIGHT = (20, 78, 115, 152)       # background right of it
IG_CONTRAST = 0.30                  # (outside - inside) / (outside + inside)
IG_DARK = 0.45                      # fraction of interior below IG_DARK_LEVEL
IG_DARK_LEVEL = 60
# ...but the decision is made on EDGE strength, not on either of those.
#
# Darkness fails when the stack fills the matrix -- measured on live frames
# where the board was full of garbage: contrast fell to 0.24 and darkness to
# 0.43, so a dark-interior test loses the game exactly when the player is
# about to die. What survives is the matrix BOUNDARY: there is a vertical
# discontinuity at the rails whether the matrix is empty or full, and whatever
# the palette. Measured:
#
#     corpus in-game      122..204     live full-stack matrix  120..155
#     corpus out-of-game   10.. 40     live menus               27.. 29
#
# A 3x margin, on both the 2020 corpus and a 2026 capture.
IG_EDGE_ROWS = (18, 80)
IG_EDGE_THRESHOLD = 80.0

# --- thresholds (see module docstring for the measurements) ---------------
CHROME_TOL = 45                     # per-channel tolerance for a chrome pixel
CHROME_THRESHOLD = 0.5              # fraction of chrome pixels that must match
GO_THRESHOLD = 0.50
# 0.60, not 0.45. The offset search below takes a max over several positions,
# which inflates scores on bright noise -- a multiple-comparisons effect. The
# banner persists ~46 frames, so per-frame recall of 0.44 still detects it with
# certainty, while 0.45 admitted a flash. Measured on a live match:
#     thr 0.45 -> banner recall 0.848, 1 false-positive frame
#     thr 0.60 -> banner recall 0.435, 0 false positives
#
# Raised again to 0.85 once the template was rebuilt from 68 LIVE banner
# frames (`tetris_banner_live.npy`) instead of the 2020 corpus. Against the
# live template the separation is far wider -- banners median 0.902, non-
# banners max 0.836 over 1196 frames -- so:
#     thr 0.75 -> recall 0.985, 1 false in 1196
#     thr 0.85 -> recall 0.706, 0 false
# The banner persists ~46 frames, so per-frame recall of 0.71 still detects
# every one; the threshold buys zero false positives for nothing.
TETRIS_THRESHOLD = 0.85
# Offset search half-width for the banner template. Kept tight for the same
# reason: every extra offset is another chance for noise to score high.
TETRIS_SEARCH = (2, 1)              # (dx, dy)
# Above this mean-absolute distance, a frame matches NONE of the four known
# out-of-game screens and is reported as UNKNOWN rather than snapped to the
# nearest one. Without it the classifier always answered with one of four, so
# the Switch home menu and the Tetris 99 mode-select screens -- neither of
# which is in the corpus -- came back labelled `title_a` or `error`.
# Corpus frames sit at p95 <= 20 from their own template; in-game frames are at
# least 54 from the nearest. A live results screen measured 44, so the usable
# window is narrow and this sits in the middle of it.
OUT_MAX_DIST = 49.0
UNKNOWN = "unknown"

# --- reward values --------------------------------------------------------
# The ONLY numbers in this file not measured from data. REWARD_GARBAGE is paid
# when the garbage counter appears, i.e. when an attack was actually sent --
# a lone single clear sends nothing and is deliberately worth zero.
REWARD_TETRIS = 1.0
# Garbage reward scales with HOW MUCH was sent: the counter's value is
# available for free now that each digit has its own template, and sending 4
# lines is genuinely four times the achievement of sending 1. Reward is
# REWARD_PER_GARBAGE * lines_sent, so 1 -> 0.25 and 4 -> 1.00.
REWARD_PER_GARBAGE = 0.25
REWARD_GARBAGE = 0.5        # fallback when the digit's value is unknown
REWARD_DEATH = -1.0

# --- state machine tuning -------------------------------------------------
# Chrome drops out on the brightest line-clear flashes (worst observed frame
# scored 0.117), so leaving the in-game state needs several consecutive
# non-chrome frames, not one. This is the difference between "the agent died"
# and "a laser went off".
DONE_PATIENCE = 4
# GO is on screen ~1s. If a frame is dropped and the glyph is missed, the
# machine would sit in PREGAME forever, so a changing playfield also starts the
# match.
PREGAME_FALLBACK_FRAMES = 3
PREGAME_FALLBACK_DELTA = 6.0
# A clear must be absent this long before it can pay again -- covers glyph
# flicker without merging two genuinely separate clears.
#
# In SECONDS, not frames, because the same frame count means different things
# at different rates: the live monitor runs ~74 fps and the env 20 Hz, so a
# 3-frame re-arm is 0.04 s in one and 0.15 s in the other. Measured on a live
# match, detection gaps within one clear were 2-5 frames (~0.05 s) while
# genuinely separate clears were ~50 frames (~0.7 s) apart, so 0.25 s sits
# clear of both.
REARM_SECONDS = 0.25
# Tetrises need a much longer window than garbage sends: four rows cannot
# legitimately be cleared twice within a couple of seconds, and the banner
# lingers long enough that a momentary dip below threshold was paying twice.
REARM_SECONDS_TETRIS = 3.0
REARM_FRAMES = 3          # retained for the frame-driven unit tests


# ---------------------------------------------------------------------------
# Colour predicates
# ---------------------------------------------------------------------------
def m_white(a):
    """Bright and desaturated: banner text and clear numerals."""
    lo, hi = a.min(axis=-1), a.max(axis=-1)
    return (lo > 180) & ((hi - lo) < 45)


def m_glyph(a):
    """Bright, not strongly coloured: UI text and numerals, whatever the tint.

    Replaces a strict white test (`min>180, max-min<45`). Measured glyph
    pixels: corpus [163,189,178] (min 127, spread 87), live capture
    [232,234,168] (min 168, spread 65) -- the live glyphs are CYAN, so the
    desaturation test rejected 100% of them and 67% of the corpus ones too.
    Requiring brightness in any channel plus a floor on the darkest channel
    keeps text while rejecting saturated tetromino blocks (a pure red block
    has min 0) and mid-grey garbage (max below 170).
    """
    lo, hi = a.min(axis=-1), a.max(axis=-1)
    return (hi > 170) & (lo > 110)


def m_cyan(a):
    """High blue+green, low red: the GO glyph."""
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    return (b > 150) & (g > 150) & (r < b - 30)


def m_glyph_local(patch, margin=45, k=None):
    """Bright RELATIVE TO ITS SURROUNDINGS, rather than in absolute terms.

    The absolute test (`max>170 and min>110`) is fine on the dark matrix but
    fails over incoming garbage: grey garbage blocks sit near 180 on every
    channel, so they pass it. The digit then merges with the background into
    one over-size component and is rejected, while the garbage itself produces
    glyph-sized components that are not digits. That is why rewards appeared
    when junk arrived and disappeared when the digit landed on blocks.

    Comparing against a heavily blurred copy of the same patch keeps the test
    working on any background, because a rendered digit is always a local
    bright spike whatever sits behind it.
    """
    g = patch.max(axis=2).astype(np.int16)
    if k is None:
        k = max(9, (min(g.shape) // 2) * 2 + 1)
    k = min(k | 1, (min(g.shape) // 2) * 2 + 1) if min(g.shape) > 4 else 3
    bg = cv2.blur(g.astype(np.float32), (k, k))
    return (g > bg + margin) & (g > 120)


def detect_content_box(frame, thresh=18):
    """Find the non-black picture inside a letterboxed capture.

    A Cam Link 4K fed by a Switch delivers 1920x1080 with the picture inset --
    measured 38px left/right and ~22px top/bottom. Resizing the whole frame
    then squeezes the game by ~2%, which is 3px in the 160-wide reference
    frame: more than enough to slide a 1px playfield rail off its column and
    make every ROI miss. Returns (y0, y1, x0, x1), or None if the frame is too
    dark to measure (a fade, or no signal).
    """
    g = frame.max(axis=2)
    cols = np.where(g.max(axis=0) > thresh)[0]
    rows = np.where(g.max(axis=1) > thresh)[0]
    if len(cols) < frame.shape[1] // 4 or len(rows) < frame.shape[0] // 4:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def detect_channel_order(frame, corpus_mean, box=None):
    """True if this frame's red and blue channels need swapping.

    Not every capture path hands OpenCV the BGR it assumes. A Cam Link 4K over
    DSHOW delivers RGB, which inverts every colour predicate here -- the warm
    Tetris 99 background arrives looking blue, chrome never matches, and the
    state machine sits out of game forever. Measured on a live results screen:
    mean distance to the corpus was 52.1 raw against 7.5 swapped.

    Decided on the game's warm bias rather than by matching a reference
    screen: red exceeds blue by +26..+78 on EVERY corpus class (in-game,
    pregame, results, both title screens, the error dialog), so a capture
    where blue exceeds red has the channels reversed. Matching against the
    corpus mean instead made the answer depend on which screen happened to be
    showing at startup -- a menu frame gave the wrong answer.
    """
    p = prepare(frame, box).astype(np.float32).mean(axis=(0, 1))
    return bool((p[2] - p[0]) < 0.0)


def prepare(frame, box=None, swap_rb=False):
    """Any capture frame -> the reference 90x160 BGR array the ROIs assume.

    `box` is a content rectangle from detect_content_box, applied before the
    resize so letterboxing does not shift the geometry. Calibrate it once and
    reuse it: measuring per frame would move the crop during a fade to black.

    `swap_rb` exchanges red and blue for captures that arrive as RGB; see
    detect_channel_order.

    INTER_AREA because it averages over the discarded pixels; INTER_LINEAR
    samples 2x2 and drops rows outright on a 1080->90 reduction, which makes
    1-2px glyph strokes appear and vanish with sub-pixel phase.
    """
    if frame.shape[0] != FRAME_H or frame.shape[1] != FRAME_W:
        if box is not None:
            y0, y1, x0, x1 = box
            frame = frame[y0:y1, x0:x1]
        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    if swap_rb:
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


def crop_region(frame, region, box=None):
    """Crop a region given in reference (90x160) coords from a full frame.

    `box` is the letterbox content rectangle, so the mapping stays correct on
    a capture whose picture is inset. Returns the native-resolution pixels --
    scaling is the caller's choice.
    """
    ry0, ry1, rx0, rx1 = region
    if frame.shape[0] == FRAME_H and frame.shape[1] == FRAME_W:
        return frame[ry0:ry1, rx0:rx1]
    y0, y1, x0, x1 = box or (0, frame.shape[0], 0, frame.shape[1])
    sy, sx = (y1 - y0) / FRAME_H, (x1 - x0) / FRAME_W
    return frame[y0 + int(ry0 * sy):y0 + int(ry1 * sy),
                 x0 + int(rx0 * sx):x0 + int(rx1 * sx)]


def _crop(a, roi):
    """Crop a SINGLE frame, (H, W) or (H, W, C)."""
    y0, y1, x0, x1 = roi
    return a[y0:y1, x0:x1]


def _batch_crop(a, roi):
    """Crop a STACK of frames, (N, H, W) or (N, H, W, C).

    Separate from _crop because plain a[y0:y1, x0:x1] on a batch slices the
    frame axis instead of the rows -- silently returning a handful of whole
    frames rather than the region, with no error.
    """
    y0, y1, x0, x1 = roi
    return a[:, y0:y1, x0:x1]


def _ncc(vec, template):
    """Zero-mean normalised correlation: scores the PATTERN, not the
    brightness. On an animated background absolute pixel values are noise --
    with the identical ROI, raw-pixel matching scored 0.518 recall where this
    scored 0.965."""
    v = vec - vec.mean()
    t = template - template.mean()
    d = np.linalg.norm(v) * np.linalg.norm(t)
    return 0.0 if d < 1e-9 else float(v @ t / d)


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------
CORPUS = {
    "neutral":    "training/tetris_environment/neutral",
    "go":         "training/tetris_pregame_opt/go",
    "pregame":    "training/tetris_pregame_opt/pregame",
    "end_screen": "training/tetris_environment/end_screen",
    "error":      "training/tetris_environment/error",
    "title_a":    "training/tetris_environment/title_a",
    "title_b":    "training/tetris_environment/title_b",
    "clr_tetris": "training/tetris_line_clears/tetris",
    "clr_none":   "training/tetris_line_clears/neutral",
}
INGAME = ("neutral", "go", "pregame", "clr_tetris")
OUTGAME = ("end_screen", "error", "title_a", "title_b")


def load_corpus(name, cap=600, seed=0, root="."):
    folder = os.path.join(root, CORPUS[name])
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    random.Random(seed).shuffle(paths)
    ims = []
    for p in paths[:cap]:
        im = cv2.imread(p)
        if im is not None and im.shape == (FRAME_H, FRAME_W, 3):
            ims.append(im)
    if not ims:
        raise SystemExit(f"no usable frames in {folder}")
    return np.stack(ims).astype(np.float32)


def build_templates(out=TEMPLATE_FILE, cap=600, root="."):
    """Derive every template from the labelled corpus and save them.

    Step 1 asks what the in-game states AGREE on; step 2 compares each state
    only against its siblings. Scoring a state against every other state wastes
    the comparison on pairs that are never confusable, and it is what stopped
    the flat version from finding anything for `neutral` at all.
    """
    data = {k: load_corpus(k, cap, root=root) for k in CORPUS}
    half = {k: v[:max(2, len(v) // 2)] for k, v in data.items()}

    mu = {k: v.mean(axis=0) for k, v in half.items()}
    var = {k: v.var(axis=0) for k, v in half.items()}

    # 1. chrome: stable within every in-game state, agreed across them, and
    #    contradicted by every out-of-game state.
    mus = np.stack([mu[k] for k in INGAME])
    vrs = np.stack([var[k] for k in INGAME])
    agree = ((mus.max(0) - mus.min(0)).max(axis=2) < 18) & \
            (np.sqrt(vrs.max(0)).max(axis=2) < 22)
    sep = np.ones((FRAME_H, FRAME_W), bool)
    for k in OUTGAME:
        sep &= np.abs(mus.mean(0) - mu[k]).max(axis=2) > 45
    chrome = agree & sep
    chrome_ref = mus.mean(0)[chrome]

    # 2. glyph templates, correlated as binary colour masks.
    tetris_t = _batch_crop(m_glyph(half["clr_tetris"]).astype(np.float32),
                           ROI_TETRIS).mean(axis=0).ravel()
    go_t = _batch_crop(m_cyan(half["go"]).astype(np.float32),
                       ROI_GO).mean(axis=0).ravel()

    # 3. out-of-game screens: raw-pixel ROI means. These screens freeze or blur
    #    the background (the error dialog leaves 95% of its pixels literally
    #    constant), so absolute values are informative here in a way they never
    #    are over live gameplay.
    out_t = {k: _batch_crop(half[k], ROI_OUT[k]).mean(axis=0).ravel()
             for k in OUTGAME}

    corpus_mean = np.mean([v.mean(axis=(0, 1, 2)) for v in half.values()], axis=0)
    print(f"corpus mean BGR: {np.round(corpus_mean, 1)}")
    np.savez_compressed(
        out, chrome_mask=chrome, chrome_ref=chrome_ref,
        tetris=tetris_t, go=go_t, corpus_mean=corpus_mean,
        **{f"out_{k}": v for k, v in out_t.items()})
    print(f"chrome pixels: {int(chrome.sum())}")
    ys, xs = np.where(chrome)
    print(f"chrome bbox:   y {ys.min()}..{ys.max()}  x {xs.min()}..{xs.max()}")
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
class TetrisMasks:
    """Stateless per-frame detectors. Every method takes a raw capture frame."""

    def __init__(self, path=TEMPLATE_FILE):
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} not found -- run `python tetris_masks.py --build` "
                f"from the repo root first.")
        z = np.load(path)
        self.chrome_mask = z["chrome_mask"]
        self.chrome_ref = z["chrome_ref"]
        self.tetris_t = z["tetris"]
        # Prefer a template averaged from LIVE banner frames when one exists:
        # the corpus template describes a 2020 build whose banner sits 2-3px
        # from where the current one renders.
        live_t = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tetris_banner_live.npy")
        if os.path.exists(live_t):
            self.tetris_t = np.load(live_t).astype(np.float32).ravel()
        self.go_t = z["go"]
        self.corpus_mean = (z["corpus_mean"] if "corpus_mean" in z.files
                            else np.array([64.0, 90.0, 110.0], np.float32))
        self.out_t = {k: z[f"out_{k}"] for k in OUTGAME}
        self.digits = None
        self.digit_labels = None
        if os.path.exists(DIGIT_FILE):
            d = np.load(DIGIT_FILE).reshape(-1, DIGIT_SIZE[1] * DIGIT_SIZE[0])
            d = d - d.mean(axis=1, keepdims=True)
            self.digits = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
            if os.path.exists(DIGIT_LABEL_FILE):
                self.digit_labels = np.load(DIGIT_LABEL_FILE)

    def digit_match(self, mask_patch):
        """(best correlation, digit value) against the known digit shapes."""
        if self.digits is None:
            return 1.0, None
        v = cv2.resize(mask_patch.astype(np.float32), DIGIT_SIZE,
                       interpolation=cv2.INTER_AREA).ravel()
        v = v - v.mean()
        sims = self.digits @ v / (np.linalg.norm(v) + 1e-6)
        k = int(np.argmax(sims))
        value = int(self.digit_labels[k]) if self.digit_labels is not None else None
        return float(sims[k]), value

    def digit_score(self, mask_patch):
        return self.digit_match(mask_patch)[0]

    # -- level 1 -----------------------------------------------------------
    def chrome_score(self, prep):
        d = np.abs(prep[self.chrome_mask].astype(np.float32) - self.chrome_ref)
        return float((d.max(axis=1) < CHROME_TOL).mean())

    def edge_score(self, prep):
        """Vertical-edge strength at the matrix rails, above the local median.

        Fill- and palette-independent: it measures that a boundary EXISTS at
        x=RAIL_L and x=RAIL_R, not what colour it is or what is inside it.
        """
        g = prep.max(axis=2).astype(np.float32)
        gx = np.abs(np.diff(g[IG_EDGE_ROWS[0]:IG_EDGE_ROWS[1], :],
                            axis=1)).mean(axis=0)

        def peak(x, w=2):
            local = gx[max(0, x - w):x + w + 1].max()
            ref = np.median(gx[max(0, x - 14):x + 15])
            return float(local - ref)

        return (peak(RAIL_L) + peak(RAIL_R)) / 2.0

    def board_occupancy(self, prep):
        """(20,10) bool grid of occupied cells, for display and diagnostics."""
        y0, y1, x0, x1 = BOARD
        rh = (y1 - y0) / BOARD_ROWS
        cw = (x1 - x0) / BOARD_COLS
        cy = np.clip((y0 + (np.arange(BOARD_ROWS) + 0.5) * rh).astype(int),
                     0, FRAME_H - 1)
        cx = np.clip((x0 + (np.arange(BOARD_COLS) + 0.5) * cw).astype(int),
                     0, FRAME_W - 1)
        return prep[cy][:, cx].max(axis=2) > BOARD_OCCUPIED

    def board_cells(self, prep):
        """Number of occupied cells in the 10x20 matrix.

        Sampled at cell centres rather than by area, so a falling piece's
        anti-aliased edges do not inflate the count.
        """
        y0, y1, x0, x1 = BOARD
        rh = (y1 - y0) / BOARD_ROWS
        cw = (x1 - x0) / BOARD_COLS
        cy = (y0 + (np.arange(BOARD_ROWS) + 0.5) * rh).astype(int)
        cx = (x0 + (np.arange(BOARD_COLS) + 0.5) * cw).astype(int)
        cells = prep[np.clip(cy, 0, FRAME_H - 1)][:, np.clip(cx, 0, FRAME_W - 1)]
        return int((cells.max(axis=2) > BOARD_OCCUPIED).sum())

    def in_game_features(self, prep):
        """(edge_score, dark_fraction). Only the first decides; the second is
        reported for diagnostics."""
        ins = _crop(prep.max(axis=2).astype(np.float32), IG_INSIDE)
        return self.edge_score(prep), float((ins < IG_DARK_LEVEL).mean())

    def in_game(self, prep):
        return self.edge_score(prep) >= IG_EDGE_THRESHOLD

    # -- level 2a ----------------------------------------------------------
    def out_screen(self, prep, with_distance=False):
        best, name = None, UNKNOWN
        for k, t in self.out_t.items():
            d = float(np.abs(_crop(prep, ROI_OUT[k]).astype(np.float32).ravel() - t).mean())
            if best is None or d < best:
                best, name = d, k
        if best is None or best > OUT_MAX_DIST:
            name = UNKNOWN
        return (name, best) if with_distance else name

    # -- level 2b ----------------------------------------------------------
    def go_score(self, prep):
        v = m_cyan(_crop(prep, ROI_GO)).astype(np.float32).ravel()
        return _ncc(v, self.go_t)

    def is_go(self, prep):
        return self.go_score(prep) >= GO_THRESHOLD

    def tetris_score(self, prep):
        """Best correlation over a small search window, not a fixed offset.

        The live UI draws the banner ~2-3px left of where the corpus does: a
        confirmed Tetris frame scored 0.335 against a 0.45 threshold, and
        +0.662 once the template was shifted 2px. Searching a few offsets
        costs a handful of dot products on a 12x36 patch and makes the test
        survive the next time the layout moves.
        """
        y0, y1, x0, x1 = ROI_TETRIS
        h, w = y1 - y0, x1 - x0
        tmpl = self.tetris_t.reshape(h, w)
        best = -1.0
        wx, wy = TETRIS_SEARCH
        for dy in range(-wy, wy + 1):
            for dx in range(-wx, wx + 1):
                yy0, xx0 = y0 + dy, x0 + dx
                if yy0 < 0 or xx0 < 0 or yy0 + h > FRAME_H or xx0 + w > FRAME_W:
                    continue
                v = m_glyph(prep[yy0:yy0 + h, xx0:xx0 + w]).astype(np.float32).ravel()
                best = max(best, _ncc(v, self.tetris_t))
        return best

    def has_tetris(self, prep):
        return self.tetris_score(prep) >= TETRIS_THRESHOLD

    def clear_numeral_hires(self, frame, box=None, swap_rb=False):
        """The garbage counter, detected on the NATIVE-resolution frame.

        At the 90x160 reference size a digit is only 4-8px tall, which is not
        enough to tell a '1' from an attack-laser sliver -- measured false
        positives were a 1x4 streak between blocks and a 3x5 flash. On a
        1080p capture the same digit is ~50-90px tall.

        Components are measured at full resolution and then converted BACK to
        reference units, so every bound tuned at 90x160 still applies and only
        the precision improves.
        """
        if frame.shape[0] == FRAME_H and frame.shape[1] == FRAME_W:
            return self.clear_numeral(frame)
        strip = crop_region(frame, NUM_STRIP, box)
        if strip.size == 0:
            return None
        if swap_rb:
            strip = strip[:, :, ::-1]
        sy = strip.shape[0] / float(NUM_STRIP[1] - NUM_STRIP[0])
        sx = strip.shape[1] / float(NUM_STRIP[3] - NUM_STRIP[2])
        m = m_glyph_local(strip).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        grey = strip.max(axis=2).astype(np.int16)
        cands = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            rw, rh = w / sx, h / sy                       # -> reference units
            rx = NUM_STRIP[2] + x / sx
            ry = NUM_STRIP[0] + y / sy
            if not (NUM_W[0] <= rw <= NUM_W[1] and NUM_H[0] <= rh <= NUM_H[1]
                    and NUM_X[0] <= rx <= NUM_X[1]
                    and area / float(w * h) > 0.35):
                continue
            comp = labels == i
            # A digit FLOATS on the dark board; a block's bright bevel is
            # welded to the block. Measured on 26 live reward events: this
            # alone cut false positives from 23 to 3.
            dil = cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8))
            ring = dil.astype(bool) & ~comp
            if not ring.any() or (grey[ring] < 70).mean() < GLYPH_RING_DARK:
                continue
            # What survived was the green KO-badge hexagons, which genuinely do
            # float on dark. Digits are white or cyan -- neutral, or blue-green
            # together. A badge is GREEN-DOMINANT.
            px = strip[comp].astype(np.float32)
            if px.size == 0:
                continue
            mean = px.mean(axis=0)
            if mean.min() < GLYPH_MIN_CHANNEL:
                continue
            if mean[1] - max(mean[0], mean[2]) > GLYPH_GREEN_BIAS:
                continue
            score, value = self.digit_match(comp[y:y + h, x:x + w])
            if score < DIGIT_NCC:
                continue
            cands.append((rx, ry, rw, rh, value))
        for i, a in enumerate(cands):
            for j, b in enumerate(cands):
                if i != j and abs(b[1] - a[1]) <= 2 and 0 < abs(b[0] - a[0]) <= 8:
                    return None
        if not cands:
            return None
        x, y, w, h, value = cands[0]
        return (int(round(x)), int(round(y)), int(round(w)), int(round(h)),
                value)

    def clear_numeral(self, prep):
        """The white digit inside the matrix, or None.

        Connected-component rather than template match: the glyph differs by
        digit and its row moves with the stack, so only its column and its size
        are stable.
        """
        y0, y1, x0, x1 = NUM_STRIP
        m = m_glyph(prep[y0:y1, x0:x1]).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cands = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            x += x0
            y += y0
            if (NUM_W[0] <= w <= NUM_W[1] and NUM_H[0] <= h <= NUM_H[1]
                    and area >= NUM_AREA and NUM_X[0] <= x <= NUM_X[1]
                    and area / float(w * h) > 0.35):
                cands.append((int(x), int(y), int(w), int(h)))
        # Reject MULTI-DIGIT numbers. The one confounder in the corpus is the
        # pregame player count -- two adjacent digits on a fixed row under the
        # blue Ready bar. Every numeral fire on `clr_none` was one of these,
        # and every genuine clear numeral is a lone digit. The state machine
        # already keeps pregame away from the reward path; this makes the
        # frame-level detector correct on its own too.
        for i, a in enumerate(cands):
            for j, b in enumerate(cands):
                if i != j and abs(b[1] - a[1]) <= 2 and 0 < abs(b[0] - a[0]) <= 8:
                    return None
        if not cands:
            return None
        # Belt-and-braces against the lobby bar: reject a candidate sitting on
        # a strongly blue background. The height floor already removes these,
        # but the bar is the one recurring structure that puts bright text
        # inside the matrix, and it costs one slice to check.
        x, y, w, h = cands[0]
        band = prep[max(0, y - 2):y + h + 2, max(0, x - 6):x + w + 6]
        if band.size:
            b, g, r = band[..., 0].astype(int), band[..., 1].astype(int), band[..., 2].astype(int)
            if (((b > 110) & (b - r > 55) & (b - g > 35)).mean()) > 0.35:
                return None
        return cands[0]

    def screen(self, frame):
        """Coarse screen label for a single frame, no temporal state."""
        prep = prepare(frame)
        if not self.in_game(prep):
            return self.out_screen(prep)
        return "go" if self.is_go(prep) else "in_game"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
TITLE_A, TITLE_B, ERROR, END = "title_a", "title_b", "error", "end_screen"
PREGAME, PLAYING = "pregame", "playing"


class TetrisState:
    """Temporal wrapper: screen state, edge-triggered reward, terminal flag.

    Frame-level detection is not enough on its own for either half of the job.
    A reward has to fire once per clear rather than once per frame the banner is
    visible, and `done` has to survive a flash frame that momentarily hides the
    chrome.
    """

    def __init__(self, masks, reward_tetris=REWARD_TETRIS,
                 reward_garbage=REWARD_GARBAGE, reward_death=REWARD_DEATH,
                 content_box=None, swap_rb=False, clock=None):
        self.m = masks
        self.clock = clock or time.monotonic
        self.rw = (reward_tetris, reward_garbage, reward_death)
        # Letterbox crop applied before every resize; see detect_content_box.
        self.box = content_box
        # Red/blue swap for RGB captures; see detect_channel_order.
        self.swap_rb = swap_rb
        self.state = PREGAME
        self._out_run = 0
        self._pf_prev = None
        self._pf_run = 0
        self._tetris_off = REARM_FRAMES
        self._clear_off = REARM_FRAMES
        # Wall-clock re-arm. `clock` is injectable so tests can drive it.
        self._off_since = {"tetris": -1e9, "garbage": -1e9}
        self._armed = {"tetris": True, "garbage": True}
        self._cells = deque(maxlen=256)      # (timestamp, cell count)

    def reset(self):
        self.__init__(self.m, *self.rw, content_box=self.box,
                      swap_rb=self.swap_rb, clock=self.clock)

    def calibrate(self, frame):
        """Measure letterbox crop AND channel order from a bright frame.

        Both are properties of the capture path rather than the game, so they
        are measured once and held. Returns (box, swap_rb).
        """
        box = detect_content_box(frame)
        if box is not None:
            self.box = box
        self.swap_rb = detect_channel_order(frame, self.m.corpus_mean, self.box)
        return self.box, self.swap_rb

    def _playfield(self, prep):
        return _crop(prep, PLAYFIELD).astype(np.float32)

    def step(self, frame):
        """Advance one frame. Returns dict(state, reward, done, event, score)."""
        prep = prepare(frame, self.box, self.swap_rb)
        rw_tetris, rw_garbage, rw_death = self.rw
        reward, done, event = 0.0, False, None
        edge, dark = self.m.in_game_features(prep)
        score = edge

        if edge < IG_EDGE_THRESHOLD:
            self._out_run += 1
            if self._out_run >= DONE_PATIENCE:
                was_playing = self.state in (PLAYING, PREGAME)
                self.state = self.m.out_screen(prep)
                if was_playing:
                    done = True
                    event = "died"
                    if self.state == END:
                        reward = rw_death
            return self._out(reward, done, event, score, dark)

        self._out_run = 0
        if self.state not in (PREGAME, PLAYING):
            self.state = PREGAME
            self._pf_prev, self._pf_run = None, 0

        if self.state == PREGAME:
            pf = self._playfield(prep)
            if self.m.is_go(prep):
                self.state = PLAYING
                event = "go"
            elif self._pf_prev is not None:
                # Fallback: a changing matrix means the match is under way even
                # if the GO glyph was never seen.
                if float(np.abs(pf - self._pf_prev).mean()) > PREGAME_FALLBACK_DELTA:
                    self._pf_run += 1
                    if self._pf_run >= PREGAME_FALLBACK_FRAMES:
                        self.state = PLAYING
                        event = "go_fallback"
                else:
                    self._pf_run = 0
            self._pf_prev = pf
            return self._out(reward, done, event, score, dark)

        # PLAYING: reward, in the priority order the clear semantics require.
        now = self.clock()
        tetris_on = self.m.has_tetris(prep)
        # The GO glyph sits inside the numeral strip and fires the shape test
        # on 100% of GO frames. GO is also what flips us into PLAYING, so
        # without this guard the match would open with a phantom clear.
        # Reward = garbage sent, read off the in-matrix counter. Board
        # occupancy is tracked only so the monitor can display it.
        self._last_cells = self.m.board_cells(prep)
        # Detected on the RAW frame where the digit is ~10x larger. The match
        # also yields the counter's VALUE, i.e. how many lines were sent.
        # A Tetris and its garbage counter are SEPARATE rewards: clearing four
        # rows is an achievement in itself, and the lines it sends are another.
        # Previously the banner suppressed the counter, so a Tetris worth 6
        # paid 1.00 instead of 1.00 + 1.50.
        hit = None
        if not self.m.is_go(prep):
            hit = self.m.clear_numeral_hires(frame, self.box, self.swap_rb)
        garbage_on = hit is not None
        self._last_garbage = hit[4] if hit and len(hit) > 4 else None

        sent = self._last_garbage
        garbage_value = (REWARD_PER_GARBAGE * sent) if sent else rw_garbage
        events = []
        for kind, on, value in (("tetris", tetris_on, rw_tetris),
                                ("garbage", garbage_on, garbage_value)):
            if on:
                self._off_since[kind] = None
                if self._armed[kind]:
                    # Accumulate: both can pay on the same frame.
                    reward += value
                    events.append(kind)
                    self._armed[kind] = False
            else:
                # Re-arm only after a CONTINUOUS absence. Timing from the
                # payout instead let a single sub-threshold frame re-pay: a
                # measured banner sat on for 0.85s and dipped to 0.444 against
                # a 0.45 threshold for one frame, which paid the same Tetris
                # three times.
                if self._off_since[kind] is None:
                    self._off_since[kind] = now
                window = (REARM_SECONDS_TETRIS if kind == "tetris"
                          else REARM_SECONDS)
                if now - self._off_since[kind] >= window:
                    self._armed[kind] = True
        if events:
            event = "+".join(events)
        return self._out(reward, done, event, score, dark)

    def _out(self, reward, done, event, score, dark=0.0):
        return {"state": self.state, "reward": reward, "done": done,
                "event": event, "edge": score, "dark": dark,
                "cells": getattr(self, "_last_cells", 0),
                "sent": getattr(self, "_last_garbage", None),
                "drop": getattr(self, "_last_drop", 0.0),
                "chrome": score}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(cap=600, root="."):
    """Measure every detector against the labelled corpus, held out.

    Templates are built from the first half of each class and every number
    below is scored on the second half, so nothing here is measured on the
    frames that produced it.
    """
    masks = TetrisMasks()
    data = {k: load_corpus(k, cap, root=root) for k in CORPUS}
    held = {k: v[max(2, len(v) // 2):] for k, v in data.items()}

    print("\n--- level 1: structural in-game test (matrix-boundary edge) ---")
    ok_in = tot_in = ok_out = tot_out = 0
    for k, v in held.items():
        feats = np.array([masks.in_game_features(f) for f in v])
        s = feats[:, 0]
        hit = feats[:, 0] >= IG_EDGE_THRESHOLD
        fam = "in " if k in INGAME or k == "clr_none" else "out"
        if fam == "in ":
            ok_in += int(hit.sum()); tot_in += len(v)
        else:
            ok_out += int((~hit).sum()); tot_out += len(v)
        print(f"  {fam} {k:11s} n={len(v):4d}  edge min={s.min():6.1f} "
              f"p1={np.percentile(s,1):6.1f} med={np.median(s):6.1f}  "
              f"fires={hit.mean():.3f}")
    print(f"  in-game recall {ok_in/max(tot_in,1):.4f} ({tot_in} frames)   "
          f"out-of-game reject {ok_out/max(tot_out,1):.4f} ({tot_out} frames)")

    print("\n--- level 2a: out-of-game screen id ---")
    cols = list(OUTGAME) + [UNKNOWN]
    conf = {k: {j: 0 for j in cols} for k in OUTGAME}
    for k in OUTGAME:
        for f in held[k]:
            conf[k][masks.out_screen(f)] += 1
    print(f"  {'truth':>11s} " + " ".join(f"{c:>11s}" for c in cols) + "   recall")
    tot = cor = 0
    for k in OUTGAME:
        row = [conf[k][c] for c in cols]
        tot += sum(row); cor += conf[k][k]
        print(f"  {k:>11s} " + " ".join(f"{n:11d}" for n in row)
              + f"   {conf[k][k]/max(sum(row),1):.3f}")
    print(f"  accuracy {cor/max(tot,1):.4f}")

    print("\n--- level 2b: GO glyph ---")
    for k in ("go", "pregame", "neutral", "clr_tetris"):
        s = np.array([masks.go_score(f) for f in held[k]])
        print(f"  {k:11s} n={len(s):4d}  med={np.median(s):+.3f} "
              f"max={s.max():+.3f}  fires={(s>=GO_THRESHOLD).mean():.3f}")

    print("\n--- reward 1: \"Tetris\" glyph ---")
    for k in ("clr_tetris", "clr_none", "neutral"):
        s = np.array([masks.tetris_score(f) for f in held[k]])
        print(f"  {k:11s} n={len(s):4d}  med={np.median(s):+.3f} "
              f"max={s.max():+.3f}  fires={(s>=TETRIS_THRESHOLD).mean():.3f}")

    print("\n--- reward 2: in-matrix numeral ---")
    for k in ("clr_tetris", "clr_none", "neutral", "pregame", "go"):
        hits = np.array([masks.clear_numeral(f) is not None for f in held[k]])
        print(f"  {k:11s} n={len(hits):4d}  fires={hits.mean():.3f}")
    print("  (the two-digit pregame PLAYER COUNT is rejected as multi-digit, which"
          "\n   took clr_none to 0.000. What remains on pregame is the single-digit"
          "\n   countdown; the state machine never evaluates reward outside PLAYING,"
          "\n   so it cannot reach the reward path either way.)")

    print("\n--- edge triggering ---")
    sm = TetrisState(masks)
    seq = list(held["clr_tetris"][:40])
    fired = sum(1 for f in seq if sm.step(f)["event"] == "tetris")
    print(f"  40 consecutive Tetris-banner frames -> {fired} reward event(s)")
    print("  (the old rewarded_steps window paid up to 7x for one clear)")
    return 0


def demo(cls="clr_tetris", cap=200, root="."):
    """Step through frames with the detector's decisions drawn on."""
    masks = TetrisMasks()
    frames = load_corpus(cls, cap, root=root).astype(np.uint8)

    def fresh(name):
        """Corpus frames are unordered samples, not an episode, so the machine
        would otherwise sit in PREGAME waiting for a GO that never comes and no
        reward would ever display. Start in-game classes already PLAYING; the
        per-frame detector scores shown alongside are state-independent
        regardless."""
        sm = TetrisState(masks)
        if name in INGAME or name == "clr_none":
            sm.state = PLAYING
        return sm

    sm = fresh(cls)
    i, S = 0, 6
    print("a/d step, [ ] class, q quit")
    names = list(CORPUS)
    while True:
        f = frames[i % len(frames)]
        r = sm.step(f)
        vis = cv2.resize(f, (FRAME_W * S, FRAME_H * S), interpolation=cv2.INTER_NEAREST)
        for roi, col in ((ROI_TETRIS, (0, 255, 0)), (ROI_GO, (255, 255, 0)),
                         (NUM_STRIP, (0, 165, 255))):
            y0, y1, x0, x1 = roi
            cv2.rectangle(vis, (x0 * S, y0 * S), (x1 * S, y1 * S), col, 1)
        num = masks.clear_numeral(prepare(f))
        if num:
            x, y, w, h = num
            cv2.rectangle(vis, (x * S, y * S), ((x + w) * S, (y + h) * S), (0, 0, 255), 2)
        txt = [f"{cls}  frame {i % len(frames)+1}/{len(frames)}",
               f"chrome {r['chrome']:.3f}  state {r['state']}",
               f"tetris {masks.tetris_score(prepare(f)):+.3f}  "
               f"go {masks.go_score(prepare(f)):+.3f}  num {'Y' if num else '-'}",
               f"reward {r['reward']:+.2f}  event {r['event']}"]
        for j, t in enumerate(txt):
            cv2.putText(vis, t, (8, 22 + j * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("tetris_masks demo", vis)
        k = cv2.waitKeyEx(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("d"):
            i += 1
        elif k == ord("a"):
            i -= 1
        elif k in (ord("["), ord("]")):
            cls = names[(names.index(cls) + (1 if k == ord("]") else -1)) % len(names)]
            frames = load_corpus(cls, cap, root=root).astype(np.uint8)
            i = 0
            sm = fresh(cls)
    cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="derive templates from the corpus")
    ap.add_argument("--eval", action="store_true", help="measure against the corpus")
    ap.add_argument("--demo", action="store_true", help="visual frame stepper")
    ap.add_argument("--class", dest="cls", default="clr_tetris", help="class for --demo")
    ap.add_argument("--cap", type=int, default=600, help="frames per class")
    ap.add_argument("--root", default=".", help="repo root holding training/")
    a = ap.parse_args()
    if a.build:
        build_templates(cap=a.cap, root=a.root)
        return 0
    if a.eval:
        return evaluate(cap=a.cap, root=a.root)
    if a.demo:
        return demo(a.cls, cap=a.cap, root=a.root)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
