#!/usr/bin/env python3
"""One canonical keyboard->Switch mapping, shared by every front-end.

`tetris_keyboard.py` (pygame) and `view_switch.py` (Win32) both build their
tables from LAYOUT, so a remap happens in exactly one place. They drifted apart
by hand several times before this existed.

Each entry: (key label, controller.py constant name, group, pygame key name,
Windows virtual-key code).
"""

LAYOUT = [
    # key    button       group     pygame key      VK code
    ("W",    "DPAD_U",    "move",   "K_w",          0x57),
    ("A",    "DPAD_L",    "move",   "K_a",          0x41),
    ("S",    "DPAD_D",    "move",   "K_s",          0x53),
    ("D",    "DPAD_R",    "move",   "K_d",          0x44),

    ("N",    "BTN_A",     "rotate", "K_n",          0x4E),
    ("M",    "BTN_B",     "rotate", "K_m",          0x4D),
    (",",    "BTN_X",     "rotate", "K_COMMA",      0xBC),
    ("/",    "BTN_Y",     "rotate", "K_SLASH",      0xBF),

    ("B",    "BTN_L",     "hold",   "K_b",          0x42),
    (".",    "BTN_R",     "hold",   "K_PERIOD",     0xBE),

    ("Enter", "BTN_PLUS",    "menu", "K_RETURN",    0x0D),
    ("Bksp",  "BTN_MINUS",   "menu", "K_BACKSPACE", 0x08),
    ("H",     "BTN_HOME",    "menu", "K_h",         0x48),
    ("C",     "BTN_CAPTURE", "menu", "K_c",         0x43),
]

# Keys a viewer/host program must NOT use for its own hotkeys.
RESERVED = {e[0].lower() for e in LAYOUT}

GROUPS = ["move", "rotate", "hold", "menu"]

# The right stick is NOT a set of independent buttons. The command word packs
# stick intensity (bits 44-51) and angle (bits 52-63) into adjacent fields, so
# adding two direction constants the way button bits are added carries out of
# intensity and into angle: RSTICK_U + RSTICK_R decodes as 91 degrees at
# intensity 254 -- very nearly centred -- instead of 45 degrees at 255.
# Arrows are therefore resolved to ONE eight-way constant per frame.
RSTICK = [
    ("Up",    "RSTICK_U", "K_UP",    0x26),
    ("Down",  "RSTICK_D", "K_DOWN",  0x28),
    ("Left",  "RSTICK_L", "K_LEFT",  0x25),
    ("Right", "RSTICK_R", "K_RIGHT", 0x27),
]

_EIGHT_WAY = {
    (0, 0): None,        (1, 0): "RSTICK_R",   (1, 1): "RSTICK_U_R",
    (0, 1): "RSTICK_U",  (-1, 1): "RSTICK_U_L", (-1, 0): "RSTICK_L",
    (-1, -1): "RSTICK_D_L", (0, -1): "RSTICK_D", (1, -1): "RSTICK_D_R",
}


def rstick_bits(controller, up, down, left, right):
    """Held arrows -> a single right-stick command value (0 when centred).

    Opposing pairs cancel, matching how the D-pad behaves.
    """
    x = (1 if right else 0) - (1 if left else 0)
    y = (1 if up else 0) - (1 if down else 0)
    name = _EIGHT_WAY[(x, y)]
    return 0 if name is None else getattr(controller, name)


def rstick_label(up, down, left, right):
    x = (1 if right else 0) - (1 if left else 0)
    y = (1 if up else 0) - (1 if down else 0)
    n = _EIGHT_WAY[(x, y)]
    return None if n is None else "R" + n.replace("RSTICK_", "")


def rows():
    """[(group, [(key label, button short name, pygame name, vk), ...]), ...]"""
    out = []
    for g in GROUPS:
        cells = [(k, b.replace("BTN_", "").replace("DPAD_", ""), pg, vk)
                 for k, b, grp, pg, vk in LAYOUT if grp == g]
        if cells:
            out.append((g, cells))
    return out


def by_vk(controller):
    """{vk code: (command bits, key label, button short name)}"""
    return {vk: (getattr(controller, btn), key,
                 btn.replace("BTN_", "").replace("DPAD_", ""))
            for key, btn, _g, _pg, vk in LAYOUT}


def by_pygame(controller, pygame):
    """{pygame key: (command bits, label)}"""
    return {getattr(pygame, pg): (getattr(controller, btn),
                                  f"{btn.replace('BTN_','').replace('DPAD_','')}")
            for _k, btn, _g, pg, vk in LAYOUT}
