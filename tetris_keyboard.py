#!/usr/bin/env python3
"""Drive the Switch from the PC keyboard through the Arduino controller link.

    python tetris_keyboard.py                 # uses SWITCH_PORT or COM5
    python tetris_keyboard.py --port COM7
    python tetris_keyboard.py --dry-run       # no hardware; prints what it would send
    python tetris_keyboard.py --list-ports

Key state is polled, not edge-triggered, so holding a key holds the button. Only
CHANGES are sent: the Switch does its own auto-repeat (DAS), so re-sending a
held direction every frame would add latency for nothing. A keepalive goes out
every KEEPALIVE_S in case a packet is dropped.
"""
import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Layout: one row for the right hand, bumpers on the outside, face buttons in
# the middle -- the same shape as a controller.
#
#     B   N   M   ,   .   /
#     L   A   B   X   R   Y
# ---------------------------------------------------------------------------
KEEPALIVE_S = 0.5
POLL_HZ = 120


def build_map(c, pygame):
    """key -> (command bits, label). Built after controller import."""
    return {
        pygame.K_w:         (c.DPAD_U, "Up / hard drop"),
        pygame.K_a:         (c.DPAD_L, "Left"),
        pygame.K_s:         (c.DPAD_D, "Down / soft drop"),
        pygame.K_d:         (c.DPAD_R, "Right"),
        pygame.K_b:         (c.BTN_L,  "L  hold"),
        pygame.K_n:         (c.BTN_A,  "A  rotate"),
        pygame.K_m:         (c.BTN_B,  "B  rotate"),
        pygame.K_COMMA:     (c.BTN_X,  "X  rotate"),
        pygame.K_PERIOD:    (c.BTN_R,  "R  hold"),
        pygame.K_SLASH:     (c.BTN_Y,  "Y  rotate"),
        pygame.K_RETURN:    (c.BTN_PLUS,    "+  start"),
        pygame.K_BACKSPACE: (c.BTN_MINUS,   "-  select"),
        pygame.K_h:         (c.BTN_HOME,    "Home"),
        pygame.K_c:         (c.BTN_CAPTURE, "Capture"),
    }


# Display rows reference the pygame key directly. Matching on the label text
# instead would light the L-bumper box whenever "Left" was held.
ROWS = [
    ("move",   [("W", "Up", "K_w"), ("A", "Left", "K_a"),
                ("S", "Down", "K_s"), ("D", "Right", "K_d")]),
    ("rotate", [("N", "A", "K_n"), ("M", "B", "K_m"),
                (",", "X", "K_COMMA"), ("/", "Y", "K_SLASH")]),
    ("hold",   [("B", "L", "K_b"), (".", "R", "K_PERIOD")]),
    ("menu",   [("Enter", "+", "K_RETURN"), ("Bksp", "-", "K_BACKSPACE"),
                ("H", "Home", "K_h"), ("C", "Capture", "K_c")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial port, e.g. COM7")
    ap.add_argument("--dry-run", action="store_true",
                    help="no hardware; print the commands instead")
    ap.add_argument("--list-ports", action="store_true")
    a = ap.parse_args()

    if a.list_ports:
        import serial.tools.list_ports as lp
        ports = list(lp.comports())
        if not ports:
            print("no serial ports found -- is the USB-to-TTL adapter plugged in?")
        for p in ports:
            print(f"  {p.device}  {p.description}")
        return 0

    # controller.py opens the port at IMPORT time, so the choice has to be made
    # before the import rather than after it.
    if a.dry_run:
        os.environ["SWITCH_PORT"] = "none"
    elif a.port:
        os.environ["SWITCH_PORT"] = a.port
    try:
        import controller as c
    except Exception as e:
        port = os.environ.get("SWITCH_PORT", "COM5")
        print(f"could not open {port}: {e}\n"
              f"  --list-ports to see what is attached\n"
              f"  --port COMn  to pick another\n"
              f"  --dry-run    to test the key mapping with no hardware",
              file=sys.stderr)
        return 1

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((520, 360))
    pygame.display.set_caption("Switch keyboard control" + ("  [DRY RUN]" if a.dry_run else ""))
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 15)
    keymap = build_map(c, pygame)
    import switch_keymap
    rstick_keys = {nm: getattr(pygame, pg) for nm, _c, pg, _vk in switch_keymap.RSTICK}

    if c.sync():
        status, ok = ("DRY RUN -- not sending" if a.dry_run else "synced"), True
    else:
        status, ok = "SYNC FAILED -- check wiring and baud", False
    print(status, flush=True)

    clock = pygame.time.Clock()
    last_cmd, last_sent, sent_n = None, 0.0, 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        pressed = pygame.key.get_pressed()
        cmd, held = 0, []
        for key, (bits, label) in keymap.items():
            if pressed[key]:
                cmd += bits
                held.append(label)
        # Arrows -> right stick, resolved as one direction (see switch_keymap).
        d = {nm: bool(pressed[k]) for nm, k in rstick_keys.items()}
        stick = switch_keymap.rstick_bits(c, d["Up"], d["Down"], d["Left"], d["Right"])
        if stick:
            cmd += stick
            held.append(switch_keymap.rstick_label(d["Up"], d["Down"],
                                                   d["Left"], d["Right"]))

        now = time.perf_counter()
        if cmd != last_cmd or now - last_sent > KEEPALIVE_S:
            c.send_cmd(cmd)
            if a.dry_run and cmd != last_cmd:
                print(f"  {c.cmd_to_packet(cmd)}  {' + '.join(held) or 'neutral'}", flush=True)
            last_cmd, last_sent = cmd, now
            sent_n += 1

        screen.fill((18, 18, 22))
        screen.blit(font.render(status, True, (120, 230, 120) if ok else (255, 90, 90)), (16, 12))
        y = 48
        for name, keys in ROWS:
            screen.blit(small.render(name, True, (140, 140, 150)), (16, y))
            x = 110
            for kname, btn, attr in keys:
                lit = pressed[getattr(pygame, attr)]
                box = pygame.Rect(x, y - 4, 88, 30)
                pygame.draw.rect(screen, (70, 150, 90) if lit else (40, 40, 48), box, border_radius=5)
                screen.blit(small.render(f"{kname}={btn}", True, (255, 255, 255)), (x + 8, y))
                x += 96
            y += 46
        screen.blit(small.render(f"packets sent {sent_n}", True, (120, 120, 130)), (16, 300))
        screen.blit(small.render("Esc or Q to quit", True, (120, 120, 130)), (16, 322))
        pygame.display.flip()
        clock.tick(POLL_HZ)

    c.send_cmd(c.NO_INPUT)          # never leave a button stuck down
    pygame.quit()
    print(f"released all inputs; {sent_n} packets sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
