#!/usr/bin/env python3
"""Isolate where the Switch controller link is broken.

    python controller_diag.py                 # find ports, scan bauds, try sync
    python controller_diag.py --loopback      # test the adapter by itself

LOOPBACK: unplug the two signal wires from the Arduino and connect the
adapter's TX directly to its own RX (leave GND alone). Bytes echoing back mean
the adapter, cable, driver and COM port are all fine, and the fault is on the
Arduino side. Nothing echoing means the adapter itself is the problem.
"""
import argparse
import sys
import time

import serial
import serial.tools.list_ports as list_ports

BAUDS = (9600, 19200, 38400, 57600, 115200)


def crc8_ccitt(old_crc, new_data):
    data = old_crc ^ new_data
    for _ in range(8):
        data = ((data << 1) ^ 0x07) & 0xFF if data & 0x80 else (data << 1) & 0xFF
    return data


def find_ports():
    ports = list(list_ports.comports())
    print("serial ports:")
    if not ports:
        print("   NONE -- the USB-to-TTL adapter is not attached to this PC.")
        print("   (If you see a HID gamepad instead, the Arduino is in the PC")
        print("    port by mistake -- the Arduino belongs in the Switch dock.)")
    for p in ports:
        print(f"   {p.device:<6} {p.description}")
    return [p.device for p in ports]


def loopback(port):
    print(f"\nloopback on {port} -- adapter TX jumpered to adapter RX:")
    probe = bytes([0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC])
    with serial.Serial(port=port, baudrate=19200, timeout=1) as s:
        time.sleep(0.2)
        s.reset_input_buffer()
        s.write(probe)
        back = s.read(len(probe))
    if back == probe:
        print("   PASS -- adapter, cable, driver and port are all good.")
        print("   The fault is on the Arduino side: wiring, sketch, or power.")
    elif back:
        print(f"   PARTIAL -- sent {list(probe)} got {list(back)}")
        print("   Bytes are being mangled: suspect a loose jumper or wrong baud.")
    else:
        print("   FAIL -- nothing came back.")
        print("   Either TX and RX are not actually jumpered, or the adapter is dead.")
    return back == probe


def scan(port):
    print(f"\nlistening on {port} for any MCU response:")
    pkt = [0x00, 0x00, 0x08, 0x80, 0x80, 0x80, 0x80, 0x00]
    crc = 0
    for d in pkt:
        crc = crc8_ccitt(crc, d)
    any_bytes = False
    for baud in BAUDS:
        try:
            s = serial.Serial(port=port, baudrate=baud, timeout=0.6)
        except Exception as e:
            print(f"   {baud:>6} : cannot open ({e})")
            continue
        time.sleep(0.25)
        s.reset_input_buffer()
        s.write(bytes([0xFF] * 9))          # force_sync preamble
        d1 = s.read(12)
        s.write(bytes(pkt + [crc]))         # a neutral packet
        d2 = s.read(4)
        s.close()
        got = list(d1) + list(d2)
        any_bytes |= bool(got)
        note = ""
        if 0xFF in got:
            note = "  <- RESP_SYNC_START, the sketch is alive"
        elif 0x90 in got:
            note = "  <- RESP_USB_ACK, fully working"
        print(f"   {baud:>6} : {len(got):2d} bytes  {[hex(b) for b in got]}{note}")
        time.sleep(0.15)
    return any_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--loopback", action="store_true")
    a = ap.parse_args()

    ports = find_ports()
    port = a.port or (ports[0] if ports else None)
    if port is None:
        return 1

    if a.loopback:
        return 0 if loopback(port) else 1

    if scan(port):
        print("\nThe MCU is transmitting. If sync still fails it is a baud or")
        print("protocol mismatch, not a wiring fault.")
    else:
        print("\nSilence at every baud -- the MCU is not transmitting at all.")
        print("In likelihood order:")
        print("  1. TX/RX not crossed. Adapter TX -> Uno pin 0 (RX),")
        print("     adapter RX -> Uno pin 1 (TX). Straight-through is the")
        print("     single most common mistake and looks exactly like this.")
        print("  2. No shared GND between adapter and Uno.")
        print("  3. The Uno has no power: it draws it from the dock, so the")
        print("     Switch must be docked and awake. Check the Uno's ON LED.")
        print("  4. The 328P sketch was never uploaded, or was erased when the")
        print("     16U2 was reflashed. The HID side working says nothing about")
        print("     the sketch -- they are two separate chips.")
        print("\nRun with --loopback to prove which side is at fault.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
