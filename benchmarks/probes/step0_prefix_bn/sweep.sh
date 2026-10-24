#!/bin/bash
PY=/y/SwitchPlayer/.venv/Scripts/python.exe
Q="/c/Users/kergi/AppData/Local/Temp/claude/Y--SwitchPlayer/ff65188e-6e1e-495e-954d-5a7a7429d10d/scratchpad/qspread"
cd "$Q"
snap() { echo "$1/$(ls -d $1/snapshots-* | xargs -n1 basename)/weights-iter$2.pth"; }
TK=Y:/SwitchPlayer-rgrow-run/benchmarks/results/mspacman-tokens-rgrow-s0-2026-10-19
RG=Y:/SwitchPlayer-rgrow-run/benchmarks/results/mspacman-rgrow-s0-2026-10-19
TKX=Y:/SwitchPlayer-rgrowx-run/benchmarks/results/mspacman-tokens-rgrow-x270-s0-2026-10-20
DS=Y:/SwitchPlayer-distill-run/benchmarks/results/mspacman-student-distill-s0-2026-10-21
FS=Y:/SwitchPlayer-distill-run/benchmarks/results/mspacman-student-fresh-s0-2026-10-21
TW=Y:/SwitchPlayer/benchmarks/results/mspacman-s0-2026-10-16
run() { echo "### $1 $(date +%T)"; $PY xrun_prefix.py "${@:2}" 2>&1 | grep -E "^\{|Error|Traceback|line [0-9]" | cut -c1-200; }
run G0 Y:/SwitchPlayer-distill-run G0 g0snaps/weights-iter005.pth g0snaps/weights-iter010.pth g0snaps/weights-iter015.pth
run TK Y:/SwitchPlayer-rgrow-run TK $(snap $TK 030) $(snap $TK 055) $(snap $TK 075) $(snap $TK 090)
run RG Y:/SwitchPlayer-rgrow-run RG $(snap $RG 030) $(snap $RG 055) $(snap $RG 075) $(snap $RG 090)
run DS Y:/SwitchPlayer-distill-run DS $(snap $DS 000) $(snap $DS 030) $(snap $DS 060) $(snap $DS 090)
run TWIN Y:/SwitchPlayer-mspacman TWIN $(snap $TW 030) $(snap $TW 055) $(snap $TW 075) $(snap $TW 090)
run FS Y:/SwitchPlayer-distill-run FS $(snap $FS 045) $(snap $FS 075)
run TKX Y:/SwitchPlayer-rgrowx-run TKX $(snap $TKX 055) $(snap $TKX 100) $(snap $TKX 140) $(snap $TKX 200)
echo "SWEEP DONE $(date +%T)"
