#!/usr/bin/env bash
# Run openflowlm-next's kernel_remarks.py over this directory's TUs and print the
# one line per variant that matters: cycles for the loop body, and the II floor.
#
# Read-only against that repo (its own rule, and this plan's): the script is
# imported, nothing is written there. Run under WSL with the ironenv venv.
#
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/code/covenant-agent/models/npu/kernels/tern_mk/remarks.sh
set -u
source /home/cyrus/ironenv142/bin/activate
KR=/mnt/c/code/openflowlm-next/open_kernels/kernel_remarks.py
cd "$(dirname "$0")"

one() {
  local src="$1"; shift
  echo "=== $src $* ==="
  python "$KR" "$src" -- "$@" 2>&1 | grep -E "II bounds|-> MII|for\.body|entry |pipelined|bankless|error:" | head -12
}

one tmk_u_t2.cc
one tmk_u_q4.cc
for blk in 1:1 1:2 1:4 1:8 2:1 2:2 2:4 4:1 4:2 8:1 4:4 8:2; do
  mb=${blk%%:*}; nb=${blk##*:}
  one tmk_mac_i8.cc "-DTMK_MB=$mb" "-DTMK_NB=$nb"
done
one tmk_mac_u4_41616.cc
