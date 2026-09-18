#!/usr/bin/env bash
# Third sweep: the MAC loops under iterator addressing, against the register
# blockings that did not spill in remarks.sh / remarks2.sh.
set -u
source /home/cyrus/ironenv142/bin/activate
KR=/mnt/c/code/openflowlm-next/open_kernels/kernel_remarks.py
cd "$(dirname "$0")"

one() {
  local src="$1"; shift
  echo "=== $src $* ==="
  python "$KR" "$src" -- "$@" 2>&1 | grep -E "II bounds|-> MII|for\.body|pipelined|error:" | head -6
}

for blk in 1:1 1:2 2:1 2:2; do
  mb=${blk%%:*}; nb=${blk##*:}
  one tmk_mac_i8_it.cc "-DTMK_MB=$mb" "-DTMK_NB=$nb"
  one tmk_mac_u4k_it.cc "-DTMK_MB=$mb" "-DTMK_NB=$nb"
done
