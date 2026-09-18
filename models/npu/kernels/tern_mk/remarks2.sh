#!/usr/bin/env bash
# Second sweep: the two wide MAC shapes, against the same register blocking as
# remarks.sh. See that script's header for how to run it.
set -u
source /home/cyrus/ironenv142/bin/activate
KR=/mnt/c/code/openflowlm-next/open_kernels/kernel_remarks.py
cd "$(dirname "$0")"

one() {
  local src="$1"; shift
  echo "=== $src $* ==="
  python "$KR" "$src" -- "$@" 2>&1 | grep -E "II bounds|-> MII|for\.body|entry |pipelined|error:" | head -8
}

for blk in 1:1 1:2 2:1 2:2 2:4 4:2; do
  mb=${blk%%:*}; nb=${blk##*:}
  one tmk_mac_u4k.cc "-DTMK_MB=$mb" "-DTMK_NB=$nb"
  one tmk_mac_i8w.cc "-DTMK_MB=$mb" "-DTMK_NB=$nb"
done
