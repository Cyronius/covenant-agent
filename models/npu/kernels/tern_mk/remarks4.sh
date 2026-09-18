#!/usr/bin/env bash
# Fourth sweep: the paired nibble expand against the three unpacks already
# measured. One number decides whether ternary can feed the native int4 MAC
# without paying for the privilege.
set -u
source /home/cyrus/ironenv142/bin/activate
KR=/mnt/c/code/openflowlm-next/open_kernels/kernel_remarks.py
cd "$(dirname "$0")"

for src in tmk_u_t2_u4_it.cc tmk_u_t2_u4p_it.cc tmk_u_t2_it.cc tmk_u_q4_it.cc; do
  echo "=== $src ==="
  python "$KR" "$src" 2>&1 | grep -E "II bounds|-> MII|for\.body|pipelined|error:" | head -6
done
