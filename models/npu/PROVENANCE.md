# Provenance — models/npu/

Third-party and shared source carried in-tree so this directory builds with no
other checkout present. Nothing here is generated at build time. Nothing here is
edited to fix bugs: bump the upstream instead and re-record it below.

Copied 2026-09-18 from `npu-prefill-engine` (git remote
`github.com/Cyronius/ggml-xdna`, commit `766e6e2`), which is where it was first
packaged into this standalone form and proven to build and run.

## vendor/xrt/include/xrt/

The NPU driver's public C++ headers. Apache-2.0, see `xrt/LICENSE` and
`xrt/NOTICE`. Version 2.20 as declared by `xrt/detail/version-slim.h`.

Pruned to one coherent upstream directory rather than a hand-picked file list.
The wrapper reaches 28 of these; the whole directory is kept so the boundary is
something upstream would recognise.

**One file is not upstream:** `xrt/detail/version-slim.h`. Upstream generates it
at build time and does not track it, so this is a minimal stand-in declaring the
version macros the ABI tag depends on. Keep those matching the driver actually
installed, and re-create this file after any driver bump.

The matching link library is **not** vendored. The driver ships none on Windows,
so `tools/gen-xrt-implib.ps1` reconstructs one from the installed driver's own
export table into `build/`. That binds to whatever is installed rather than to a
pinned copy, which is what we want.

## vendor/xrt-shim/

`xrt_shim.h` and `xrt_shim.cpp`: a small plain-C surface over the driver's C++
API covering device, context, kernel, buffers, run and runlist, so callers that
are not C++17-with-the-driver's-headers can drive the NPU.

Author: Cyrus Attoun.

**This file now exists in three places** and is byte-identical in all of them:

| repo | path |
|---|---|
| phlegm | `npu-engine/xrt-shim/` |
| npu-prefill-engine | `vendor/xrt-shim/` |
| here | `models/npu/vendor/xrt-shim/` |

A fix belongs in all three or the copies diverge silently. If that becomes
annoying, the right answer is one of them becoming the source and the others
taking it by submodule, not three hand-maintained copies.

## tools/

- `gen-xrt-implib.ps1` — rebuilds the Windows link library from the installed
  driver. Same origin and commit as the above.
- `bench-dispatch.cpp` — measures what one NPU submission costs, three ways:
  a fresh run each time, one run object restarted, and many runs in a single
  runlist. This is the instrument behind the dispatch-cost figures the planner
  design's cost model uses, and the nearest starting point for the hardware
  probe that design calls bet 4, whether on-chip state survives a weight reload.

## Verification

Built and run on this box on 2026-09-18 against the installed driver
(32.00.20102.3931), driving a real kernel from openflowlm-next. See
`README.md` for the command and the result.

## What is deliberately not here

The llama.cpp backend that `npu-prefill-engine` exists for. This directory needs
the NPU driver plumbing, not the scheduler seam. Prefill and hybrid-mode
measurements stay in that repo; this one owns the new model and its own
benchmarks.
