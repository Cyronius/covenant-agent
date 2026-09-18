# models/npu — driving the NPU from this repo

What is here is the plumbing that lets this repo talk to the AMD XDNA2 NPU
without any other checkout being present. It is not a model and not a kernel.
It exists because step 5 of `.claude/plans/npu-native-planner.md` needs a way to
run a kernel and time it, and step 6 needs a thin driver of our own.

Copied from `npu-prefill-engine` on 2026-09-18, where it was first packaged into
this form. See [PROVENANCE.md](PROVENANCE.md) for what came from where, and for
the warning that the wrapper now exists in three repos at once.

| path | what |
|---|---|
| `vendor/xrt/` | the NPU driver's public headers, pinned at version 2.20 |
| `vendor/xrt-shim/` | a small plain-C surface over those headers: device, context, kernel, buffers, run, runlist |
| `tools/gen-xrt-implib.ps1` | rebuilds the Windows link library, which the driver does not ship, from the installed driver itself |
| `tools/bench-dispatch.cpp` | measures what one NPU submission costs |
| `CMakeLists.txt` | builds the above; stops cleanly if the driver is absent |

Nothing binary is tracked. `build/` is ignored.

## Build

Needs the AMD NPU driver installed, plus MSVC Build Tools 2022 and CMake.

```
powershell -File tools\gen-xrt-implib.ps1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The first step reads the installed driver's export table and writes a link
library into `build/`, so the build binds to whatever driver is on the machine
rather than to a pinned copy. Without the driver, the configure step says so and
stops rather than failing confusingly.

## Run

`bench-dispatch` takes a built kernel directory, an iteration count, and the
sizes in bytes of the buffers that kernel binds. It reports the same work three
ways, which separates the fixed cost of submitting from the cost of building a
submission.

```
build\bench-dispatch.exe <kernel-build-dir> 50 10485760 1048576 8388608
```

Verified on this box on 2026-09-18 against driver 32.00.20102.3931, driving
openflowlm-next's prefill kernel built at 8192 by 2048 with 256 tokens:

| submission shape | per dispatch |
|---|---|
| a fresh run object each time | 6.33 ms |
| one run object restarted | 5.93 ms |
| fifty runs in one submission | 5.83 ms |

**Read those numbers carefully.** That kernel moves 19 MB per pass and is doing
real work, so almost all of the time above is the kernel, not the submission.
The fixed cost of submitting is what you get from a kernel that does nearly
nothing, and measured that way it is about a tenth of a millisecond, falling to
a few hundredths when runs are batched. The planner design's cost model uses the
small number, and the run above is a check that the whole path works end to
end, not a measurement of overhead.

## Where the measurements live

Prefill and hybrid-mode numbers for this box stay in `npu-prefill-engine`, which
owns that investigation. This repo owns the new model and its own benchmarks.
When this directory produces a number about the model, it belongs in `results/`
here, under the same rule as everything else: the harness is the only source of
accuracy numbers.
