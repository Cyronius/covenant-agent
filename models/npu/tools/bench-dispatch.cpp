// How much does one NPU dispatch cost?
//
// Everything in the granularity argument hangs off this number. The scheduler
// hands us one matmul per split, so the per-split submit+wait latency - not the
// kernel's own runtime - decides whether op-granularity offload can work at all.
//
// Uses the rot13 smoke design because it does almost no work: what is left
// is the fixed cost. Measures three submission shapes:
//
//   sequential  start/wait per run          - what graph_compute does today
//   runlist     N runs, one execute/wait    - what one split could do
//   reuse       one run object, restarted   - how much is run construction
//
// Traces: XDNA-DISPATCH-COST

#include "xrt_shim.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static std::vector<uint8_t> read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t) n) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
    return buf;
}

static void die(const char * what) {
    fprintf(stderr, "%s: %s\n", what, xrtsh_last_error());
    exit(1);
}

int main(int argc, char ** argv) {
    const std::string design = argc > 1
        ? argv[1]
        : "../openflowlm-next/open_kernels/designs/rot13/build";

    const int iters = argc > 2 ? atoi(argv[2]) : 200;

    // Remaining args are buffer sizes, bound at args 3.. in order. Defaults to
    // rot13's two 1 KiB buffers. Contents do not matter here - we are timing
    // submission, not checking results.
    std::vector<size_t> bufsizes;
    for (int i = 3; i < argc; i++) {
        bufsizes.push_back((size_t) atoll(argv[i]));
    }
    if (bufsizes.empty()) {
        bufsizes = { 1024, 1024 };
    }

    xrtsh_dev dev = xrtsh_device_open(0);
    if (!dev) die("device_open");

    char name[256];
    if (xrtsh_device_name(dev, name, sizeof(name)) > 0) {
        printf("device: %s\n", name);
    }

    xrtsh_ctx ctx = xrtsh_hwctx_create(dev, (design + "/final.xclbin").c_str());
    if (!ctx) die("hwctx_create");

    xrtsh_kernel k = xrtsh_kernel_create_xclbin(ctx, "MLIR_AIE");
    if (!k) die("kernel_create_xclbin");

    const std::vector<uint8_t> insts = read_file(design + "/insts.bin");
    const int nwords = (int) (insts.size() / 4);

    xrtsh_bo ibo = xrtsh_bo_create_instr(dev, k, insts.size());
    if (!ibo) die("bo_create_instr");
    xrtsh_bo_write(ibo, insts.data(), insts.size(), 0);
    xrtsh_bo_sync(ibo, 1);

    std::vector<xrtsh_bo> bufs;
    size_t total_bytes = 0;
    for (size_t sz : bufsizes) {
        xrtsh_bo b = xrtsh_bo_create(dev, sz);
        if (!b) die("bo_create");
        memset(xrtsh_bo_map(b), 0, sz);
        xrtsh_bo_sync(b, 1);
        bufs.push_back(b);
        total_bytes += sz;
    }

    printf("design: %s\n", design.c_str());
    printf("instrs: %d words (%zu bytes)\n", nwords, insts.size());
    printf("bufs:   %zu totalling %.2f MiB\n", bufs.size(), total_bytes/1048576.0);
    printf("iters:  %d\n\n", iters);

    auto make_run = [&]() {
        xrtsh_run r = xrtsh_run_create(k);
        if (!r) die("run_create");
        xrtsh_run_set_arg_int(r, 0, 3);
        xrtsh_run_set_arg_bo (r, 1, ibo);
        xrtsh_run_set_arg_int(r, 2, nwords);
        for (size_t i = 0; i < bufs.size(); i++) {
            xrtsh_run_set_arg_bo(r, (int) (3 + i), bufs[i]);
        }
        return r;
    };

    // warm up - the first dispatch after a context switch pays extra
    {
        xrtsh_run r = make_run();
        for (int i = 0; i < 5; i++) {
            xrtsh_run_start(r);
            if (xrtsh_run_wait(r) != 4) die("warmup run");
        }
        xrtsh_run_free(r);
    }

    // 1. build a run, start, wait - one at a time
    {
        const clk::time_point t0 = clk::now();
        for (int i = 0; i < iters; i++) {
            xrtsh_run r = make_run();
            xrtsh_run_start(r);
            if (xrtsh_run_wait(r) != 4) die("sequential run");
            xrtsh_run_free(r);
        }
        const double ms = ms_since(t0);
        printf("sequential (new run each)  %8.3f ms total  %7.3f ms/dispatch\n", ms, ms/iters);
    }

    // 2. one run object, restarted - isolates run construction from submit
    {
        xrtsh_run r = make_run();
        const clk::time_point t0 = clk::now();
        for (int i = 0; i < iters; i++) {
            xrtsh_run_start(r);
            if (xrtsh_run_wait(r) != 4) die("reuse run");
        }
        const double ms = ms_since(t0);
        printf("reuse (one run restarted)  %8.3f ms total  %7.3f ms/dispatch\n", ms, ms/iters);
        xrtsh_run_free(r);
    }

    // 3. all runs in one runlist - what a coalesced split could submit
    {
        std::vector<xrtsh_run> runs;
        runs.reserve(iters);
        for (int i = 0; i < iters; i++) {
            runs.push_back(make_run());
        }

        xrtsh_runlist rl = xrtsh_runlist_create(ctx);
        if (!rl) die("runlist_create");
        for (xrtsh_run r : runs) {
            if (xrtsh_runlist_add(rl, r) < 0) die("runlist_add");
        }

        const clk::time_point t0 = clk::now();
        if (xrtsh_runlist_execute(rl) < 0) die("runlist_execute");
        if (xrtsh_runlist_wait(rl) < 0) die("runlist_wait");
        const double ms = ms_since(t0);
        printf("runlist (%d in one submit) %8.3f ms total  %7.3f ms/dispatch\n", iters, ms, ms/iters);

        xrtsh_runlist_free(rl);
        for (xrtsh_run r : runs) xrtsh_run_free(r);
    }

    for (xrtsh_bo b : bufs) xrtsh_bo_free(b);
    xrtsh_bo_free(ibo);
    xrtsh_kernel_free(k);
    xrtsh_hwctx_free(ctx);
    xrtsh_device_free(dev);
    return 0;
}
