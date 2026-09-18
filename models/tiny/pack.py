"""Bundle everything the pod needs into one archive.

Cloud time is the expensive thing, so nothing should be figured out while the
meter runs. This produces a single tarball that unpacks and runs.

What goes in, and why each piece:

  tiny/*.py               the experiment
  data_cache_struct/        pre-tokenized tensors (per-line token tensors, graph
                            edges, canvas targets, keywords.json), so the pod
                            never parses the 500 MB corpus or trains a tokenizer
  core/, harness/context.py covenant-agent's parser, typechecker and compiler.
                            Pure stdlib Python, 214 KB, and shipping it means
                            the compiler repair loop runs on the pod instead of
                            being the one arm that has to wait for the laptop.

What stays behind: the Node sandbox. Executing a program against world state is
the only step that needs it, and that is scoring, which happens after the pod is
shut down.

    python pack.py --out pod_bundle.tar.gz
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from corpus import COVENANT

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache_struct")
    ap.add_argument("--out", default="pod_bundle.tar.gz")
    ap.add_argument("--no-cache", action="store_true",
                    help="code only, for when the cache is already uploaded")
    args = ap.parse_args()

    cache = HERE / args.cache
    if not args.no_cache and not (cache / "config.json").exists():
        raise SystemExit(f"no cache at {cache}; run prep.py first")

    out = Path(args.out)
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for py in sorted(HERE.glob("*.py")):
            tar.add(py, arcname=f"tiny/{py.name}")
            n += 1
        for extra in ("README.md", "pod.md", "run_phase1.sh", "run_step1.sh", "selftest.sh"):
            p = HERE / extra
            if p.exists():
                tar.add(p, arcname=f"tiny/{extra}")
                n += 1

        # The compiler, so repair runs on the pod.
        for py in sorted((COVENANT / "core").glob("*.py")):
            tar.add(py, arcname=f"core/{py.name}")
            n += 1
        tar.add(COVENANT / "harness" / "__init__.py", arcname="harness/__init__.py")
        tar.add(COVENANT / "harness" / "context.py", arcname="harness/context.py")
        n += 2

        if not args.no_cache:
            for f in sorted(cache.iterdir()):
                if f.suffix in (".pt", ".json") or f.name == "rows.pkl":
                    tar.add(f, arcname=f"tiny/{args.cache}/{f.name}")
                    n += 1

    mb = out.stat().st_size / 1e6
    print(f"{n} files -> {out} ({mb:.0f} MB)")
    print("\non the pod:")
    print(f"  tar xzf {out.name} && cd tiny")
    print("  pip install torch tokenizers")
    print(f"  CACHE={args.cache} bash run_step1.sh")
    print("\nbring back: out/ (generations and curves/)")


if __name__ == "__main__":
    main()
