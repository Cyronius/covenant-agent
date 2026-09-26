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
import io
import tarfile
from pathlib import Path

from corpus import COVENANT

HERE = Path(__file__).parent

# Everything in the bundle runs on Linux, and a shell script with CRLF line
# endings dies there on `set: -: invalid option` -- which reads like a broken
# flag rather than a broken file. Editors on this machine produce CRLF without
# being asked, and .gitattributes only fixes what git touches, not what tarfile
# reads off the disk. So the bundle normalises text on the way in.
TEXT_SUFFIXES = {".sh", ".py", ".md", ".json"}
CRLF, LF = bytes((13, 10)), bytes((10,))


def add_text(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    """Add a text file with LF endings, whatever it looks like on disk."""
    data = src.read_bytes().replace(CRLF, LF)
    info = tar.gettarinfo(str(src), arcname=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def add(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    if src.suffix in TEXT_SUFFIXES:
        add_text(tar, src, arcname)
    else:
        tar.add(src, arcname=arcname)


def add_slim(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    """A cache split with only the stage columns (stage_pretrain.COLS)."""
    import torch
    from stage_pretrain import COLS
    d = torch.load(src, mmap=True)
    buf = io.BytesIO()
    torch.save({k: v.clone() for k, v in d.items() if k in COLS}, buf)
    info = tarfile.TarInfo(arcname)
    info.size = buf.tell()
    buf.seek(0)
    tar.addfile(info, buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", default=["data_cache_struct"],
                    help="one or more cache directories")
    ap.add_argument("--stage-only", action="store_true",
                    help="ship only the columns stage_pretrain / vocab_report "
                         "read (a third of a split cache's size)")
    ap.add_argument("--general", default=None, metavar="DIR",
                    help="ship data/general's texts, names and siblings as "
                         "general/ beside tiny/ (run_vocab.sh)")
    ap.add_argument("--out", default="pod_bundle.tar.gz")
    ap.add_argument("--no-cache", action="store_true",
                    help="code only, for when the cache is already uploaded")
    ap.add_argument("--with", dest="extra", action="append", default=[],
                    metavar="PATH",
                    help="ship another file under tiny/, at the same relative "
                         "path (e.g. pretrained stage weights, "
                         "runs/stages_d256/stages.pt)")
    args = ap.parse_args()

    for c in args.cache:
        if not args.no_cache and not (HERE / c / "config.json").exists():
            raise SystemExit(f"no cache at {HERE / c}; run prep.py first")

    out = Path(args.out)
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for py in sorted(HERE.glob("*.py")):
            add(tar, py, f"tiny/{py.name}")
            n += 1
        for extra in ("README.md", "pod.md", "run_phase1.sh", "run_step1.sh",
                      "run_step2.sh", "run_step3.sh", "run_step4.sh", "run_vocab.sh", "run_size.sh", "run_planner.sh", "run_diag.sh", "run_s6split.sh", "run_s6off2.sh",
                      "selftest.sh"):
            p = HERE / extra
            if p.exists():
                add(tar, p, f"tiny/{extra}")
                n += 1

        for extra in args.extra:
            p = HERE / extra
            if not p.exists():
                raise SystemExit(f"--with {extra}: no such file under {HERE}")
            add(tar, p, f"tiny/{Path(extra).as_posix()}")
            n += 1

        # The compiler, so repair runs on the pod.
        for py in sorted((COVENANT / "core").glob("*.py")):
            add(tar, py, f"core/{py.name}")
            n += 1
        add(tar, COVENANT / "harness" / "__init__.py", "harness/__init__.py")
        add(tar, COVENANT / "harness" / "context.py", "harness/context.py")
        n += 2

        if not args.no_cache:
            for c in args.cache:
                for f in sorted((HERE / c).iterdir()):
                    if f.name == "general.pt":
                        continue            # rebuilt on the pod from general/
                    if f.suffix in (".pt", ".json") or f.name == "rows.pkl":
                        if args.stage_only and f.stem in ("train", "val", "test", "holdout"):
                            add_slim(tar, f, f"tiny/{c}/{f.name}")
                        else:
                            add(tar, f, f"tiny/{c}/{f.name}")
                        n += 1
        if args.general:
            for name in ("texts.jsonl", "names.jsonl", "siblings.jsonl"):
                add(tar, Path(args.general) / name, f"general/{name}")
                n += 1

    mb = out.stat().st_size / 1e6
    print(f"{n} files -> {out} ({mb:.0f} MB)")
    print("\non the pod:")
    print(f"  tar xzf {out.name} && cd tiny")
    print("  pip install torch tokenizers")
    print("  bash selftest.sh " + args.cache[0] + "   # 2 minutes, catches what an hour in would not")
    print(f"  CACHE={args.cache[0]} bash run_step2.sh   # or run_step1.sh, run_step3.sh, run_vocab.sh")
    print("\nbring back: out/ (generations and curves/)")


if __name__ == "__main__":
    main()
