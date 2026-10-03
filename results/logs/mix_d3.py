"""The "add, don't displace" corpus (tiny-general-agent-menu.md D3;
.claude/plans/staged-decoder-experts.md step 0b).

R24's mixes held the corpus at 30,000 rows, so every new row pushed an old
one out, and its 3-12 point losses on the old exams mix lost rows with
genuine interference. This corpus keeps every clt_train row and adds every
new row R24's 30% mix used: 30,000 + 9,034.

  python results/logs/mix_d3.py        # writes data/d3_train.jsonl
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "clt_train.jsonl"
NEW = [ROOT / "data" / f for f in ("famb_episodes.jsonl", "wsb_episodes.jsonl",
                                    "rmb_episodes.jsonl", "service_retail.jsonl",
                                    "service_airline.jsonl")]


def lines(path: Path) -> list:
    return [ln if ln.endswith("\n") else ln + "\n"
            for ln in path.open(encoding="utf-8") if ln.strip()]


def main() -> None:
    rows = lines(BASE)
    n_base = len(rows)
    for p in NEW:
        rows += lines(p)
    random.Random(20261002).shuffle(rows)
    out = ROOT / "data" / "d3_train.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(rows)
    print(f"{out.name}: {len(rows)} rows ({n_base} clt_train + {len(rows) - n_base} new)")


if __name__ == "__main__":
    main()
