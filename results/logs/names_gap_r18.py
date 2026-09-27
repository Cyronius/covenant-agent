"""R18: goal success on the S6 plain exam split by whether the task's tools
carry real names or meaningless ones (the 15% of exam rows tagged
`opaque-names`), for A0 and SPt at each share of S6 training rows.

  cd models/tiny && python ../../results/logs/names_gap_r18.py
"""
import json
import pickle

rows = {r["id"]: r for r in pickle.load(open("data_cache_s6off/rows.pkl", "rb"))["holdout"]}
RUNS = {"A0 0% S6, seed 0": "runs/pod_s6split/out/s6off_A0_plain",
        "A0 0% S6, seed 1": "runs/pod_s6off2/out/s6off_A0s1_plain",
        "A0 25% S6": "runs/pod_s6frac/out/s6d25_A0_plain",
        "A0 100% S6 (R13)": "runs/pod_planner/out/s6_A0_holdout_plain",
        "SPt 0% S6": "runs/pod_s6off2/out/s6off_SPt_plain",
        "SPt 25% S6": "runs/pod_s6frac/out/s6d25_SPt_plain",
        "SPt 50% S6": "runs/pod_s6frac/out/s6d50_SPt_plain",
        "SPt 100% S6 (R13)": "runs/pod_planner/out/s6_SPt_holdout_plain"}
for name, run in RUNS.items():
    sc = json.load(open(run + ".score.json"))["rows"]
    tag = lambda s: "opaque-names" in rows[s["task_id"].split("#")[0]]["tags"]  # noqa: E731
    op, nm = [s for s in sc if tag(s)], [s for s in sc if not tag(s)]
    g = lambda xs: sum(x["goal"] for x in xs) / len(xs)  # noqa: E731
    print(f"{name:18s} real names {g(nm):.1%} (n={len(nm)})   meaningless names {g(op):.1%} "
          f"(n={len(op)})   gap {100 * (g(nm) - g(op)):+.1f}")
