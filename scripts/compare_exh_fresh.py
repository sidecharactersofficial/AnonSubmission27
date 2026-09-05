#!/usr/bin/env python3
"""Compare freshly reproduced EXH K=1 eval files (*.fresh.json) against the
archived ones, with pre-registered tolerances.

Reproduction of `eval_deepfool_k1.py` uses an UNSEEDED random-start PGD init,
so per-sample outcomes jitter across runs. Deterministic quantities are compared
exactly; stochastic ones under generous, pre-registered tolerances:
  - n_total, clean_correct, attacked: exact (batch order is fixed);
  - k0_survivors / k1_survivors:        within +-0.75% relative;
  - k1_pct_of_attacked:                 within +-0.5 pp absolute.
Archived files embed checkpoint_sha256; it is compared exactly (this pins each
sweep to the exact weights in models/).

Usage:
  python3 scripts/compare_exh_fresh.py --attack pgd40 [--dataset cicids2017]
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["pgd40", "deepfool"], required=True)
    p.add_argument("--dataset", default=None)
    args = p.parse_args()

    fresh = glob.glob(
        f"results/foolbox/exh_k1_{args.attack}_*_seed*.fresh.json")
    if args.dataset:
        safe = args.dataset.replace("-", "_")
        fresh = [f for f in fresh if f"{safe}_" in f]

    if not fresh:
        print(f"no {args.attack} .fresh files matched; run the corresponding "
              f"reproduce-multiseed-* target first")
        return 1

    ok = True
    n = 0
    for fresh_path in fresh:
        base = fresh_path.replace(".fresh.json", ".json")
        if not os.path.exists(base):
            print(f"WARN no archive for {os.path.basename(fresh_path)}")
            continue
        f = json.load(open(fresh_path))
        a = json.load(open(base))
        n += 1
        if f["checkpoint_sha256"] != a["checkpoint_sha256"]:
            print(f"{os.path.basename(base)}: SHA MISMATCH")
            ok = False
            continue
        if f["n_total"] != a["n_total"] or f["clean_correct"] != a["clean_correct"]:
            print(f"{os.path.basename(base)}: n_total/clean MISMATCH "
                  f"({f['n_total']},{f['clean_correct']}) vs "
                  f"({a['n_total']},{a['clean_correct']})")
            ok = False
            continue
        for key in ("k0_survivors", "k1_survivors"):
            d = abs(f[key] - a[key])
            tol = max(1.0, 0.0075 * max(a[key], 1))
            if d > tol:
                print(f"{os.path.basename(base)}: {key} {f[key]} vs {a[key]} "
                      f"(|d|={d:.0f} > tol {tol:.0f}) MISMATCH")
                ok = False
        if abs(f["k1_pct_of_attacked"] - a["k1_pct_of_attacked"]) > 0.5:
            print(f"{os.path.basename(base)}: k1_pct {f['k1_pct_of_attacked']} "
                  f"vs {a['k1_pct_of_attacked']} MISMATCH")
            ok = False

    print(f"compare_exh_fresh: compared {n} files "
          f"({'ALL PASS' if ok else 'MISMATCHES FOUND'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())