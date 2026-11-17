"""The honest DFL estimate, and which way of picking from a decoder search generalises.

For each DFL match in turn, the search's candidates are ranked on the other six (held-out evidence
from fresh/run_lomo.py, merged truth), a pick is made by each rule, and the pick is scored on the
match left out. Decoder ranking and scoring use external DFL matches.

    uv run python fresh/run_nested.py --set nogap [--seeds 0,10,20]

`voteK` decodes with each of the K best candidates and lets them vote frame by frame.
"""

import argparse
import json
from multiprocessing import Pool

import bench
import config
import data_io
import numpy as np
import public
from run_adapt import median_params, to_decode

RULES = ("rank1", "median3", "median5", "median10", "vote5", "vote10", "vote20")


def _decode(loglik, runs, c, durations=None, seen=None, q=None):
    return bench.decode_runs(loglik * c["scale"], runs, to_decode(c),
                             tables=bench.dur_tables(durations, c),
                             gain=None if seen is None else bench.gains(seen, q, c))


def _held(args):
    m, tag, merge, merged, candidates = args
    lomo = np.load(data_io.PUBLIC_CACHE / f"lomo-{tag}-1.npz")
    match_id = data_io.DFL_MATCHES[m]
    _, truth, runs, _ = data_io.dfl(match_id)
    target = public.merge_short(truth, merge)
    others = [j for j in range(merged.shape[1]) if j != m]
    order = np.argsort(-merged[:, others].mean(1), kind="stable")
    loglik = lomo[match_id].astype(np.float64)
    durations = bench.dfl_durations(exclude=m) if bench.DUR else None
    seen = public.dfl_frames(match_id, m)[0][:, 0] > 0.5
    q = bench.dfl_changes(match_id)
    out = {}
    for rule in RULES:
        if rule.startswith("vote"):
            top = [candidates[i] for i in order[:int(rule[4:])]]
            codes = bench.vote([_decode(loglik, runs, c, durations, seen, q) for c in top], runs)
        else:
            k = 1 if rule == "rank1" else int(rule.replace("median", ""))
            c = (candidates[order[0]] if k == 1
                 else median_params([candidates[i] for i in order[:k]]))
            codes = _decode(loglik, runs, c, durations, seen, q)
        out[rule] = (bench.spell(target, codes), bench.spell(truth, codes))
    return match_id, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="nogap")
    parser.add_argument("--merge", type=float, default=3.0)
    parser.add_argument("--n", type=int, default=0, help="use only the first n candidates")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--seed", type=int, default=None,
                        help="decoder search seed (default: config.DECODER_SEED)")
    args = parser.parse_args()
    tag = bench.evidence_tag(args.set, bench.parse_seeds(args.seeds))
    seed = config.DECODER_SEED if args.seed is None else args.seed
    saved = np.load(data_io.OUT / f"decoder-{bench.dtag(tag)}-m{args.merge:g}-s{seed}-all.npz")
    merged = saved["merged"]
    candidates = [json.loads(c) for c in saved["candidates"]]
    if args.n:
        merged, candidates = merged[:args.n], candidates[:args.n]
    jobs = [(m, tag, args.merge, merged, candidates) for m in range(merged.shape[1])]
    with Pool(bench.WORKERS) as pool:
        results = pool.map(_held, jobs)
    print(f"evidence {bench.dtag(tag)}, merge {args.merge:g} s, {len(candidates)} candidates; held-out DFL "
          "spellF1 (merged truth / raw truth) per picking rule:")
    for rule in RULES:
        values = np.array([out[rule] for _, out in results])
        print(f"  {rule:<9} merged {values[:, 0].mean():.3f} (sd {values[:, 0].std():.3f})  raw "
              f"{values[:, 1].mean():.3f}   per match "
              + " ".join(f"{v:.3f}" for v in values[:, 0]))


if __name__ == "__main__":
    main()
