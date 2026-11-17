"""Experiment A, step 2: the decoder chosen on DFL only.

Random search over the evidence scale and the decoder's costs and minimum lengths, scored on the
held-out DFL evidence of fresh/run_lomo.py against DFL's truth with spells shorter than `--merge`
seconds merged (public.merge_short). The top candidates are then applied to the match with a model
trained on all seven DFL matches, reporting the whole-match counts. The public window is not read.

    uv run python fresh/run_decoder.py --set all --merge 3
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import bench
import data_io
import numpy as np
import public

from possession.decode import DecodeParams

_STATE = {}


def sample_candidates(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({
            "scale": float(np.exp(rng.uniform(np.log(0.03), np.log(1.0)))),
            "team_switch": float(np.exp(rng.uniform(np.log(1.0), np.log(40.0)))),
            "to_dead": float(np.exp(rng.uniform(np.log(1.0), np.log(40.0)))),
            "from_dead": float(np.exp(rng.uniform(np.log(1.0), np.log(40.0)))),
            "min_possession": int(rng.integers(1, 61)),
            "min_dead": int(rng.integers(1, 151)),
            "shift": int(rng.integers(-12, 13)),
        })
    if bench.DUR:  # a separate stream, so the other numbers are the same candidates as before
        extra = np.random.default_rng(seed + 1000)
        for c in out:
            c["dur_weight"] = float(extra.uniform(0.0, 3.0))
    if bench.GAIN:
        extra = np.random.default_rng(seed + 2000)
        for c in out:
            c["switch_seen"] = float(extra.uniform(-0.8, 0.8))
    if bench.CHANGE:
        extra = np.random.default_rng(seed + 3000)
        for c in out:
            c["change_weight"] = float(extra.uniform(0.0, 2.0))
    return out


def params_of(c):
    return DecodeParams(team_switch=c["team_switch"], to_dead=c["to_dead"],
                        from_dead=c["from_dead"], min_possession=c["min_possession"],
                        min_dead=c["min_dead"], shift=c["shift"])


def _init(tag, l2, merge):
    lomo = np.load(data_io.PUBLIC_CACHE / f"lomo-{tag}-{l2:g}.npz")
    items = []
    for i, match_id in enumerate(data_io.DFL_MATCHES):
        _, truth, runs, _ = data_io.dfl(match_id)
        items.append((lomo[match_id].astype(np.float64), truth,
                      public.merge_short(truth, merge) if merge else truth, runs,
                      bench.dfl_durations(exclude=i, merge=merge) if bench.DUR else None,
                      public.dfl_frames(match_id, i)[0][:, 0] > 0.5, bench.dfl_changes(match_id)))
    _STATE["items"] = items


def _score(c):
    params = params_of(c)
    merged, raw = [], []
    for loglik, truth, target, runs, durations, seen, q in _STATE["items"]:
        codes = bench.decode_runs(loglik * c["scale"], runs, params,
                                  tables=bench.dur_tables(durations, c),
                                  gain=bench.gains(seen, q, c))
        merged.append(bench.spell(target, codes))
        raw.append(bench.spell(truth, codes))
    return float(np.mean(merged)), float(np.mean(raw)), merged, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="all")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--merge", type=float, default=3.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--search-only", action="store_true",
                        help="write the DFL search and skip the final prediction on the provided match")
    parser.add_argument("--candidates-from", type=Path,
                        help="reuse the exact candidate values stored in an earlier search file")
    parser.add_argument("--seeds", default="0", help="the evidence of run_lomo.py --seeds")
    args = parser.parse_args()
    seeds = bench.parse_seeds(args.seeds)
    tag = bench.evidence_tag(args.set, seeds)
    began = time.time()
    if args.candidates_from is None:
        candidates = sample_candidates(args.n, args.seed)
    else:
        with np.load(args.candidates_from, allow_pickle=False) as source:
            candidates = [json.loads(candidate) for candidate in source["candidates"]]
        if len(candidates) != args.n:
            parser.error("--n must match the number of stored candidates")
    with Pool(bench.WORKERS, initializer=_init, initargs=(tag, args.l2, args.merge)) as pool:
        results = pool.map(_score, candidates, chunksize=4)
    order = np.argsort([-r[0] for r in results], kind="stable")
    np.savez(data_io.OUT / f"decoder-{bench.dtag(tag)}-m{args.merge:g}-s{args.seed}-all.npz",
             merged=np.array([r[2] for r in results]), raw=np.array([r[3] for r in results]),
             candidates=np.array([json.dumps(c) for c in candidates]))
    if args.search_only:
        print(f"DFL search complete: {len(candidates)} candidates, {time.time() - began:.0f} s")
        return

    # the match: a model on all seven DFL matches, the same columns
    from run_lomo import FEATURE_SETS
    names = FEATURE_SETS[args.set]
    columns = None if names is None else bench.columns_of(names)
    model = bench.train_seeds(columns, seeds, l2=args.l2)
    x, _ = bench.match_frames()
    loglik = model.log_likelihood(x if columns is None else x[:, columns])
    all_durations = bench.dfl_durations(merge=args.merge) if bench.DUR else None
    q = None
    if bench.CHANGE:
        import changes
        q = changes.probabilities(changes.train(), x)
    print(f"evidence {bench.dtag(tag)}, merge {args.merge:g} s, {args.n} candidates, "
          f"{time.time() - began:.0f} s. Top {args.top} by DFL spellF1 (merged truth):")
    print("rank  dfl_merged dfl_raw | match poss stops live% | params")
    best = []
    for rank, i in enumerate(order[:args.top]):
        c = candidates[i]
        codes = bench.dead_before_kickoff(bench.decode_runs(
            loglik * c["scale"], bench.GRADED_RUNS, params_of(c),
            tables=bench.dur_tables(all_durations, c), gain=bench.gains(x[:, 0] > 0.5, q, c)))
        n_poss, n_dead, live, graded = bench.counts(codes)
        print(f"{rank:>4}  {results[i][0]:.3f}      {results[i][1]:.3f}   | "
              f"{n_poss:>4} {n_dead:>4} {100 * live / graded:.0f}% | "
              + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in c.items()))
        best.append({"rank": rank, "dfl_merged": results[i][0], "dfl_raw": results[i][1],
                     "per_match": results[i][2], "params": c})
    out = data_io.OUT / f"decoder-{bench.dtag(tag)}-m{args.merge:g}-s{args.seed}.json"
    out.write_text(json.dumps(best, indent=1) + "\n")


if __name__ == "__main__":
    main()
