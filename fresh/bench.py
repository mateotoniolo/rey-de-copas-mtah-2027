"""Shared pieces of the experiments: datasets, training, decoding, scoring. Aggregates only."""

import os
from dataclasses import replace

import config
import ctrl
import data_io
import feats
import hdecode
import learn
import longctx
import numpy as np
import public
import setpiece
import shape
import teamctx
import touch

from possession.decode import DecodeParams, shift_changes, viterbi
from possession.official import FPS, GRADED_RUNS, KICKOFFS, A, D, H, X, pm, score, segments

GAIN = config.GAIN  # fresh/config.json
DUR = config.DUR  # fresh/config.json
CORRECT = config.CORRECT  # fresh/config.json
CHANGE = config.CHANGE  # fresh/config.json
WORKERS = int(os.environ.get("FRESH_WORKERS", "4"))  # the main session uses the rest of the CPU
SUBSAMPLE = 3  # train on every third frame: neighbours are nearly copies
# even (mirror-invariant) bends of the "long" set's columns, x^2 / (1 + |x|): a nonlinear model
# that stays a deterministic logistic regression (idea 3, feature set "bend")
_LONG_SET = [n for n in feats.NAMES if not n.startswith("gap_len")
             and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
BEND_NAMES = tuple(f"bend_{n}" for n in _LONG_SET)
NAMES = feats.NAMES + longctx.NAMES + teamctx.NAMES + BEND_NAMES + touch.NAMES + shape.NAMES + ctrl.NAMES + setpiece.NAMES  # every column a feature set may name, feats' first


def match_frames():
    """(features, attack sign) of the match, cached next to the match data."""
    cache = data_io.MATCH_CACHE / "feats-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            x, sign = saved["x"], saved["sign"]
    else:
        tracking, _, _ = data_io.match()
        x, sign = feats.compute(tracking, GRADED_RUNS, FPS, data_io.PITCH)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x, sign=sign)
    return np.hstack([with_bends(np.hstack([x, match_long(), match_team(sign)])), match_touch(),
                      match_shape(sign), match_ctrl(), match_setp()]), sign


def with_bends(x):
    """Append the bend columns to a feats + longctx + teamctx matrix."""
    base = feats.NAMES + longctx.NAMES + teamctx.NAMES
    part = x[:, [base.index(n) for n in _LONG_SET]].astype(np.float64)
    return np.hstack([x, (part * part / (1 + np.abs(part))).astype(np.float32)])


def match_touch():
    """The touch columns of the match (fresh/touch.py), cached."""
    cache = data_io.MATCH_CACHE / "touch-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _ = data_io.match()
    x = touch.compute(tracking.ball, tracking.players, tracking.team, FPS)
    np.savez(cache, x=x)
    return x


def dfl_touch(match_id, seed, degraded=None):
    if degraded is not None:
        return touch.compute(degraded.ball, degraded.players, degraded.team, FPS)
    cache = data_io.PUBLIC_CACHE / f"touch-{match_id}-{seed}-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, truth, runs, _ = data_io.dfl(match_id)
    deg = public.degrade(tracking, runs, seed, truth=truth)
    x = touch.compute(deg.ball, deg.players, deg.team, FPS)
    np.savez(cache, x=x)
    return x


def match_ctrl():
    """The E4 columns of the match (fresh/ctrl.py), cached."""
    cache = data_io.MATCH_CACHE / "ctrl-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _ = data_io.match()
    x = ctrl.compute(tracking.ball, tracking.players, tracking.team, FPS)
    np.savez(cache, x=x)
    return x


def dfl_ctrl(match_id, seed, degraded=None):
    if degraded is not None:
        return ctrl.compute(degraded.ball, degraded.players, degraded.team, FPS)
    cache = data_io.PUBLIC_CACHE / f"ctrl-{match_id}-{seed}-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, truth, runs, _ = data_io.dfl(match_id)
    deg = public.degrade(tracking, runs, seed, truth=truth)
    x = ctrl.compute(deg.ball, deg.players, deg.team, FPS)
    np.savez(cache, x=x)
    return x


def match_setp():
    """The E9 columns of the match (fresh/setpiece.py), cached."""
    cache = data_io.MATCH_CACHE / "setp-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _ = data_io.match()
    x = setpiece.compute(tracking.players, tracking.team, FPS, data_io.PITCH)
    np.savez(cache, x=x)
    return x


def dfl_setp(match_id, seed=None, degraded=None):
    """Players only, so the degradation (which changes the ball) does not change them."""
    cache = data_io.PUBLIC_CACHE / f"setp-{match_id}-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _, pitch = data_io.dfl(match_id)
    players = tracking.players if degraded is None else degraded.players
    x = setpiece.compute(players, tracking.team, FPS, pitch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x)
    return x


def match_shape(sign):
    """The team-shape columns of the match (fresh/shape.py), cached."""
    cache = data_io.MATCH_CACHE / "shape-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _ = data_io.match()
    x = shape.compute(tracking.players, tracking.team, sign, FPS)
    np.savez(cache, x=x)
    return x


def dfl_shape(match_id, seed, sign, degraded=None):
    if degraded is not None:
        return shape.compute(degraded.players, degraded.team, sign, FPS)
    cache = data_io.PUBLIC_CACHE / f"shape-{match_id}-{seed}-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, truth, runs, _ = data_io.dfl(match_id)
    deg = public.degrade(tracking, runs, seed, truth=truth)
    x = shape.compute(deg.players, deg.team, sign, FPS)
    np.savez(cache, x=x)
    return x


def match_team(sign):
    """The team-context columns of the match (fresh/teamctx.py), cached."""
    cache = data_io.MATCH_CACHE / "team-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _ = data_io.match()
    x = teamctx.compute(tracking.ball, tracking.players, tracking.team, sign, FPS)
    np.savez(cache, x=x)
    return x


def dfl_team(match_id, seed, sign, degraded=None):
    """Team-context columns of a DFL match under the degradation of `seed` (public.degrade), or of
    an already degraded tracking (not cached)."""
    if degraded is not None:
        return teamctx.compute(degraded.ball, degraded.players, degraded.team, sign, FPS)
    cache = data_io.PUBLIC_CACHE / f"team-{match_id}-{seed}-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, truth, runs, _ = data_io.dfl(match_id)
    deg = public.degrade(tracking, runs, seed, truth=truth)
    x = teamctx.compute(deg.ball, deg.players, deg.team, sign, FPS)
    np.savez(cache, x=x)
    return x


def match_long():
    """The player-only long-context columns of the match (fresh/longctx.py), cached."""
    cache = data_io.MATCH_CACHE / "long-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _ = data_io.match()
    x = longctx.compute(tracking.players, FPS, data_io.PITCH)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x)
    return x


def dfl_long(match_id):
    """The long-context columns of a DFL match: players only, so the same for every degradation."""
    cache = data_io.PUBLIC_CACHE / f"long-{match_id}-v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"]
    tracking, _, _, pitch = data_io.dfl(match_id)
    x = longctx.compute(tracking.players, FPS, pitch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x)
    return x


def dfl_set(seed=0):
    """[(match_id, x, truth, runs)] for the seven DFL matches, each degraded with its own seed."""
    out = []
    for i, match_id in enumerate(data_io.DFL_MATCHES):
        x, sign, truth, runs = public.dfl_frames(match_id, seed + i)
        out.append((match_id, np.hstack([with_bends(np.hstack([x, dfl_long(match_id),
                                                     dfl_team(match_id, seed + i, sign)])),
                                         dfl_touch(match_id, seed + i),
                                         dfl_shape(match_id, seed + i, sign),
                                         dfl_ctrl(match_id, seed + i),
                                         dfl_setp(match_id, seed + i)]),
                    truth, runs))
    return out


def sample_set(seed=0):
    """Metrica's sample games 1-3 as extra training matches. Their own ball
    blanks (out of play) are filled on the straight line first, then the DFL degradation of the
    run (state-free gaps) is applied, so a gap says nothing about the state. Cached."""
    import ctrl

    from possession.tracking import Tracking
    out = []
    for number in (1, 2, 3):
        cache = data_io.PUBLIC_CACHE / f"metrica-{number}-{seed + 10 + number}-v2.npz"
        tracking, truth, runs, pitch = data_io.sample(number)
        if cache.exists():
            with np.load(cache) as saved:
                out.append((f"metrica-{number}", saved["x"], truth, runs))
            continue
        filled = Tracking(ctrl.project(tracking.ball).astype(np.float32), tracking.players,
                          tracking.team)
        deg = public.degrade(filled, runs, seed + 10 + number, truth=truth)
        x, sign = feats.compute(deg, runs, FPS, pitch)
        x = np.hstack([with_bends(np.hstack([x, longctx.compute(tracking.players, FPS, pitch),
                                             teamctx.compute(deg.ball, deg.players, deg.team,
                                                             sign, FPS)])),
                       touch.compute(deg.ball, deg.players, deg.team, FPS),
                       shape.compute(deg.players, deg.team, sign, FPS),
                       ctrl.compute(deg.ball, deg.players, deg.team, FPS),
                       setpiece.compute(tracking.players, tracking.team, FPS, pitch)])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x)
        out.append((f"metrica-{number}", x, truth, runs))
    return out


def training_rows(items, columns=None, mirror=True, subsample=SUBSAMPLE):
    xs, ys = [], []
    for _, x, truth, _ in items:
        keep = np.flatnonzero(truth != X)[::subsample]
        rows, labels = x[keep], truth[keep].astype(np.int64)
        xs.append(rows)
        ys.append(labels)
        if mirror:
            flipped, swapped = public.mirror_rows(rows, labels)
            xs.append(flipped)
            ys.append(swapped)
    x = np.concatenate(xs)
    if columns is not None:
        x = x[:, columns]
    return x, np.concatenate(ys)


CLASSIFIER = config.CLASSIFIER  # "logistic" (learn.py), "gbm" or "net" (classifiers.py)


def _window_rows(items, columns, mirror, subsample=SUBSAMPLE):
    import classifiers
    zs, ys = [], []
    for _, x, truth, _ in items:
        keep = np.flatnonzero(truth != X)[::subsample]
        labels = truth[keep].astype(np.int64)
        views = [(x, labels)]
        if mirror:
            flipped, swapped = public.mirror_rows(x, truth)
            views.append((flipped, swapped[keep].astype(np.int64)))
        for xx, yy in views:
            xx = xx if columns is None else xx[:, columns]
            zs.append(classifiers.windows(xx)[keep])
            ys.append(yy)
    return np.concatenate(zs), np.concatenate(ys)


def train(items, columns=None, l2=1.0, mirror=True):
    if CLASSIFIER == "ens3":  # gbm, net and logistic, log-likelihoods averaged
        import classifiers
        x, y = training_rows(items, columns, mirror)
        return Averaged([classifiers.fit_gbm(x, y),
                         classifiers.fit_net(*_window_rows(items, columns, mirror)),
                         learn.fit(x, y, l2=l2)])
    if CLASSIFIER == "net":
        import classifiers
        return classifiers.fit_net(*_window_rows(items, columns, mirror))
    x, y = training_rows(items, columns, mirror)
    if CLASSIFIER == "gbm":
        import classifiers
        return classifiers.fit_gbm(x, y)
    return learn.fit(x, y, l2=l2)


class Averaged:
    """Several models read as one: the mean of their log-likelihoods (and log-posteriors)."""

    def __init__(self, models):
        self.models = list(models)

    def log_likelihood(self, x):
        return np.mean([m.log_likelihood(x) for m in self.models], axis=0)

    def log_proba(self, x):
        return np.mean([m.log_proba(x) for m in self.models], axis=0)


def train_seeds(columns, seeds=(0,), exclude=None, l2=1.0):
    """One model per DFL degradation seed (matches degraded with seed + i), averaged; `exclude`
    leaves one match out by index."""
    models = []
    for seed in seeds:
        items = [it for i, it in enumerate(dfl_set(seed)) if i != exclude]
        if config.METRICA:
            items += sample_set(seed)
        models.append(train(items, columns, l2))
    return Averaged(models)


def evidence_tag(set_name, seeds=(0,)):
    """Names the held-out evidence and the decoder search of a feature set and seed list."""
    seeds = tuple(seeds)
    if CLASSIFIER != "logistic":
        set_name = f"{set_name}-{CLASSIFIER}"
    if config.METRICA:
        set_name = f"{set_name}-mx"
    return set_name if seeds == (0,) else f"{set_name}-s{'_'.join(str(s) for s in seeds)}"


def parse_seeds(text):
    return tuple(int(s) for s in str(text).split(","))


def vote(predictions, runs):
    """Frame-wise majority of H, A and D inside `runs`, ties to the lowest code; X elsewhere.
    Integer counting over exact labels, so it keeps the output byte for byte."""
    out = np.array(predictions[0], dtype=np.int8, copy=True)
    for start, stop in runs:
        counts = np.zeros((stop - start, 3), dtype=np.int32)
        for codes in predictions:
            window = np.asarray(codes[start:stop])
            for code in (H, A, D):
                counts[:, code] += window == code
        out[start:stop] = np.argmax(counts, axis=1)
    return out


def columns_of(names):
    return [NAMES.index(n) for n in names]


def dtag(tag, dur=None):
    """The decoder search's file tag: the evidence tag, plus "-dur" for the duration decoder."""
    out = f"{tag}-dur" if (DUR if dur is None else dur) else tag
    out = f"{out}-gain" if GAIN and (DUR if dur is None else dur) else out
    out = f"{out}-chg" if CHANGE and (DUR if dur is None else dur) else out
    degradation = int(os.environ.get("FRESH_TRAIN_DEGRADE") or config.DEGRADE)
    if (DUR if dur is None else dur) and degradation != 1:
        out = f"{out}-g{degradation}"
    if (DUR if dur is None else dur) and config.DUR_MERGE is not None:
        out = f"{out}-d{config.DUR_MERGE:g}"
    if (DUR if dur is None else dur) and config.DUR_SCALE != 1.0:
        out = f"{out}-ds{config.DUR_SCALE:g}"
    return out


def durations_merge(merge):
    """The merge seconds of the truth the duration prior is fitted on (config.DUR_MERGE wins)."""
    return merge if config.DUR_MERGE is None else config.DUR_MERGE


def switch_gain(seen, c):
    """(n, 3) nats off a team-to-team hand-over: switch_seen x team_switch where the ball is seen
    (smoothed over 1 s), so the search can make changes cheaper or dearer where they are visible."""
    if "switch_seen" not in c:
        return None
    visible = feats.rolling_mean(np.asarray(seen, dtype=np.float64), 12, 12)
    g = float(c["switch_seen"]) * float(c["team_switch"]) * np.nan_to_num(visible)
    return np.column_stack([g, g, np.zeros_like(g)])


def change_gain(q, c):
    """(n, 3) nats off a team-to-team hand-over: change_weight x team_switch x the change model's
    P(a hand-over into that team) (changes.py), so a change costs less where a pass or a reception
    by the other team is seen."""
    if q is None or "change_weight" not in c:
        return None
    g = float(c["change_weight"]) * float(c["team_switch"]) * np.asarray(q, dtype=np.float64)
    return np.column_stack([g[:, 0], g[:, 1], np.zeros(len(g))])


def gains(seen, q, c):
    """switch_gain and change_gain together, or None when neither applies."""
    parts = [g for g in (switch_gain(seen, c), change_gain(q, c)) if g is not None]
    return None if not parts else parts[0] if len(parts) == 1 else parts[0] + parts[1]


def dfl_changes(match_id):
    """The held-out change evidence of a DFL match (changes.py), or None without FRESH_CHANGE."""
    if not CHANGE:
        return None
    with np.load(data_io.PUBLIC_CACHE / "lomo-change.npz") as saved:
        return saved[match_id]


def dfl_durations(exclude=None, merge=None):
    """hdecode.fit on DFL's merged truth (config.MERGE unless `merge`), every match but `exclude`."""
    merge = durations_merge(config.MERGE if merge is None else merge)
    truths, runs_list = [], []
    for i, match_id in enumerate(data_io.DFL_MATCHES):
        if i == exclude:
            continue
        _, truth, runs, _ = data_io.dfl(match_id)
        truths.append(public.merge_short(truth, merge))
        runs_list.append(runs)
    fitted = hdecode.fit(truths, runs_list)
    if config.DUR_SCALE != 1.0:
        shift = float(np.log(config.DUR_SCALE))
        fitted = {key: [mu + shift, sigma] for key, (mu, sigma) in fitted.items()}
    return fitted


def dur_tables(durations, c):
    """Bonus tables for candidate `c` (its dur_weight and minimum lengths), or None."""
    if durations is None or "dur_weight" not in c:
        return None
    mins = [int(c["min_possession"]), int(c["min_possession"]), int(c["min_dead"])]
    return hdecode.bonus_tables(durations, float(c["dur_weight"]), mins)


def decode_runs(scores, runs, params, n=None, tables=None, gain=None):
    """Full-length codes: Viterbi inside each run (duration-aware when `tables`), X elsewhere."""
    out = np.full(n or len(scores), X, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = (viterbi(scores[start:stop], params) if tables is None
                           else hdecode.viterbi(scores[start:stop], params, tables,
                                                None if gain is None else gain[start:stop]))
    return shift_changes(out, runs, params.shift) if params.shift else out


def dead_before_kickoff(codes):
    """The ball is dead before each half's kickoff (football's rules; the kickoff frames are the
    official anchors in vendor/mtah-2027/games.py)."""
    out = codes.copy()
    for (start, stop), kickoff in zip(GRADED_RUNS, KICKOFFS, strict=True):
        end = min(max(kickoff - 1, start), stop)
        part = out[start:end]
        part[part != X] = D
    return out


def spell(truth, pred):
    return score(truth, pred)["spell_f1"]


def counts(codes, runs=GRADED_RUNS):
    """(possessions, stoppages, live minutes, graded minutes) of a decoded array."""
    segs = segments(codes, runs)
    n_poss = sum(1 for _, _, k in segs if k in (H, A))
    n_dead = sum(1 for _, _, k in segs if k == D)
    live = np.isin(codes, (H, A)).sum() / FPS / 60
    graded = (codes != X).sum() / FPS / 60
    return n_poss, n_dead, live, graded


def public_score(codes):
    s = score(data_io.load_public_gt(), codes)
    return s["spell_f1"], s["macro_f1"], s["n_pred"], s["n_gt"]


def bootstrap_public(codes, n=1000):
    r = pm.evaluate(data_io.load_public_gt(), codes, bootstrap_n=n)
    return r["spell_bootstrap_low"], r["spell_bootstrap_high"]


def decode_params(**kw):
    base = DecodeParams(team_switch=10.0, to_dead=8.0, from_dead=8.0, min_possession=12,
                        min_dead=50, shift=0)
    return replace(base, **kw)
