"""Public football made to look like the match's tracking, and its truths made comparable.

Version 1 uses label-free aggregates of the provided tracking (fresh/stats.py):
- the ball is blurred by a centred mean plus a little jitter, so its one-frame velocity changes
  match the match's (median ~0.23 m/s, 99th percentile ~1.5 m/s; DFL raw: 0.25 and 21);
- the ball is hidden in long gaps, regardless of the state, on ~41% of frames, with gap lengths
  drawn from a log-normal matching the match's gaps (median ~11 s, 90th percentile ~34 s, none
  shorter than 1 s). Hiding regardless of the state is deliberate: what a gap says about a stoppage
  in this match is learned from the match itself (fresh/adapt.py), not assumed from another
  provider.

The submitted variant A uses version 4, whose fixed settings are defined below.

`merge_short` smooths DFL's possession flag: a spell shorter than T stays with the team that had
the ball before it ("possession stays with the last team that had it", CHALLENGE.md). T is
chosen on DFL alone (`config.json` merge_seconds).
"""

import data_io
import feats
import numpy as np

from possession.official import FPS, A, D, H, X, segments
from possession.tracking import AWAY, HOME, Tracking

BLUR = 21  # frames
JITTER = 0.004  # metres
HIDE_SHARE = 0.41
GAP_MEDIAN = 10.7  # seconds
GAP_SIGMA = 0.85  # log-normal sigma: median 10.7 s puts the 90th percentile near 32 s
MIN_GAP = 1.0  # seconds
# version 3: blur window (frames) and jitter (m), an alternate of version 1 not used by default
V3_BLUR_JITTER = (21, 0.05)
# Version 4 is the fixed degradation for the external-only variant. Round values for 25 Hz
# optical tracking in general: one second of smoothing, a few centimetres of noise, the ball
# missing about half the time, and gaps of several seconds. Not estimated from this match.
V4_BLUR = 25  # frames
V4_JITTER = 0.05  # metres
V4_HIDE_SHARE = 0.5
V4_GAP_MEDIAN = 8.0  # seconds
V4_GAP_SIGMA = 1.0
V4_MIN_GAP = 0.5  # seconds


def blur_ball(ball, rng, window=BLUR, jitter=JITTER):
    half = window // 2
    out = np.column_stack([feats.rolling_mean(ball[:, i].astype(np.float64), half, half)
                           for i in (0, 1)])
    out = out + rng.normal(0.0, jitter, size=out.shape)
    out[~np.isfinite(ball[:, 0])] = np.nan
    return out.astype(np.float32)


def hide_ball(ball, runs, rng, share=HIDE_SHARE, gap_median=GAP_MEDIAN, gap_sigma=GAP_SIGMA,
              min_gap=MIN_GAP):
    """Long gaps placed at random inside each run until ~`share` of its frames are hidden."""
    out = ball.copy()
    mean_gap = gap_median * np.exp(gap_sigma ** 2 / 2) * FPS
    mean_shown = mean_gap * (1 - share) / share
    for start, stop in runs:
        t = start + int(rng.exponential(mean_shown))
        while t < stop:
            gap = max(min_gap, rng.lognormal(np.log(gap_median), gap_sigma))
            span = round(float(gap) * FPS)
            out[t:min(t + span, stop)] = np.nan
            t += span + max(1, int(rng.exponential(mean_shown)))
    return out


class degradation:
    """Within the block, DFL inputs are degraded (and cached) as `version`: swaps config.DEGRADE
    and data_io.PUBLIC_CACHE, which every cached DFL input reads at call time."""

    def __init__(self, version):
        self.version = version

    def __enter__(self):
        import config
        self.saved = config.DEGRADE, data_io.PUBLIC_CACHE
        config.DEGRADE = self.version
        data_io.PUBLIC_CACHE = data_io.OUT / ("cache" if self.version == 1 else f"cache-v{self.version}")
        return self

    def __exit__(self, *exc):
        import config
        config.DEGRADE, data_io.PUBLIC_CACHE = self.saved


def test_version():
    """The degradation of the held-out match's inputs (config.TEST_DEGRADE, else DEGRADE)."""
    import config
    return config.DEGRADE if config.TEST_DEGRADE is None else config.TEST_DEGRADE


def test_suffix():
    """File suffix of held-out evidence scored on another degradation's inputs ("" if none)."""
    import config
    return "" if test_version() == config.DEGRADE else f"-test{test_version()}"


def degrade(tracking, runs, seed, hide=True, truth=None, version=None):
    """Version 1: blur, then state-free gaps. Version 2 (config.DEGRADE): the frozen stoppage ball
    replaced by a moving one, blur, then state-dependent gaps whose rates come from gap_targets.
    Version 4: version 1's shape with the fixed V4 constants."""
    import os

    import config
    forced = os.environ.get("FRESH_TRAIN_DEGRADE")
    if version is None and forced:
        version = int(forced)
    else:
        version = config.DEGRADE if version is None else version
    rng = np.random.default_rng(seed)
    if version in (1, 3, 4):
        if version == 4:
            window, jitter = V4_BLUR, V4_JITTER
        elif version == 3:
            window, jitter = V3_BLUR_JITTER
        else:
            window, jitter = BLUR, JITTER
        ball = blur_ball(tracking.ball, rng, window, jitter)
        if hide:
            if version == 4:
                ball = hide_ball(ball, runs, rng, V4_HIDE_SHARE, V4_GAP_MEDIAN, V4_GAP_SIGMA,
                                 V4_MIN_GAP)
            else:
                ball = hide_ball(ball, runs, rng)
        return Tracking(ball, tracking.players, tracking.team)
    if truth is None:
        raise ValueError("degradation 2 needs the DFL truth")
    ball = blur_ball(unfreeze(tracking.ball, truth, runs, rng), rng)
    if hide:
        p_dead, p_live = gap_targets()
        ball = hide_by_state(ball, truth, runs, rng, p_dead, p_live)
    return Tracking(ball, tracking.players, tracking.team)


WAYPOINT_SD = 4.0  # metres: the ball's detour between where play stopped and where it restarts
WALK_SD = 0.02  # metres per frame: carried / rolled ball noise


def unfreeze(ball, truth, runs, rng):
    """DFL freezes the ball at every stoppage, which the match's tracking never does (frozen
    frames: ~42% of DFL, ~0.2% of the match's seen ball). Each stoppage's ball instead travels a
    curve from where play stopped to where it restarts, through a random waypoint, plus a pinned
    random walk."""
    out = ball.astype(np.float64).copy()
    n_frames = len(ball)
    for a, b, k in segments(truth, runs):
        n = b - a
        if k != D or n < 2:
            continue
        p0 = out[a - 1] if a > 0 else out[a]
        p1 = out[b] if b < n_frames else out[b - 1]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        way = (p0 + p1) / 2 + rng.normal(0.0, WAYPOINT_SD, 2)
        s = np.linspace(0.0, 1.0, n + 2)[1:-1, None]
        path = (1 - s) ** 2 * p0 + 2 * s * (1 - s) * way + s ** 2 * p1
        walk = np.cumsum(rng.normal(0.0, WALK_SD, (n, 2)), axis=0)
        walk -= s * walk[-1]  # pinned: ends where the restart is
        out[a:b] = path + walk
    return out.astype(ball.dtype)


def hide_by_state(ball, truth, runs, rng, p_dead, p_live):
    """Each stoppage mostly hidden (from 0-3 s after it starts to 0-1 s before it ends) with a
    probability set so ~p_dead of dead frames are missing; long random gaps in play cover ~p_live
    of live frames. The same scheme as the gap bench (fresh/run_simgap.py), on a given ball."""
    out = ball.copy()
    mean_gap = GAP_MEDIAN * np.exp(GAP_SIGMA ** 2 / 2) * FPS
    mean_shown = mean_gap * (1 - p_live) / max(p_live, 1e-3)
    hidden = np.zeros(len(ball), dtype=bool)
    for start, stop in runs:
        for a, b, k in segments(truth, [(start, stop)]):
            if k != D:
                continue
            if rng.random() < min(1.0, p_dead / 0.85):
                lo = a + int(rng.uniform(0, 3) * FPS)
                hi = b - int(rng.uniform(0, 1) * FPS)
                hidden[lo:max(lo, hi)] = True
        t = start + int(rng.exponential(mean_shown))
        while t < stop:
            span = int(max(MIN_GAP, rng.lognormal(np.log(GAP_MEDIAN), GAP_SIGMA)) * FPS)
            hidden[t:t + span] |= np.isin(truth[t:t + span], (H, A))
            t += span + max(1, int(rng.exponential(mean_shown)))
    out[hidden] = np.nan
    return out


def gap_targets():
    """(P(missing | dead), P(missing | live)) on the match, from its EM posteriors under variant
    B's decoder (run_simgap.match_targets, version-1 evidence), computed once and cached next to
    the match data."""
    import json
    path = data_io.MATCH_CACHE / "gap-targets-v1.json"
    if path.exists():
        return tuple(json.loads(path.read_text()))
    raise FileNotFoundError(f"{path} missing: run `python run_simgap.py --targets` first")


def mirror_rows(matrix, truth):
    """The same frames with home and away exchanged: team-signed columns negated, H and A
    swapped. Exact for every column of feats.NAMES."""
    import bench
    import ctrl
    import shape
    import teamctx
    import touch
    names = bench.NAMES[:matrix.shape[1]]
    flipped = matrix.copy()
    for name in feats.TEAM_SIGNED + teamctx.SIGNED + touch.SIGNED + shape.SIGNED + ctrl.SIGNED:
        if name in names:
            flipped[:, names.index(name)] *= -1
    swapped = np.where(truth == H, A, np.where(truth == A, H, truth)).astype(truth.dtype)
    return flipped, swapped


def merge_short(truth, seconds):
    """Every H/A spell shorter than `seconds` given to the team of the live spell right before it
    (or, after a stoppage, right after it), left to right, so a run of brief touches stays with
    whoever last held the ball for long enough."""
    out = truth.copy()
    limit = seconds * FPS
    for a, b in _runs(truth):
        merged = []  # [start, stop, label], adjacent spells of one label joined as we go
        for s, e, k in segments(truth, [(a, b)]):
            prev = merged[-1] if merged else None
            joined = prev is not None and prev[1] == s
            if k in (H, A) and e - s < limit and joined and prev[2] in (H, A):
                prev[1] = e  # a brief touch: the ball stays with the last team
            elif joined and prev[2] == k:
                prev[1] = e
            else:
                merged.append([s, e, k])
        # a brief spell right after a stoppage goes to the team that follows it
        for i, (s, e, k) in enumerate(merged):
            if (k in (H, A) and e - s < limit and i + 1 < len(merged)
                    and merged[i + 1][0] == e and merged[i + 1][2] in (H, A)):
                merged[i][2] = merged[i + 1][2]
        for s, e, k in merged:
            out[s:e] = k
    return out


def _runs(truth):
    graded = truth != X
    starts, stops = feats.runs_of(graded)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def rates(truth):
    """(possessions per live minute, stoppages per graded minute, dead share)."""
    segs = segments(truth, _runs(truth))
    live = np.isin(truth, (H, A)).sum() / FPS / 60
    graded = (truth != X).sum() / FPS / 60
    n_poss = sum(1 for _, _, k in segs if k in (H, A))
    n_dead = sum(1 for _, _, k in segs if k == D)
    return n_poss / live, n_dead / graded, 1 - live / graded


def dfl_frames(match_id, seed, hide=True):
    """(features, attack sign, truth, runs) of one degraded DFL match, cached."""
    cache = data_io.PUBLIC_CACHE / f"dfl-{match_id}-{seed}-{int(hide)}.npz"
    tracking, truth, runs, pitch = data_io.dfl(match_id)
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"], saved["sign"], truth, runs
    x, sign = feats.compute(degrade(tracking, runs, seed, hide, truth=truth), runs, FPS, pitch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, sign=sign)
    return x, sign, truth, runs


def sample_frames(number, seed):
    """(features, attack sign, truth, runs) of a Metrica sample game, ball blurred only."""
    cache = data_io.PUBLIC_CACHE / f"sample-{number}-{seed}.npz"
    tracking, truth, runs, pitch = data_io.sample(number)
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"], saved["sign"], truth, runs
    x, sign = feats.compute(degrade(tracking, runs, seed, hide=False, version=1), runs, FPS, pitch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, sign=sign)
    return x, sign, truth, runs


__all__ = ["AWAY", "HOME"]
