"""The fresh pipeline end to end: fit (label-free) and predict (deterministic).

Variants, each adding one step to the previous:
- A: the submission. The configured classifier (a gradient-boosted tree) trained on the seven
  DFL matches under the fixed version-4 degradation (mirrored, no gap columns), decoded with the
  median of the ten best decoders of a search on DFL's held-out evidence against DFL's truth
  merged at config.MERGE seconds; the ball dead before each kickoff. No EM on the provided match.
- B: not submitted. A plus EM on the match: the ball's gaps and the broadcast cutting away, with
  tables learned from the match's own unlabelled data (fresh/adapt.py). Run it with --variant B.
- C: B with the evidence half DFL model, half a model self-trained on the match, out of block
  (fresh/selftrain.py).

Suffixes, in this order: `e` averages models trained on three DFL degradation seeds (0, 10, 20),
with the decoder search run on that averaged evidence; `v` lets the ten best decoders of the search
vote frame by frame instead of decoding once with their median (e.g. "Bev").

`fit` writes everything learned to fresh/model-<variant>.json: model weights, decoder numbers and
EM tables, no frame of the match. `predict_match` reads it back, so the submission is regenerated
from the repository byte for byte (the Viterbi runs on integer scores; the vote counts labels).
"""

import json
import os

import adapt
import bench
import changes
import classifiers
import config
import corrector
import data_io
import learn
import numpy as np
import selftrain
from run_adapt import median_params, ranked, to_decode
from run_lomo import FEATURE_SETS

from possession.official import GAME, GRADED_RUNS

SET = config.SET  # evidence feature set, fresh/config.json
COLUMNS = bench.columns_of(FEATURE_SETS[SET])
CHANNELS = ("gap", "broadcast")
TOP = 10  # decoders whose median is used, or who vote
MERGE = config.MERGE  # fresh/config.json merge_seconds, chosen on DFL only
SEED_ENSEMBLE = (0, 10, 20)
SELF_WEIGHT = 0.5  # C: share of the self-trained model in the evidence
FINAL = "A"  # the submitted variant (README.md). B remains available as --variant B.
VARIANTS = ("A", "B", "C", "Ae", "Av", "Aev", "Be", "Bv", "Bev")


def parse(variant):
    """(base letter, seeds, vote?) of a variant name."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {VARIANTS}")
    seeds = SEED_ENSEMBLE if "e" in variant[1:] else (0,)
    return variant[0], seeds, "v" in variant[1:]


def artifact_path(variant):
    suffix = (("" if SET == "nogap" else f"-{SET}") + ("-dur" if bench.DUR else "")
              + ("-gain" if bench.GAIN and bench.DUR else "")
              + ("-chg" if bench.CHANGE and bench.DUR else "")
              + ("-cor" if bench.CORRECT else "")
              + ("" if bench.CLASSIFIER == "logistic" else f"-{bench.CLASSIFIER}"))
    return data_io.FRESH / f"model-{variant}{suffix}.json"


def _channels(base):
    if base == "A":
        return []
    tracking, tags, live = data_io.match()
    seen = np.isfinite(tracking.ball[:, 0])
    return adapt.channels_for(seen, live, tags, CHANNELS)


def _loglik(body, x):
    models = [classifiers.from_json(m) for m in body["dfl_models"]]
    loglik = bench.Averaged(models).log_likelihood(x[:, COLUMNS])
    if body.get("self_models"):
        own = np.zeros_like(loglik)
        for (start, stop), model in zip(body["self_blocks"], body["self_models"], strict=True):
            own[start:stop] = learn.Logistic.from_json(model).log_likelihood(
                x[start:stop, COLUMNS])
        loglik = SELF_WEIGHT * own + (1 - SELF_WEIGHT) * loglik
    return loglik


def change_evidence(body, x):
    """The change model's hand-over probabilities on the match (changes.py), or None."""
    if not body.get("change_model"):
        return None
    return changes.probabilities(learn.Logistic.from_json(body["change_model"]), x)


def _evidence(loglik, c, channels, tables):
    """A member's evidence: the model's log-likelihood and the channels, both at its scale."""
    base = loglik * c["scale"]
    if not channels:
        return base
    return base + adapt._channel_scores(channels, [np.array(t) for t in tables], c["scale"])


def fit(variant=FINAL, log=print):
    """Fit `variant`. Variant A trains on DFL under degradation 4 and does not adapt to the match."""
    base, seeds, voting = parse(variant)
    if base != "A":
        return _fit(base, seeds, voting, variant, log)
    import data_io
    previous = os.environ.get("FRESH_TRAIN_DEGRADE")
    previous_cache = data_io.PUBLIC_CACHE
    os.environ["FRESH_TRAIN_DEGRADE"] = "4"
    data_io.PUBLIC_CACHE = data_io.OUT / "cache-v4"
    try:
        return _fit(base, seeds, voting, variant, log)
    finally:
        data_io.PUBLIC_CACHE = previous_cache
        if previous is None:
            os.environ.pop("FRESH_TRAIN_DEGRADE", None)
        else:
            os.environ["FRESH_TRAIN_DEGRADE"] = previous


def _fit(base, seeds, voting, variant, log):
    candidates = ranked(bench.evidence_tag(SET, seeds), MERGE, seed=config.DECODER_SEED)
    em_decoder = median_params(candidates[:TOP])
    members = candidates[:TOP] if voting else [em_decoder]
    models = bench.train_seeds(COLUMNS, seeds).models
    body = {"game": GAME, "variant": variant, "seeds": list(seeds), "members": members,
            "em_decoder": em_decoder, "dfl_models": [m.to_json() for m in models],
            "channels": [] if base == "A" else list(CHANNELS), "tables": [],
            "durations": bench.dfl_durations(merge=MERGE) if bench.DUR else None,
            "change_model": changes.train().to_json() if bench.CHANGE else None}
    if bench.CORRECT:
        model, threshold = corrector.train(bench.evidence_tag(SET, seeds), MERGE)
        body["corrector"] = {"model": model.to_json(), "threshold": threshold}
        log(f"segment corrector learned on DFL: threshold {threshold}")
    log(f"evidence: {len(models)} DFL model(s), seeds {list(seeds)}; "
        f"{'vote of the ' + str(TOP) if voting else 'median of the ' + str(TOP)} best decoders")
    log("decoder for EM: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in em_decoder.items()))
    if base == "A":
        return body
    x, _ = bench.match_frames()
    channels = _channels(base)
    params = to_decode(em_decoder)
    loglik = _loglik(body, x)
    evidence, tables, history = adapt.em(loglik * em_decoder["scale"], channels, params,
                                         em_decoder["scale"])
    body["tables"] = [t.tolist() for t in tables]
    log(f"EM: {len(history)} iterations, posterior dead share {history[-1][2]:.3f}")
    if base == "C":
        gamma = adapt.posterior(evidence, params)
        blocks, own = selftrain.fit_blocks(x, gamma, GRADED_RUNS, COLUMNS)
        body["self_blocks"] = blocks
        body["self_models"] = [m.to_json() for m in own]
        loglik = _loglik(body, x)
        _, tables, history = adapt.em(loglik * em_decoder["scale"], channels, params,
                                      em_decoder["scale"])
        body["tables"] = [t.tolist() for t in tables]
        log(f"EM after self-training: {len(history)} iterations, dead share {history[-1][2]:.3f}")
    return body


def save(body, variant):
    artifact_path(variant).write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")


def load(variant):
    path = artifact_path(variant)
    if not path.exists():
        return None
    body = json.loads(path.read_text())
    if "members" not in body or (bench.CORRECT and "corrector" not in body):
        return None  # an artifact from before the variants or the corrector: refit
    if body.get("game") != GAME:
        return None  # EM tables learned on another match: refit on this one (label-free)
    return body


def predict_match(variant=FINAL, body=None):
    """Full-length codes for the match from the saved artifacts (fitted on the spot if absent)."""
    if variant == "final":
        variant = FINAL
    body = body or load(variant)
    if body is None:
        body = fit(variant)
        save(body, variant)
    x, _ = bench.match_frames()
    channels = _channels(variant[0])
    loglik = _loglik(body, x)
    q = change_evidence(body, x)
    decoded = [bench.decode_runs(_evidence(loglik, c, channels, body["tables"]), GRADED_RUNS,
                                 to_decode(c), tables=bench.dur_tables(body.get("durations"), c),
                                 gain=bench.gains(x[:, 0] > 0.5, q, c))
               for c in body["members"]]
    codes = decoded[0] if len(decoded) == 1 else bench.vote(decoded, GRADED_RUNS)
    if body.get("corrector"):
        codes = corrector.apply(body, codes, loglik * body["em_decoder"]["scale"], x, GRADED_RUNS)
    return bench.dead_before_kickoff(codes)
