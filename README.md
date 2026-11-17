# Rey de Copas — More than a Hack 2027

Team **Rey de Copas** (Mateo Toniolo, Gabriel Canavesio, Santiago Vidoz).

The submission is variant A. A gradient-boosted tree is trained on the seven external DFL matches with the fixed version-4 degradation. It applies 25-frame smoothing, 5 cm jitter, ball occlusion on about half the frames, and log-normal gaps with an 8 s median. A duration-aware integer Viterbi decoder turns the evidence into possessions. A segment corrector trained on DFL can hand a segment it judges wrong to its neighbour.

The decoder search and the examples used to train the corrector also use DFL evidence under the fixed version-4 degradation. The search uses 240 candidates with seed 0 and ranks them on DFL with a 2 s merge threshold. The decoder uses the normalized median of the ten best candidates. Variant A uses the provided match as input for prediction. Training and tuning use external data.

`uv run python fresh/predict.py --variant B` is an alternative and is not submitted. It uses expectation-maximisation tables fit on BAR-ATH's unlabelled tracking and homography. That fit does not use ground truth, the public window, hand labels, or video. It writes `fresh/out/fresh-B-setp-dur-cor-gbm.csv`. The SHA-256 of that earlier file, which is not the submission, is `0fa36ddbb2707f2c032a67cb742ab7dcb2dda427bfb3ec251f4f6146f9f16999`.

The match data is not in this repository. It is never sent to a hosted AI service.

Our code is MIT, copyright 2026 Rey de Copas (see `LICENSE`). The DFL tracking used for training is not included. It is CC BY 4.0 and stays under its own terms: credit the Deutsche Fußball Liga (DFL) and cite Bassek et al. 2025.

## Requirements

- Python 3.11, installed and run with [uv](https://docs.astral.sh/uv/)
- The competition match bundle, in `data/` (gitignored) or pointed to by `MTAH_DATA_DIR`
- The seven public DFL / IDSSE matches, under `external/dfl/` (gitignored)

The DFL matches are Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, 2025, CC BY 4.0, figshare [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177). Place each match's `matchinformation` and `positions_raw_observed` XML in `external/dfl/`.

## Reproduce the submission

```bash
git clone --recursive https://github.com/mateotoniolo/rey-de-copas-mtah-2027
cd rey-de-copas-mtah-2027
uv sync --locked
uv run python fresh/predict.py
uv run python fresh/predict.py --refit
```

Both commands write `fresh/out/fresh-A-setp-dur-cor-gbm.csv`. Expected SHA-256:

```
387fc93bcf44f6b7b339cd0e7ad445d28b11b067c2c57ff06622b93cfb9a687a
```

`--refit` learns the saved model again from the external matches and the tracked version-4 decoder search, then writes the same file. `fresh/out/` is gitignored.

To repeat the decoder search on DFL, generate the held-out evidence and run the same search:

```bash
FRESH_DEGRADE=4 uv run python fresh/run_lomo.py --set setp --seeds 0
FRESH_DEGRADE=4 uv run python fresh/run_decoder.py --set setp --merge 2 --n 240 --seed 0 --search-only --candidates-from fresh/artifacts/decoder-setp-gbm-dur-m2-s0-all.npz
```

The command reuses only the original, seeded candidate values from the older artifact. It computes all rankings from v4 evidence. The tracked v4 search artifact retains candidate parameters and the rankings needed to refit. Full DFL scores stay in the ignored output directory.

Check the CSV with the pinned official validator:

```bash
uv run python vendor/mtah-2027/evaluation/validate_submission.py fresh/out/fresh-A-setp-dur-cor-gbm.csv
```

## Compliance

The submitted pipeline uses no hand labels or other human-produced labels of the provided match. The public five-minute window and the match video were not used for training, tuning, or selection. Variant A uses no expectation-maximisation on the provided match. Its classifier, decoder search, duration prior and corrector are derived from external DFL matches using the procedure above.

## Layout

| path | what |
| --- | --- |
| `fresh/` | the pipeline and the saved model |
| `src/possession/` | tracking readers, the integer decoder, and the submission writer |
| `vendor/mtah-2027/` | the official scorer and validator, pinned |
