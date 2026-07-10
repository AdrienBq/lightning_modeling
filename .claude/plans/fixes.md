# Fix Plan — `lightning_modelling` paper demo repo

Progress tracker for the clean-up pass. **Resume at Step 3.**

Status legend: ✅ done · ⏳ in progress · ⬜ not started

- ✅ Step 1 — Latent correctness bugs (src)
- ✅ Step 2 — Notebook eval alignment → extremes-only (nb01, nb02)
- ✅ Step 3 — Docs / README factual fixes
- ✅ Step 4 — Dead code removal (src)  [kept BaselineModel + some comments per user]
- ⬜ Step 5 — Dead code removal (notebooks)   ← **RESUME HERE**
- ⬜ Step 6 — Optimization
- ⬜ Step 7 — Docstrings & comments

---

## Context

Public companion repo to a lightning-modelling paper (ERA5-derived predictors → binary
lightning target). Readers clone it, read the `src` package, and run the three reproduction
notebooks, so the bar is: no latent bugs, no dead code, non-obvious transforms documented,
docs factually correct.

Key facts established during the audit (verified against the working tree):
- **Kernel**: repaired — a `lightning_modelling` (double-"l") kernelspec points at
  `/homedata/aburq/.venvs/lightning_modelling/bin/python`; the broken single-"l"
  `lightning_modeling` spec was removed. All three notebooks reference `lightning_modelling`.
- **Demo data**: only the **49 extreme-day** `.pt` files are shipped (out of the original 5843
  days). Files are named by global id (`sample_000152.pt` = id 152).
- **Dataset indexing is POSITIONAL and intended** (do NOT change `CustomPTDataset` file lookup):
  `sample_files[sample_ids[idx]]`, with `sample_files` = sorted list of the 49 shipped files.
  So passing `sample_ids=range(len(test_extremes))` makes entry `i` = the i-th shipped file,
  which is id-sorted-aligned with `test_extremes.iloc[i]` and `EXTREMES_SEASONS[i]`.

User decisions:
- Scope: do everything, **step by step** (check in between steps).
- Eval runs on **only the 49 extreme days** (via the extremes dataset/loader).
- `metrics.py` "IoU" is intentionally **Dice** → renamed, formula unchanged.

---

## ✅ Step 1 — Latent correctness bugs (src)  [DONE]

- `trainer.py:273` `if i + 1 % 100 == 0:` → `if (i + 1) % 100 == 0:` (precedence).
- `trainer.py` `test()`: added `season = None` before the loop → no `NameError` when `seasons=None`.
- `dataset.py` `__getitem__`: `self.transform = None` set unconditionally in `__init__` → no
  `AttributeError` when scaler missing.
- `dataset.py` sample-file listing wrapped in `sorted(...)` → deterministic order.
- `dataset.py` scaler defaults fixed `scalers/…` → `data/scaler/scaler_full.pkl` in
  `CustomPTDataset.__init__` and `create_train_test`. (Third site in `create_train_val_test`
  left for deletion in Step 4.)
- `metrics.py` `IoU` class → **`Dice`** (`self.iou`→`self.dice`, print label, JSON key
  `"IoU"`→`"Dice"`), with docstring stating `Dice = 2*(I+s)/(P+T-I+s)`. Formula unchanged.
- Note: original audit bug A4 (`histogram_binning_recalibration`) was already fixed in the
  working tree before this pass.
- Notebook `trainer.metrics.iou.iou` → `.dice.dice` was fixed & committed by the user.

Verified: all three files parse, package imports from the venv, `Dice` runs.

## ✅ Step 2 — Notebook eval alignment → extremes-only (nb01, nb02)  [DONE]

`dataset.py` NOT changed (positional indexing is intended). Notebook-only fix:
- **nb01**: added `CustomPTDataset` import; built `TEST_EXTREMES_DATASET` with
  `sample_ids=range(len(test_extremes))`; plot loop iterates `TEST_EXTREMES_DATASET`.
- **nb02**: `TEST_EXTREMES_DATASET` now uses `sample_ids=range(len(test_extremes))` (was
  `test_extremes_ids`, which `IndexError`ed); eval loop + shape probe use `TEST_EXTREMES_DATASET`;
  kept `test_extremes_ids` for indexing `EXTREMES_SEASONS` (global ids, correct).

Verified: both notebooks execute end-to-end from `notebooks/`; nb02 prints metrics for all 5
models; iterating all 49 extreme days raises no error.

⚠️ **Open config choice (not a bug):** nb02 "Configure the trainers" sets `EARLY_STOP=True,
N_EARLY=2`, so only the first 2 extreme days are evaluated by default. To reproduce the paper's
Table-5 numbers over all 49 days, set `EARLY_STOP=False`. Left as-is pending user preference.

---

## ✅ Step 3 — Docs / README factual fixes   [DONE]

- **data/samples/README.md** — fixed lat/lon swap; verified tensor shape `(24, 6, 101, 149)` →
  now reads "(24 timesteps, 6 channels, 101 latitudes, 149 longitudes)".
- **README.md (root)** — Installation section clarified: `pip install .` / `uv sync` (portable,
  CPU-friendly) vs `requirements.txt` (authors' exact pinned GPU freeze); noted they're not
  equivalent. (User decision: "document the difference".)
- **Placeholders** — the three `[INSERT …]` left untouched (User decision: "skip for now").
- **Scaler path** — no doc referenced `scalers` (plural); only stale plural is in the dead
  `create_train_val_test` (removed in Step 4). Nothing to fix here.

## ✅ Step 4 — Dead code removal (src)   ← RESUME HERE

- **dataset.py**: remove module-level `device`; remove `create_train_val_test` (never called;
  also carries a broken scaler default); remove redundant local `import pandas as pd`. **KEEP**
  local `import json` (NOT redundant — json not imported at module top).
- **models.py**: remove `BaselineModel` (never instantiated).
- **metrics.py**: remove unused `SCALER_PATH`, `DATES`; fix `def save(self: Path):` →
  `def save(self):`; drop commented block (~226-227).
- **trainer.py**: remove module-level `device` (shadowed by `self.device`); remove dead
  `start_time = time.time()` in `test()` ONLY (the `train()`/`platt` ones are live); remove unused
  `__init__` attrs `full_dataset, events_start_date, events_start_hour, geopt_path, data_path,
  n_years`; `pos_weight` is a `train()` local used only in a comment; decide whether to drop the
  never-called `recalibrate()`; remove commented lines (~6, 117, 128, 282).
- **common_path.py**: remove unused `LIBRARY_NAME`.
- **plots.py**: remove commented-out blocks (~117-121, 233-234, 239, 248, 396, 418-420,
  504-505, 582 [582 is a dangling truncated comment]).
- Verify with `ruff`/`pyflakes` that no live reference was cut.

## ✅ Step 5 — Dead code removal (notebooks)

- **nb02 eval-loop cell**: remove `pred_events_maps = []` (unused), `start_time_model` (unread →
  makes `import time` dead, remove that too), unused enumerate index `j`, and commented
  `# pbar = tqdm(...)` (tqdm not imported).
- **nb02 load-models cell**: remove dead `logreg_name = "logreg"` (constructor uses the literal).
  Note nb01 *does* use `logreg_name` — leave nb01 (or align both to one style).
- **nb03**: the loader cell is a verbatim copy of nb02's; drop the objects nb03 doesn't use and
  their now-orphaned cell-3 imports (`DataLoader`, `CustomPTDataset`, `get_seasons`) — but KEEP
  `create_train_test`. Remove the empty trailing cell.

## ⬜ Step 6 — Optimization

- **metrics.py** `StreamingAUC.update`: replace the per-pixel Python loop (hot path via
  `update_all`) with vectorized `np.add.at` / `np.bincount` over `bin_indices` split by `y_true`.
  Biggest win.
- **Channel-masking block duplicated 5×**: `models.py` forwards (attr `remove_vars`) ×3 and
  `deter_architecture.py` forwards (attr `removed_features`) ×2. Factor into one helper; normalize
  the differing attribute names/guards.
- **dataset.py** `create_train_test`: **4** season branches (summer/winter/spring/autumn), not 5;
  collapse to a `{season: months}` dict + one path; compute the `month` column once.
- **plots.py**: `_create_colors`, `_extract_metadata`, and the `base_colors` list are byte-
  identical across `LightningPlotMultiModel` and `SaliencyPlotPaper` — hoist to a shared base.
- **models.py**: `torch.tensor(numpy_array)` (×3) → `torch.from_numpy(...)`.
- **trainer.py**: `for idx, epoch in enumerate(range(num_epochs))` → `for epoch in range(...)`
  (`idx == epoch` always; update the `f"epoch_{idx}"`/checkpoint uses).

## ⬜ Step 7 — Docstrings & comments

- **metrics.py**: docstring `StreamingAUC` (histogram-cumsum trapezoid AUC/AP + `first_nonzero_index`
  trimming), `Dice` (done in Step 1), `DeterMetrics` (note `ets` is an *attribute*, not a method —
  document ETS + random-hits `tr` term), `FractionalScores` (FBS/FBSS), `DevianceScore`
  (deviance skill + null/climatology reference).
- **deter_architecture.py**: docstring `PlattScaling`, `HistogramBinning`; state `num_bins`
  (default 44) must equal the 44-entry `bin_edges` list; note it's hardcoded regardless of a
  `num_bins` override.
- **models.py**: docstring the wrapper classes; document the masking convention once — trigger is
  `len(remove_vars) == 5` (remove all five features → replace input with a single all-ones
  channel), `len < 5` drops those channels (NOT "index 5").
- **dataset.py**: docstring `create_train_test` / `get_seasons`; explain the 2008-starts-a-day-late
  special case; promote the `5843` magic number (full-dataset day count; demo ships only 49 days)
  to a named constant with a comment.
- **plots.py**: class docstrings for `LightningPlotMultiModel` / `SaliencyPlotPaper` (both do all
  work as an `__init__` side effect); name/explain layout magic numbers
  (`set_extent([-5,20,30,55])`, colourbar offsets).
- **common_path.py**: document that `ROOT_PATH = Path.cwd().parents[0]` is CWD-dependent (assumes
  running from a first-level subdir like `notebooks/`).
- **notebooks**: add a concluding/summary markdown cell to each; add a markdown intro to nb03's
  saliency section explaining the "pick day → pick lat/lon/hour → plot" flow.

---

## Verification (run after each step; full pass at the end)

- Execute all three notebooks top-to-bottom **from `notebooks/`** with the `lightning_modelling`
  kernel/venv. Confirm nb02 reproduces the Table-5 metrics over the 49 extreme days (set
  `EARLY_STOP=False` for the full run), nb01/nb03 render figures without errors.
  - nbconvert is not installed in the venv; to execute cells use an exec-based runner with
    `/homedata/aburq/.venvs/lightning_modelling/bin/python` and strip `%` magics.
- `ruff` / `pyflakes` over `src/` after Steps 4-6 to confirm no dead symbol removed a live use.
- Grep sweep after any rename/extraction to confirm all references updated.
