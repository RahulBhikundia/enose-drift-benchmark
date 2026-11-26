# Does Flatness Help Gas Sensors Survive Drift?

CSL7620 Machine Learning course project, IIT Jodhpur, November 2026.
Report (CVPR format): [`reports/final_report_cvpr.pdf`](reports/final_report_cvpr.pdf). Longer version with more explanation: [`reports/final_report.pdf`](reports/final_report.pdf).

The project is a leakage-free benchmark of sharpness-aware optimizers (SAM, MSAM, SAGM, DISAM and our SA-MSAM variant), domain-generalization methods, few-shot recalibration and label-free test-time adaptation under three real kinds of electronic-nose drift:

| Dataset | Drift type | Held-out unit |
|---|---|---|
| UCI Gas Sensor Array Drift (#224) | sensor ageing (10 batches, 36 months) | later batches |
| SmellNet-Base | session to session (6 sessions) | one session |
| UCI Twin Gas Sensor Arrays (#361) | device to device (5 units) | one unit |

## Main findings

1. No sharpness-aware or domain-generalization method reliably beats AdamW.
2. MSAM costs the same as SGD but is unstable.
3. SA-MSAM does not beat a random-direction control.
4. Validation-based model selection is unreliable under drift.
5. The remedy that works depends on the drift type. Device and session shift can be fixed with label-free test-time adaptation; ageing needs a few labelled samples from the new period.

## Repository layout

```
src/                     library code
  data_uci.py, data_smellnet.py, data_twin.py   loaders + features (one scaler fit on training domains only)
  models.py              MLP (UCI, Twin), 1-D CNN and GRU (SmellNet)
  optim.py               SAM, MSAM, SA-MSAM, SAGM, DISAM
  sharpness.py           sharpness measures
  train.py               one training run -> one JSON file (the unit everything else builds on)
scripts/
  data/                  download_data.py (fetch + checksum), leakage audits (audit_*.py)
  experiments/           tuning grid, HP selection, frozen test runs, selection study, few-shot, TTA, GRU/feature
                         ablation, shift diagnosis, export_checkpoints.py
  analysis/              tables, bootstrap CIs, Wilcoxon + Holm, Friedman
  figures/               the four report figures
  report/                builders for reports/final_report.pdf
  orchestration/         Windows launchers used for the long unattended runs (restart-safe chains)
tests/                   test_optim.py: our MSAM matches the official implementation step by step
results/                 every run as JSON (tuning/, final/, selection/, fewshot/, tta/, ...), analysis/ tables, logs
checkpoints/             seed-0 weights of SGD/AdamW/SAM/MSAM for every split + manifest.csv
figures/                 report figures (PDF + PNG) and audit plots
notes/                   decisions.md (dated decisions, pre-registrations), progress_log.md, audit_findings.md
reports/                 final_report_cvpr.pdf (submission), cvpr/ (LaTeX source), final_report.pdf
data/README.md           dataset sources, versions, checksums, assumptions
environment.yml          conda environment;  requirements.txt (ranges)  /  requirements.lock.txt (exact versions used)
```

## Reproducing

### 1. Environment

Tested on Windows 11, Python 3.12.8, PyTorch 2.5.1 + CUDA 12.1, one NVIDIA GTX 1650 (4 GB). Every run also works on CPU (slower).

```bash
conda env create -f environment.yml && conda activate enose-drift
# or: python -m venv .venv && .venv/Scripts/pip install -r requirements.lock.txt
```

### 2. Data

The datasets are not in the repository (about 3 GB). All three are public; the script downloads them and checks the SHA-256 checksums listed in `data/README.md`.

```bash
python scripts/data/download_data.py
python scripts/data/audit_uci.py; python scripts/data/audit_smellnet.py; python scripts/data/audit_twin.py   # leakage audits
```

### 3. Quick check (about 15 minutes)

```bash
python -m src.train --dataset uci --split main --opt msam --hp '{"lr":0.03,"rho":1.0}' --epochs 24 --seed 0 --out run.json
python scripts/experiments/export_checkpoints.py --load   # reload the shipped checkpoints and re-score them on the test splits
git clone https://github.com/MarlonBecker/MSAM external/msam_code && python -m pytest tests   # MSAM vs official code
```

### 4. Full pipeline (about 18 GPU-hours of training, 7,194 runs)

The tables and figures can be rebuilt from the shipped `results/` without retraining (step 5).

```bash
python scripts/experiments/run_grid.py --stage 1          # tuning: 12 configs x 3 seeds per method, validation only (stages 1-4)
python scripts/experiments/select_hp.py --stage 1         # freeze selections -> results/tuning/selected_stage*.json
python scripts/experiments/run_final.py                   # frozen test runs, 5 seeds (UCI, SmellNet)
python scripts/experiments/run_twin.py --phase tune; python scripts/experiments/run_twin.py --phase tune2   # Twin: tune
python scripts/experiments/run_final.py --datasets twin twinreg --seeds 0 1 2 3 4 5 6 7 8 9                 # Twin: 10 seeds
python scripts/experiments/baseline_rf.py
python scripts/experiments/run_selection_study.py
python scripts/experiments/run_fewshot.py --mode tune; python scripts/experiments/run_fewshot.py --mode final
python scripts/experiments/shift_diagnosis.py
python scripts/experiments/run_tta.py --mode tune; python scripts/experiments/run_tta.py --mode select
python scripts/experiments/run_tta.py --mode final --seeds 0 1 2 3 4; python scripts/experiments/run_tta.py --mode report
python scripts/experiments/run_smellfeat.py smellfeat; python scripts/experiments/run_smellfeat.py smellgru
python scripts/experiments/tta_sensitivity.py
```

The exact order and arguments used for the reported runs are in `scripts/orchestration/after_*.cmd` and `notes/progress_log.md`. Finished runs are skipped on restart.

### 5. Tables, statistics, figures, report

```bash
python scripts/analysis/analyze_final.py; python scripts/analysis/stats_tests.py; python scripts/analysis/analyze_selection.py
python scripts/analysis/compare_smellfeat.py
python scripts/figures/fig_main.py; python scripts/figures/fig_fewshot.py; python scripts/figures/fig_tta.py
cd reports/cvpr && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Checkpoints

`checkpoints/<dataset>/<split>/<opt>_s0.pt` holds the weights of the seed-0 model of SGD, AdamW, SAM and MSAM for every held-out split (97 files; 3 original runs diverged; about 20 MB in total, small enough to keep in the repository). Each file stores the `state_dict` and the exact arguments; `checkpoints/manifest.csv` lists size, SHA-256 prefix, the score in the report and the score of the re-trained checkpoint. Every other model in the study is small (35k to 67k parameters, seconds to minutes to train) and is fully specified by its JSON in `results/`, so it can be re-trained with the one-line command above instead of being stored.

## Reproducibility statement

- **Protocol.** Every method gets the same tuning budget (12 configurations x 3 seeds) scored on validation data only. Selections are frozen in `results/tuning/selected_*.json` before any test run; each method is then tested once with 5 seeds (10 on Twin). Statistics treat the held-out split as the unit of analysis. Scalers are fit on training domains only; methods that use unlabelled target data are labelled transductive.
- **Artifacts.** Every training run (7,194) is stored as a JSON with its arguments, seed, versions, GPU, timing, learning curve, per-domain metrics and test predictions. All tables and figures are regenerated from these files.
- **Determinism.** Seeds are fixed for PyTorch and NumPy; cuDNN kernels are not forced to be deterministic. Re-training the 97 checkpointed runs (`checkpoints/manifest.csv`) reproduces all 73 MLP runs (UCI, Twin) exactly. The SmellNet CNN differs by a median of 0.018 macro-F1, below the seed-to-seed spread, and one MSAM run collapsed on re-training (0.733 to 0.187), the instability the report describes. Claims are therefore made over seeds and splits, never from a single run.
- **Data.** Public datasets with pinned versions and checksums (`data/README.md`). Two ordering assumptions (SmellNet session order, Twin repetition order) are stated there and in the report.
- **Declaration.** We declare that all results in the report were produced by the code in this repository, on the data described above, with the protocol described above, and that no test-set result was used to choose a method, a hyperparameter or an epoch count.

