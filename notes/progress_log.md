# Progress log

Newest first. If a session restarts, read this file, `notes/decisions.md` and `notes/audit_findings.md` to resume.

## 2026-11-12 (early hours)
- [x] Stage-2 tuning done; `scripts/select_hp.py --stage 2` written to `results/tuning/selected_stage2.json`
- [x] `scripts/go_nogo.py` writes the validation-only comparison table to `results/tuning/go_nogo.csv`
- Result: no SA-MSAM variant beats MSAM on either dataset (UCI −0.007 to −0.017; SmellNet −0.012 to −0.016), and the extrapolated and disagree variants don't beat the random-direction control. Recency weighting is clearly best on SmellNet (0.855 ± 0.003 vs MSAM 0.817 ± 0.024); on UCI everything is within noise (0.926–0.947)
- [x] User decided NO-GO: SA-MSAM is reported as a negative result / ablation (see decisions.md)
- [~] RUNNING (detached, started ~00:20): `run_final.py`, 5 seeds × all splits. Core methods (sgd adamw sam msam recency gi, 450 jobs) go to `results/final_core.log`, then the SA-MSAM ablations (3 methods) go to `results/final_samsam.log`. Resumable: rerunning skips finished JSONs
- [x] `scripts/analyze_final.py` writes tables to `results/analysis/` and scatter plots to `figures/final_sharpness_vs_f1_*.png`. First look at 01:45 (core methods 425/450; SA-MSAM not run yet):
  - UCI: MSAM has the best average (0.781) and mean rank; the other methods are 0.753–0.764
  - SmellNet LOSO: recency, the best on validation, ranks near the bottom on test (0.769), and MSAM is the lowest (0.754). Validation→test reversal to investigate. "Recency" is ill-defined when the held-out session sits in the middle (loso1–5)
  - Sharpness vs test F1 within split: weakly negative for the Hessian and adversarial measures (flatter is better, ρ ≈ −0.25 to −0.43), inconsistent across splits; random-perturbation sharpness shows no relationship
- [x] 02:30: core runs finished (450/450); SA-MSAM at 215/225. Full SmellNet core results:
  - MSAM collapses: avg 0.704, last in rank. On loso6 the seeds score 0.19 / 0.00 / 0.70 / 0.61 / 0.76, yet `diverged` is False. The selected SmellNet config (lr 0.03, ρ 0.3) is unstable; validation SD (0.024) already hinted at this
  - Recency is also bad on loso6 (0.725 vs SGD 0.779), so the "middle held-out session" explanation does NOT explain it. Recency overfit the validation split (γ = 0.3 puts nearly all weight on session 4, which suits val session 5)
  - SA-MSAM disagree/random (lr 0.01) beat MSAM (lr 0.03) on SmellNet test, most likely because of the lower lr, not the direction. Don't claim it as a method win
- [ ] TODO: report failed runs (F1 near 0) explicitly, with median + IQR alongside mean ± sd; add a divergence check on final loss/accuracy
- [ ] NEXT: re-run analysis when SA-MSAM completes; then (per-split test macro-F1, sharpness vs shift-robustness correlation), RF + TabPFN baselines, mixed-batch ablation

## 2026-11-11 (evening)
- [x] SmellNet protocol checked (`external/smellnet_code`, read only). Session suffix = "day" in their code; windows 50/100/500 with stride = window/2; real-time test data NOT released. Our RF gets 82–83% on the official split vs the paper's 58.5% (verify with our CNN)
- [x] `src/optim.py`: SAM, MSAM, SA-MSAM (extrap / disagree / random). `tests/test_optim.py` all pass; MSAM matches the official code to 1e-10
- [x] `src/models.py`, `src/sharpness.py`, `src/train.py` (one run → one JSON)
- [x] Pilots: UCI and SmellNet runs take seconds each
- [~] RUNNING: stage-1 tuning (`scripts/run_grid.py --stage 1`, log in `results/tuning_stage1.log`), then `scripts/select_hp.py --stage 1`
- [x] Baselines added to train.py: `--method recency` (RIDER-style) and `--method gi` (GI-style finite-difference variant, time input); `--sampler mixed` for the ablation
- [x] Stage-2 grid now = SA-MSAM ×3 + recency + gi (12 configs each; baselines on AdamW)
- [ ] NEXT: stage-2 tuning, then the go/no-go on validation (ask the user)
- [ ] TODO baselines: RF (UCI + SmellNet), Drift-Resilient TabPFN (UCI); mixed-vs-domain batch ablation
- [ ] Final runs: 5 seeds × (UCI main + roll2..9, SmellNet loso1..6) with --sharpness

## 2026-11-11 (afternoon)
- Environment, data, loaders, audits, decisions (see decisions.md and audit_findings.md)

## 2026-11-12 ~02:40: final runs COMPLETE (675/675)
- Full tables in `results/analysis/analysis_full.txt` and `results/analysis/table_test_f1_*.csv`
- UCI avg test F1: MSAM is best, 0.781 (rank 2.9); the others are 0.736–0.764
- SmellNet avg: SA-MSAM random 0.835, disagree 0.826, SGD 0.824, SAM 0.822 ... recency 0.762, MSAM 0.704 (last)
- 4 runs collapsed (F1 < 0.3), none flagged as diverged: smell loso6 MSAM s0 and s1; uci roll8 samsam_disagree s2 and samsam_extrap s3
- Sharpness–F1 within split: adversarial ρ = −0.39 (smell, 4/6 sig), −0.20 (uci, 4/9 sig); Hessian weaker; random none
- [~] RUNNING: sensitivity check, MSAM on SmellNet at lr 0.01 (ρ 0.3, 62 epochs), 5 seeds × loso1–6, in `results/sensitivity/`. Reported separately; frozen selections are NOT changed
- [x] 02:50: sensitivity check DONE. MSAM on SmellNet at lr 0.01 averages 0.833 test F1 (loso1–6: 0.618/0.895/0.910/0.960/0.814/0.804), 0 collapses in 30 runs, vs 0.704 at the selected lr 0.03. Conclusions: (1) MSAM's SmellNet failure is learning-rate instability, not the method; (2) the SA-MSAM "win" (random 0.835) is explained by its lower lr: the matched-lr MSAM scores the same. The primary results stay as frozen (validation-selected); this is reported as a post-hoc sensitivity analysis and disclosed as such
- [x] Implemented stage-3 methods (src/optim.py: SAGM, DISAM; src/train.py: --norm domain, coral, groupdro, collapse check). Old methods reproduce bit-for-bit (msam/sam/recency/gi checked); optimizer tests pass; every new method smoke-tested. Backups: src/*.py.bak
- [~] RUNNING (detached): `run_grid.py --stage 3` (432 jobs) then `select_hp.py --stage 3`, logged to `results/tuning_stage3.log`
- [~] Chain `scripts/after_stage3.cmd` (detached) waits for `selected_stage3.json`, then runs the final test runs for the 6 new methods (`results/final_stage3.log`), the RF baseline (`results/final_rf.log`) and `analyze_final.py`, and writes ALLDONE. NOTE: detached jobs must be started from the PowerShell tool (Start-Process); launching from Git Bash gets killed when the shell exits
- [x] analyze_final.py now reports median [IQR] and collapse counts (`table_robust_*.csv`)
- [ ] TODO: Drift-Resilient TabPFN (UCI), a SAGM unit test, and checking go_nogo for stage 3 once tuning finishes
- [x] 03:40 overnight safety: `scripts/watchdog.ps1` (detached, log `results/watchdog.log`) restarts stage-3 tuning or the follow-up chain if either dies (max 3 restarts each) and exits on ALLDONE. AC sleep/hibernate = Never; on AC power; Windows Update active hours set to 3–21 (restart pending!)
- Morning outputs (expected ~9 AM): `results/analysis/validation_all_methods.txt` (validation comparison, ~5:45), `results/analysis/analysis_full.txt` (all test tables), `results/analysis/table_robust_*.csv`, `figures/final_sharpness_vs_f1_*.png`

## 2026-11-12 morning: stage 3 COMPLETE (ALLDONE 07:08; no crashes; 1200 final runs)
- SmellNet test avg: RF 0.874 (window-summary features) > CORAL 0.853 (best deep method; also best on val, 0.847) > SA-MSAM random 0.835 > the rest at 0.80–0.83; MSAM 0.704, recency 0.762. domnorm did NOT help (0.806)
- UCI test avg: MSAM 0.781, domnorm 0.772 (best median, 0.792), the rest 0.75–0.77; DISAM 0.689 (4 collapsed runs; selected lr 0.1); RF 0.729 (0.094 on batch 8)
- Validation → test is unreliable on UCI: GroupDRO (val 0.964) and DISAM (val 0.959) were the top two on validation and are mid / last on test
- Sharpness–F1 weakens with more methods: adversarial ρ −0.21 (smell) / −0.15 (uci); Hessian ≈ 0 on SmellNet
- Emerging story: under real sensor drift no method reliably beats ERM; validation-based selection under shift is unreliable; sharpness is only weakly predictive (cf. Gulrajani & Lopez-Paz 2021, "In Search of Lost DG")
- [ ] NEXT: paired significance tests vs ERM; val-vs-test scatter figure; TabPFN; report outline

## 2026-11-12 late morning: Phase A started
- [~] Stage 4 (CC-PCA, drift augmentation, SWA): chain `scripts/after_stage4.cmd` + `scripts/watchdog4.ps1`, started 10:50
- [~] Selection study (`scripts/run_selection_study.py`, 1152 runs: all 12 configs of 8 methods on the main test split + a 2nd val fold "tune2"): queued after stage 4 via `scripts/after_selection.cmd` + `scripts/watchdogsel.ps1`. Not run in parallel: only 0.8 GB RAM free
- [x] `scripts/stats_tests.py` (per-split units, bootstrap CI, Wilcoxon + Holm, Friedman), results in `results/analysis/stats_vs_adamw_*.csv`:
  - SmellNet: CORAL beats AdamW on 6/6 splits (+0.043, CI [0.032, 0.056]); RF 6/6 (+0.064). MSAM is worse on 6/6. Holm p ≥ 0.47 is a structural floor (n=6, 15 tests): report CIs and win counts
  - UCI: no method's CI excludes 0 except DISAM (worse, −0.075). Friedman p = 0.045, driven by RF and DISAM
- [x] Stage-4 validation selections: SmellNet CC-PCA 0.858 (best val of any method so far), driftaug 0.797, SWA 0.789; UCI CC-PCA 0.940, driftaug 0.944, SWA 0.923. Final runs in progress (`results/final_stage4.log`)
- [x] Phase B few-shot (`scripts/run_fewshot.py`): adaptation hps tuned on val and frozen in `results/fewshot/selected.json`. Validation: UCI none 0.911 → ft_all 0.980 (k=5/class); SmellNet none 0.763 → ft_all 0.899 (first 2 windows). Affine HURTS on SmellNet (0.595). Hypothesis: the first windows are the odour onset (transient), so their class means don't match steady-state means. Check later
- [~] RUNNING (detached): few-shot final, 15 splits × 5 seeds, writes `results/fewshot/final_*.csv`, log `results/fewshot_final.log`

## 2026-11-12 afternoon
- [x] Stage 4 final done. Test vs AdamW: CC-PCA SmellNet −0.065 (0/6; the best val score of any method, 0.858 → 2nd-worst on test!), UCI +0.049 (7/9; best avg rank). driftaug: SmellNet −0.030 (0/6), UCI ≈ 0. SWA degenerate (averaging never started; UCI SWA == AdamW) → `swa_full` fix (see decisions)
- [x] Twin dataset: download (sha256 E8801DE3...), `src/data_twin.py` (resampled cache `data/processed/twin_raw.npz`, features `twin_features.csv`), audit `scripts/audit_twin.py` → `results/audit_twin/summary.txt`
- [x] train.py: datasets twin/twinreg, regression path (MSE, R²/RMSE/MAE, `score` key); UCI still bit-exact; backups src/*.bak3
- [~] Running (detached + watchdogs): selection study (~19:00), few-shot final (~13:20), Twin pipeline after few-shot (`results/twin_pipeline.log`, ~19:15), swa_full after the selection study (~19:30)
- [ ] TODO: swa_full for twin/twinreg (rerun select_swa_full.py after twin tuning, then run_final --datasets twin twinreg --methods swa_full --seeds 0..9); few-shot for Twin; selection-study analysis script; RF-vs-CNN on SmellNet features; shift diagnosis; figures

### 2026-11-12 15:20 status
- Few-shot final: 71/~75 runs (last split smell loso6), finishing ~15:50. Twin chain auto-starts after it.
- Selection study: 485/1152, slowed to ~85 runs/h on SmellNet CNN under RAM pressure (~230 MB free; a VS Code Jupyter kernel is also open). Revised ETA ~23:00-00:00.
- swa_full finals wait for selection ALLDONE -> ETA ~01:00-02:00 Nov 13. Twin pipeline ETA ~22:00-23:00.

### 2026-11-12 17:00 status
- Few-shot final DONE (15:5x). Twin tuning + tune2 DONE; Twin final 571/1500 (fast, ETA ~18:45).
- Twin validation: twin (classification) saturated (val 0.98-1.00 for most methods) -> little room to separate methods;
  twinreg: domnorm val R2 0.888 vs 0.62-0.71 for all others (largest single-method effect seen on validation anywhere).
- Re-ran select_swa_full.py (now includes twin/twinreg; uci/smell picks unchanged). Added scripts/after_swatwin.cmd
  (waits for twin ALLDONE + final_swafull ALLDONE, then swa_full twin/twinreg 10 seeds, then analysis + stats). Launched PID 6776.
- Selection study 914/1152, ETA ~17:45; swa_full UCI/SmellNet ETA ~19:30; everything ETA ~20:00.

### 2026-11-12 22:10 ALL QUEUED RUNS DONE (selection study, few-shot final, Twin 1500 + RF, swa_full all 4 datasets, analysis + stats)
Test, mean diff vs AdamW [wins/splits]:
- SmellNet: RF +0.064 [6/6], CORAL +0.043 [6/6], swa_full +0.031 [6/6] (SWA fix works), SA-MSAM random +0.025 [6/6]; MSAM -0.106, CC-PCA -0.065.
- UCI: nothing reliable; CC-PCA +0.049 [7/9]; DISAM -0.075.
- Twin (gas): saturated; MSAM / SA-MSAM collapse on test (-0.24..-0.26, 0/5) despite val 0.996-1.000 -> another val->test reversal.
- Twinreg (R2): domnorm +0.224 [4/5, CI 0.05-0.41, avg rank 2.2] -> validation finding HOLDS on test; RF +0.126; CORAL +0.031 [5/5].
  CAVEAT: domnorm standardizes each unit with its own (unlabeled) statistics -> transductive; needs an unlabeled batch from the new device. Must be stated.
TODO: twinreg samsam_disagree/random mean diffs are -2714 / -7634: a few seeds produce exploding regression outputs (test R2 << 0) that the
  collapse rule (classification-only F1 threshold) does not catch -> report medians / add a regression divergence rule (e.g. test R2 < -1 or non-finite).
TODO: analysis of selection study + few-shot results; stats label says "macro-F1" for twinreg (should be R2).

### 2026-11-13 first look at few-shot final + selection study (scripts/selection_quicklook.py; preliminary)
- Few-shot (test): fine-tuning on a few labelled samples from the new batch/session is the biggest gain in the project.
  UCI none 0.754-0.764 -> ft_all 0.854/0.929/0.952 (k=1/5/10), wins 23/27; SmellNet 0.825-0.839 -> ft_all 0.881/0.917/0.944 (k=1/2/4), 18/18.
  Affine recalibration: no help on UCI, hurts SmellNet (0/18).
- Selection study, Spearman(val, test) across the 12 configs within each method, main test split:
  UCI: adamw -0.97, recency -0.67, disam -0.49, groupdro -0.43, coral -0.04, sgd 0.28, sam 0.43, msam 0.86. Regret (best test - test of val-chosen) up to 0.13.
  SmellNet: mostly positive (sam 0.93, disam 0.91, msam 0.84) but coral -0.07, recency -0.08; msam regret 0.148.
  Second validation fold (fold2) agrees with fold1 far better than test does -> extra val folds do not fix it.

### 2026-11-13 analysis fixes + selection study analysis
- analyze_final.py: twinreg runs whose TRAINING failed (diverged / train R2 < 0.5) now scored R2 = 0 (= predicting the train mean),
  raw value kept in test_f1_raw. Runs that trained fine keep negative test R2. Twinreg SA-MSAM diffs now -0.06/-0.12 (were -2714/-7634).
  stats label says "test R2" for twinreg. UCI/SmellNet/Twin numbers unchanged. Old outputs kept as *_pre_regfix.txt.
- scripts/analyze_selection.py -> results/analysis/selection_summary.csv, selection_full.txt, figures/selection_val_vs_test.{png,pdf}.
  Test = ONE split per dataset (UCI main, SmellNet loso6), 12 configs x 3 seeds, 8 methods.
  Mean over methods: Spearman(val,test) UCI -0.13 (5/8 methods negative; AdamW -0.97), SmellNet +0.55 (+0.39 among configs that trained).
  Second validation fold agrees with the first (UCI +0.34, SmellNet +0.77) better than test does.
  Regret (oracle test - test of picked config): val rule UCI 0.066 / SmellNet 0.040; random pick 0.102 / 0.163 (inflated by diverging configs).
  Validation avoids catastrophic configs but cannot rank the working ones on UCI; on UCI AdamW/DISAM/GroupDRO the val pick is worse than a random pick.
  Risk-averse rule (mean - sd over seeds) fixes SmellNet (0.040 -> 0.013; MSAM 0.148 -> 0.004) but not UCI (0.076). Two-fold rule: no help.

### 2026-11-13 ~04:00 Twin few-shot, shift diagnosis, main figure
- run_fewshot.py extended to twin/twinreg (shots = k random recordings PER GAS from the held-out unit; 10 seeds; adaptation hp
  tuned on Twin validation, merged into results/fewshot/selected.json, UCI/SmellNet entries verified unchanged; backups *_pre_twin.*).
  Twin gas ID: none 0.91 -> affine 0.97-0.98, ft_last 0.97-0.985 (k=1-5). Twinreg R2: none 0.55 -> affine/ft 0.67-0.69 at k=5;
  at k=1 ft_all hurts (0.45), ft_last/affine help. Unlike SmellNet, AFFINE recalibration works on Twin.
- scripts/shift_diagnosis.py (oracle per-feature corrections using test labels, diagnostic only; seeds 0-2):
  twin    none 0.911  standardize(label-free) 0.965  offset_oracle 0.952  affine_oracle 0.993
  twinreg none 0.576  standardize 0.692  offset_oracle 0.501  affine_oracle 0.713
  uci     none 0.759  standardize 0.694  offset_oracle 0.785  affine_oracle 0.693
  => Twin unit-to-unit shift is almost entirely per-feature gain/offset (calibration-type); UCI ageing drift is NOT
     (even an oracle affine fit with all test labels does not help). Explains why domnorm wins on Twin only.
  SmellNet (+ first-window vs random-window affine -> onset-transient test) running detached (results/shift_smell.log).
- scripts/fig_main.py -> figures/main_effects.{png,pdf}: forest plot, difference from AdamW with 95% CI + win counts, 4 datasets.

### 2026-11-13 ~05:00 SmellNet shift diagnosis done + features-vs-model experiment started
- SmellNet: none 0.836, standardize 0.802, offset_oracle 0.796, affine_oracle 0.796 (0/6 splits improve),
  affine from first 4 windows 0.738 vs from 4 random windows 0.802.
  => SmellNet session shift is NOT per-feature gain/offset (even the oracle with all labels hurts). Onset-transient
     hypothesis partly supported: first windows are worse than random windows (-0.064), but random windows don't help either.
  => Summary across datasets: Twin (device-to-device) = calibration-type shift, fixed by per-feature gain/offset (and by
     label-free standardization); UCI (ageing) and SmellNet (sessions) are not -> need retraining (few-shot fine-tuning).
- Bug fixed: shift_diagnosis.csv had mixed column counts (SmellNet 2 extra cols); repaired (raw kept as
  shift_diagnosis_raw_mixed.csv), script now writes fixed columns.
- NOTE: SmellNet has 50 substance classes (not 6). run_smellfeat.py first crashed on this (MLP n_cls=6); fixed to 50.
- scripts/run_smellfeat.py: dataset "smellfeat" = SmellNet windows -> 30 RF summary features -> MLP; AdamW + CORAL, same
  12x3 tuning, frozen loso1-6 x 5 seeds. Question: is RF's SmellNet win the features or the model? Running (~25 min).
- figures/fewshot.{png,pdf} added.

### 2026-11-13 features vs model on SmellNet DONE (scripts/compare_smellfeat.py)
Test macro-F1 mean over 6 splits (diff vs CNN-raw AdamW, wins/6):
  CNN raw AdamW 0.810 | CNN raw CORAL 0.853 (+0.043, 6/6) | MLP 30 feat AdamW 0.842 (+0.032, 6/6) |
  MLP 30 feat CORAL 0.866 (+0.056, 6/6) | RF 30 feat 0.874 (+0.064, 6/6)
=> Half of RF's lead is the summary features (MLP on the same features closes +0.032 of +0.064); feature summary and
   CORAL stack (+0.056), nearly matching RF. "Simple features + simple alignment" ~ RF; the raw CNN is the weak baseline.
STATUS: all items on the "still to do" list are done except optional (more seeds, 4th dataset) and writing.

### 2026-11-13 late: diagnose-then-adapt (user's choice for remaining experiment time)
- Lit check: SAM already combined with domain adaptation for e-nose (Sensors 2024, CDAN+SAM) -> our "first SAM on e-nose"
  claim must be narrowed; label-free test-time adaptation (TTA) for e-nose drift looks open.
- scripts/run_tta.py: TTA on the frozen AdamW model: standardize, feat_align, ent_affine (IM-trained input gain/offset),
  shot_im (SHOT-style IM feature adaptation); fewshot ft_all reference; seeds 0-2. hp from validation (results/tta/selected.json).
- Distribution-level shift diagnostics FAILED on validation (linear / RF / excess-shift all uninformative; logged in decisions.md).
  Model-based criteria (IM change, cross-seed agreement change) evaluated instead; on validation the agreement criterion is
  fooled on UCI (standardize -> seeds agree more but are wrong).
- Validation lead: shot_im SmellNet 0.710 -> 0.825 (label-free!), ent_affine 0.765; standardize best on Twin; nothing label-free helps UCI.
- RUNNING: run_tta.py --mode final (all test splits, 3 seeds) -> results/tta/final.csv, then report -> results/tta/report_final.txt.

### 2026-11-14 00:05 TTA final DONE (results/tta/final.csv, criteria_final.csv, report_final.txt; 3 seeds; hp frozen from validation)
Label-free test-time adaptation vs none (mean diff over splits [95% bootstrap CI], wins):
- SmellNet: shot_im +0.069 [+0.049,+0.087] 6/6 ; ent_affine +0.042 6/6 ; standardize -0.033 0/6 ; fewshot(k=4 labelled) +0.096 6/6
- Twin gas: shot_im +0.070 [0.000,0.144] 4/5 ; feat_align +0.073 ; fewshot(k=2) +0.047 -> label-free BEATS 2-shot labelled (4/5)
- Twin conc.: feat_align +0.121 [+0.025,+0.218] 5/5 ; fewshot(k=2) -0.096 (2 recordings/gas too few for regression)
- UCI: shot_im +0.027 [+0.001,+0.053] 5/9 ; standardize -0.073 ; fewshot(k=5) +0.176 8/9 -> ageing drift still needs labels
IM criterion (apply standardize iff IM objective drops): correct on 6/6 SmellNet, 5/5 Twin domains, 54% UCI.
=> HEADLINE CANDIDATE: label-free TTA (SHOT-style information maximization) recovers ~70% of the few-shot gain on day-to-day
   drift and matches/beats few-shot on device-to-device drift, with zero labels; ageing drift (UCI) is the exception.
TODO: more seeds (5) for TTA; add TTA to report/figures; lit check on TTA for gas sensors.

### 2026-11-14 01:35 overnight: TTA rerun with 5 seeds (user request)
- run_tta.py --seeds option added; final mode resumes (seeds 0-2 kept, adds 3-4). 3-seed outputs backed up as
  results/tta/final_3seeds.csv, report_final_3seeds.txt. Chain: final -> report (results/tta/report_final.txt) -> ALLDONE in
  results/tta_final5.log. Watchdog scripts/watchdogtta.ps1 (restart up to 3x; log results/watchdogtta.log). ETA ~03:00-03:30.
- Plan agreed in spirit: freeze new experiments; next = understanding sessions with the user (see memory).

## 2026-11-14 14:25 — 5-seed TTA rerun finished (ALLDONE 01:45; watchdog exited cleanly, no restarts)
- results/tta/report_final.txt (5 seeds) vs report_final_3seeds.txt: headline numbers stable within ~0.01 except twinreg feat_align (0.736 -> 0.712).
- 5-seed means (none -> best label-free / few-shot): smell 0.851 -> shot_im 0.918 (6/6) / 0.943; twin 0.910 -> feat_align 0.982, shot_im 0.978 / 0.962;
  twinreg 0.604 -> feat_align 0.712 (4/5) / 0.552; uci 0.763 -> shot_im 0.784 (5/13) / 0.929.
- IM criterion: still 6/6 smell, 5/5 twin, but UCI dropped 54% -> 31% (worse than chance). Agreement criterion UCI 38%.
- Conclusion unchanged: label-free TTA works for session (SmellNet) and device (Twin) shift, not ageing (UCI).

## 2026-11-14 17:30 — architecture check + TTA sensitivity (user-approved)
- Ablations already in hand were listed for the write-up (main: features-vs-raw, few-shot K, TTA ladder, shift diagnosis).
- NEW smellgru: GRUNet (2-layer GRU, hidden 64, time-mean pool) on the same raw SmellNet windows, same protocol as smellfeat
  (AdamW + CORAL, 12 configs x 3 seeds on tune, finals loso1-6 x 5 seeds). src/models.py, src/train.py (backups .bak5);
  scripts/run_smellfeat.py now takes DS argv. Running detached; watchdog scripts/watchdoggru.ps1 -> results/watchdoggru.log, results/gru.log.
- TTA sensitivity (scripts/tta_sensitivity.py -> results/analysis/tta_sensitivity.csv; validation only): shot_im gain on SmellNet stable for
  lr 1e-4..1e-3 (+0.09..+0.11), collapses at lr 1e-2 (-0.07). ent_affine safe for lr <= 1e-2. Twin validation unit saturates (+0.014 all).
  UCI: every TTA config HURTS on validation (-0.01..-0.08), consistent with "label-free TTA does not fix ageing".

## 2026-11-14 19:35 — GRU architecture check DONE (finished 18:23, no restarts)
SmellNet test macro-F1, mean over loso1-6 x 5 seeds: CNN 0.810 / +CORAL 0.853; GRU 0.820 (+0.010, 4/6) / +CORAL 0.856;
MLP-30feat 0.842 / +CORAL 0.866; RF 0.874. Architecture barely matters (GRU ~ CNN); CORAL adds ~+0.04 to either;
features matter more. Conclusions are not a CNN artifact. scripts/compare_smellfeat.py now includes GRU.

## 2026-11-15 — decision: try calibration-layer method (option 1) from 2026-11-17; pre-registered in decisions.md. Nothing running.

## 2026-11-26 — deadline 27 Nov: option 1 (calibration layer) DROPPED (never started; no time). Focus: final report, README, slides.

## 2026-11-26 — submission package
- reports/final_report.pdf (11 pp; scripts/make_final_report.py + scripts/final_report_body.py): academic structure (abstract, intro,
  related work, data/audit, methods, protocol, results 6.1-6.9 incl. cost table, TTA, GRU ablation, hypotheses H1-H4, limitations, refs).
  Author line is a placeholder "[Student name] · [Roll number]" -> user must fill. Ref [17] lacks authors.
- figures/tta.{png,pdf} (scripts/fig_tta.py). README.md written.

## 2026-11-26 — CVPR-format report: reports/cvpr/ (official cvpr-org author-kit, pagenumbers mode; main.tex, main.bib, fig/), compiled with MiKTeX -> reports/final_report_cvpr.pdf (7 pp: ~6 content + refs). Author block placeholders [Student Name]/[Roll Number]/[email].
