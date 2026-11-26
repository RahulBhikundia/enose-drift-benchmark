# Text of the final report; executed by scripts/report/make_final_report.py (helpers P, H1, H2, B, T, BOX, FIGURE).
# Numbers come from results/analysis/*, results/tta/report_final.txt, results/gru.log, results/analysis/tta_sensitivity.csv.

# ================================================================ title
P("Does Flatness Help Gas Sensors Survive Drift? A Leakage-Free Benchmark of Sharpness-Aware, Domain-Generalization "
  "and Test-Time Adaptation Methods under Real Electronic-Nose Drift", "title")
P("Rahul Bhikundia (M26CSA006) · Gourav (M26CSE018) · Jharendra Jogi (M26MAC005) · CSL7620 Machine Learning, IIT Jodhpur · November 2026", "auth")

BOX("<b>Abstract.</b> Metal-oxide gas sensors drift: they age, and two copies of the same device never respond "
    "identically, so a classifier trained on yesterday’s data degrades on tomorrow’s. Flat minima, found by "
    "sharpness-aware minimization (SAM) and its cheap variant Momentum-SAM (MSAM), are often claimed to generalize "
    "better under distribution shift. We test that claim on real drift. After auditing three public datasets that "
    "cover three drift types — sensor ageing (UCI Gas Sensor Array Drift), session-to-session shift (SmellNet) and "
    "device-to-device shift (UCI Twin Gas Sensor Arrays) — we compare about 20 methods (SGD, AdamW, SAM, MSAM, "
    "SAGM, DISAM, a drift-aware MSAM variant we propose, CORAL, GroupDRO, normalization, augmentation, SWA, a random "
    "forest) under one frozen protocol: equal tuning budget, selection on validation only, one test pass, 5–10 seeds, "
    "statistics over held-out splits. <b>Findings.</b> (1) No sharpness-aware or domain-generalization method "
    "reliably beats AdamW; MSAM is as cheap as SGD but unstable, and our drift-aware variant does not beat a "
    "random-direction control. (2) Validation-based model selection is unreliable under drift (four "
    "validation→test reversals; on UCI the validation and test rankings are slightly anti-correlated). (3) What does "
    "help depends on the drift type. An oracle per-sensor calibration diagnosis shows that device shift is "
    "calibration-type, and label-free test-time adaptation matches labelled recalibration there (Twin "
    "0.910→0.982 F1) and recovers most of it for session shift (SmellNet 0.851→0.918). Sensor ageing is different: "
    "only a few labelled samples from the new period help (UCI 0.763→0.929, label-free at most 0.784). We conclude "
    "that the remedy should be matched to the drift type, and give a simple diagnosis for doing so.")

# ================================================================ 1 intro
H1("1. Introduction")
P("An electronic nose (e-nose) is an array of chemical sensors whose joint response is a fingerprint of a gas. "
  "Over weeks and months the sensors’ baselines and sensitivities change (ageing, poisoning, humidity and "
  "temperature effects), and manufacturing tolerances make every unit respond slightly differently. A model "
  "trained on one period or one device therefore sees a shifted input distribution in deployment. Recalibrating "
  "sensors with reference gases is expensive, so a training-time fix would be valuable.")
P("Sharpness-aware minimization (SAM) [1] seeks weights whose loss stays low in a whole neighbourhood (a "
  "“flat” minimum), and flatness is frequently linked to robustness under distribution shift [4, 5]. SAM costs a "
  "second forward/backward pass per step; Momentum-SAM (MSAM) [2] perturbs along the momentum direction and is "
  "almost free. The project brief asked whether MSAM gives a favourable accuracy-versus-cost trade-off when a "
  "gas classifier is trained on earlier batches and tested on later ones, and stressed that the UCI benchmark is "
  "temporally confounded [12], so a careful audit had to come first.")
P("We broadened the question to all three drift types available in public data and to the method families a "
  "practitioner would actually try. <b>Contributions:</b>")
B(["A leakage-audited benchmark of the SAM family (SAM, MSAM, SAGM, DISAM), domain-generalization (DG) methods and "
   "classical baselines on three real drift types, under one frozen validation-only protocol (§3–§5).",
   "A drift-aware MSAM variant (SA-MSAM) evaluated against a random-direction control, with a pre-registered "
   "go/no-go decision; it is a negative result (§6.3).",
   "Evidence that validation-based selection is unreliable under drift and that sharpness barely predicts robustness "
   "(§6.4, §6.5).",
   "A shift diagnosis based on oracle per-sensor calibration, and a controlled comparison showing that the same "
   "label-free test-time adaptation (TTA) works for device and session shift but not for ageing (§6.7, §6.8).",
   "Ablations on features vs architecture (CNN, GRU, MLP on features, random forest) and on TTA hyperparameters "
   "(§6.9)."])

# ================================================================ 2 related work
H1("2. Related work")
P("<b>Sharpness-aware optimization.</b> SAM [1] minimizes the worst-case loss in an ℓ<sub>2</sub> ball around the "
  "weights. MSAM [2] replaces the gradient-ascent perturbation by the negative momentum direction, removing the "
  "extra pass. SAGM [4] aligns the gradients of the empirical and perturbed losses for domain generalization; "
  "DISAM [5] balances per-domain losses inside the perturbation. These methods were evaluated mostly on image DG "
  "benchmarks. SAM has been used for e-nose drift inside a conditional domain-adversarial network [17], but we "
  "found no head-to-head comparison with DG and adaptation methods across several real drift types.")
P("<b>Domain generalization and model selection.</b> CORAL [6] aligns second-order feature statistics; GroupDRO "
  "[7] up-weights the worst domain; SWA [8] averages weights along the trajectory. DomainBed [9] showed that, with "
  "a fair selection protocol, plain empirical risk minimization is hard to beat on images; our results confirm "
  "this on sensor data.")
P("<b>E-nose drift compensation.</b> Vergara et al. [10] introduced the UCI drift dataset with classifier "
  "ensembles; many domain-adaptation methods followed, e.g. domain-adaptation extreme learning machines [11] and "
  "adversarial approaches [17]. Dennler et al. [12] showed that the UCI data are confounded by acquisition time. "
  "Fonollosa et al. [13] introduced the Twin dataset and used direct standardization with labelled transfer "
  "samples. SmellNet [14] is a recent large real-world dataset. Fu et al. [18] report that drift-compensation "
  "rankings depend on chronology and on the availability of target data, in line with our selection results.")
P("<b>Test-time adaptation.</b> Tent [15] minimizes prediction entropy at test time; SHOT [16] adapts the feature "
  "extractor with an information-maximization (IM) objective while freezing the classifier. Online TTA has "
  "recently been proposed for e-nose drift [19, 20]. We do not propose a new TTA method; we ask on which drift "
  "types standard ones work, and why.")

# ================================================================ 3 data
H1("3. Datasets and leakage audit")
T([["", "UCI Gas Sensor Array Drift [10]", "SmellNet-Base [14]", "UCI Twin Gas Sensor Arrays [13]"],
   ["Drift type", "Sensor <b>ageing</b>, 36 months, 10 batches", "<b>Session</b> shift, 6 sessions on different days",
    "<b>Device</b> shift, 5 nominally identical units"],
   ["Input", "16 sensors × 8 features = 128", "6 channels, windows of 100 steps (≈1 Hz), stride 50",
    "8 sensors × 8 features = 64 (extracted by us)"],
   ["Task", "6 gases (macro-F1)", "50 substances (macro-F1)", "(a) 4 gases (macro-F1); (b) log-concentration (R²)"],
   ["Test splits", "Train 1–5 → test 6–10, plus 8 rolling one-step-ahead splits", "Leave one session out (6)",
    "Leave one unit out (5)"],
   ["Tuning split", "Train 1–3 → validate 4–5", "Train 1–4 → validate 5", "Train units 1–3 → validate unit 4"],
   ["Model", "MLP 128–256–128–6", "1-D CNN, 3 conv layers, width 64", "MLP 64(+4)–256–128–out"]],
  [2.4, 4.9, 4.6, 5.1])
P("We use all of SmellNet-Base (50 substances × 6 sessions = 300 recordings); its mixture and GC-MS subsets "
  "address other tasks. Before any method was run, each dataset was audited:")
B(["<b>Random splits leak heavily.</b> UCI: 99.4% accuracy with a random 80/20 split vs 47–77% when training on "
   "the past and testing on the future. SmellNet: 99.9% on random windows vs ≈82% on an unseen session, because "
   "overlapping windows of one recording fall on both sides. Twin concentration: R² 0.81–0.94 on random splits, "
   "negative on some unseen units. <b>All results below use domain-held-out splits</b>, with scalers fitted on "
   "training domains only.",
   "<b>Drift is strong and not monotonic.</b> A linear classifier separates UCI batch 1 from every later batch at "
   "97–100%, yet batch 10 moves back towards batch 1. SmellNet sessions and Twin units are identifiable from "
   "the readings at ≥99.5%.",
   "<b>Class coverage.</b> Toluene is absent from UCI batches 3–5, so macro-F1 is computed over the classes "
   "present in each test batch. The pre-exposure baseline needed for the test of Dennler et al. [12] is not in the "
   "128-feature release, so we do not claim to have run it.",
   "<b>Twin preprocessing.</b> 82 of 640 recordings are irregularly sampled; all were resampled to a uniform grid "
   "and gas on/off times were detected inside plausible windows, then checked on every file."])

# ================================================================ 4 methods
H1("4. Methods")
T([["Group", "Methods", "Idea"],
   ["Baselines", "SGD + momentum, <b>AdamW</b> [3] (reference), random forest", "Plain training; RF on summary features"],
   ["Sharpness-aware", "SAM [1], MSAM [2], SAGM [4], DISAM [5]", "Seek flat minima (SAGM/DISAM add domain terms)"],
   ["Proposed (ablation)", "SA-MSAM: extrapolated, disagreement and random directions", "Perturb along an estimated drift direction"],
   ["Domain generalization", "CORAL [6], GroupDRO [7], mixed-domain batches", "Align statistics / protect the worst domain"],
   ["Time-aware", "Recency weighting, time as input", "Weight or condition on time (UCI, SmellNet)"],
   ["Input-level", "Per-domain normalization*, CC-PCA, drift augmentation", "Standardize each domain; remove or simulate drift"],
   ["Weight averaging", "SWA [8] (early-stopped and full-length)", "Average weights along training"],
   ["Few-shot (labels)", "Affine recalibration, fine-tune last layer, fine-tune all", "k labelled samples from the new domain"],
   ["Test-time, label-free*", "Standardize, feature alignment, entropy input-affine, SHOT-IM [16]",
    "Adapt a frozen model on unlabelled target data"]],
  [3.1, 6.6, 7.3])
P("* Per-domain normalization and all test-time methods use unlabelled data from the target domain "
  "(transductive); they are reported separately from methods that never see it.", "cap")
P("<b>SA-MSAM.</b> MSAM perturbs the weights along the momentum. SA-MSAM instead perturbs along an estimate of the "
  "drift direction in gradient space: the difference between the average gradients of the newest and oldest "
  "training domains (extrapolated), or the direction of maximal disagreement between domains. A random unit "
  "direction of the same norm is the control.")
P("<b>Label-free test-time adaptation.</b> Given the frozen source model and the unlabelled samples of one target "
  "domain: <i>standardize</i> re-standardizes each input feature with the target mean and standard deviation; "
  "<i>feature alignment</i> matches the mean and standard deviation of the penultimate features to the source "
  "statistics; <i>entropy input-affine</i> learns a per-feature gain and offset on the input by minimizing the IM "
  "loss (mean prediction entropy minus entropy of the mean prediction); <i>SHOT-IM</i> [16] updates the feature "
  "extractor with the same loss and a frozen classifier. TTA hyperparameters (learning rate, steps) were chosen on "
  "the validation split only. The reference is few-shot fine-tuning of all layers with k labelled target samples "
  "(k = 5 per class on UCI, the first 4 windows per recording on SmellNet, 2 recordings per gas on Twin).")
P("<b>Shift diagnosis.</b> To ask what kind of shift each dataset has, we correct the test inputs with per-feature "
  "maps of increasing power, the strongest being an <i>oracle</i> gain + offset fitted with the test labels. It "
  "is a diagnostic upper bound for calibration-type correction and is never used for selection. If even the "
  "oracle calibration does not help, the shift is not a per-sensor calibration error.")

# ================================================================ 5 protocol
H1("5. Experimental protocol")
B(["<b>Equal tuning budget.</b> Every method gets 12 hyperparameter configurations × 3 seeds on the tuning split. "
   "The configuration with the best mean validation score is selected; the number of epochs is the median "
   "best-validation epoch.",
   "<b>Freeze, then test once.</b> Selected settings are frozen before any test run; every method is reported, "
   "including collapses. Test runs use 5 seeds (UCI, SmellNet) or 10 seeds (Twin) on every test split.",
   "<b>Fairness.</b> All SAM-family methods share an SGD-momentum base; our MSAM implementation matches the "
   "official code to 1e-10. No BatchNorm is used (it interacts with perturbed passes). Cost is reported as "
   "forward/backward passes and wall-clock time on one laptop GPU.",
   "<b>Statistics.</b> The unit of analysis is the held-out split, not the seed. Against AdamW we report the mean "
   "difference, a 95% bootstrap confidence interval over splits, the number of splits won, Wilcoxon signed-rank "
   "tests with Holm correction, and Friedman tests. Collapsed runs count (Twin-concentration runs whose training "
   "diverged are scored R² = 0, the score of predicting the mean).",
   "<b>Reproducibility.</b> Pinned environment; every run writes a JSON file with its configuration, learning curve "
   "and metrics; all decisions are dated in notes/decisions.md. In total ≈7,200 training runs."])

# ================================================================ 6 results
story.append(PageBreak())
H1("6. Results")
H2("6.1 No training-time method reliably beats AdamW")
FIGURE("main_effects.png",
       "Figure 1. Test score of each method minus AdamW, per dataset. Dot = mean over held-out splits; line = 95% "
       "bootstrap CI; filled = CI excludes zero; right column = splits where the method beat AdamW.")
T([["Method", "UCI (F1)", "SmellNet (F1)", "Twin gas (F1)", "Twin conc. (R²)"],
   ["AdamW (reference)", "0.764", "0.810", "0.913", "0.554"],
   ["SAM / MSAM", "0.759 / 0.781", "0.822 / <b>0.704</b>", "0.932 / <b>0.652</b>", "<b>0.234</b> / 0.470"],
   ["SA-MSAM (disagreement / random)", "0.736 / 0.759", "0.826 / 0.835", "0.659 / 0.679", "0.418 / 0.457"],
   ["SAGM / DISAM", "0.766 / <b>0.689</b>", "0.820 / 0.820", "0.946 / 0.922", "0.472 / 0.563"],
   ["CORAL", "0.763", "<b>0.853</b>", "0.942", "0.585"],
   ["Per-domain normalization*", "0.772", "0.806", "<b>0.977</b>", "<b>0.779</b>"],
   ["CC-PCA", "<b>0.813</b>", "0.745", "0.904", "0.473"],
   ["SWA (full length)", "0.763", "0.841", "0.917", "0.394"],
   ["Random forest", "0.729", "<b>0.874</b>", "0.960", "0.680"]],
  [5.0, 2.8, 3.0, 3.0, 3.2])
P("Table 1. Mean test score over all held-out splits and seeds (bold = notable best or worst). "
  "* transductive.", "cap")
B(["<b>UCI (ageing):</b> nothing reliably beats AdamW. CC-PCA has the best mean (+0.049, 7/9 splits) but its CI "
   "includes zero and it fails on SmellNet; DISAM is clearly worse (−0.075).",
   "<b>SmellNet (session):</b> the random forest (+0.064) and CORAL (+0.043) win on all 6 splits; full-length SWA "
   "+0.031 (6/6). MSAM collapses (−0.106) at its validation-selected learning rate.",
   "<b>Twin (device):</b> gas identification is near ceiling and serves as a control, except that MSAM and SA-MSAM "
   "collapse (≈ −0.25, 0/5). For concentration, per-device normalization gives +0.224 R² (4/5, CI 0.05–0.41); "
   "SAM is worst (−0.31).",
   "<b>Significance:</b> with 5–9 splits no pairwise difference survives Holm correction (on SmellNet the smallest "
   "attainable corrected p is ≈0.6), so we rely on effect sizes, CIs and win counts. Friedman tests reject “all "
   "methods equal” on every dataset."])

H2("6.2 Cost: MSAM is as cheap as SGD, SAM costs about twice as much")
T([["Median per epoch", "UCI", "SmellNet", "Twin gas", "Twin conc."],
   ["SGD + momentum", "59 passes · 114 ms", "45 · 137 ms", "10 · 29 ms", "10 · 28 ms"],
   ["AdamW", "59 · 126 ms", "45 · 137 ms", "10 · 28 ms", "10 · 31 ms"],
   ["MSAM", "59 · 176 ms", "45 · 143 ms", "10 · 29 ms", "10 · 36 ms"],
   ["SAM", "118 · 358 ms", "90 · 291 ms", "20 · 46 ms", "20 · 52 ms"],
   ["SAGM / DISAM", "118 / 114 · 256 / 381 ms", "90 / 84 · 279 / 360 ms", "20 / 16 · 54 / 71 ms",
    "20 / 16 · 60 / 78 ms"]],
  [3.6, 3.6, 3.4, 3.2, 3.2])
P("Table 2. Forward/backward passes and wall-clock time per epoch (median over final runs; one laptop GPU). "
  "MSAM delivers the promised saving over SAM, but the saving matters little because neither improves "
  "robustness (§6.1).", "cap")

H2("6.3 SA-MSAM (proposed variant): a negative result")
P("At the pre-registered validation go/no-go, before any test run, no SA-MSAM variant beat MSAM (UCI −0.007 to "
  "−0.017; SmellNet −0.012 to −0.016), and neither drift-direction estimate beat the <b>random-direction "
  "control</b>, so the method was not pursued further and is reported as an ablation. Test results agree: where "
  "SA-MSAM looks better than MSAM (SmellNet), the difference is explained by its lower selected learning rate, not "
  "by the direction. A plausible reason is that the drift in these datasets has no consistent direction (UCI batch "
  "10 reverts towards batch 1; SmellNet shift is day-to-day), so there is nothing to extrapolate.")

H2("6.4 Validation-based selection is unreliable under drift")
B(["<b>Four validation→test reversals:</b> recency weighting and MSAM on SmellNet, CC-PCA (best SmellNet "
   "validation score 0.858, second-worst on test), and MSAM on Twin gas (validation 0.996, test collapse).",
   "<b>Selection study</b> (all 12 configurations of 8 methods evaluated on test, 1,152 runs; one test split per "
   "dataset, so indicative only): on UCI validation and test rankings are slightly opposed (Spearman −0.13; −0.97 "
   "for AdamW), and for AdamW, DISAM and GroupDRO the validation pick was worse than a random working "
   "configuration. On SmellNet the ranking transfers better (+0.55).",
   "<b>Mitigation:</b> selecting by mean − standard deviation over seeds reduced the selection regret on SmellNet "
   "from 0.040 to 0.013 and removed MSAM’s collapse; it did not help on UCI. A second validation fold did not help "
   "either: the two folds agree with each other far more than with the test."])
FIGURE("selection_val_vs_test.png",
       "Figure 2. Each dot is one hyperparameter configuration (mean of 3 seeds): validation score (x) vs test "
       "score (y). Ring = configuration chosen by validation; dashed line = best attainable test score; ρ = "
       "Spearman correlation (in brackets: configurations that trained).", width=15.5)

H2("6.5 Flatness barely predicts robustness")
P("Within each split we correlated three sharpness measures (adversarial, top Hessian eigenvalue, random "
  "perturbation) with test score over all final runs. The correlations are weak and inconsistent: adversarial "
  "sharpness ρ ≈ −0.15 (UCI), −0.29 (SmellNet), +0.08 (Twin gas), −0.13 (Twin concentration); random-perturbation "
  "sharpness ≈ 0. Flatter models are at most slightly better, which undercuts flatness as the route to drift "
  "robustness.")

H2("6.6 A few labelled samples help on every dataset")
FIGURE("fewshot.png",
       "Figure 3. Test score after recalibrating the frozen AdamW model with k labelled samples from the new batch, "
       "session or device. Each recalibration method was tuned on the validation split.")
P("Fine-tuning all layers on a handful of labelled samples gives the largest gains of the project: UCI 0.76→0.95 "
  "(10 samples per class), SmellNet 0.83→0.94 (first 4 windows per recording), Twin gas 0.91→0.985, Twin "
  "concentration 0.55→0.69. A per-feature gain + offset recalibration works on Twin, does little on UCI, and "
  "<i>hurts</i> on SmellNet (0/18 runs improve).")

H2("6.7 Shift diagnosis: what kind of drift is it?")
T([["Dataset", "No correction", "Label-free standardize", "Oracle offset", "Oracle gain + offset", "Reading"],
   ["Twin gas (F1)", "0.911", "0.965", "0.952", "<b>0.993</b>", "Calibration-type"],
   ["Twin conc. (R²)", "0.576", "0.692", "0.501", "<b>0.713</b>", "Mostly calibration-type"],
   ["UCI (F1)", "0.759", "0.694", "0.785", "0.693", "Not calibration-type"],
   ["SmellNet (F1)", "0.836", "0.802", "0.796", "0.796", "Not per-feature input calibration"]],
  [2.6, 2.0, 3.0, 1.9, 2.6, 4.9], shade=[(1, GOOD), (2, GOOD), (3, BAD), (4, BAD)])
P("Table 3. Per-feature input corrections on the test domain. “Oracle” maps use test labels and are a "
  "diagnostic upper bound only.", "cap")
P("Device-to-device shift is almost entirely a per-sensor scale and offset error: an oracle gain + offset "
  "recovers Twin gas identification to 0.993. For ageing (UCI), even the oracle calibration does not help, so "
  "the change is not a per-sensor calibration error: the response pattern itself changes. SmellNet is also not "
  "repaired by per-feature calibration of the raw input, but §6.8 shows that it <i>is</i> largely repaired by "
  "adaptation inside the network, so its session shift is simple at the level of learned features rather than "
  "raw readings.")

H2("6.8 Label-free test-time adaptation works for device and session shift, not for ageing")
FIGURE("tta.png",
       "Figure 4. Gain over the unadapted AdamW model (mean over test splits; 5 seeds). Blue: label-free methods "
       "using only unlabelled data of the target domain; orange: few-shot fine-tuning with labelled target "
       "samples. Entropy-based methods do not apply to regression.")
T([["Dataset (drift)", "No adaptation", "Best label-free", "Few-shot (labels)", "Label-free wins / splits"],
   ["Twin gas (device)", "0.910", "<b>0.982</b> (feature align)", "0.962", "3/5 (other 2 at ceiling)"],
   ["Twin conc. (device)", "0.604", "<b>0.712</b> (feature align)", "0.552", "4/5"],
   ["SmellNet (session)", "0.851", "0.918 (SHOT-IM)", "<b>0.943</b>", "6/6"],
   ["UCI (ageing)", "0.763", "0.784 (SHOT-IM)", "<b>0.929</b>", "5/9"]],
  [3.6, 2.4, 4.0, 3.0, 4.0], shade=[(1, GOOD), (2, GOOD), (3, GOOD), (4, BAD)])
P("Table 4. Label-free TTA vs few-shot fine-tuning (results/tta/report_final.txt; 5 seeds). A 3-seed run gives "
  "the same numbers within 0.01, except Twin-concentration feature alignment (0.736 with 3 seeds).", "cap")
B(["<b>Device shift (Twin):</b> label-free adaptation matches or beats few-shot fine-tuning, consistent with the "
   "calibration-type diagnosis. Two labelled recordings per gas are too few to fine-tune the regression model "
   "reliably, whereas aligning feature statistics needs no labels.",
   "<b>Session shift (SmellNet):</b> SHOT-IM recovers about 70% of the few-shot gain (+0.067 of +0.091) without "
   "labels and improves all 6 sessions. Plain input standardization hurts, matching §6.7.",
   "<b>Ageing (UCI):</b> the best label-free method gains only +0.022, against +0.166 for few-shot fine-tuning, "
   "and every TTA configuration lowers the validation score on UCI (§6.9).",
   "<b>Deciding when to adapt.</b> A label-free rule — standardize only if this lowers the IM objective — chose "
   "correctly on 6/6 SmellNet and 5/5 Twin domains but on only 31% of UCI domains (worse than chance); a "
   "cross-seed agreement rule did no better (38%). Unlabelled data cannot reveal whether an ageing shift is being "
   "repaired."])
BOX("<b>Why ageing resists label-free adaptation.</b> Without target labels, a change in the input distribution "
    "can come from the sensors (drift) or from the gases (a different class mix or different concentrations). "
    "Label-free methods assume that the class structure is preserved up to a simple transformation — true for a "
    "calibration offset between devices, false when ageing changes how each sensor responds to each gas. The two "
    "explanations cannot be told apart from unlabelled data alone, so for ageing some labelled target samples "
    "are needed.")

H2("6.9 Ablations")
T([["SmellNet model (same windows, splits and protocol)", "Plain (AdamW)", "With CORAL"],
   ["1-D CNN on raw windows (main model)", "0.810", "0.853"],
   ["GRU (2 layers, 64 units) on raw windows", "0.820", "0.856"],
   ["MLP on 30 window summary features", "0.842", "0.866"],
   ["Random forest on the same 30 features", "0.874", "—"]],
  [8.6, 4.2, 4.2])
P("Table 5. Architecture vs features (mean test macro-F1 over 6 held-out sessions × 5 seeds).", "cap")
B(["<b>Architecture barely matters; features and alignment do.</b> Replacing the CNN by a GRU changes the score by "
   "+0.010 (4/6 sessions), and by +0.003 with CORAL. CORAL adds ≈+0.04 to either network, and simple summary "
   "features (mean, standard deviation, minimum, maximum and slope per channel) beat both raw-signal networks. "
   "The conclusions are therefore not an artifact of the CNN.",
   "<b>TTA hyperparameter sensitivity</b> (validation split only): SHOT-IM on SmellNet gains +0.09 to +0.11 for "
   "learning rates 1e-4 to 1e-3 and 50–200 steps, and fails only at 1e-2 (−0.07). On UCI every TTA configuration "
   "lowers the validation score (−0.01 to −0.08). The TTA conclusions do not hinge on a lucky setting.",
   "<b>Seeds:</b> 3-seed and 5-seed TTA runs give the same conclusions (Table 4).",
   "<b>SWA and early stopping clash:</b> stopping at the best validation epoch ended training before averaging "
   "began, so early-stopped SWA was effectively AdamW. Full-length SWA helps on SmellNet (+0.031, 6/6). Both "
   "variants are reported.",
   "<b>Few-shot budget and layers</b> (Figure 3): gains grow with k; fine-tuning all layers is the most reliable "
   "recalibration, and affine recalibration is competitive only on Twin."])

# ================================================================ 7 discussion
H1("7. Discussion")
H2("Answers to the hypotheses of the project brief")
T([["Hypothesis", "Verdict"],
   ["H1: MSAM matches or beats SAM in macro-F1 under chronological evaluation",
    "Mixed. MSAM ≥ SAM on UCI (0.781 vs 0.759), but it collapses on SmellNet and Twin gas at its selected "
    "learning rate and is below SAM there. Neither beats AdamW reliably."],
   ["H2: MSAM needs less training time than SAM", "Supported: half the passes and about half the time per epoch of "
    "SAM on UCI and SmellNet, the same cost as SGD (Table 2)."],
   ["H3: rankings differ between random and temporal splits", "Strongly supported: random splits give ≈99%+ "
    "accuracy on all three datasets, far above every held-out-domain score (§3)."],
   ["H4: any benefit varies by class and batch", "Supported: effects differ by split and dataset, and validation "
    "winners often lose on test (§6.4)."]],
  [7.0, 10.0])
H2("Match the remedy to the drift type")
P("Across ≈7,200 runs, the method family mattered far less than the type of drift. Training-time robustness "
  "(flatness, DG regularizers) gives at most small, dataset-specific gains. When the shift is calibration-type "
  "— different devices, and different sessions once represented in feature space — cheap label-free adaptation "
  "on unlabelled target data recovers most of the loss. When it is not (ageing), labelled target samples are "
  "required. The oracle-calibration diagnosis, run on a small labelled set from the new condition, is a "
  "practical way to decide which case applies before choosing a deployment strategy.")

# ================================================================ 8 limitations
H1("8. Limitations")
B(["Few held-out domains (5–9 per dataset) limit statistical power; no pairwise test survives multiple-comparison "
   "correction, so claims rest on effect sizes, CIs and win counts.",
   "The selection study uses one test split per dataset and is indicative.",
   "Some baselines (e.g. random forest, full-length SWA) were added after early test results were seen; this is "
   "disclosed in notes/decisions.md, and they were tuned with the same budget.",
   "Per-domain normalization and all TTA methods are transductive: they need unlabelled data from the new "
   "domain. This is realistic for a deployed device but is a different setting from pure domain generalization.",
   "The UCI release lacks the pre-exposure baseline needed to rule out the temporal confound of [12]; the "
   "SmellNet session order is assumed chronological.",
   "Small models only (MLP, 1-D CNN, GRU). SmellNet GPU runs are reproducible to ±0.005 (cuDNN "
   "non-determinism), below the seed spread.",
   "The literature search was focused (web search and Semantic Scholar), not systematic; prior e-nose TTA methods "
   "[19, 20] were not re-implemented."])

# ================================================================ 9 conclusion
H1("9. Conclusion")
P("Under a leakage-free, validation-only protocol, sharpness-aware optimizers — including the cheap MSAM and our "
  "drift-aware variant — do not make gas-sensor classifiers robust to real drift, and validation scores are an "
  "unreliable guide under drift. The more useful question is not which training method to use but what kind of "
  "drift is present. Device and session shifts can be absorbed by label-free test-time adaptation; ageing cannot, "
  "and a few labelled samples from the new period were the most effective remedy we found. A simple "
  "oracle-calibration diagnosis tells the cases apart.")
P("<b>Code and data availability.</b> All code (src/, scripts/), per-run results (results/), figures and the dated "
  "decision log (notes/decisions.md) are in the project folder; all datasets are public.", "cap")

# ================================================================ references
H1("References")
for r in [
    "[1] P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-aware minimization for efficiently improving generalization. ICLR, 2021.",
    "[2] M. Becker, F. Altrock, B. Risse. Momentum-SAM: Sharpness aware minimization without computational overhead. arXiv:2401.12033, 2024.",
    "[3] I. Loshchilov, F. Hutter. Decoupled weight decay regularization. ICLR, 2019.",
    "[4] P. Wang, Z. Zhang, Z. Lei, L. Zhang. Sharpness-aware gradient matching for domain generalization. CVPR, 2023.",
    "[5] R. Zhang et al. Domain-inspired sharpness-aware minimization under domain shifts. ICLR, 2024.",
    "[6] B. Sun, K. Saenko. Deep CORAL: Correlation alignment for deep domain adaptation. ECCV Workshops, 2016.",
    "[7] S. Sagawa, P. W. Koh, T. B. Hashimoto, P. Liang. Distributionally robust neural networks for group shifts. ICLR, 2020.",
    "[8] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, A. G. Wilson. Averaging weights leads to wider optima and better generalization. UAI, 2018.",
    "[9] I. Gulrajani, D. Lopez-Paz. In search of lost domain generalization. ICLR, 2021.",
    "[10] A. Vergara, S. Vembu, T. Ayhan, M. A. Ryan, M. L. Homer, R. Huerta. Chemical gas sensor drift compensation using classifier ensembles. Sensors and Actuators B, 166:320–329, 2012.",
    "[11] L. Zhang, D. Zhang. Domain adaptation extreme learning machines for drift compensation in E-nose systems. IEEE Trans. Instrumentation and Measurement, 64(7), 2015.",
    "[12] N. Dennler, S. Rastogi, J. Fonollosa, A. van Schaik, M. Schmuker. Drift in a popular metal oxide sensor dataset reveals limitations for gas classification benchmarks. Sensors and Actuators B, 361:131668, 2022.",
    "[13] J. Fonollosa, L. Fernández, A. Gutiérrez-Gálvez, R. Huerta, S. Marco. Calibration transfer and drift counteraction in chemical sensor arrays using Direct Standardization. Sensors and Actuators B, 236:1044–1053, 2016.",
    "[14] D. Feng et al. SmellNet: A large-scale dataset for real-world smell recognition. arXiv:2506.00239, 2025.",
    "[15] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell. Tent: Fully test-time adaptation by entropy minimization. ICLR, 2021.",
    "[16] J. Liang, D. Hu, J. Feng. Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation. ICML, 2020.",
    "[17] Smooth conditional domain adversarial network with SAM for e-nose drift compensation. Sensors, 24(4):1319, 2024. [authors to be added]",
    "[18] Fu et al. Protocol-aware reproducibility case study of electronic-nose drift compensation across chronological target periods. Microchemical Journal, 2026.",
    "[19] Liang et al. An online test-time adaptive drift compensation network for electronic nose. IEEE Trans. Instrumentation and Measurement, 2025. doi:10.1109/TIM.2025.3591198.",
    "[20] Liang, Chen, Yang. Online test-time adaptation for electronic nose drift compensation. EEI, 2024. doi:10.1109/EEI63073.2024.10696456."]:
    P(r, "ref")
