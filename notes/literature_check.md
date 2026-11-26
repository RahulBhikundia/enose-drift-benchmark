# Focused literature check (2026-11-14, web search; not exhaustive: Scholar / IEEE Xplore / ScienceDirect full-text check still advised)

## Question: is label-free test-time adaptation (TTA) for e-nose drift new?

### Unsupervised domain adaptation (UDA) for e-nose drift: LARGE existing literature (must cite, and compare against)
UDA trains WITH labelled source data AND unlabelled target data together (re-training per target).
- DAELM, Zhang & Zhang 2015, domain adaptation extreme learning machines: https://arxiv.org/pdf/1505.06405
- Balanced Distribution Adaptation for MOS drift: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8153337/
- CDAN + SAM (smooth conditional domain adversarial), Sensors 2024: https://mdpi.com/1424-8220/24/4/1319
  (also: SAM is already used for e-nose drift, inside a UDA method)
- AMDS-PFFA, attention multi-source UDA, 2024: https://arxiv.org/html/2409.13167v1
- Label-free drift compensation via domain adversarial network (HATN-DA), Sens. Actuators B 2023:
  https://www.sciencedirect.com/science/article/abs/pii/S0925400523004240
- PUDA, prototype-optimized UDA with a dynamic Transformer: https://www.sciencedirect.com/science/article/abs/pii/S095741742402311X
- Semi-supervised adversarial DA CNN: https://www.sciencedirect.com/science/article/abs/pii/S0925400524013728
- Open-set adversarial DA for e-nose: https://www.sciencedirect.com/science/article/abs/pii/S0169743925002394
- Knowledge distillation for drift, 2025 (notes earlier benchmarks lack statistical validation): https://arxiv.org/html/2507.17071v1
- Iterative RF + incremental domain-adversarial (real-time): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12471586/

### Target-free / domain generalization for gas sensors
- TDACNN, target-domain-free DA CNN: https://arxiv.org/abs/2110.07509
- Using context to adapt to sensor drift (2020): https://arxiv.org/pdf/2003.07292

### Calibration transfer (Twin dataset's own paper)
- Fonollosa et al. 2016, Direct Standardization with paired transfer samples (supervised), mapping slave units to a master
  unit and countering drift: https://diposit.ub.edu/dspace/handle/2445/115587 ; dataset: https://archive.ics.uci.edu/dataset/361/twin+gas+sensor+arrays
  No unsupervised / label-free adaptation on Twin found.

### TTA for time series in general (cite as the general area)
- ACCUP, TTA for time-series classification, KDD 2025 (says TTA for time series "largely unexplored"): https://arxiv.org/pdf/2501.01472
- TTA for forecasting: DynaTTA / TTFBench (ICML 2025): https://icml.cc/virtual/2025/48077 ; TAFAS: https://arxiv.org/html/2501.04970v1
- Tent (entropy minimization), arXiv 2006.10726; SHOT (Liang et al., ICML 2020): the methods we use.

### SmellNet
- SmellNet paper, v1 to v4 (2025): https://arxiv.org/abs/2506.00239v4. No session-shift adaptation work found.

## Verdict
- NOT new: label-free (unsupervised) adaptation to e-nose drift in general (many UDA papers); SAM for e-nose drift (CDAN+SAM 2024);
  calibration transfer on Twin (supervised DS).
- Apparently new (no paper found): SOURCE-FREE TEST-TIME adaptation of a deployed e-nose model (no retraining, no source data,
  adapt per new device/session/batch), evaluated across three drift types against few-shot labelled recalibration under a
  frozen, validation-only protocol; the drift-type-dependent result (works for day-to-day and device-to-device, not ageing);
  the leakage-free benchmark of SAM-family + DG methods; and the IM-based switch for when standardization helps.
- Reviewer risk: "why not just use UDA?" -> must add at least one UDA baseline (e.g. DANN or Deep-CORAL with unlabelled target)
  and argue the deployment advantage of TTA (no source data / no retraining on device).

## UPDATE 2026-11-14: Publish or Perish (Semantic Scholar via pop8query; Google Scholar CLI blocked by CAPTCHA) + user's Google Scholar results
Raw results: results/litsearch/ (s2_*.csv, merged_s2.csv; 13 queries).
### TTA for e-nose drift ALREADY EXISTS -> our "first TTA for e-nose" claim is WRONG; drop it.
- Liang, Chen, Yang. "Online Test-Time Adaptation for Electronic Nose Drift Compensation", EEI 2024. DOI 10.1109/EEI63073.2024.10696456
  (teacher-student distillation, continual updates on unlabelled drift data; public datasets)
- Liang et al. "An Online Test-Time Adaptive Drift Compensation Network for Electronic Nose" (OTTA-DriftNet), IEEE TIM 2025.
  DOI 10.1109/TIM.2025.3591198 (source model plus online TTA, teacher-student KD, attention + GRU)
- Luo & Sun. "Stage-Adaptive Test-Time Experts for Strict Multi-Target Drift in Electronic Nose Systems", ISOEN 2026 (UCI drift, 62.95% avg acc)
- Monori 2026 PhD thesis (Royal Holloway): continual test-time adaptation for odour recognition
- Zou & Yan. "Robust Pseudolabel Subspace Learning for E-Nose Drift Compensation", IEEE Sensors Letters 2026 (unsupervised; temporal + device drift)
### Benchmark / protocol / few-shot angles are also partly occupied
- Fu et al. "Protocol-aware reproducibility case study of electronic-nose drift compensation across chronological target periods",
  Microchemical Journal 2026 (rankings depend on chronology and target-data availability) -> overlaps our protocol/selection finding; MUST READ
- "A Reproducible Benchmark for Gas Sensor Array Classification" (Sensors 2025, Holm-corrected tests; not drift)
- "Drift-Robust Lightweight Deep Learning on Open Gas Sensor Benchmarks: A Reproducible Architecture Study" (Molecules 2026)
- "Metric Network for E-Nose Drift Compensation: Few-Shot Learning" (IEEE Sensors J. 2025) -> few-shot under drift exists
- Meta-learning (MAML) for long-term drift (ISOEN 2026); TSFM embeddings for e-nose (arXiv 2026)
- "Domain Generalization ... e-Nose Cross-Platform Identification of Ignitable Liquids" (2024/2025) -> DG on e-nose exists

### Revised novelty (what still looks open; verify by reading Liang 2025, Luo 2026, Fu 2026 in full)
1. Cross-drift-type evidence: the same label-free TTA works for day-to-day (SmellNet) and device-to-device (Twin) shift but
   NOT for ageing (UCI), explained by the shift diagnosis (calibration-type vs not). UNVERIFIED: which datasets the prior TTA papers use (abstracts say "public datasets"; Luo 2026 = UCI drift) -> check full texts.
2. Sharpness-aware family (SAM/MSAM/SAGM/DISAM, incl. our SA-MSAM with random-direction control) vs DG vs TTA vs few-shot
   under ONE frozen validation-only protocol: flatness does not buy drift robustness.
3. Validation selection failures under drift (4 reversals + selection study); must be positioned against Fu et al. 2026.
4. TTA on Twin (device-to-device) and SmellNet: no prior TTA work found on these datasets.

## UPDATE 2026-11-14 (2): focused check of revised claim 1 (label-free TTA works for day-to-day + device-to-device, not ageing)
Sources: web search + Semantic Scholar via PoP (results/litsearch/s2_2*.csv).
- Which datasets Liang 2024 / OTTA-DriftNet 2025 use: still UNKNOWN (abstracts: "a public dataset"); Luo & Sun 2026 = UCI drift.
  OTTA-DriftNet pages: https://ieeexplore.ieee.org/document/11087654/ , https://www.researchgate.net/publication/393933417
- Twin dataset IS already used for UNSUPERVISED DA (not TTA): HATN-DA 2023 (S0925400523004240; uses the twin set's 5
  "batches"/days, 97.5-100%), OSADM open-set (S0957417424006237), DAST (S0957417422013847), PTADA regression 2025
  (S0925400525016314), WAFDAN 2025 (S0950705125018052). Need to check whether they treat UNITS or DAYS as domains.
  -> "first label-free adaptation on Twin" is NOT claimable; "first test-time (source-free) adaptation on Twin" not contradicted.
- Drift + device shift handled jointly WITH transfer samples (supervised): "Improving the performance of drifted/shifted e-nose
  systems by cross-domain transfer using common transfer samples", Sens. Actuators B 2021 (S092540052031501X);
  global calibration models across units, Sens. Actuators B 2022.
- 2026 review of drift-compensation algorithms (Chemosensors 14(6):143, doi 10.3390/chemosensors14060143) lists device differences
  and ageing as distinct drift sources; no head-to-head of one label-free method across ageing vs session vs device found.
- SmellNet: no adaptation / TTA work found.
VERDICT on claim 1: survives only in this narrowed form: "a controlled comparison showing the SAME label-free TTA helps for
session-to-session and device-to-device shift but not for ageing drift, with a diagnosis (oracle per-feature calibration)
explaining why". Not claimable: TTA for e-nose (Liang 2024/25), label-free adaptation on Twin (HATN-DA etc.).
Must read before final wording: OTTA-DriftNet (datasets), HATN-DA (Twin domain definition), the 2026 review.
