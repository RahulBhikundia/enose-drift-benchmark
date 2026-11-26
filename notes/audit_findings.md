# Audit findings (Day 1, 2026-11-11)

Scripts: `scripts/audit_uci.py`, `scripts/audit_smellnet.py`. Raw outputs: `results/audit_*`. Figures: `figures/audit_*`.

## UCI Gas Drift
1. **Class coverage:** Toluene is absent from batches 3, 4 and 5 and has only 5 samples in batch 2. A "train on batches 1–5" split has 79 Toluene samples, all but 5 from batch 1. Batch 10 is perfectly balanced (600 per class).
2. **File order is not acquisition order.** Batches 1–6 are sorted by class (5–14 label runs), and batch 10 alternates classes perfectly (3,600 runs). No within-batch temporal analysis is possible.
3. **Dennler-style baseline test isn't possible with UCI #224:** it has no pre-exposure (baseline) features. The report must say so.
4. **Drift is strong:** within the same gas, a linear classifier separates batch 1 from any later batch at 0.97–1.00 balanced accuracy.
5. **Drift is not monotonic:** the feature shift from batch 1 grows up to batch 9 (median |SMD| 1.6) but drops for batch 10 (0.8), which was recorded after the sensors had been off for about 5 months.
6. **Batch ID barely predicts the gas** (24.7% vs 21.6% for always guessing the majority class), so at batch level the label mix and time are only weakly entangled.
7. **A random split massively overstates performance:** logistic regression gets 99.4% on a random 80/20 split vs 47–77% when trained on batches 1–5 and tested on 6–10 (macro-F1 0.42–0.66). This strongly supports H3.

## SmellNet-Base (random forest on 60 s window summaries, chance = 2%)
| Protocol | Acc | Macro-F1 |
|---|---|---|
| Random 80/20 windows (leaky) | 0.999 | 0.999 |
| Official: train sessions 1–5, test session 6 | 0.820 | 0.793 |
| Leave-one-session-out (mean of 6) | 0.882 | 0.866 |
| First 60 s only, LOSO | 0.783 | 0.742 |
| Identify session within substance (random split) | 0.997 | n/a |

1. **Random-window leakage is severe** (99.9% vs 82%). Overlapping windows from the same session end up on both sides of the split.
2. **Hard sessions:** held out, session 1 scores 65%, session 6 scores 82%, and sessions 2–4 score about 99%.
   - Session 1 starts from a much lower baseline and rises over the first ~100 s. Sessions 2–6 start already elevated. This looks like a **protocol difference** (for example, sensor warm-up or clean-air recovery before session 1), not gradual drift.
   - Session 5 sits lower on most channels; session 6 has an LPG offset of about +4.
   - So SmellNet's shift is **day-to-day, not a smooth trend.**
3. **Day signature:** within each substance, the session is identifiable at 99.7%, so every day has a strong, distinct signature. (This uses a random split, so it is optimistic, but the signal is clearly there.)
4. **The first-60-s test is inconclusive as a confound test.** In sessions 2–6 the odour already seems present at t = 0, so the first minute contains real odour signal. Only session 1 has a visible clean-air baseline.
5. **To verify:** our random forest's 82% on the official session-6 test is higher than the paper's 58.5–65% for deep models. Check their exact protocol (channels, windowing, normalisation, window-level vs session-level scoring) before claiming anything.
6. Session index → chronological day is still an assumption; the session statistics neither confirm nor contradict it.

## Implications for the method (SA-MSAM)
- **Neither dataset shows monotonic drift** (UCI batch 10 reverts toward batch 1; SmellNet is day-to-day). The "extrapolate the drift direction forward" assumption may not hold.
- Keep two variants and let validation decide (Nov 19 go/no-go):
  (a) **extrapolated** direction: newest-domain gradient EMA minus oldest-domain gradient EMA;
  (b) **non-extrapolated** direction: the direction of greatest disagreement between domain gradients (for example, the gradient EMA of the hardest current domain minus the mean). This is closer to DISAM in spirit, but uses MSAM's zero-cost mechanism.
- UCI split design must handle Toluene's absence from batches 3–5. Options: report macro-F1 over the classes present, or use batches 1–2 plus a stratified part of the data for training. Decide before any test runs.
