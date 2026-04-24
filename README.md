# Anki Spaced-Repetition Causal Inference Project

**Author:** Siddharth Pottapatri  
**Data:** January 2025 – April 2026 (449 days, 52,119 reviews)  
**Deck:** Human Japanese Intermediate Shared  

---

## Project Files

| File | Description |
|------|-------------|
| `anki_causal_analysis.ipynb` | Main notebook: EDA, ITS, PSM (retention), Survival |
| `psm_response_time.ipynb` | PSM v1: response time on hard cards (flawed definition) |
| `psm_response_time_v2.ipynb` | PSM v2: response time on hard cards (corrected definition) |
| `assign_experiment_tags.py` | Script to tag 50/50 treatment/control cards for 60-day experiment |
| `experiment_tags_import.txt` | Anki-ready import file for experiment tags |
| `experiment_assignment_log.csv` | Record of note ID → treatment/control assignment |
| `fig1_daily_reviews.png` | Daily review volume + retention rate over time |
| `fig2_heatmap.png` | Retention rate by hour and day of week |
| `fig3_retention_by_maturity.png` | Retention rate by card maturity band |
| `fig4_its.png` | Interrupted time series with counterfactual |
| `fig5_psm_overlap.png` | Propensity score common support |
| `fig6_balance.png` | Covariate balance before/after PSM |
| `fig7_km_survival.png` | Kaplan-Meier curves by ease factor quartile |
| `fig8_cox_hr.png` | Cox PH hazard ratio forest plot |
| `fig9_schoenfeld.png` | Cox PH proportional hazards assumption check |
| `v2_fig1_temporal.png` | Temporal distribution of hard reviews (v2) |
| `v2_fig2_distributions.png` | Response time distributions pre/post matching (v2) |
| `v2_fig3_comparison.png` | ATE comparison: v1 vs v2 definition |

---

## Data Source

My personal Anki SQLite database at `C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2`.

**Key schema facts for reproducibility:**
- `cards.factor` and `revlog.factor` store ease factor as integer × 1000 (e.g. 2500 = 2.50). Divide by 1000 before use.
- `revlog.factor = 0` means the card was still in the initial learning phase at that review — exclude from any ease-factor analysis.
- Anki 2.1.45+ uses a custom `unicase` collation. Register a no-op collation with `con.create_collation('unicase', lambda a, b: (a > b) - (a < b))` before reading tables, otherwise `sqlite3` raises `OperationalError`.
- `revlog.time` is capped at exactly 60,000 ms by Anki (any review taking longer is recorded as 60s). These are not valid response-time measurements.
- `revlog.id` is a Unix timestamp in milliseconds.

---

## Phase 1 — Exploratory Data Analysis

**File:** `anki_causal_analysis.ipynb`, Cells 1–8

### What I did
I connected to my Anki SQLite database, extracted the `cards`, `revlog`, and `decks` tables, and built a feature-engineered DataFrame. Key features I created: datetime components (hour, day of week), a retention flag (`ease > 1`), ease factor (dividing stored integers by 1000), log-interval, and days since last review.

### Key findings

| Metric | Value |
|--------|-------|
| Date range | 2025-01-28 → 2026-04-23 (449 days) |
| Total reviews | 52,119 |
| Unique cards | 9,037 |
| Average reviews/day | 116 |
| **Overall retention rate** | **99.2%** |
| Median time per card | 4.2 seconds |

**The 99.2% retention rate is the single most important fact in this dataset.** It means almost every review is a success, which severely limits the statistical power of any analysis using binary retention as an outcome — there is almost no variation left to explain.

**Retention by card maturity** (Figure 3): Cards in the initial learning phase show the most variance. Cards with intervals over 60 days maintain the highest and most stable retention, confirming that Anki's spaced-repetition algorithm is working as intended.

---

## Phase 2 — Interrupted Time Series (ITS)

**File:** `anki_causal_analysis.ipynb`, Cells 9–12

### What I did
I modelled my daily retention rate as a segmented regression with a level change (β₂) and slope change (β₃) at a detected intervention point. I used heteroscedasticity-robust standard errors (HC3). The intervention point was auto-detected as the date of the largest single-day review volume shift in my data — September 13, 2025.

**Model:**  
`retention_t = β₀ + β₁·t + β₂·D_t + β₃·(t − t*)·D_t + ε`

where `D_t = 1` if after intervention.

### Key findings

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| β₂ (level change) | −0.014 | Retention dropped ~1.4 pp at the intervention |
| β₃ (slope change) | −0.000136/day | Slight post-intervention deceleration in trend |
| R² | 0.265 | Model explains 26.5% of variance in daily retention |
| Durbin-Watson | — | Check notebook output for autocorrelation |

**Caveat:** ITS assumes no simultaneous confounders at the breakpoint. I detected the intervention algorithmically rather than pre-specifying it, so any life event coinciding with September 2025 could explain the level change.

---

## Phase 3 — Propensity Score Matching: Morning vs Evening (Binary Retention)

**File:** `anki_causal_analysis.ipynb`, Cells 13–20

### What I did
I estimated the average treatment effect (ATE) of reviewing in the morning (06:00–12:00) versus the evening (18:00–24:00) on binary retention (`ease > 1`). I controlled for ease factor, log-interval, deck, and log-days since last review. I used logistic regression for propensity score estimation, 1:1 nearest-neighbour matching with a 0.2-SD caliper on log-odds, and Rosenbaum bounds for sensitivity analysis.

### Key findings

| Metric | Value |
|--------|-------|
| Morning reviews | 1,375 |
| Evening reviews | 20,428 |
| Morning retention | 99.20% |
| Evening retention | 99.27% |
| **ATE** | **−0.07 percentage points** |
| 95% CI | [−0.65 pp, +0.51 pp] |
| t-test p | 0.827 |
| Rosenbaum Γ at p=0.05 | Never reached — all p = 1.0 |

**Result: Not significant.** The ATE of −0.07 pp is statistically and practically zero. The Rosenbaum bounds show the null result is extremely robust — even a confounder that tripled my odds of reviewing in the morning would not change the conclusion.

**Why nothing is significant here:** With 99.2% retention, there is essentially no variation in the outcome to detect. This is a real finding: time of day does not affect whether I remember a card.

---

## Phase 4 — Survival Analysis: Time to First Lapse

**File:** `anki_causal_analysis.ipynb`, Cells 21–26

### What I did
I built a survival dataset where each card's "time at risk" is its total review count (`reps`), and the event is receiving at least one lapse (`lapses ≥ 1`). The classic Anki leech threshold (lapses ≥ 8) was unusable — only 1 card in my dataset had reached it, as my deck is relatively young. I fitted Kaplan-Meier curves stratified by ease factor quartile and a Cox proportional hazards model with ease factor as the primary covariate.

### Key findings

**Lapse rate by ease factor quartile:**

| Quartile | Cards | Lapse rate |
|----------|-------|------------|
| Q1 — Hardest (ef ≤ 2.65) | 2,742 | **7.5%** |
| Q2 | 2,445 | 0.6% |
| Q3 | 2,890 | 0.5% |
| Q4 — Easiest | 925 | 0.4% |

**Cox PH:** HR = 2.06, p ≈ 0.000. A one-unit increase in ease factor is associated with a 2.06× change in hazard of first lapse.

**Log-rank test (Q1 vs Q4):** Significant — my harder cards lapse significantly sooner.

**Caveat:** The proportional hazards assumption is mildly violated (Schoenfeld test p = 0.025), meaning the hazard ratio is not constant over time. The effect of ease factor on lapse risk changes as cards age.

**This is the strongest finding in the project.** A small cluster of hard cards (bottom ease-factor quartile) is responsible for nearly all my lapses. These cards are identifiable from early in their review history.

---

## PSM Extension — Morning vs Evening Response Time on Hard Cards

### Why I looked at response time
Binary retention was flat at 99.2%. Response time is a continuous, more sensitive outcome that captures how quickly and fluently I retrieve information — even on reviews I ultimately get right.

---

### v1: Hard cards defined by current ease factor (flawed)

**File:** `psm_response_time.ipynb`

**Definition:** Hard card = `cards.factor ≤ 2.65` (bottom 25% of *current* ease factors as of April 2026).

**Problem I discovered:** This definition looks backwards. Cards that are currently hard had 82% of their reviews concentrated in 2025, when I was just starting out. I had no morning hard-card reviews at all in Jan–Apr 2025. The morning/evening comparison was heavily confounded by my learning curve — early-2025 reviews were slower for me regardless of time of day.

**Results:**
- Matched pairs: 533
- ATE median: **+1,111 ms** (morning slower), 95% CI [+557, +1,634]
- Mann-Whitney p < 0.0001

**Verdict:** Significant, but not trustworthy — the hard-card definition conflated "hard card today" with "card I reviewed when I was a beginner."

---

### v2: Hard reviews defined at review time (corrected)

**File:** `psm_response_time_v2.ipynb`

**Definition:** A review is hard if `revlog.factor < 2.5` at the moment of that review. The value 2.5 is Anki's default starting ease factor — any card below it has been actively downgraded by the algorithm due to repeated failures. This is temporally correct and semantically meaningful.

**Additional confounder added:** `month_idx` (calendar month as a numeric index) to partially absorb the remaining learning-curve effect.

**Results:**
- Hard reviews (morning + evening): 4,252
- Matched pairs: 309
- ATE (mean diff): **+508 ms**, 95% CI [−858, +1,844], t-test p = 0.498
- ATE (median diff): **+1,084 ms**, 95% CI [+209, +1,924], Mann-Whitney p = 0.0045

**Verdict:** My morning reviews on hard cards remain significantly slower by ~1 second even with the corrected definition. The mean-based ATE is no longer significant (p = 0.498) because the mean is sensitive to a handful of very long reviews; the median-based ATE is more robust to skew and remains significant.

**Residual concern:** `month_idx` SMD after matching = 0.117 (slightly above the 0.1 balance threshold), meaning matched morning and evening reviews are still not perfectly balanced on calendar time. This is a structural limitation — I had zero morning hard-card reviews before May 2025, so there is no counterfactual to match against from that early period.

---

## Overall Summary of Findings

| Analysis | Result | Confidence |
|----------|--------|------------|
| ITS: Did Sep 2025 shift change retention? | Small negative level change (−1.4 pp) | Low — intervention detected algorithmically, not pre-specified |
| PSM: Does morning review improve retention? | No effect (ATE = −0.07 pp) | High — retention is 99.2%, no variation to detect |
| Survival: Do harder cards lapse sooner? | Yes — Q1 cards lapse at 7.5% vs 0.4% for easiest | High — large, significant, consistent across KM and Cox |
| PSM v2: Are morning hard-card reviews slower? | Yes — morning ~1,084 ms slower (median) | Moderate — significant but residual temporal confound remains |

**The big picture:**  
My retention is excellent across the board (99.2%). The meaningful variation in my data is not *whether* I remember cards but *which* cards are structurally difficult. A small group of hard cards (ease factor degraded below 2.5) concentrates almost all my lapses and takes noticeably longer to answer in the morning than in the evening. Whether that slowness represents cognitive sluggishness or more careful processing cannot be determined from response time alone.

---

## 60-Day A/B Experiment — Read Aloud vs Silent Review (Active)

**Status: RUNNING** — started April 2026, ends ~June 2026

**Files:** `assign_experiment_tags.py` · `experiment_tags_import.txt` · `experiment_assignment_log.csv` · `mark_treatment_cards.py`

### Design

| | Treatment | Control |
|---|---|---|
| Tag | `exp::treatment` | `exp::control` |
| Notes | 1,745 | 1,745 |
| Intervention | Card marked with ★ — I read the sentence out loud before rating | No change — reviewed silently as normal |
| Assignment | Random (seed=42), 50/50 split across all 3,490 notes in the deck | |

The `★` marker was added to the English field of all treatment cards using `mark_treatment_cards.py`, which writes directly to the Anki SQLite database. The marker is the only visible difference between groups — no content was changed.

### Research Question
Does reading a sentence card out loud during review cause a measurable reduction in lapse rate and response time over a 60-day window, compared to identical cards reviewed silently?

### Causal Assumptions
- **Stable Unit Treatment Value Assumption (SUTVA):** reviewing a treatment card does not affect my performance on control cards in the same session.
- **No anticipation:** my behaviour on control cards is not affected by knowing that treatment cards exist.
- **Randomisation holds:** assignment was purely random; no systematic difference between groups at baseline (verifiable from `experiment_assignment_log.csv` — both groups draw from the same deck and note type).

### Primary Outcomes (measured at day 60)
1. **Lapse rate** — proportion of reviews where ease = 1 (Again), by group
2. **Response time (ms)** — median time per review, by group

### Analysis Plan
At the end of the 60 days I will run a PSM analysis on the logged note IDs, using the same methodology from `psm_response_time_v2.ipynb`, comparing treatment vs control on both outcomes. Because assignment was randomised, propensity score matching is a robustness check rather than a necessity — a simple two-sample test (Mann-Whitney for response time, proportion test for lapse rate) on the randomised groups is the primary analysis.

### Baseline Balance
Both groups were drawn from the same deck and note type with a single random shuffle. Baseline ease factor distributions should be identical by design; this will be verified at analysis time using the pre-experiment ease factors stored in `experiment_assignment_log.csv`.

---

## What I'd Do Next

1. **Restrict to 2026 only** — the cleanest fix for the learning-curve confound. ~14k reviews at stable skill level.
2. **Analyse the 60-day experiment results** — compare treatment vs control on lapse rate and response time using PSM on the tagged groups.
3. **Dose-response within session** — does my response time degrade as I review card #50 vs card #1 in a session? Model review number within session as the running variable.
4. **Card content analysis** — the `notes` table stores the actual Japanese text. Hard cards likely share linguistic features (word length, JLPT level, phonetic similarity) that explain why they're structurally difficult.
5. **Survival analysis on a longer horizon** — re-run in 12 months when more cards will have accumulated enough lapses to make the classic leech threshold (≥ 8) usable.

---

## Technical Notes

**Environment:** Python 3.11, Windows 11

**Dependencies:**
```
pandas, numpy, matplotlib, seaborn
statsmodels          # ITS segmented regression
scikit-learn         # logistic regression, nearest-neighbour matching
scipy                # t-test, Mann-Whitney U, Rosenbaum bounds
lifelines            # Kaplan-Meier, Cox PH
```

**Statistical decisions log:**

| Decision | Rationale |
|----------|-----------|
| t-test over Z-test | Population variance unknown; estimated from data. At n > 300 the difference is negligible (< 0.01 in p-value). |
| Mann-Whitney as primary test for response time | My response times are heavily right-skewed. Mann-Whitney makes no distributional assumption and tests stochastic dominance directly. |
| Exclude revlog.time = 60,000 ms | Anki's hard cap records any review exceeding 60s as exactly 60,000 ms. These are not real response times. |
| Use lapses ≥ 1 instead of ≥ 8 for survival | Only 1 card reached the classic leech threshold of 8 lapses in 449 days. First-lapse is a more common, earlier, and still-meaningful event. |
| Use revlog.factor (not cards.factor) for hard-review definition | cards.factor is a current snapshot; revlog.factor captures card difficulty at the moment of review. |
| Add month_idx as PSM confounder in v2 | Partially absorbs the learning-curve trend that remained after switching to the at-review-time hard-review definition. |
