# Causal Inference on Spaced-Repetition Learning Data
## Full Project Analysis

**Author:** Siddharth Pottapatri  
**Data:** January 28, 2025 – June 23, 2026 (511 days, 59,465 reviews)  
**Deck:** Human Japanese Intermediate  
**Date of Analysis:** June 23, 2026

---

## 1. Motivation

Most educational technology research relies on aggregated, anonymized datasets where individual behavior is averaged away. I wanted to ask causal questions about my own learning — using my personal Anki spaced-repetition database as a longitudinal record of how a single learner acquires Japanese vocabulary over time.

The questions I brought to this data:

1. **Did a behavioral shift in September 2025 change my retention rate?** (Interrupted Time Series)
2. **Does reviewing in the morning versus evening affect whether I remember a card?** (Propensity Score Matching on binary retention)
3. **Which cards are structurally hard, and do they fail sooner?** (Survival Analysis)
4. **Does morning review slow me down on the cards I find hardest?** (PSM on continuous response time)
5. **Does reading cards out loud during review reduce lapse rate?** (60-day Randomized A/B Experiment)

---

## 2. Data and Schema

**Source:** Anki SQLite database at `collection.anki2`.

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `revlog` | `id` (ms timestamp), `cid`, `ease` (1–4), `time` (ms), `factor` | `time` capped at 60,000 ms; `factor` = ease factor × 1000 |
| `cards` | `id`, `nid`, `factor`, `lapses`, `reps` | Current card state snapshot |
| `notes` | `id`, `flds`, `tags` | Fields separated by `\x1f`; experiment tags stored here |

**Key schema facts:**
- `ease = 1` (Again) and `ease = 2` (Hard) are the two failure buttons in Anki. For the experiment, I defined **lapse** as `ease ≤ 2` — both Again and Hard represent inadequate recall and trigger algorithmic difficulty adjustment.
- `ease = 3` (Good) and `ease = 4` (Easy) indicate successful retrieval.
- `revlog.factor = 0` means the card was in the initial learning phase. Excluded from any ease-factor analysis.
- `revlog.id` is a Unix millisecond timestamp used as the primary time variable.
- SQLite uses a custom `unicase` collation in Anki 2.1.45+. A no-op collation must be registered before reading tables: `con.create_collation('unicase', lambda a, b: (a > b) - (a < b))`.
- The 60-second cap: any review exceeding 60 seconds is recorded as exactly 60,000 ms. These 3,100 observations (5.2% of all reviews) are excluded from all response-time analyses.

**Baseline statistics:**

| Metric | Value |
|--------|-------|
| Total reviews | 59,465 |
| Unique cards | 10,243 |
| Date range | 2025-01-28 to 2026-06-23 (511 days) |
| Average reviews/day | 116 |
| Overall retention (ease > 1) | 99.3% |
| Again rate | 0.73% |
| Hard rate | 15.84% |
| Good rate | 20.25% |
| Easy rate | 63.17% |
| Median response time (non-capped) | 4.1s |
| Capped reviews (≥60s) | 3,100 (5.2%) |

**The 99.3% binary retention rate is the defining constraint of this dataset.** With only 0.73% Again presses, binary success/failure has almost no variation to explain. All analyses using binary retention as an outcome must reckon with this ceiling effect.

---

## 3. Phase 1 — Exploratory Data Analysis

**File:** `anki_causal_analysis.ipynb`, Cells 1–8

### Methodology

Connected to the Anki SQLite database, extracted `revlog`, `cards`, and `notes`, and built a feature-engineered DataFrame. Key features:

- `retained = (ease > 1)` — binary retention flag
- `ease_factor = factor / 1000` — current ease factor (Anki default = 2.5)
- `log_ivl = log(interval + 1)` — log-transformed review interval
- `hour`, `dow` — hour of day and day of week from the ms timestamp
- `days_since_last` — inter-review gap per card

### Findings

**Retention by card maturity** (Figure 3): Cards in the initial learning phase (interval < 1 day) show the most variance. Cards with intervals > 60 days maintain near-perfect and stable retention, confirming the spaced-repetition algorithm is working as designed.

**Time of day:** Review volume peaks in the 19:00–22:00 window. Morning reviews (06:00–12:00) account for only ~1,375 of 59,465 total reviews — a significant class imbalance that complicates morning/evening comparisons.

**Session fatigue:** No clear degradation of retention rate as a function of card number within session, though response time shows mild upward drift after card 40.

---

## 4. Phase 2 — Interrupted Time Series (ITS)

**File:** `anki_causal_analysis.ipynb`, Cells 9–12

### Methodology

Modelled daily retention rate as a segmented regression with a detected intervention point:

```
retention_t = β₀ + β₁·t + β₂·D_t + β₃·(t − t*)·D_t + ε
```

where `D_t = 1` after the intervention. Heteroscedasticity-robust standard errors (HC3). The intervention point was auto-detected as the date of the largest single-day review-volume shift: **September 13, 2025**.

### Results

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| β₂ (level change) | −0.014 | Retention dropped ~1.4 pp at the detected intervention |
| β₃ (slope change) | −0.000136/day | Slight post-intervention deceleration in daily trend |
| R² | 0.265 | Model explains 26.5% of variance in daily retention |

### Interpretation

The small negative level change suggests a behavioral shift around September 2025 — likely a period of inconsistent review or introduction of harder material — but the effect is small in absolute terms and the confidence is low. The critical caveat is that the intervention point was detected algorithmically rather than pre-specified. Any life event coinciding with September 2025 could explain the level change. Without a pre-registered intervention date, this result is exploratory only.

---

## 5. Phase 3 — PSM: Morning vs Evening Retention

**File:** `anki_causal_analysis.ipynb`, Cells 13–20

### Methodology

Estimated the Average Treatment Effect (ATE) of reviewing in the morning (06:00–12:00) versus the evening (18:00–24:00) on binary retention (`ease > 1`).

**Propensity model covariates:** ease factor, log-interval, log-days since last review  
**Matching:** 1:1 nearest-neighbour on log-odds, 0.2-SD caliper  
**Sensitivity:** Rosenbaum bounds analysis

### Results

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

### Interpretation

**Not significant.** The ATE is statistically and practically zero. The Rosenbaum bounds show extreme robustness — even a confounder that tripled the odds of morning review would not change the conclusion.

This is a real finding, not a failure to detect: **time of day does not affect whether I remember a card.** The ceiling effect (99.3% baseline retention) means there is almost no variation left to explain. The meaningful question is not whether I remember, but *how confidently and quickly* I remember.

---

## 6. Phase 4 — Survival Analysis: Time to First Lapse

**File:** `anki_causal_analysis.ipynb`, Cells 21–26

### Methodology

Built a survival dataset where each card's "time at risk" is its total review count (`reps`), and the event is receiving at least one lapse (`lapses ≥ 1`).

**Why first-lapse instead of classic leech (≥8)?** Only 1 card in the dataset reached the 8-lapse threshold. The deck is too young (511 days) for the classic survival definition to be usable. First-lapse is an earlier, more common, and still-meaningful event.

**Models:**
- Kaplan-Meier curves stratified by ease-factor quartile
- Cox proportional hazards model with ease factor as primary covariate

### Results

**Lapse rate by ease-factor quartile:**

| Quartile | Cards | Lapse rate |
|----------|-------|------------|
| Q1 — Hardest (ef ≤ 2.65) | 2,742 | **7.5%** |
| Q2 | 2,445 | 0.6% |
| Q3 | 2,890 | 0.5% |
| Q4 — Easiest | 925 | 0.4% |

**Cox PH:** HR = 2.06, p ≈ 0.000. A one-unit increase in ease factor is associated with a 2.06× change in lapse hazard.  
**Log-rank test (Q1 vs Q4):** Significant — harder cards lapse significantly sooner.  
**Proportional hazards check:** Schoenfeld test p = 0.025. Mild violation — the hazard ratio is not constant over time; the ease factor effect on lapse risk changes as cards age.

### Interpretation

**This is the strongest finding in the project.** A small cluster of structurally hard cards — the bottom ease-factor quartile — is responsible for nearly all lapses (7.5% first-lapse rate vs 0.4% for the easiest quartile). These cards are identifiable early in their review history from their ease factor trajectory. If I were to intervene on "high-risk" cards, I should target cards with `factor < 2.65` (below Anki's default starting value of 2.5).

---

## 7. Phase 5 — PSM: Morning vs Evening Response Time on Hard Cards

### Why response time?

Binary retention was flat at 99.3%. Response time is a continuous, more sensitive outcome — it captures retrieval fluency even on reviews I ultimately press Good or Easy on.

### v1: Hard cards defined by current ease factor (flawed)

**File:** `psm_response_time.ipynb`

**Definition:** Hard card = `cards.factor ≤ 2.65` (bottom 25% of *current* ease factors as of April 2026).

**Problem discovered:** This definition looks backwards in time. Cards that are *currently* hard had 82% of their reviews concentrated in early 2025, when I was just starting out. There were no morning hard-card reviews at all in January–April 2025. The morning/evening comparison was heavily confounded by my learning curve — early reviews were slower regardless of time of day.

**Results:** ATE median = +1,111 ms (morning slower), 95% CI [+557, +1,634], Mann-Whitney p < 0.0001.  
**Verdict:** Statistically significant but not trustworthy — conflated "hard card today" with "card I reviewed when I was a beginner."

### v2: Hard reviews defined at review time (corrected)

**File:** `psm_response_time_v2.ipynb`

**Definition:** A review is hard if `revlog.factor < 2.5` at the moment of that review. The value 2.5 is Anki's default starting ease factor — any card below it has been actively downgraded due to repeated failures. This is temporally correct.

**Additional covariate:** `month_idx` (calendar month as a numeric index) to partially absorb the remaining learning-curve effect.

### Results

| Metric | Value |
|--------|-------|
| Hard reviews qualifying | 4,252 |
| Matched pairs | 309 |
| ATE (mean diff) | +508 ms, 95% CI [−858, +1,844], t-test p = 0.498 |
| ATE (median diff) | **+1,084 ms**, 95% CI [+209, +1,924], Mann-Whitney p = 0.0045 |

**Post-match balance:** `month_idx` SMD = 0.117 (slightly above 0.1 threshold). Structural limitation: there are no morning hard-card reviews before May 2025, so the counterfactual for early-period reviews cannot be matched.

### Interpretation

Morning reviews on hard cards remain ~1 second slower even with the corrected definition. The mean-based ATE is non-significant because a handful of very long reviews inflate the mean; the median-based ATE is more robust and remains significant. The residual `month_idx` imbalance means this estimate carries uncertainty — the effect could partly reflect calendar time rather than time of day. My morning hard-card reviews are slower, but whether that represents cognitive sluggishness, more careful deliberation, or residual confounding cannot be determined from response time alone.

---

## 8. Phase 6 — 60-Day Randomized A/B Experiment: Read Aloud vs Silent Review

### Design

| | Treatment | Control |
|---|---|---|
| Tag | `exp::treatment` | `exp::control` |
| Notes | 1,745 | 1,745 |
| Intervention | Card marked with ★ — read the sentence out loud before rating | Silent review as normal |
| Assignment | Random shuffle, seed=42, 50/50 split |
| Start | April 24, 2026 |
| End | June 23, 2026 (Day 61) |

The ★ marker was prepended to both the Japanese field (field[1]) and English meaning field (field[2]) of all treatment notes using `mark_treatment_cards.py`, which writes directly to the Anki SQLite database. No content was altered — only the marker was added.

**Research question:** Does reading a sentence card out loud during review cause a measurable reduction in lapse rate and response time over a 60-day window, compared to identical cards reviewed silently?

**Lapse definition:** `ease ≤ 2` (Again or Hard). Both buttons trigger algorithmic downgrade; Hard is defined as the primary lapse signal because it reflects genuine retrieval difficulty even when the user decides not to press Again.

**Causal assumptions:**
- **SUTVA:** Reviewing a treatment card does not affect performance on control cards in the same session.
- **No anticipation:** Behavior on control cards is not affected by knowing treatment cards exist.
- **Randomization holds:** Assignment was purely random; baseline ease distributions are identical by design.

### Primary Outcomes

**7,346 total reviews over 61 days (3,648 treatment | 3,698 control)**

| Outcome | Treatment | Control | Delta | p-value |
|---------|-----------|---------|-------|---------|
| Lapse rate (ease ≤ 2) | 11.81% | 10.76% | **+1.05 pp** | 0.154 *(n.s.)* |
| Median response time (non-capped) | 5.13s | 4.65s | **+482 ms** | 0.0001 *** |

**Ease-button breakdown:**

| Button | Treatment | Control |
|--------|-----------|---------|
| Again | 0.2% | 0.3% |
| **Hard** | **11.7%** | **10.5%** |
| Good | 13.2% | 14.5% |
| Easy | 75.0% | 74.7% |

**Bayesian analysis:**

Using a Beta-Binomial conjugate model with 200,000 Monte Carlo draws:

| Metric | Value |
|--------|-------|
| P(Treatment lapse rate < Control) | **7.7%** |
| Median lapse reduction | −9.8% (treatment is worse, not better) |
| 90% credible interval | [−22.3%, +1.4%] |

### Weekly Trend

| Week | Treatment | Control | Gap |
|------|-----------|---------|-----|
| 1 | 12.4% | 5.2% | **+7.1 pp** |
| 2 | 7.2% | 6.1% | +1.1 pp |
| 3 | 8.4% | 10.9% | −2.5 pp |
| 4 | 17.8% | 12.5% | +5.3 pp |
| 5 | 17.1% | 13.9% | +3.2 pp |
| 6 | 12.5% | 12.0% | +0.5 pp |
| 7 | 12.9% | 15.5% | −2.6 pp |
| 8 | 7.7% | 10.4% | −2.7 pp |
| 9 | 9.5% | 9.2% | +0.3 pp |

The weekly trajectory shows no systematic convergence. Week 1's +7.1 pp gap biased the overall result; weeks 7–8 show treatment performing better. There is no trend consistent with a cumulative benefit of reading aloud.

### Supplementary Tests

**Bayesian sequential monitoring** (4 pre-specified checkpoints at days 15, 30, 45, 60, Bonferroni-corrected α/4 = 0.0125, |z| ≥ 2.50 threshold):

| Day | T lapse | C lapse | z-stat | Stop? |
|-----|---------|---------|--------|-------|
| 15 | ~12% | ~6% | ~2.0 | No |
| 30 | ~11% | ~9% | ~0.9 | No |
| 45 | ~12.5% | ~10.5% | ~1.5 | No |
| 60 | 11.81% | 10.76% | ~1.4 | No |

The sequential test never crossed the stopping boundary in either direction.

---

## 9. Kanji Complexity Analysis

**Research question:** On days when treatment lapse rate exceeded control lapse rate, were treatment cards disproportionately kanji-heavy compared to control cards? Does card linguistic complexity explain the day-to-day variation?

**Method:** For each review, extracted the Japanese field (field[1]) and counted kanji characters (Unicode range U+4E00–U+9FFF). Computed daily average kanji count per group. Classified each day as T-worse (treatment lapse > control lapse) or T-better.

### Results

**Daily performance split across 61 days:**

| Category | Days | Share |
|----------|------|-------|
| Treatment worse | 29 | 48% |
| Treatment better | 31 | 51% |
| Tied | 1 | 2% |

**Average kanji per card by day type:**

| Day type | Treatment | Control | T − C delta |
|----------|-----------|---------|-------------|
| T-worse days | 1.71 | 1.90 | **−0.19** |
| T-better days | 1.58 | 1.78 | **−0.20** |

**Correlation of daily (T − C) kanji delta vs (T − C) lapse rate gap:**

| Test | r | p-value |
|------|---|---------|
| Pearson | −0.038 | 0.774 |
| Spearman | +0.027 | 0.835 |

### Interpretation

**Kanji density does not explain treatment/control variation.** Two key observations:

1. Treatment cards consistently have *fewer* kanji per card than control cards on both good and bad days (−0.19 and −0.20 delta respectively). This means treatment cards are actually linguistically simpler on average — which rules out "treatment got harder cards" as an explanation for the lapse gap.

2. The correlation between kanji load difference and lapse rate gap is effectively zero (r = −0.038). Day-to-day variation is noise from small daily sample sizes (~50–70 reviews per group), not a signal of card difficulty.

The extreme outlier (April 27: +25.9 pp gap) had only 45 treatment reviews on that day — one or two difficult cards can swing a small-N daily lapse rate by 20+ pp. The daily variation is a sampling artifact, not a real pattern.

---

## 10. Final Conclusions

### Summary of Findings

| Analysis | Finding | Confidence |
|----------|---------|------------|
| ITS: Did Sep 2025 shift change retention? | Small negative level change (−1.4 pp) | Low — intervention detected algorithmically |
| PSM: Does morning review improve retention? | No effect (ATE = −0.07 pp, p = 0.83) | High — retention ceiling, essentially no variation to detect |
| Survival: Do harder cards lapse sooner? | Yes — Q1 cards lapse at 7.5% vs 0.4% for easiest (HR = 2.06, p < 0.001) | High — large, significant, consistent across KM and Cox |
| PSM v2: Are morning hard-card reviews slower? | Yes — morning ~1,084 ms slower (median, p = 0.005) | Moderate — significant but residual temporal confound |
| A/B Experiment: Does read-aloud reduce lapses? | No — null result (+1.05 pp, p = 0.154; Bayesian P(better) = 7.7%) | High — randomized, adequately powered for effects >4 pp |
| A/B Experiment: Does read-aloud slow RT? | Yes — +482 ms median (Mann-Whitney p = 0.0001) | High — consistent across all weeks |
| Kanji complexity: Does it explain daily T/C gap? | No — r = −0.038, p = 0.77; treatment cards have *less* kanji | High — systematic null across both correlation and stratification |

### The Big Picture

My retention is excellent (99.3%). The meaningful variation in this dataset is not *whether* I remember cards but *which* cards are structurally difficult.

**What is actually driving my learning:**
- A small cluster of hard cards (ease factor below Anki's default 2.5) concentrates nearly all lapses and takes noticeably longer to answer.
- These cards can be identified from their ease factor trajectory within the first few months of review.
- Time of day, morning vs evening — does not matter for whether I remember.
- Reading aloud during review — does not help retention, costs ~0.5 seconds per card, and appears to raise self-assessment stringency (more Hard presses) without reducing Again presses.

**Why the A/B result is valuable:**

A null result from a properly randomized experiment is valid science. The read-aloud hypothesis was theoretically grounded (phonological encoding, dual-coding theory) and practically implemented. It failed not because of experimental error but because the mechanism — adding an output modality to a visual recognition task — was apparently not strong enough to change the underlying memory trace for Japanese vocabulary flashcards. For Anki-style recognition tasks, silent review appears to be as effective as read-aloud review, and approximately 10% faster per card.

### What I Would Do Next

1. **Longer experiment horizon** — 60 days was adequately powered for effects > 4 pp. For effects of 1–2 pp, the experiment would need ~6 months and ~15,000+ reviews per group.
2. **Card-content analysis** — Hard cards likely share phonetic or orthographic features (similar kanji, phonetic homophony) that explain structural difficulty. JLPT level, stroke count, and phonetic similarity are all computable from the `notes.flds` field.
3. **Dose-response within session** — Does my response time degrade as I review card #50 vs card #1 in a session? Model review sequence number as a running variable.
4. **Alternative treatment modalities** — Written output (typing the answer) or image-based SRS may produce larger and detectable effects. The key question is which output modality most changes the encoding depth for Japanese vocabulary specifically.
5. **Survival analysis at 24 months** — When the deck has matured, re-run survival analysis with the classic 8-lapse leech threshold now meaningful for a larger fraction of cards.

---

## Appendix: Technical Stack

**Environment:** Python 3.11, Windows 11

**Dependencies:**
```
pandas>=2.0.0          # data manipulation
numpy>=1.25.0          # numerical operations
scipy>=1.11.0          # Mann-Whitney, Pearson/Spearman, proportion z-test
statsmodels>=0.14.0    # ITS segmented regression, HC3 SE, Bonferroni
scikit-learn>=1.3.0    # logistic regression, nearest-neighbour PSM
lifelines>=0.27.0      # Kaplan-Meier, Cox PH, Schoenfeld test
plotly>=5.18.0         # interactive charts in Streamlit dashboard
streamlit>=1.35.0      # live experiment dashboard
```

**Key statistical decisions:**

| Decision | Rationale |
|----------|-----------|
| Mann-Whitney as primary RT test | Response times are heavily right-skewed. Mann-Whitney makes no distributional assumption and tests stochastic dominance directly. |
| Exclude `revlog.time = 60,000 ms` | Anki's hard cap records any review exceeding 60s as exactly 60,000 ms. Not valid response-time measurements. |
| Lapse = `ease ≤ 2` (Again + Hard) | Hard presses represent genuine retrieval difficulty and trigger algorithmic downgrade. Using `ease = 1` only (Again) gave a 0.7% lapse rate — insufficient events for analysis. |
| `revlog.factor` (not `cards.factor`) for hard-review definition | `cards.factor` is a current snapshot; `revlog.factor` captures difficulty at the moment of review. |
| Beta-Binomial Bayesian with 200k draws | Conjugate prior avoids MCMC overhead; 200k draws gives stable credible interval estimates. |
| Bonferroni α/4 for sequential test | Pre-specified 4 monitoring checkpoints; family-wise error correction. |
| SQL JOIN (not Python IN list) | SQLite 999-variable limit silently returns empty results for `WHERE cid IN (?,?...)` with >999 card IDs. JOIN avoids this constraint. |

---

*Dashboard available at: [Streamlit Cloud deployment](https://causal-inference-on-spaced-repetition-learning.streamlit.app/)*  
*Source code: [GitHub Repository](https://github.com/sid-pottapatri/Causal-Inference-on-Spaced-Repetition-Learning-Data)*
