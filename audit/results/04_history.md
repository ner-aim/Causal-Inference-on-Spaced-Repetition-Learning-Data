### Repository state

```
HEAD e07e50d310bca7a957b600ab1c39da7a2dbf5d7d
git status --short: ?? audit/
git diff HEAD --stat -- dashboard.py export_data.py: (no differences: audited code equals committed code)
```

### Commit log (every commit message in this repo)

```
e07e50d 2026-07-10 updated
24396c8 2026-06-23 updated
c508eb0 2026-06-14 updated
834afea 2026-06-08 updated
17f3e41 2026-06-07 updated
cbe2f9a 2026-06-02 updated
932ba64 2026-05-22 updated
ca6f97c 2026-05-19 updated
7727b2d 2026-05-18 updated
6613d51 2026-05-18 updated
f8d59d4 2026-05-18 updated
21455be 2026-05-15 updated
8e35ed8 2026-05-10 updated
03aeda6 2026-05-09 updated
a09da48 2026-05-04 updated
bbdcc7e 2026-05-03 updated
d947dae 2026-05-02 updated
50f6542 2026-05-01 Updated files
d7550da 2026-04-30 Updated files
93139ed 2026-04-26 Updated Export data
111faac 2026-04-25 Update dashboard.py
2cf5976 2026-04-25 Update dashboard.py
39624eb 2026-04-25 Updated Experiment metrics
77ed41c 2026-04-24 Update dashboard.py
1b1f8b6 2026-04-24 Updated Dashboard
cf7b6f0 2026-04-24 Updated Files
5a967f2 2026-04-24 New Project for Causal inference
ab291be 2026-04-24 Initial commit
```

### git log -S / -G for the four terms

```
git log --all -S'leak'      -> 0 commits
    (none)
git log --all -i -G'leak'   -> 0 commits
    (none)
git log --all -S'timing'      -> 0 commits
    (none)
git log --all -i -G'timing'   -> 0 commits
    (none)
git log --all -S'shift'      -> 4 commits
    e07e50d 2026-07-10 updated
    39624eb 2026-04-25 Updated Experiment metrics
    cf7b6f0 2026-04-24 Updated Files
    5a967f2 2026-04-24 New Project for Causal inference
git log --all -i -G'shift'   -> 4 commits
    e07e50d 2026-07-10 updated
    39624eb 2026-04-25 Updated Experiment metrics
    cf7b6f0 2026-04-24 Updated Files
    5a967f2 2026-04-24 New Project for Causal inference
git log --all -S'lag'      -> 2 commits
    e07e50d 2026-07-10 updated
    5a967f2 2026-04-24 New Project for Causal inference
git log --all -i -G'lag'   -> 2 commits
    e07e50d 2026-07-10 updated
    5a967f2 2026-04-24 New Project for Causal inference
```

### Matching lines inside the shift/lag hit commits

```
term=shift commit=e07e50d: 4 matching +/- lines (non-notebook files)
    +1. **Did a behavioral shift in September 2025 change my retention rate?** (Interrupted Time Series)
    +where `D_t = 1` after the intervention. Heteroscedasticity-robust standard errors (HC3). The intervention point was auto-detected as the date of the largest single-day r
    +The small negative level change suggests a behavioral shift around September 2025 — likely a period of inconsistent review or introduction of harder material — but the e
    +| ITS: Did Sep 2025 shift change retention? | Small negative level change (−1.4 pp) | Low — intervention detected algorithmically |
term=shift commit=39624eb: 1 matching +/- lines (non-notebook files)
    -        "Intervention auto-detected at **September 13, 2025** — largest single-day volume shift."
term=shift commit=cf7b6f0: 1 matching +/- lines (non-notebook files)
    +        "Intervention auto-detected at **September 13, 2025** — largest single-day volume shift."
term=shift commit=5a967f2: 2 matching +/- lines (non-notebook files)
    +I modelled my daily retention rate as a segmented regression with a level change (β₂) and slope change (β₃) at a detected intervention point. I used heteroscedasticity-r
    +| ITS: Did Sep 2025 shift change retention? | Small negative level change (−1.4 pp) | Low — intervention detected algorithmically, not pre-specified |
term=lag commit=e07e50d: 1 matching +/- lines (non-notebook files)
    +- `retained = (ease > 1)` — binary retention flag
term=lag commit=5a967f2: 1 matching +/- lines (non-notebook files)
    +I connected to my Anki SQLite database, extracted the `cards`, `revlog`, and `decks` tables, and built a feature-engineered DataFrame. Key features I created: datetime c
```

### Working-tree grep for leak|timing|shift|lag (case-insensitive)

| file | where | text |
|---|---|---|
| assign_experiment_tags.py | line 241 | 3. File → Import  (or Ctrl+Shift+I). |
| ANALYSIS.md | line 17 | 1. **Did a behavioral shift in September 2025 change my retention rate?** (Interrupted Time Series) |
| ANALYSIS.md | line 99 | where `D_t = 1` after the intervention. Heteroscedasticity-robust standard errors (HC3). The intervention point was auto-detected as the date of the l |
| ANALYSIS.md | line 111 | The small negative level change suggests a behavioral shift around September 2025 — likely a period of inconsistent review or introduction of harder m |
| ANALYSIS.md | line 356 | | ITS: Did Sep 2025 shift change retention? | Small negative level change (−1.4 pp) | Low — intervention detected algorithmically | |
| README.md | line 76 | I modelled my daily retention rate as a segmented regression with a level change (β₂) and slope change (β₃) at a detected intervention point. I used h |
| README.md | line 198 | | ITS: Did Sep 2025 shift change retention? | Small negative level change (−1.4 pp) | Low — intervention detected algorithmically, not pre-specified | |
| anki_causal_analysis.ipynb | cell 0 (markdown), line 14 | | 2 | Interrupted Time Series (ITS) | Did behavioural changes shift retention trends? | |
| anki_causal_analysis.ipynb | cell 9 (markdown), line 11 | **Intervention identification:** We detect natural breakpoints by looking for structural shifts in daily review volume (≥2 std deviations) or multi-da |
| anki_causal_analysis.ipynb | cell 10 (code), line 9 | ts['prev_date'] = ts['date'].shift(1) |
| anki_causal_analysis.ipynb | cell 14 (code), line 24 | psm_df['prev_ts'] = psm_df.groupby('cid')['ts'].shift(1) |
| psm_response_time.ipynb | cell 3 (code), line 15 | df['prev_ts'] = df.groupby('cid')['ts'].shift(1) |
| psm_response_time_v2.ipynb | cell 4 (code), line 6 | df['prev_ts']       = df.groupby('cid')['ts'].shift(1) |

### Pickaxe on the tokens that carry the v1 -> v2 change

```
-S'revlog.factor': e07e50d 2026-07-10, 39624eb 2026-04-25, cf7b6f0 2026-04-24, 5a967f2 2026-04-24
-S'cards.factor': e07e50d 2026-07-10, 5a967f2 2026-04-24
-S'at review time': e07e50d 2026-07-10, 39624eb 2026-04-25, cf7b6f0 2026-04-24, 5a967f2 2026-04-24
-S'current snapshot': e07e50d 2026-07-10, 5a967f2 2026-04-24
-S'point-in-time': (none)
-S'days_since': e07e50d 2026-07-10, 5a967f2 2026-04-24
-S'lastIvl': 5a967f2 2026-04-24
-S'.shift(': 5a967f2 2026-04-24
-S'run_dml': e07e50d 2026-07-10
```

### v1 / v2 notebooks: git history against file timestamps

| file | added in commit | commits touching it | file mtime (Eastern) |
|---|---|---|---|
| psm_response_time.ipynb | 5a967f2 2026-04-24 13:59:56 -0400 | 1 | 2026-04-23 15:14 |
| psm_response_time_v2.ipynb | 5a967f2 2026-04-24 13:59:56 -0400 | 1 | 2026-04-23 15:37 |
| anki_causal_analysis.ipynb | 5a967f2 2026-04-24 13:59:56 -0400 | 1 | 2026-04-23 14:44 |
| psm_response_time_executed.ipynb | 5a967f2 2026-04-24 13:59:56 -0400 | 1 | 2026-04-23 15:15 |
| psm_response_time_v2_executed.ipynb | 5a967f2 2026-04-24 13:59:56 -0400 | 1 | 2026-04-23 15:37 |
