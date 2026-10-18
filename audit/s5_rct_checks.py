"""s5_rct_checks.py - audit of the 60-day read-aloud RCT (the headline result; not a DML/PSM analysis).

1. Reproduce the reported numbers (7,346 reviews; lapse 11.81% vs 10.76%; z-test p; Mann-Whitney p; Bayesian 7.7%).
2. Treatment timing: when did the star marker actually land on the treatment notes, and how many reviews of the
   treatment arm precede it?
3. Inference that respects the unit of randomisation: assignment was by NOTE, the z-test treats REVIEWS as independent.
   Cluster-robust SEs and a re-randomisation test over all 3,490 notes (B=5000).
4. Placebo checks: pre-experiment lapse rate and response time of the same notes (outcome measured before the
   treatment existed), and kanji count / deck position of the notes (cannot be affected).

Writes results/05_rct.json, results/05_rct.md, figures/rct_rerandomisation.png.

Usage (repo root):  ANKI_DB=<collection copy> python audit/s5_rct_checks.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

import audit_lib as L

L.utf8_stdout()
warnings.filterwarnings("ignore")
B = 5000
SEED = 20260422
START_UTC = pd.Timestamp("2026-04-24 04:00")  # local midnight of 2026-04-24 (EDT): what dashboard.py's datetime.combine(...).timestamp() gives
END_UTC = pd.Timestamp("2026-06-24")          # reproduces the reported 7,346 reviews exactly
OUT, MD = {}, []


def md(title, df=None, text=None, fmt="{:.4g}"):
    MD.append(f"### {title}\n")
    if text:
        MD.append(text + "\n")
    if df is not None:
        MD.append(L.md_table(df, fmt) + "\n")
    print(f"\n## {title}")
    if text:
        print(text)
    if df is not None:
        print(L.md_table(df, fmt))


con = L.connect()
notes = pd.read_sql("SELECT id, tags, mod, flds FROM notes", con)
rev = pd.read_sql("SELECT r.id, r.cid, r.ease, r.time, c.nid FROM revlog r JOIN cards c ON r.cid = c.id", con)
con.close()

notes["jp"] = notes["flds"].map(lambda f: f.split("\x1f")[1] if "\x1f" in f else "")
notes["kanji"] = notes["jp"].map(L.count_kanji)
notes["deck_pos"] = notes["id"].rank(pct=True)
exp_notes = notes[notes["tags"].str.contains("exp::")].copy()
exp_notes["treat"] = exp_notes["tags"].str.contains("exp::treatment").astype(int)  # same rule as dashboard.py
rev["ts"] = pd.to_datetime(rev["id"], unit="ms")
rev = rev.merge(exp_notes[["id", "treat"]].rename(columns={"id": "nid"}), on="nid", how="inner")
rev["lapse"] = (rev["ease"] <= 2).astype(int)
rev["valid_rt"] = rev["time"] != L.ANKI_CAP_MS  # dashboard.py: capped == (time == 60000) is excluded, nothing else
rev["log_rt"] = np.where(rev["valid_rt"], np.log(rev["time"].clip(lower=1)), np.nan)
OUT["n_notes"] = {"treatment": int(exp_notes["treat"].sum()), "control": int((1 - exp_notes["treat"]).sum())}

# star marker timing (notes.mod is rewritten by mark_treatment_cards.py)
tmod = pd.to_datetime(exp_notes.loc[exp_notes["treat"] == 1, "mod"], unit="s")
marker_ts = tmod.mode().iloc[0]
n_at_marker = int((tmod == marker_ts).sum())
has_star = exp_notes.loc[exp_notes["treat"] == 1, "jp"].str.startswith("★").mean()
OUT["marker"] = {"marker_utc": marker_ts.isoformat(), "notes_written_at_that_second": n_at_marker, "share_treatment_notes_with_star_now": float(has_star),
                 "analysis_start_utc": START_UTC.isoformat()}
print("marker written:", marker_ts, "to", n_at_marker, "notes; share with star now:", has_star)

post = rev[(rev["ts"] >= START_UTC) & (rev["ts"] < END_UTC)].copy()
pre = rev[rev["ts"] < START_UTC].copy()
post_marker = post[post["ts"] >= marker_ts].copy()


def headline(frame, label):
    t, c = frame[frame["treat"] == 1], frame[frame["treat"] == 0]
    tl, cl = int(t["lapse"].sum()), int(c["lapse"].sum())
    z, p = proportions_ztest([tl, cl], [len(t), len(c)])
    tr, cr = t.loc[t["valid_rt"], "time"].to_numpy(), c.loc[c["valid_rt"], "time"].to_numpy()
    mw = stats.mannwhitneyu(tr, cr, alternative="two-sided").pvalue
    lm = smf.ols("lapse ~ treat", data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame["nid"]})
    fr = frame[frame["valid_rt"]]
    rm = smf.ols("log_rt ~ treat", data=fr).fit(cov_type="cluster", cov_kwds={"groups": fr["nid"]})
    return {"sample": label, "reviews T": len(t), "reviews C": len(c), "lapse % T": 100 * tl / len(t), "lapse % C": 100 * cl / len(c),
            "diff pp": 100 * (tl / len(t) - cl / len(c)), "z-test p (reviews independent)": float(p),
            "cluster-by-note p (lapse)": float(lm.pvalues["treat"]), "median RT T s": float(np.median(tr) / 1000), "median RT C s": float(np.median(cr) / 1000),
            "Mann-Whitney p": float(mw), "cluster-by-note p (log RT)": float(rm.pvalues["treat"]), "notes with reviews": int(frame["nid"].nunique())}


h_all = headline(post, "as reported: 2026-04-24 to 2026-06-23")
h_pm = headline(post_marker, "from the moment the star marker existed")
md("1. Reported result and the same test with note-level clustering", pd.DataFrame([h_all, h_pm]), fmt="{:.4g}",
   text="Reported: 7,346 reviews (3,648 T / 3,698 C), lapse 11.81% vs 10.76%, p = 0.154; median RT 5.13 s vs 4.65 s.")
OUT["headline"] = [h_all, h_pm]

# Bayesian as in dashboard.py (Beta(1+lapses, 1+non-lapses), 200k draws, seed 42)
t_, c_ = post[post["treat"] == 1], post[post["treat"] == 0]
rng = np.random.default_rng(42)
p_t = rng.beta(1 + t_["lapse"].sum(), 1 + (len(t_) - t_["lapse"].sum()), size=200_000)
p_c = rng.beta(1 + c_["lapse"].sum(), 1 + (len(c_) - c_["lapse"].sum()), size=200_000)
OUT["bayes_P_treatment_better"] = float(np.mean(p_t < p_c))
print("Bayesian P(treatment better) =", OUT["bayes_P_treatment_better"])

# --- treatment timing
pre_marker_t = post[(post["treat"] == 1) & (post["ts"] < marker_ts)]
pre_marker_all = post[post["ts"] < marker_ts]
timing = pd.DataFrame([{"quantity": "treatment-arm reviews in the reported window dated before the marker existed", "value": len(pre_marker_t)},
                       {"quantity": "as % of all treatment-arm reviews in the reported window", "value": 100 * len(pre_marker_t) / (post["treat"] == 1).sum()},
                       {"quantity": "control-arm reviews in the same pre-marker span", "value": int((pre_marker_all["treat"] == 0).sum())},
                       {"quantity": "span in days between analysis start and marker", "value": (marker_ts - START_UTC).total_seconds() / 86400},
                       {"quantity": "lapse % treatment arm, pre-marker span", "value": 100 * pre_marker_t["lapse"].mean()},
                       {"quantity": "lapse % control arm, pre-marker span", "value": 100 * pre_marker_all[pre_marker_all["treat"] == 0]["lapse"].mean()}])
md("2. The star marker landed after the analysis window opened", timing, fmt="{:.4g}",
   text=f"notes.mod of the treatment notes: {n_at_marker:,} of {int(exp_notes['treat'].sum()):,} rewritten at {marker_ts} UTC. The analysis counts "
        f"every review from {START_UTC} UTC.")
OUT["timing"] = timing.to_dict(orient="records")

# --- re-randomisation over all notes (the unit of assignment)
exp_notes = exp_notes.reset_index(drop=True)
assign = exp_notes["treat"].to_numpy()
perms = np.empty((B, len(assign)), dtype=np.int8)
rng = np.random.default_rng(SEED)
for b in range(B):
    perms[b] = rng.permutation(assign)


def note_counts(frame):
    g = frame.groupby("nid").agg(n=("lapse", "size"), lapses=("lapse", "sum"), n_rt=("valid_rt", "sum"), sum_logrt=("log_rt", "sum"))
    return g.reindex(exp_notes["id"].to_numpy(), fill_value=0)


def ri_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)

    def stat(T):
        nt, dt = T @ num, T @ den
        return nt / dt - (num.sum() - nt) / (den.sum() - dt)
    real = float(stat(assign.astype(float)[None, :])[0])
    null = stat(perms.astype(float))
    return real, null, float((1 + np.sum(np.abs(null) >= abs(real))) / (1 + B))


rows, nulls = [], {}
for label, frame in [("reported window", post), ("from marker onward", post_marker), ("pre-experiment (same notes, before 2026-04-24)", pre)]:
    cn = note_counts(frame)
    real, null, p = ri_ratio(cn["lapses"], cn["n"])
    rows.append({"window": label, "outcome": "lapse rate (pooled)", "real diff": real * 100, "null sd": null.std(ddof=1) * 100, "unit": "pp", "re-randomisation p": p})
    nulls[(label, "lapse")] = (real, null)
    real2, null2, p2 = ri_ratio(cn["sum_logrt"], cn["n_rt"])
    rows.append({"window": label, "outcome": "mean log response time", "real diff": real2, "null sd": null2.std(ddof=1), "unit": "log RT", "re-randomisation p": p2})
for label, col in [("kanji count of note", "kanji"), ("deck position of note", "deck_pos")]:
    v = exp_notes[col].to_numpy(float)
    real, null, p = ri_ratio(v, np.ones_like(v))
    rows.append({"window": "(cannot be affected)", "outcome": label, "real diff": real, "null sd": null.std(ddof=1), "unit": "", "re-randomisation p": p})
tri = pd.DataFrame(rows)
md("3-4. Re-randomisation over all 3,490 notes (B=5000) and placebo outcomes", tri, fmt="{:.4g}",
   text="Statistic: treatment minus control. The null re-draws the 1,745/1,745 split over notes and recomputes the same statistic.")
OUT["rerandomisation"] = rows

# like-for-like re-randomisation for the reported response-time statistic (pooled median difference, capped excluded)
def ri_median(frame):
    fr = frame[frame["valid_rt"]]
    idx = pd.Index(exp_notes["id"]).get_indexer(fr["nid"])
    t_real = fr["treat"].to_numpy()
    rt = fr["time"].to_numpy(float)
    real = float(np.median(rt[t_real == 1]) - np.median(rt[t_real == 0]))
    null = np.empty(B)
    for b in range(B):
        T = perms[b][idx]
        null[b] = np.median(rt[T == 1]) - np.median(rt[T == 0])
    return real, null, float((1 + np.sum(np.abs(null) >= abs(real))) / (1 + B))


rows_m = []
for label, frame in [("reported window", post), ("from marker onward", post_marker), ("pre-experiment (same notes)", pre)]:
    real, null, p = ri_median(frame)
    rows_m.append({"window": label, "statistic": "pooled median RT difference (ms)", "real diff": real, "null sd": null.std(ddof=1),
                   "null 2.5%": np.percentile(null, 2.5), "null 97.5%": np.percentile(null, 97.5), "re-randomisation p": p})
    nulls[(label, "median_rt")] = (real, null)
tm = pd.DataFrame(rows_m)
md("3b. Re-randomisation test on the REPORTED response-time statistic (pooled median difference)", tm, fmt="{:.4g}",
   text="Reported: +482 ms, Mann-Whitney p = 0.0001 (treats 7,346 reviews as independent). Here the 3,490 notes are re-assigned instead.")
OUT["rerandomisation_median_rt"] = rows_m

# difference-in-differences against the same notes' pre-experiment gap (same re-randomisations)
cn_post, cn_pre = note_counts(post), note_counts(pre)


def did_stat(T):
    def gap(cn):
        num, den = cn["lapses"].to_numpy(float), cn["n"].to_numpy(float)
        nt, dt = T @ num, T @ den
        return nt / dt - (num.sum() - nt) / (den.sum() - dt)
    return gap(cn_post) - gap(cn_pre)


real_did = float(did_stat(assign.astype(float)[None, :])[0])
null_did = did_stat(perms.astype(float))
did = {"post gap pp": 100 * nulls[("reported window", "lapse")][0], "pre gap pp": 100 * nulls[("pre-experiment (same notes, before 2026-04-24)", "lapse")][0],
       "difference-in-differences pp": 100 * real_did, "null sd pp": 100 * null_did.std(ddof=1),
       "re-randomisation p": float((1 + np.sum(np.abs(null_did) >= abs(real_did))) / (1 + B))}
md("3c. Lapse-rate gap after minus the gap before the experiment (same notes)", pd.DataFrame([did]), fmt="{:.4g}",
   text="Both gaps are treatment minus control on the same 3,490 notes; the pre-period reviews cannot reflect the treatment.")
OUT["did_lapse"] = did

# pre-period descriptive balance
pre_tab = pd.DataFrame([{"arm": "treatment" if k == 1 else "control", "pre-period reviews": len(g), "lapse %": 100 * g["lapse"].mean(),
                         "median RT s": float(np.median(g.loc[g["valid_rt"], "time"]) / 1000)} for k, g in pre.groupby("treat")])
md("4b. The same notes before the experiment existed", pre_tab, fmt="{:.4g}")
OUT["pre_period"] = pre_tab.to_dict(orient="records")

L.write_json("05_rct.json", OUT)
with open(L.RESULTS / "05_rct.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(MD))

plt = L.mpl_style()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
real, null = nulls[("reported window", "lapse")]
real_pm, _ = nulls[("from marker onward", "lapse")]
ax.hist(null * 100, bins=50, color=L.GREY, label=f"{B} re-randomisations")
ax.axvline(real * 100, color=L.BLUE, lw=2.2, label="reported window")
ax.axvline(real_pm * 100, color=L.ORANGE, lw=2.2, label="from marker onward")
ax.set_xlabel("treatment minus control lapse rate (percentage points)")
ax.set_ylabel("re-randomisations")
ax.set_title("Post-period lapse-rate difference", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax = axes[1]
real_pre, null_pre = nulls[("pre-experiment (same notes, before 2026-04-24)", "lapse")]
ax.hist(null_pre * 100, bins=50, color=L.GREY, label=f"{B} re-randomisations")
ax.axvline(real_pre * 100, color=L.BLUE, lw=2.2, label="real arms, before the experiment began")
ax.set_xlabel("treatment minus control lapse rate (percentage points)")
ax.set_title("Placebo: same notes, pre-experiment reviews", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(L.FIGURES / "rct_rerandomisation.png")
plt.close(fig)
