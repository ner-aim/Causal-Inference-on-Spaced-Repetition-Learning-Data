"""s3_leakage.py - timing, leakage and validation-design checks.

A. When does each revlog field get written? (empirical test of revlog.factor / revlog.ivl semantics)
B. Which analysis rows are in the sample only because of the review's own answer?
C. DML cross-fitting folds: how much of a test fold's card / note / future is in its training fold?
D. DML under alternative fold designs (grouped by note, time-blocked, in-fold scaling) and cluster-robust SEs.
E. Sensitivity: clock read in US Eastern time instead of UTC; point-in-time features (pre-answer factor, lastIvl,
   true previous review). Diagnostics only - the audited estimates are not replaced.
F. Survival/Cox: reconstruction of the notebook's card table from revlog and circularity checks.
G. Inventory of scalers, encoders, imputers, splits and rolling windows in the audited source.

Writes results/03_leakage.json, results/03_leakage.md, figures/leakage_timing.png, figures/leakage_dml_robustness.png.

Usage (repo root):  ANKI_DB=<collection copy> python audit/s3_leakage.py
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.metrics import roc_auc_score

import audit_lib as L

L.utf8_stdout()
warnings.filterwarnings("ignore")
N_JOBS = 16
N_ORDERS = 100
SEED_O = 20260421
REPO = L.AUDIT_DIR.parent
OUT = {}
MD = []


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


# =============================================================================================
# A. revlog field timing
# =============================================================================================
revlog, cards, notes, decks, crt = L.load_tables()
rl_all = revlog.copy()
rl_all = rl_all.sort_values(["cid", "id"])
g = rl_all.groupby("cid", sort=False)
for src, dst in [("factor", "prev_factor"), ("type", "prev_type"), ("ease", "prev_ease"), ("ivl", "prev_ivl")]:
    rl_all[dst] = g[src].shift(1)
rl_all["dfac"] = rl_all["factor"] - rl_all["prev_factor"]
m = (rl_all["type"] == 1) & (rl_all["prev_type"] == 1) & (rl_all["factor"] > 0) & (rl_all["prev_factor"] > 0)
d = rl_all[m]
by_cur = d.groupby("ease")["dfac"].agg(["mean", "std", "count"]).reset_index().rename(columns={"ease": "ease of THIS review"})
by_prev = d.groupby("prev_ease")["dfac"].agg(["mean", "std", "count"]).reset_index().rename(columns={"prev_ease": "ease of PREVIOUS review"})
md("A1. Change in revlog.factor against the previous row of the same card (review -> review rows)",
   by_cur, text=f"{len(d):,} consecutive review rows. If revlog.factor were the factor BEFORE the answer, the change would line up with the "
                "previous row's button instead.")
md("A1b. Same change, grouped by the PREVIOUS row's button", by_prev)
OUT["A1_dfactor_by_current_ease"] = by_cur.to_dict(orient="records")
OUT["A1_dfactor_by_previous_ease"] = by_prev.to_dict(orient="records")

r1 = rl_all[rl_all["type"] == 1]
ivl_tab = r1.groupby("ease").agg(rows=("ivl", "size"), share_ivl_negative=("ivl", lambda s: float((s < 0).mean())),
                                  median_ivl=("ivl", "median"), median_lastIvl=("lastIvl", "median")).reset_index()
md("A2. revlog.ivl by button (type 1 = review rows)", ivl_tab,
   text="ivl is the interval assigned by this answer; every Again row carries ivl = -600 (the 10-minute relearning step, in seconds).")
OUT["A2_ivl_by_ease"] = ivl_tab.to_dict(orient="records")
m2 = (rl_all["prev_ivl"].notna()) & (rl_all["type"] == 1) & (rl_all["prev_type"] == 1) & (rl_all["prev_ivl"] > 0)
share_link = float((rl_all.loc[m2, "lastIvl"] == rl_all.loc[m2, "prev_ivl"]).mean())
OUT["A2_share_lastIvl_equals_previous_row_ivl"] = share_link
OUT["A2_n"] = int(m2.sum())
print(f"share lastIvl_t == ivl_(t-1): {share_link:.4f} (n={int(m2.sum()):,})")

# A3. hour-of-day, UTC against Eastern
local = pd.to_datetime(rl_all["id"], unit="ms", utc=True).dt.tz_convert(L.LOCAL_TZ)
hours = pd.DataFrame({"hour": range(24),
                      "reviews_utc_clock": pd.to_datetime(rl_all["id"], unit="ms").dt.hour.value_counts().sort_index().reindex(range(24), fill_value=0).to_numpy(),
                      "reviews_eastern_clock": local.dt.hour.value_counts().sort_index().reindex(range(24), fill_value=0).to_numpy()})
OUT["A3_hour_counts"] = hours.to_dict(orient="records")
crt_utc = pd.Timestamp(crt, unit="s")
OUT["A3_col_crt_utc"] = crt_utc.isoformat()
OUT["A3_col_crt_implied_utc_offset_hours_if_rollover_4am"] = float(4 - crt_utc.hour - crt_utc.minute / 60)
print("col.crt =", crt_utc, "-> implied UTC offset at creation if the default 4am rollover:", OUT["A3_col_crt_implied_utc_offset_hours_if_rollover_4am"])

# =============================================================================================
# B. rows that are in the sample because of the answer itself
# =============================================================================================
rl_nb, cards_, notes_, decks_, meta = L.prepare(L.CUTOFF_NOTEBOOKS)
rl_dml, *_ = L.prepare(L.CUTOFF_DML)
v2_df, v2_trim = L.build_v2(rl_nb)
dml_f = L.build_dml(rl_dml)
p3 = L.build_phase3(rl_nb)


def entered_by_answer(frame, label):
    pf = frame["prev_factor"]
    cat = np.select([pf.isna(), pf == 0, pf < 2500, pf >= 2500],
                    ["no previous row", "previous row is a learning step (factor 0)", "already hard before this answer", "NOT hard before this answer"],
                    default="?")
    t = pd.crosstab(cat, frame["morning"].map({1: "morning", 0: "evening"}))
    t["total"] = t.sum(axis=1)
    t = t.reset_index().rename(columns={"row_0": "pre-answer factor status"})
    t.columns = ["pre-answer factor status"] + list(t.columns[1:])
    tot = t[["evening", "morning", "total"]].sum()
    for c in ["evening", "morning", "total"]:
        t[c + " %"] = 100 * t[c] / tot[c]
    md(f"B. {label}: status of each sample row's factor BEFORE its own answer", t, fmt="{:.1f}")
    ent = frame[cat == "NOT hard before this answer"]
    return {"table": t.to_dict(orient="records"),
            "ease_of_rows_that_entered": {int(k): int(v) for k, v in ent["ease"].value_counts().sort_index().items()}}


OUT["B_v2"] = entered_by_answer(v2_trim, "PSM v2 sample (ef from revlog.factor < 2.5)")
OUT["B_dml"] = entered_by_answer(dml_f, "DML sample (revlog.factor < 2500)")

# B2. Phase 3: how well does a covariate predict the outcome it is supposed to precede?
auc_ivl = roc_auc_score(1 - p3["retained"], p3["log_ivl"])
tab = pd.crosstab(p3["retained"].map({0: "retained = 0 (Again)", 1: "retained = 1"}), (p3["ivl"] < 0).map({True: "ivl < 0", False: "ivl > 0"}))
md("B2. Phase 3: outcome `retained` against the covariate `ivl` sign", tab.reset_index(),
   text=f"AUC of log_ivl for predicting retained == 0: {auc_ivl:.4f}. log_ivl is log1p(|revlog.ivl|), written after the answer.")
OUT["B2_phase3_auc_log_ivl_vs_lapse"] = float(auc_ivl)
OUT["B2_phase3_crosstab"] = tab.to_dict()

# =============================================================================================
# C. DML fold structure
# =============================================================================================
Xd, Dd, Yd = dml_f[L.DML_FEATURES].to_numpy(float), dml_f["morning"].to_numpy(), dml_f["log_rt"].to_numpy()
cid_d = dml_f["cid"].to_numpy()
nid_d = dml_f["nid"].to_numpy(float)
grp_note = np.where(np.isnan(nid_d), -cid_d.astype(float), nid_d)
ts_d = dml_f["id"].to_numpy()
folds0 = L.make_folds(len(Yd))


def fold_structure(folds):
    rows = []
    for k, (tr, te) in enumerate(folds):
        max_tr = pd.Series(ts_d[tr]).groupby(cid_d[tr]).max()
        min_tr = pd.Series(ts_d[tr]).groupby(cid_d[tr]).min()
        later = max_tr.reindex(cid_d[te]).to_numpy() > ts_d[te]
        earlier = min_tr.reindex(cid_d[te]).to_numpy() < ts_d[te]
        rows.append({"fold": k, "test rows": len(te), "% test rows whose CARD is in training": 100 * np.isin(cid_d[te], cid_d[tr]).mean(),
                     "% whose NOTE is in training": 100 * np.isin(grp_note[te], grp_note[tr]).mean(),
                     "% with a LATER review of the same card in training": 100 * np.nanmean(later),
                     "% with an EARLIER review of the same card in training": 100 * np.nanmean(earlier),
                     "% of training rows dated after the test fold's first row": 100 * np.mean(ts_d[tr] > ts_d[te].min())})
    return pd.DataFrame(rows)


fs = fold_structure(folds0)
cards_per_note = cards_.groupby("nid").size()
md("C. Audited DML folds (KFold, shuffle=True, random_state=42, row level): what leaks across the train/test boundary", fs, fmt="{:.1f}",
   text=f"{len(Yd):,} rows from {len(np.unique(cid_d)):,} cards ({len(np.unique(grp_note)):,} notes); reviews per card mean "
        f"{len(Yd) / len(np.unique(cid_d)):.2f}; cards per note in the collection: {cards_per_note.value_counts().sort_index().to_dict()}.")
OUT["C_fold_structure"] = fs.to_dict(orient="records")
OUT["C_sample"] = {"rows": int(len(Yd)), "cards": int(len(np.unique(cid_d))), "notes": int(len(np.unique(grp_note))),
                   "reviews_per_card": float(len(Yd) / len(np.unique(cid_d))), "n_revlog_total": int(len(revlog))}
OUT["C_cards_per_note_distribution"] = {int(k): int(v) for k, v in cards_per_note.value_counts().sort_index().items()}

# =============================================================================================
# D. DML under alternative fold designs
# =============================================================================================
designs = {
    "audited: row-level KFold, scaler on all rows": dict(folds=folds0, scale="global"),
    "row-level KFold, scaler fit inside each training fold": dict(folds=folds0, scale="in_fold"),
    "folds grouped by note (no note in both train and test)": dict(folds=L.grouped_folds(grp_note), scale="global"),
    "folds grouped by card": dict(folds=L.grouped_folds(cid_d), scale="global"),
    "5 contiguous time blocks (test block never overlaps train in time)": dict(folds=L.blocked_time_folds(len(Yd)), scale="global"),
}
rows_d = []
for kind in ["ridge", "gbm"]:
    for name, kw in designs.items():
        res, *_ = L.dml_fit(Yd, Dd, Xd, kind, folds=kw["folds"], scale=kw["scale"], clusters=cid_d)
        ms = L.theta_to_ms(res["theta"], res["se"], Yd)
        res_n, *_ = L.dml_fit(Yd, Dd, Xd, kind, folds=kw["folds"], scale=kw["scale"], clusters=grp_note)
        rows_d.append({"learner": kind, "design": name, "theta": res["theta"], "ATE ms": ms["ate_ms"], "SE iid": res["se"],
                       "SE cluster(card)": res["se_cluster"], "SE cluster(note)": res_n["se_cluster"],
                       "p iid": res["p"], "p cluster(card)": res["p_cluster"], "p cluster(note)": res_n["p_cluster"]})
td = pd.DataFrame(rows_d)
md("D. DML estimate under alternative fold designs, with cluster-robust standard errors", td, fmt="{:.4g}",
   text="Same samples, same learners. Only the folds, the scaling step or the SE changes.")
OUT["D_dml_designs"] = td.to_dict(orient="records")

# =============================================================================================
# E. Sensitivity: Eastern clock and point-in-time features
# =============================================================================================
SPECS = {
    "S0 audited (UTC clock, post-answer features)": dict(hour_col="hour_utc", timing="post", days_mode="subset"),
    "S1 point-in-time features (UTC clock)": dict(hour_col="hour_utc", timing="pre", days_mode="true_prev"),
    "S2 Eastern clock (post-answer features)": dict(hour_col="hour_local", timing="post", days_mode="subset"),
    "S3 Eastern clock + point-in-time features": dict(hour_col="hour_local", timing="pre", days_mode="true_prev"),
}


def psm_spec(builder, rl, cov, ycol, kind, **kw):
    built = builder(rl, **kw)
    frame = (built[1] if isinstance(built, tuple) else built).reset_index(drop=True)
    X_sc = L.scale_all(frame[cov].to_numpy(float))
    treat = frame["morning"].to_numpy().astype(np.int8)
    y = frame[ycol].to_numpy(float)
    aud = L.psm_order_worker(np.arange(len(treat)), X_sc, treat, y, kind)
    rng = np.random.default_rng(SEED_O)
    res = Parallel(n_jobs=N_JOBS)(delayed(L.psm_order_worker)(rng.permutation(len(treat)), X_sc, treat, y, kind) for _ in range(N_ORDERS))
    key = "median_diff" if kind == "cont" else "mean_diff"
    arr = np.array([r[key] for r in res])
    p = np.array([r["p"] for r in res])
    return {"n": int(len(treat)), "n_morning": int(treat.sum()), "n_evening": int((treat == 0).sum()), "n_pairs": aud["n_pairs"],
            "audited_order": aud, "orders_stat": key, "orders_mean": float(arr.mean()), "orders_sd": float(arr.std(ddof=1)),
            "orders_min": float(arr.min()), "orders_max": float(arr.max()), "orders_share_p_lt_05": float(np.mean(p < 0.05))}


rows_e = []
OUT["E"] = {}
for sname, kw in SPECS.items():
    rec = {}
    pv2 = psm_spec(L.build_v2, rl_nb, L.V2_COVARIATES, "time", "cont", **kw)
    p3s = psm_spec(L.build_phase3, rl_nb, L.P3_COVARIATES, "retained", "binary", **kw)
    rec["PSM v2"], rec["PSM Phase 3"] = pv2, p3s
    dkw = {k: v for k, v in kw.items() if k != "days_mode"}
    fdml = L.build_dml(rl_dml, **dkw)
    Xs_, Ds_, Ys_, cs_ = fdml[L.DML_FEATURES].to_numpy(float), fdml["morning"].to_numpy(), fdml["log_rt"].to_numpy(), fdml["cid"].to_numpy()
    for kind in ["ridge", "gbm"]:
        res, *_ = L.dml_fit(Ys_, Ds_, Xs_, kind, clusters=cs_)
        ms = L.theta_to_ms(res["theta"], res["se"], Ys_)
        rec[f"DML {kind}"] = {"n": res["n"], "n_morning": int(Ds_.sum()), "theta": res["theta"], "ate_ms": ms["ate_ms"],
                              "p_iid": res["p"], "p_cluster_card": res["p_cluster"]}
    OUT["E"][sname] = rec
    rows_e.append({"specification": sname,
                   "PSM v2: n (morning/evening)": f"{pv2['n_morning']}/{pv2['n_evening']}",
                   "PSM v2 median diff ms (audited order)": pv2["audited_order"]["median_diff"],
                   "PSM v2 MW p (audited order)": pv2["audited_order"]["p"],
                   "PSM v2 median diff across 100 row orders: mean [min, max]": f"{pv2['orders_mean']:.0f} [{pv2['orders_min']:.0f}, {pv2['orders_max']:.0f}]",
                   "Phase 3 ATE pp": 100 * p3s["audited_order"]["mean_diff"], "Phase 3 p": p3s["audited_order"]["p"],
                   "DML ridge ATE ms (p cluster)": f"{rec['DML ridge']['ate_ms']:+.0f} (p {rec['DML ridge']['p_cluster_card']:.2g})",
                   "DML gbm ATE ms (p cluster)": f"{rec['DML gbm']['ate_ms']:+.0f} (p {rec['DML gbm']['p_cluster_card']:.2g})",
                   "DML n (morning/evening)": f"{rec['DML ridge']['n_morning']}/{rec['DML ridge']['n'] - rec['DML ridge']['n_morning']}"})
te = pd.DataFrame(rows_e)
md("E. Sensitivity of each estimate to the clock and to point-in-time features", te, fmt="{:.4g}",
   text="Diagnostics only. The specifications change the sample as well as the covariates, so rows are not like-for-like replacements of the audited run.")

# =============================================================================================
# F. Survival / Cox reconstruction
# =============================================================================================
rl_s, *_ = L.prepare(L.CUTOFF_NOTEBOOKS)
r = rl_s.sort_values(["cid", "id"]).copy()
r["is_lapse"] = ((r["type"] == 1) & (r["ease"] == 1)).astype(int)
r["lapse_no"] = r.groupby("cid")["is_lapse"].cumsum()
r["row_no"] = r.groupby("cid").cumcount() + 1
grp = r.groupby("cid")
card = pd.DataFrame({"reps": grp.size(), "lapses": grp["is_lapse"].sum(), "factor_final": grp["factor"].last() / 1000})
first = r[(r["is_lapse"] == 1) & (r["lapse_no"] == 1)].set_index("cid")
card["reps_at_first_lapse"] = first["row_no"].reindex(card.index)
card["factor_before_first_lapse"] = (first["prev_factor"] / 1000).reindex(card.index)
surv = card[(card["reps"] >= 1)].copy()
surv["ease_factor"] = surv["factor_final"].replace(0, np.nan)
surv = surv.dropna(subset=["ease_factor"])
surv["event"] = (surv["lapses"] >= 1).astype(int)
surv["duration"] = surv["reps"].clip(lower=1)
surv["q"] = pd.qcut(surv["ease_factor"], q=4, labels=["Q1 (Hardest)", "Q2", "Q3", "Q4 (Easiest)"])
qt = surv.groupby("q", observed=True)["event"].agg(lapses="sum", total="count", rate="mean").reset_index()
reported = pd.DataFrame({"q": ["Q1 (Hardest)", "Q2", "Q3", "Q4 (Easiest)"], "reported lapses": [205, 15, 13, 4], "reported total": [2742, 2445, 2890, 925]})
qt = qt.merge(reported, on="q")
md("F1. Reconstruction of the notebook's quartile table from revlog (cutoff 2026-04-24)", qt, fmt="{:.4g}",
   text=f"Cards: {len(surv):,} (notebook 9,002); events: {int(surv['event'].sum())} (notebook 237). The cards table of 2026-04-23 no longer exists, so "
        "this is a reconstruction (reps = revlog rows; lapses = Again on review rows; factor = last post-answer factor).")
OUT["F1_reconstruction"] = {"cards": int(len(surv)), "events": int(surv["event"].sum()), "quartiles": qt.to_dict(orient="records")}

surv["ease_factor_c"] = surv["ease_factor"] - surv["ease_factor"].mean()
cph = CoxPHFitter(penalizer=0.1)
cph.fit(surv[["duration", "event", "ease_factor_c"]], duration_col="duration", event_col="event")
cs = cph.summary.loc["ease_factor_c"]
OUT["F2_cox_as_audited"] = {"coef": float(cs["coef"]), "HR": float(cs["exp(coef)"]), "p": float(cs["p"])}
print(f"Cox (reconstructed, as audited spec): coef={cs['coef']:+.3f} HR={cs['exp(coef)']:.3f} p={cs['p']:.2g}")

ev = surv[surv["event"] == 1].copy()
ev["post_event_reps"] = ev["reps"] - ev["reps_at_first_lapse"]
q1_thr = surv.loc[surv["q"] == "Q1 (Hardest)", "ease_factor"].max()
ev["in_Q1_by_final_factor"] = ev["ease_factor"] <= q1_thr
ev["in_Q1_by_pre_lapse_factor"] = ev["factor_before_first_lapse"] <= q1_thr
circ = {"q1_threshold": float(q1_thr), "n_events": int(len(ev)),
        "events_in_Q1_by_final_factor": int(ev["in_Q1_by_final_factor"].sum()),
        "events_in_Q1_by_factor_before_first_lapse": int(ev["in_Q1_by_pre_lapse_factor"].sum()),
        "events_with_reviews_after_first_lapse_counted_in_duration": int((ev["post_event_reps"] > 0).sum()),
        "median_reps_total_for_events": float(ev["reps"].median()), "median_reps_at_first_lapse": float(ev["reps_at_first_lapse"].median()),
        "median_factor_change_at_first_lapse_pts": float(((ev["factor_before_first_lapse"] - ev["ease_factor"]) * 1000).median())}
OUT["F3_circularity"] = circ
md("F3. Is the lapse feeding its own predictor? (event cards only)", pd.DataFrame([circ]).T.reset_index().rename(columns={"index": "quantity", 0: "value"}))

# time-to-first-lapse done as the notebook's text describes it: duration = reviews until first lapse, factor before that lapse
alt = surv.copy()
alt["duration_alt"] = np.where(alt["event"] == 1, alt["reps_at_first_lapse"], alt["reps"]).astype(float)
alt["ease_pre"] = np.where(alt["event"] == 1, alt["factor_before_first_lapse"], alt["ease_factor"])
alt = alt.dropna(subset=["duration_alt", "ease_pre"])
alt = alt[alt["ease_pre"] > 0]
alt["ease_pre_c"] = alt["ease_pre"] - alt["ease_pre"].mean()
cph2 = CoxPHFitter(penalizer=0.1)
cph2.fit(alt[["duration_alt", "event", "ease_pre_c"]], duration_col="duration_alt", event_col="event")
c2 = cph2.summary.loc["ease_pre_c"]
OUT["F4_cox_time_to_first_lapse_pre_lapse_factor"] = {"coef": float(c2["coef"]), "HR": float(c2["exp(coef)"]), "p": float(c2["p"]), "n": int(len(alt))}
print(f"Cox (duration = reviews to first lapse, factor before first lapse): coef={c2['coef']:+.3f} HR={c2['exp(coef)']:.3f} p={c2['p']:.2g}")

# =============================================================================================
# G. Inventory of scalers / encoders / imputers / splits / rolling windows in the audited source
# =============================================================================================
PATTERNS = {"scaler fit on full data": r"StandardScaler|fit_transform|MinMaxScaler",
            "categorical encoder": r"OneHotEncoder|LabelEncoder|OrdinalEncoder|get_dummies|TargetEncoder",
            "train/test or CV split": r"train_test_split|KFold|GroupKFold|TimeSeriesSplit|cross_val|cv\s*=",
            "imputation with a full-sample statistic": r"fillna\(",
            "quantile cut on full data": r"qcut|quantile\(",
            "rolling window": r"rolling\(",
            "shuffle / random state": r"shuffle|random_state"}
files = {"dashboard.py": None, "export_data.py": None}
hits = []
for fname in files:
    for i, line in enumerate(open(REPO / fname, encoding="utf-8").read().splitlines(), 1):
        for lab, pat in PATTERNS.items():
            if re.search(pat, line):
                hits.append({"file": fname, "where": f"line {i}", "kind": lab, "code": line.strip()[:110]})
for nb in ["anki_causal_analysis", "psm_response_time", "psm_response_time_v2"]:
    d_ = json.load(open(REPO / f"{nb}.ipynb", encoding="utf-8"))
    for ci, c in enumerate(d_["cells"]):
        if c["cell_type"] != "code":
            continue
        for ln, line in enumerate("".join(c["source"]).splitlines(), 1):
            for lab, pat in PATTERNS.items():
                if re.search(pat, line):
                    hits.append({"file": nb + ".ipynb", "where": f"cell {ci}, line {ln}", "kind": lab, "code": line.strip()[:110]})
hits_df = pd.DataFrame(hits)
md("G. Source inventory (regex over the audited .py files and notebook code cells)", hits_df)
OUT["G_inventory"] = hits

L.write_json("03_leakage.json", OUT)
with open(L.RESULTS / "03_leakage.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(MD))

# =============================================================================================
# Figures
# =============================================================================================
plt = L.mpl_style()
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
ax = axes[0]
names = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
ax.bar([names[e] for e in by_cur["ease of THIS review"]], by_cur["mean"], color=L.BLUE, width=0.6)
ax.set_title("factor change vs previous row,\ngrouped by THIS row's button")
ax.set_ylabel("mean change in revlog.factor (points)")
for x, v in enumerate(by_cur["mean"]):
    ax.text(x, v + (8 if v >= 0 else -8), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=L.INK2)
ax = axes[1]
ax.bar([names[int(e)] for e in by_prev["ease of PREVIOUS review"]], by_prev["mean"], color=L.ORANGE, width=0.6)
ax.set_title("same change, grouped by the\nPREVIOUS row's button")
ax.set_ylim(*axes[0].get_ylim())
ax = axes[2]
ax.bar(hours["hour"] - 0.2, hours["reviews_utc_clock"], width=0.4, color=L.ORANGE, label="UTC clock (what the analyses used)")
ax.bar(hours["hour"] + 0.2, hours["reviews_eastern_clock"], width=0.4, color=L.BLUE, label="US Eastern clock")
for lo_, hi_ in [(6, 11), (18, 23)]:
    ax.axvspan(lo_ - 0.5, hi_ + 0.5, color="#e8e7e2", alpha=0.55, zorder=0)
ax.set_title("reviews by hour of day\n(grey = the 6-11 and 18-23 windows)")
ax.set_xlabel("hour of day")
ax.set_ylabel("reviews")
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
fig.savefig(L.FIGURES / "leakage_timing.png")
plt.close(fig)

# DML / PSM robustness forest (all in ms)
items = []
for _, rrow in td.iterrows():
    items.append((f"DML {rrow['learner']} | {rrow['design']}", rrow["ATE ms"], rrow["learner"]))
for sname, rec in OUT["E"].items():
    items.append((f"PSM v2 median diff | {sname}", rec["PSM v2"]["audited_order"]["median_diff"], "psm"))
    for kind in ["ridge", "gbm"]:
        items.append((f"DML {kind} | {sname}", rec[f"DML {kind}"]["ate_ms"], kind))
col = {"ridge": L.BLUE, "gbm": L.ORANGE, "psm": "#1baf7a"}
fig, ax = plt.subplots(figsize=(10, 0.28 * len(items) + 1.2))
ypos = np.arange(len(items))[::-1]
for y, (lab, val, k) in zip(ypos, items):
    ax.scatter(val, y, s=40, color=col[k], zorder=3, edgecolor=L.SURFACE, linewidth=0.8)
    ax.text(val + 25, y, f"{val:+.0f}", va="center", fontsize=7.5, color=L.INK2)
ax.axvline(0, color=L.INK2, lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels([i[0] for i in items], fontsize=7)
ax.set_xlabel("estimated morning minus evening effect on response time (ms)")
for k, lab in [("ridge", "DML ridge/logistic"), ("gbm", "DML gradient boosting"), ("psm", "PSM v2 (median diff)")]:
    ax.scatter([], [], color=col[k], label=lab)
ax.set_xlim(-50, 1350)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(L.FIGURES / "leakage_dml_robustness.png")
plt.close(fig)
print("\nwrote results/03_leakage.json, results/03_leakage.md and two figures")
