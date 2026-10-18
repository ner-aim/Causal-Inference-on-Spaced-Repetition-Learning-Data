"""build_report.py - assemble audit/REPORT.md from audit/REPORT.template.md and audit/results/*.json.

Every {{key}} in the template is replaced by a value computed here from the JSON the audit scripts wrote.
Unknown keys raise an error that lists all of them, so no placeholder can reach the report unresolved.
Other placeholders:  {{include:results/<file>}}  {{fig:<png>|<caption>}}  {{code:<script>}}

Usage (repo root):  python audit/build_report.py
"""
import json
import re
import sys

import numpy as np
import pandas as pd

import audit_lib as L

L.utf8_stdout()
A = L.AUDIT_DIR
R = L.RESULTS
FENCE = "`" * 3


def J(name):
    with open(R / name, encoding="utf-8") as fh:
        return json.load(fh)


rep, ov, pl, lk = J("00_replication.json"), J("01_overlap.json"), J("02_placebo.json"), J("03_leakage.json")
hi, rc, cb = J("04_history.json"), J("05_rct.json"), J("06_cloud_branch.json")
RES = pl["results"]
EXT = pl["extras"]
E = lk["E"]
S0, S1, S2, S3 = [k for k in E]


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------
def n0(x):
    return f"{x:,.0f}"


def d(x, nd=1):
    return f"{x:.{nd}f}"


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def pv(p):
    p = float(p)
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.4f}"
    return f"{p:.1e}"


def rng(lo, hi, nd=1):
    return f"{lo:.{nd}f}-{hi:.{nd}f}"


def seed_in(path, pattern):
    return re.search(pattern, (A / path).read_text(encoding="utf-8")).group(1)


# ---------------------------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------------------------
F = {}
meta = rep["meta"]
F.update(head7=hi["head"][:7], db_sha=meta["db_sha256"], db_bytes=n0(meta["db_bytes"]), py=meta["python"], numpy=meta["numpy"],
         pandas=meta["pandas"], scipy=meta["scipy"], sklearn=meta["sklearn"])
F["seed_u"], F["seed_s"] = pl["seed_unrestricted"], pl["seed_within_month"]
F["seed_o"] = pl["seed_row_order"]
F["seed_o3"] = seed_in("s3_leakage.py", r"SEED_O\s*=\s*(\d+)")
F["seed_rct"] = seed_in("s5_rct_checks.py", r"SEED\s*=\s*(\d+)")
F["repl_n"] = rep["n_checks"]
F["repl_fail"] = rep["n_fail"]
F["repl_n_pass"] = rep["n_checks"] - rep["n_fail"]
F["B"] = pl["B"]
F["p_floor"] = f"{1 / (pl['B'] + 1):.3f}"

# --- sensitivity specifications
def spec(s):
    return E[s]


F["s0_v2_med"], F["s1_v2_med"], F["s2_v2_med"], F["s3_v2_med"] = [n0(spec(s)["PSM v2"]["audited_order"]["median_diff"]) for s in (S0, S1, S2, S3)]
F["s0_v2_p"], F["s1_v2_p"] = pv(spec(S0)["PSM v2"]["audited_order"]["p"]), pv(spec(S1)["PSM v2"]["audited_order"]["p"])
F["s1_v2_pairs"] = n0(spec(S1)["PSM v2"]["n_pairs"])
for tag, s in zip(("s0", "s1", "s2", "s3"), (S0, S1, S2, S3)):
    F[f"{tag}_ridge"] = n0(spec(s)["DML ridge"]["ate_ms"])
    F[f"{tag}_gbm"] = n0(spec(s)["DML gbm"]["ate_ms"])
    F[f"{tag}_p3"] = sg(100 * spec(s)["PSM Phase 3"]["audited_order"]["mean_diff"], 2)
    F[f"{tag}_p3_p"] = pv(spec(s)["PSM Phase 3"]["audited_order"]["p"])
for s in (S0, S1, S2, S3):
    assert spec(s)["PSM v2"]["audited_order"]["median_diff"] > 0 and spec(s)["DML ridge"]["ate_ms"] > 0 and spec(s)["DML gbm"]["ate_ms"] > 0
psm_vals = [spec(s)["PSM v2"]["audited_order"]["median_diff"] for s in (S0, S1, S2, S3)]
dml_vals = [spec(s)[k]["ate_ms"] for s in (S0, S1, S2, S3) for k in ("DML ridge", "DML gbm")]
F["psm_spec_min"], F["psm_spec_max"] = n0(min(psm_vals)), n0(max(psm_vals))
F["dml_spec_min"], F["dml_spec_max"] = n0(min(dml_vals)), n0(max(dml_vals))
F["v2_reported"] = n0(spec(S0)["PSM v2"]["audited_order"]["median_diff"])
F["v2_p"] = pv(spec(S0)["PSM v2"]["audited_order"]["p"])

# --- overlap
OVT = {"v2": "PSM v2 (UTC hours, as audited)", "p3": "PSM Phase 3 (UTC hours, as audited)",
       "ridge": "DML ridge (UTC hours, as audited)", "gbm": "DML gbm (UTC hours, as audited)", "east": "PSM v2 (Eastern hours, sensitivity)"}


def band(k):
    return ov[OVT[k]]["band"]


def prange(k):
    b = band(k)
    return rng(min(b["morning"]["min"], b["evening"]["min"]), max(b["morning"]["max"], b["evening"]["max"]), 3)


def month_stats(k):
    mt = pd.DataFrame(ov[OVT[k]]["month_table"])
    z = mt[mt["morning"] == 0]
    return len(z), len(mt), (z["mean_ps"].mean() if len(z) else float("nan")), mt[mt["morning"] > 0]["mean_ps"].mean()


F["ov_v2_auc"], F["ov_ridge_auc"], F["ov_gbm_auc"] = d(band("v2")["auc"], 3), d(band("ridge")["auc"], 3), d(band("gbm")["auc"], 3)
F["p3_auc"] = d(lk["B2_phase3_auc_log_ivl_vs_lapse"], 3)
F["ov_v2_range"], F["ov_ridge_range"], F["ov_east_range"] = prange("v2"), prange("ridge"), prange("east")
F["ov_p3_out"], F["ov_p3_pct"] = n0(band("p3")["outside_total"]), d(band("p3")["outside_pct"], 1)
F["ov_p3_deck0"] = n0(ov[OVT["p3"]]["dropped_rows"]["deck_enc_zero"])
F["ov_gbm_n"], F["ov_gbm_pct"] = n0(band("gbm")["outside_total"]), d(band("gbm")["outside_pct"], 1)
F["ov_east_morning"], F["ov_east_n"] = n0(band("east")["n_treated"]), n0(band("east")["n"])
zv, mv, psz, psn = month_stats("v2")
F["ov_v2_zero"], F["ov_v2_months"], F["ov_v2_ps_zero"], F["ov_v2_ps_nonzero"] = zv, mv, d(psz, 4), d(psn, 4)
zg, mg, psgz, psgn = month_stats("gbm")
F["ov_gbm_zero"], F["ov_gbm_months"], F["ov_gbm_ps_zero"], F["ov_gbm_ps_nonzero"] = zg, mg, d(psgz, 4), d(psgn, 4)
p3o = ov[OVT["p3"]]
F["p3_full_pp"] = sg(100 * p3o["est_full"]["mean_diff"], 3)
F["p3_keep_pp"] = sg(100 * p3o["est_trimmed_keep_ps"]["mean_diff"], 3)
F["p3_refit_pp"] = sg(100 * p3o["est_trimmed"]["mean_diff"], 3)
null_sd_pp = 100 * RES["PSM Phase 3"]["retained"]["unrestricted"]["null_sd"]
F["p3_null_sd_pp"] = d(null_sd_pp, 2)
F["p3_shift_sd"] = d(abs(p3o["est_trimmed"]["mean_diff"] - p3o["est_full"]["mean_diff"]) * 100 / null_sd_pp, 1)
gbm_o = ov[OVT["gbm"]]
F["trim_gbm_drop"], F["trim_gbm_clip"] = n0(gbm_o["est_trim_drop"]["ate_ms"]), n0(gbm_o["est_trim_clip"]["ate_ms"])
F["trim_gbm_delta"] = n0(max(abs(gbm_o["est_trim_drop"]["ate_ms"] - gbm_o["est_full"]["ate_ms"]), abs(gbm_o["est_trim_clip"]["ate_ms"] - gbm_o["est_full"]["ate_ms"])))
v1txt = None
import json as _json
nb1 = _json.load(open(A.parent / "psm_response_time_executed.ipynb", encoding="utf-8"))
for c in nb1["cells"]:
    if c["cell_type"] == "code":
        t = "".join("".join(o.get("text", [])) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", [])) for o in c.get("outputs", []))
        if "PS model train accuracy" in t:
            v1txt = t
acc = re.search(r"train accuracy:\s*([\d.]+)", v1txt).group(1)
mins = [float(x) for x in re.findall(r"(?:Evening|Morning)\s+\d+\.0\s+[\d.]+\s+[\d.]+\s+([\d.]+)", v1txt)]
maxs = [float(x) for x in re.findall(r"(?:Evening|Morning)\s+\d+\.0\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", v1txt)]
F["v1_acc"], F["v1_ps_range"] = acc, rng(min(mins), max(maxs), 3)

# --- placebo
F["perm_v2"] = pv(RES["PSM v2"]["time"]["unrestricted"]["perm_p"])
F["perm_v2_m"] = pv(RES["PSM v2"]["time"]["within_month"]["perm_p"])
F["perm_p3"] = pv(RES["PSM Phase 3"]["retained"]["unrestricted"]["perm_p"])
plac = [(m, o, r) for m in ("PSM v2", "DML ridge", "DML gbm") for o, r in RES[m].items() if not r["is_real_outcome"]]
F["placebo_max_z"] = d(max(abs(r["unrestricted"]["z_centered"]) for _, _, r in plac), 2)
F["placebo_min_p"] = pv(min(r["unrestricted"]["perm_p"] for _, _, r in plac))
k3 = RES["PSM Phase 3"]["kanji_count"]
q3 = RES["PSM Phase 3"]["nid_rank_pct"]
F["p3_kanji"], F["p3_kanji_p"] = sg(k3["unrestricted"]["real"], 3), pv(k3["mann_whitney_p_on_pairs"])
F["p3_pos"], F["p3_pos_p"] = sg(q3["unrestricted"]["real"], 3), pv(q3["mann_whitney_p_on_pairs"])
F["p3_kanji_nullmean"], F["p3_pos_nullmean"] = sg(k3["unrestricted"]["null_mean"], 3), sg(q3["unrestricted"]["null_mean"], 3)
F["p3_kanji_pc"], F["p3_pos_pc"] = pv(k3["unrestricted"]["perm_p_centered"]), pv(q3["unrestricted"]["perm_p_centered"])
ro = EXT["row_order"]["PSM v2"]
F["ro_B"] = ro["B"]
F["ro_aud"] = n0(ro["audited_order"]["median_diff"])
F["ro_mean"], F["ro_sd"] = n0(ro["median_diff"]["mean"]), n0(ro["median_diff"]["sd"])
F["ro_min"], F["ro_max"] = n0(ro["median_diff"]["min"]), n0(ro["median_diff"]["max"])
F["ro_sig"] = d(100 * ro["share_p_below_0.05"], 1)
F["ro_nonsig"] = d(100 * (1 - ro["share_p_below_0.05"]), 1)
F["ro_ctrl"] = n0(ro["n_unique_controls"]["mean"])
md2, md3 = EXT["matching_diagnostics"]["PSM v2"], EXT["matching_diagnostics"]["PSM Phase 3"]
F["ov_v2_tied"] = d(100 * md2["share_of_rows_sharing_a_ps_value_with_another_row"], 1)
F["p3_unique"], F["p3_pairs"], F["p3_maxuse"] = n0(md3["n_unique_controls_used"]), n0(md3["n_pairs"]), n0(md3["max_times_one_control_used"])
F["p3_mc_pos"], F["p3_ac_pos"], F["p3_t_pos"] = d(md3["mean_deck_position_matched_controls"], 3), d(md3["mean_deck_position_all_controls"], 3), d(md3["mean_deck_position_all_treated"], 3)

# --- leakage
hrs = pd.DataFrame(lk["A3_hour_counts"])
F["utc_morning"] = n0(hrs.loc[hrs["hour"].between(6, 11), "reviews_utc_clock"].sum())
F["east_morning"] = n0(hrs.loc[hrs["hour"].between(6, 11), "reviews_eastern_clock"].sum())
F["n_revlog"] = n0(lk["C_sample"]["n_revlog_total"])
top = lambda c: ", ".join(str(int(h)) for h in hrs.sort_values(c, ascending=False)["hour"].head(3))
F["top3_utc"], F["top3_east"] = top("reviews_utc_clock"), top("reviews_eastern_clock")
F["crt_utc"] = lk["A3_col_crt_utc"].replace("T", " ")
a1 = {int(r["ease of THIS review"]): r for r in lk["A1_dfactor_by_current_ease"]}
F["df_again"], F["df_hard"], F["df_good"], F["df_easy"] = [("0" if abs(a1[e]["mean"]) < 0.5 else sg(a1[e]["mean"], 0)) for e in (1, 2, 3, 4)]
F["lastivl_share"] = d(100 * lk["A2_share_lastIvl_equals_previous_row_ivl"], 2)
F["lastivl_n"] = n0(lk["A2_n"])


def entered(key):
    rows = {r["pre-answer factor status"]: r for r in lk[key]["table"]}
    r = rows["NOT hard before this answer"]
    return n0(r["total"]), d(r["total %"], 1), d(r["evening %"], 1), d(r["morning %"], 1)


F["v2_entered"], F["v2_entered_pct"], F["v2_entered_evening"], F["v2_entered_morning"] = entered("B_v2")
F["dml_entered"], F["dml_entered_pct"], F["dml_entered_evening"], F["dml_entered_morning"] = entered("B_dml")
ct = lk["B2_phase3_crosstab"]
F["p3_lapse_neg"] = n0(ct["ivl < 0"]["retained = 0 (Again)"])
F["p3_ret_neg"] = n0(ct["ivl < 0"]["retained = 1"])
F["p3_ret_n"] = n0(ct["ivl < 0"]["retained = 1"] + ct["ivl > 0"]["retained = 1"])
cs = lk["C_sample"]
F["reviews_per_card"] = d(cs["reviews_per_card"], 2)
F["cards_per_note"] = ", ".join(str(k) for k in lk["C_cards_per_note_distribution"])
fs = pd.DataFrame(lk["C_fold_structure"])
col = lambda s: [c for c in fs.columns if c.startswith(s)][0]
F["fold_card"] = rng(fs[col("% test rows whose CARD")].min(), fs[col("% test rows whose CARD")].max())
F["fold_note"] = rng(fs[col("% whose NOTE")].min(), fs[col("% whose NOTE")].max())
F["fold_later"] = rng(fs[col("% with a LATER")].min(), fs[col("% with a LATER")].max())
F["fold_after"] = rng(fs[col("% of training rows dated after")].min(), fs[col("% of training rows dated after")].max())
dd = pd.DataFrame(lk["D_dml_designs"])


def design(learner, key):
    return dd[(dd["learner"] == learner) & (dd["design"].str.startswith(key))].iloc[0]


F["des_ridge_base"] = n0(design("ridge", "audited")["ATE ms"])
F["des_ridge_note"] = n0(design("ridge", "folds grouped by note")["ATE ms"])
F["des_ridge_scale"] = n0(design("ridge", "row-level KFold, scaler fit inside")["ATE ms"])
F["des_gbm_base"] = n0(design("gbm", "audited")["ATE ms"])
F["des_gbm_note"] = n0(design("gbm", "folds grouped by note")["ATE ms"])
F["blk_ridge"], F["blk_ridge_p"] = n0(design("ridge", "5 contiguous")["ATE ms"]), pv(design("ridge", "5 contiguous")["p cluster(card)"])
F["blk_gbm"], F["blk_gbm_p"] = n0(design("gbm", "5 contiguous")["ATE ms"]), pv(design("gbm", "5 contiguous")["p cluster(card)"])
ratios = np.concatenate([(dd["SE cluster(card)"] / dd["SE iid"] - 1).to_numpy(), (dd["SE cluster(note)"] / dd["SE iid"] - 1).to_numpy()])
F["se_ratio_max"] = d(100 * ratios.max(), 1)

# --- survival
f1 = lk["F1_reconstruction"]
qs = {r["q"]: r for r in f1["quartiles"]}
F["surv_cards"], F["surv_events"] = n0(f1["cards"]), n0(f1["events"])
nbtxt = None
nb0 = _json.load(open(A.parent / "anki_causal_analysis_executed.ipynb", encoding="utf-8"))
for c in nb0["cells"]:
    if c["cell_type"] == "code":
        t = "".join("".join(o.get("text", [])) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", [])) for o in c.get("outputs", []))
        if "number of events observed" in t:
            nbtxt = t
m_events = int(re.search(r"number of events observed\s*=\s*(\d+)", nbtxt).group(1))
m_obs = int(re.search(r"number of observations\s*=\s*(\d+)", nbtxt).group(1))
cm = re.search(r"ease_factor_c\s+([\-\d.]+)\s+([\d.]+)", nbtxt)
F["nb_cards"], F["nb_events"] = n0(m_obs), n0(m_events)
F["cox_hr"] = cm.group(2)
F["nb_coef"] = cm.group(1)
F["q1_diff"] = n0(abs(qs["Q1 (Hardest)"]["total"] - qs["Q1 (Hardest)"]["reported total"]))
F["cox_coef"] = sg(lk["F2_cox_as_audited"]["coef"], 3)
F["cox_hr_recon"] = d(lk["F2_cox_as_audited"]["HR"], 2)
f3 = lk["F3_circularity"]
F["cox_post_all"] = n0(f3["events_with_reviews_after_first_lapse_counted_in_duration"])
F["med_reps_total"], F["med_reps_first"] = n0(f3["median_reps_total_for_events"]), n0(f3["median_reps_at_first_lapse"])
F["med_fall"] = n0(f3["median_factor_change_at_first_lapse_pts"])
F["q1_events"] = n0(f3["events_in_Q1_by_final_factor"])
F["q1_share_final"] = d(100 * f3["events_in_Q1_by_final_factor"] / f3["n_events"], 1)
F["q1_share_pre"] = d(100 * f3["events_in_Q1_by_factor_before_first_lapse"] / f3["n_events"], 1)
f4 = lk["F4_cox_time_to_first_lapse_pre_lapse_factor"]
F["cox_alt_coef"], F["cox_alt_hr"], F["cox_alt_p"] = sg(f4["coef"], 3), d(f4["HR"], 3), pv(f4["p"])
F["q1_card_share"] = d(100 * qs["Q1 (Hardest)"]["total"] / f1["cards"], 1)

# --- history
F["h_shift_n"] = len(hi["pickaxe"]["shift"]["git_log_S"])
F["h_lag_n"] = len(hi["pickaxe"]["lag"]["git_log_S"])

# --- RCT
hd = {h["sample"]: h for h in rc["headline"]}
h_all = hd["as reported: 2026-04-24 to 2026-06-23"]
h_pm = hd["from the moment the star marker existed"]
F["rct_n"] = n0(h_all["reviews T"] + h_all["reviews C"])
F["rct_nt"], F["rct_nc"] = n0(h_all["reviews T"]), n0(h_all["reviews C"])
F["rct_lt"], F["rct_lc"] = d(h_all["lapse % T"], 2), d(h_all["lapse % C"], 2)
F["rct_z_p"] = pv(h_all["z-test p (reviews independent)"])
F["rct_mw_p"] = pv(h_all["Mann-Whitney p"])
F["rct_cl_p"] = pv(h_all["cluster-by-note p (lapse)"])
F["rct_cl_rt_p"] = pv(h_all["cluster-by-note p (log RT)"])
F["rct_bayes"] = d(100 * rc["bayes_P_treatment_better"], 1) + "%"
F["rct_diff"], F["rct_diff_pm"] = d(h_all["diff pp"], 2), d(h_pm["diff pp"], 2)
F["rct_rt"] = n0(1000 * (h_all["median RT T s"] - h_all["median RT C s"]))
F["marker_ts"] = rc["marker"]["marker_utc"].replace("T", " ")
F["marker_n"], F["marker_total"] = n0(rc["marker"]["notes_written_at_that_second"]), n0(rc["n_notes"]["treatment"])
tim = {t["quantity"]: t["value"] for t in rc["timing"]}
F["pm_n"] = n0(tim["treatment-arm reviews in the reported window dated before the marker existed"])
F["pm_pct"] = d(tim["as % of all treatment-arm reviews in the reported window"], 1)
F["marker_days"] = d(tim["span in days between analysis start and marker"], 1)
rer = {(r["window"], r["outcome"]): r for r in rc["rerandomisation"]}
medrt = {r["window"]: r for r in rc["rerandomisation_median_rt"]}
F["rct_ri_med_p"] = pv(medrt["reported window"]["re-randomisation p"])
pre = rer[("pre-experiment (same notes, before 2026-04-24)", "lapse rate (pooled)")]
F["pre_gap"], F["pre_p"] = d(pre["real diff"], 2), pv(pre["re-randomisation p"])
F["pre_rt"] = n0(medrt["pre-experiment (same notes)"]["real diff"])
F["pre_rt_p"] = pv(medrt["pre-experiment (same notes)"]["re-randomisation p"])
F["rct_did"], F["rct_did_p"] = sg(rc["did_lapse"]["difference-in-differences pp"], 2), pv(rc["did_lapse"]["re-randomisation p"])
F["rct_kanji_p"] = pv(rer[("(cannot be affected)", "kanji count of note")]["re-randomisation p"])
F["rct_pos_p"] = pv(rer[("(cannot be affected)", "deck position of note")]["re-randomisation p"])

# --- deployed branch
F["cloud_n"] = n0(cb["description"]["cloud_branch_rows"])
ce = {r["learner"]: r for r in cb["cloud_branch_estimates"]}
F["cloud_ridge"], F["cloud_gbm"] = n0(ce["ridge"]["ATE ms"]), n0(ce["gbm"]["ATE ms"])
F["cloud_last"] = cb["description"]["parquet_last_review_date"]


# ---------------------------------------------------------------------------------------------
# virtual includes (tables rebuilt from JSON, or sections of the scripts' markdown files)
# ---------------------------------------------------------------------------------------------
def sections(md_name):
    text = (R / md_name).read_text(encoding="utf-8")
    parts = re.split(r"(?m)^### ", text)
    return {p.split("\n", 1)[0].strip(): p.split("\n", 1)[1].strip() if "\n" in p else "" for p in parts if p.strip()}


def bump(md):
    return re.sub(r"(?m)^### ", "#### ", md)


def sect_block(sec, prefix):
    out = []
    for title, body in sec.items():
        if title.startswith(prefix):
            out.append(f"#### {title}\n\n{body}")
    return "\n\n".join(out)


V = {}
ex_pairs = [("PSM v2", "matched pairs"), ("PSM v2", "ATE median diff (ms)"), ("PSM v2", "median-diff CI low"), ("PSM v2", "median-diff CI high"),
            ("PSM v2", "Mann-Whitney p"), ("PSM Phase 3", "matched pairs"), ("PSM Phase 3", "ATE (retention)"), ("PSM Phase 3", "t-test p"),
            ("DML ridge", "theta (log RT)"), ("DML ridge", "ATE (ms)"), ("DML gbm", "theta (log RT)"), ("DML gbm", "ATE (ms)")]
chk = pd.DataFrame(rep["checks"])
ex = pd.concat([chk[(chk["analysis"] == a) & (chk["quantity"] == q)] for a, q in ex_pairs])
V["results/00_replication_excerpt.md"] = L.md_table(ex[["analysis", "quantity", "reported", "reproduced", "match", "source"]], "{:.6g}")
V["results/00_replication.md"] = (R / "00_replication.md").read_text(encoding="utf-8")

s02 = sections("02_placebo.md")
V["results/02_placebo_a.md"] = s02["(a) Real estimate against the shuffled-label null"]
V["results/02_placebo_b.md"] = s02["(b) and (c) Placebo outcomes"]
mdf = pd.DataFrame(EXT["matching_diagnostics"])
mdf.index.name = "quantity"
mdf = mdf.reset_index()
mdf["quantity"] = mdf["quantity"].str.replace("_", " ")
rows = []
for name, r in EXT["row_order"].items():
    key = "median_diff" if name == "PSM v2" else "mean_diff"
    rows.append({"analysis": name, "estimate": r["label"], "audited order": r["audited_order"][key], "mean over orders": r[key]["mean"],
                 "sd": r[key]["sd"], "min": r[key]["min"], "max": r[key]["max"], "share of orders with p<0.05": r["share_p_below_0.05"],
                 "orders": r["B"]})
V["results/02_placebo_c.md"] = ("**Matching diagnostics with the real labels**\n\n" + L.md_table(mdf, "{:.4g}") +
                                 "\n\n**Row-order sensitivity** (same data and propensity model; only the order rows reach the matcher changes)\n\n" +
                                 L.md_table(pd.DataFrame(rows), "{:.4g}"))
s03 = sections("03_leakage.md")
for letter in "ABCDEFG":
    V[f"results/03_{letter}.md"] = sect_block(s03, letter)
V["results/01_overlap.md"] = (R / "01_overlap.md").read_text(encoding="utf-8")
V["results/04_history.md"] = bump((R / "04_history.md").read_text(encoding="utf-8"))
V["results/05_rct.md"] = bump((R / "05_rct.md").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------
template = (A / "REPORT.template.md").read_text(encoding="utf-8")
CODE_BLOCKS = {}
missing = []
used = set()


def repl(m):
    key = m.group(1).strip()
    if key.startswith("include:"):
        name = key[len("include:"):]
        if name in V:
            return V[name]
        return (A / name).read_text(encoding="utf-8")
    if key.startswith("fig:"):
        name, cap = key[len("fig:"):].split("|", 1)
        cap = re.sub(r"\{\{([^}]+)\}\}", lambda mm: str(F[mm.group(1)]), cap)
        assert (L.FIGURES / name).exists(), name
        return f"![{cap}](figures/{name})\n\n*{cap}*"
    if key.startswith("code:"):
        name = key[len("code:"):]
        code = (A / name).read_text(encoding="utf-8").rstrip("\n")
        fence4 = "`" * 4
        assert fence4 not in code
        CODE_BLOCKS[name] = f"### A.{name}\n\n{fence4}python\n{code}\n{fence4}"
        return f"@@CODE:{name}@@"
    if key in F:
        used.add(key)
        return str(F[key])
    missing.append(key)
    return m.group(0)


out = re.sub(r"\{\{([^}]+)\}\}", repl, template)
# placeholders inside figure captions
for m in re.finditer(r"\{\{([^}]+)\}\}", out):
    if m.group(1) not in missing:
        missing.append(m.group(1))
if missing:
    print("UNRESOLVED PLACEHOLDERS:", sorted(set(missing)))
    sys.exit(1)
assert len(re.findall(r"@@CODE:[A-Za-z0-9_.]+@@", out)) == len(CODE_BLOCKS)
for name, block in CODE_BLOCKS.items():
    out = out.replace(f"@@CODE:{name}@@", block)
(A / "REPORT.md").write_text(out, encoding="utf-8")
unused = sorted(set(F) - used - {"nb_coef"})
print(f"wrote audit/REPORT.md ({len(out):,} characters, {len(used)} facts used)")
if unused:
    print("facts computed but not used in the template:", unused)
