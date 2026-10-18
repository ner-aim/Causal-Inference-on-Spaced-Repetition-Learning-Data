"""s0_replication.py - can the audited numbers be reproduced from the Anki collection?

Re-runs the audited code (via audit_lib) at the snapshot that matches each reported run and
compares with the figures printed in the executed notebooks / the DML run of 2026-07-10.
Writes results/00_replication.json and results/00_replication.md.

Usage (from the repo root):   ANKI_DB=<path to collection.anki2 copy> python audit/s0_replication.py
"""
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats

import audit_lib as L

L.utf8_stdout()

rows = []  # (analysis, quantity, reported, reproduced, tolerance, source)


def check(analysis, quantity, reported, reproduced, tol, source):
    ok = (abs(float(reproduced) - float(reported)) <= tol)
    rows.append({"analysis": analysis, "quantity": quantity, "reported": reported,
                 "reproduced": round(float(reproduced), 6), "tolerance": tol, "match": "PASS" if ok else "FAIL",
                 "source": source})


meta_out = {
    "db_path": str(L.DB_PATH), "db_sha256": L.sha256_file(L.DB_PATH), "db_bytes": L.DB_PATH.stat().st_size,
    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    "scipy": scipy.__version__, "sklearn": sklearn.__version__,
}

# ------------------------------------------------------------------ PSM v2
rl_nb, cards, notes, decks, meta = L.prepare(L.CUTOFF_NOTEBOOKS)
meta_out.update(meta)
SRC_V2 = "psm_response_time_v2_executed.ipynb"
df, df_trim = L.build_v2(rl_nb)
check("PSM v2", "hard reviews (ef<2.5)", 4252, len(df), 0, SRC_V2 + " cell 2")
check("PSM v2", "morning hard reviews", 319, df["morning"].sum(), 0, SRC_V2 + " cell 2")
check("PSM v2", "evening hard reviews", 3933, (df["morning"] == 0).sum(), 0, SRC_V2 + " cell 2")
check("PSM v2", "capped at 60s", 214, df["capped"].sum(), 0, SRC_V2 + " cell 2")
check("PSM v2", "after excluding capped", 4038, len(df_trim), 0, SRC_V2 + " cell 4")
check("PSM v2", "morning after exclusion", 309, df_trim["morning"].sum(), 0, SRC_V2 + " cell 4")

X = df_trim[L.V2_COVARIATES].to_numpy(float)
treat = df_trim["morning"].to_numpy()
X_sc = L.scale_all(X)
ps, t_pos, c_pos, caliper, lr = L.psm_fit_match(X_sc, treat)
check("PSM v2", "PS model train accuracy", 0.923, lr.score(X_sc, treat), 0.0005, SRC_V2 + " cell 6")
for name, rep, got in zip(L.V2_COVARIATES, [-0.0125, 0.0336, 0.0447, 0.0310], lr.coef_[0]):
    check("PSM v2", f"PS coef {name}", rep, got, 0.00005, SRC_V2 + " cell 6")
check("PSM v2", "PS mean, evening", 0.0765, ps[treat == 0].mean(), 0.00005, SRC_V2 + " cell 6")
check("PSM v2", "PS mean, morning", 0.0770, ps[treat == 1].mean(), 0.00005, SRC_V2 + " cell 6")
check("PSM v2", "PS max, evening", 0.0974, ps[treat == 0].max(), 0.00005, SRC_V2 + " cell 6")
check("PSM v2", "PS max, morning", 0.0940, ps[treat == 1].max(), 0.00005, SRC_V2 + " cell 6")
check("PSM v2", "PS min, evening", 0.0603, ps[treat == 0].min(), 0.00005, SRC_V2 + " cell 6")
check("PSM v2", "caliper", 0.01745, caliper, 0.000005, SRC_V2 + " cell 7")
check("PSM v2", "matched pairs", 309, len(t_pos), 0, SRC_V2 + " cell 7")

y = df_trim["time"].to_numpy(float)
mt, ev = y[t_pos], y[c_pos]
check("PSM v2", "matched median, morning (ms)", 5497, np.median(mt), 0.5, SRC_V2 + " cell 7")
check("PSM v2", "matched median, evening (ms)", 4413, np.median(ev), 0.5, SRC_V2 + " cell 7")
eff = L.pair_effects(mt, ev)
check("PSM v2", "ATE mean diff (ms)", 508, eff["mean_diff"], 0.5, SRC_V2 + " cell 12")
check("PSM v2", "ATE median diff (ms)", 1084, eff["median_diff"], 0.5, SRC_V2 + " cell 12")
rng = np.random.default_rng(42)
n = len(mt)
boot_mean, boot_med = [], []
for _ in range(10_000):  # notebook cell 12, same seed and draw order
    idx = rng.integers(0, n, n)
    boot_mean.append(mt[idx].mean() - ev[idx].mean())
    boot_med.append(np.median(mt[idx]) - np.median(ev[idx]))
lo_m, hi_m = np.percentile(boot_mean, [2.5, 97.5])
lo_d, hi_d = np.percentile(boot_med, [2.5, 97.5])
check("PSM v2", "mean-diff CI low", -858, lo_m, 0.5, SRC_V2 + " cell 12")
check("PSM v2", "mean-diff CI high", 1844, hi_m, 0.5, SRC_V2 + " cell 12")
check("PSM v2", "median-diff CI low", 209, lo_d, 0.5, SRC_V2 + " cell 12")
check("PSM v2", "median-diff CI high", 1924, hi_d, 0.5, SRC_V2 + " cell 12")
_, p_t = stats.ttest_ind(mt, ev)
u, p_mw = stats.mannwhitneyu(mt, ev, alternative="two-sided")
check("PSM v2", "t-test p", 0.4979, p_t, 0.00005, SRC_V2 + " cell 12")
check("PSM v2", "Mann-Whitney U", 54044, u, 0.5, SRC_V2 + " cell 12")
check("PSM v2", "Mann-Whitney p", 0.0045, p_mw, 0.00005, SRC_V2 + " cell 12")
for name, rep in zip(L.V2_COVARIATES, [0.014531, 0.032767, 0.095352, 0.116792]):
    got = L.smd(X[t_pos, L.V2_COVARIATES.index(name)], X[c_pos, L.V2_COVARIATES.index(name)])
    check("PSM v2", f"SMD after matching: {name}", rep, got, 0.000005, SRC_V2 + " cell 8")

# ------------------------------------------------------------------ PSM Phase 3 (retention)
SRC_P3 = "anki_causal_analysis_executed.ipynb"
p3 = L.build_phase3(rl_nb)
check("PSM Phase 3", "PSM dataset rows", 21803, len(p3), 0, SRC_P3 + " cell 14")
check("PSM Phase 3", "morning rows", 1375, p3["morning"].sum(), 0, SRC_P3 + " cell 14")
check("PSM Phase 3", "evening rows", 20428, (p3["morning"] == 0).sum(), 0, SRC_P3 + " cell 14")
X3 = p3[L.P3_COVARIATES].to_numpy(float)
t3 = p3["morning"].to_numpy()
X3_sc = L.scale_all(X3)
ps3, tp3, cp3, cal3, lr3 = L.psm_fit_match(X3_sc, t3)
check("PSM Phase 3", "PS model train accuracy", 0.937, lr3.score(X3_sc, t3), 0.0005, SRC_P3 + " cell 15")
check("PSM Phase 3", "caliper", 0.0306, cal3, 0.00005, SRC_P3 + " cell 17")
check("PSM Phase 3", "matched pairs", 1375, len(tp3), 0, SRC_P3 + " cell 17")
r3 = p3["retained"].to_numpy(float)
m_ret, e_ret = r3[tp3], r3[cp3]
ate3 = m_ret.mean() - e_ret.mean()
check("PSM Phase 3", "ATE (retention)", -0.0007, ate3, 0.00005, SRC_P3 + " cell 19")
t_stat3, p_val3 = stats.ttest_ind(m_ret, e_ret)
check("PSM Phase 3", "t statistic", -0.219, t_stat3, 0.0005, SRC_P3 + " cell 19")
check("PSM Phase 3", "t-test p", 0.8267, p_val3, 0.00005, SRC_P3 + " cell 19")
rng = np.random.default_rng(42)
boots = []
for _ in range(5000):  # notebook cell 19
    idx = rng.integers(0, len(m_ret), len(m_ret))
    boots.append(m_ret[idx].mean() - e_ret[idx].mean())
lo3, hi3 = np.percentile(boots, [2.5, 97.5])
check("PSM Phase 3", "bootstrap CI low", -0.0065, lo3, 0.00005, SRC_P3 + " cell 19")
check("PSM Phase 3", "bootstrap CI high", 0.0051, hi3, 0.00005, SRC_P3 + " cell 19")

# ------------------------------------------------------------------ DML
rl_dml, _, _, _, meta_dml = L.prepare(L.CUTOFF_DML)
dml = L.build_dml(rl_dml)
SRC_D = "DML run of 2026-07-10 (conversation log) / dashboard.py run_dml()"
check("DML", "hard reviews", 4763, len(dml), 0, SRC_D)
check("DML", "morning", 530, dml["morning"].sum(), 0, SRC_D)
check("DML", "evening", 4233, (dml["morning"] == 0).sum(), 0, SRC_D)
Xd = dml[L.DML_FEATURES].to_numpy(float)
Dd = dml["morning"].to_numpy()
Yd = dml["log_rt"].to_numpy()
folds = L.make_folds(len(Yd))
dml_out = {}
for kind, rep_theta, rep_se, rep_ms in [("ridge", 0.1380, 0.0344, 767), ("gbm", 0.1634, None, 920)]:
    res, Y_res, D_res, g_hat = L.dml_fit(Yd, Dd, Xd, kind, folds=folds)
    ms = L.theta_to_ms(res["theta"], res["se"], Yd)
    check("DML " + kind, "theta (log RT)", rep_theta, res["theta"], 0.00005, SRC_D)
    if rep_se is not None:
        check("DML " + kind, "SE", rep_se, res["se"], 0.00005, SRC_D)
    check("DML " + kind, "ATE (ms)", rep_ms, ms["ate_ms"], 0.5, SRC_D)
    if kind == "ridge":
        check("DML ridge", "CI low (ms)", 379, ms["ci_lo_ms"], 0.5, SRC_D)
        check("DML ridge", "CI high (ms)", 1182, ms["ci_hi_ms"], 0.5, SRC_D)
    else:
        check("DML gbm", "CI low (ms)", 511, ms["ci_lo_ms"], 0.5, SRC_D)
        check("DML gbm", "CI high (ms)", 1359, ms["ci_hi_ms"], 0.5, SRC_D)
    dml_out[kind] = {**res, **ms}

table = pd.DataFrame(rows)
n_fail = int((table["match"] == "FAIL").sum())
print(table.to_string(index=False))
print(f"\n{len(table) - n_fail} PASS, {n_fail} FAIL of {len(table)} checks")

out = {"meta": meta_out, "n_checks": len(table), "n_fail": n_fail, "checks": table.to_dict(orient="records"), "dml": dml_out}
L.write_json("00_replication.json", out)
with open(L.RESULTS / "00_replication.md", "w", encoding="utf-8") as fh:
    fh.write(L.md_table(table[["analysis", "quantity", "reported", "reproduced", "match", "source"]], "{:.6g}"))
    fh.write(f"\n\n{len(table) - n_fail} of {len(table)} checks reproduce within the printed precision.\n")
