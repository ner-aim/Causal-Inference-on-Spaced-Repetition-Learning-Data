"""s6_cloud_branch.py - does the deployed (Streamlit Cloud) DML run on the same sample as the local one?

dashboard.py load_hard_reviews_dml() has two branches. The local branch filters in SQL to
factor > 0 AND factor < 2500 AND time > 0 AND time < 60000. The CLOUD_MODE branch reads data/revlog.parquet and applies no
such filter. This script runs the CLOUD_MODE branch literally on the committed data/revlog.parquet and reports the sample
and the DML estimates it produces.

Does not need the Anki collection. Writes results/06_cloud_branch.json and results/06_cloud_branch.md.

Usage (repo root):  python audit/s6_cloud_branch.py
"""
import subprocess

import numpy as np
import pandas as pd

import audit_lib as L

L.utf8_stdout()
REPO = L.AUDIT_DIR.parent

head = subprocess.run(["git", "diff", "HEAD", "--stat", "--", "data/revlog.parquet"], cwd=REPO, capture_output=True, text=True).stdout.strip()
tracked = subprocess.run(["git", "ls-files", "data/revlog.parquet"], cwd=REPO, capture_output=True, text=True).stdout.strip()

df = pd.read_parquet(REPO / "data" / "revlog.parquet")
n_file = len(df)
# ---- literal copy of the CLOUD_MODE branch of load_hard_reviews_dml() (dashboard.py lines 156-167) and its tail (186-188)
df["ts"] = pd.to_datetime(df["id"], unit="ms")
df["hour"] = df["ts"].dt.hour
df["month_idx"] = (df["ts"].dt.year - df["ts"].dt.year.min()) * 12 + df["ts"].dt.month
df["log_rt"] = np.log(df["time"].clip(1))
df["ease_factor"] = df.get("ease_factor", df["factor"] / 1000 if "factor" in df.columns else np.nan)
df["log_ivl"] = np.log(df["ivl"].abs().clip(1)) if "ivl" in df.columns else 0.0
df = df[df["hour"].between(6, 12) | df["hour"].between(18, 24)].copy()
df["morning"] = df["hour"].between(6, 12).astype(int)
cloud = df.dropna(subset=["ease_factor", "log_ivl", "month_idx", "log_rt"])

desc = {"parquet_rows": int(n_file), "parquet_last_review_date": str(pd.to_datetime(cloud["ts"]).max().date()),
        "tracked_in_git": bool(tracked), "diff_vs_HEAD": head or "(none)",
        "cloud_branch_rows": int(len(cloud)), "cloud_branch_morning": int(cloud["morning"].sum()),
        "share_with_factor_ge_2500": float((cloud["factor"] >= 2500).mean()),
        "share_with_factor_zero": float((cloud["factor"] == 0).mean()),
        "share_capped_60s": float((cloud["time"] == L.ANKI_CAP_MS).mean()),
        "share_time_zero": float((cloud["time"] == 0).mean())}
print(desc)

rows = []
X = cloud[L.DML_FEATURES].to_numpy(float)
D = cloud["morning"].to_numpy()
Y = cloud["log_rt"].to_numpy()
cid = cloud["cid"].to_numpy()
for kind in ["ridge", "gbm"]:
    res, *_ = L.dml_fit(Y, D, X, kind, clusters=cid)
    ms = L.theta_to_ms(res["theta"], res["se"], Y)
    rows.append({"learner": kind, "n": res["n"], "theta": res["theta"], "ATE ms": ms["ate_ms"], "95% CI low": ms["ci_lo_ms"], "95% CI high": ms["ci_hi_ms"],
                 "p iid": res["p"], "p cluster(card)": res["p_cluster"]})
tab = pd.DataFrame(rows)
print(L.md_table(tab))

# local branch on the same snapshot date for comparison (needs the Anki copy; skipped if unavailable)
local_rows = []
try:
    rl, *_ = L.prepare(L.CUTOFF_DML)
    loc = L.build_dml(rl)
    for kind in ["ridge", "gbm"]:
        res, *_ = L.dml_fit(loc["log_rt"].to_numpy(), loc["morning"].to_numpy(), loc[L.DML_FEATURES].to_numpy(float), kind)
        ms = L.theta_to_ms(res["theta"], res["se"], loc["log_rt"].to_numpy())
        local_rows.append({"learner": kind, "n": res["n"], "theta": res["theta"], "ATE ms": ms["ate_ms"], "branch": "local (hard reviews, as reported)"})
except Exception as e:  # collection not available on this machine
    local_rows.append({"learner": "n/a", "branch": f"not run: {type(e).__name__}"})
ltab = pd.DataFrame(local_rows)
print(L.md_table(ltab))

L.write_json("06_cloud_branch.json", {"description": desc, "cloud_branch_estimates": rows, "local_branch_estimates": local_rows})
with open(L.RESULTS / "06_cloud_branch.md", "w", encoding="utf-8") as fh:
    fh.write(L.md_table(pd.DataFrame([desc]).T.reset_index().rename(columns={"index": "quantity", 0: "value"})) + "\n\n")
    fh.write("Cloud branch (what the deployed dashboard computes):\n\n" + L.md_table(tab) + "\n\n")
    fh.write("Local branch (what was reported):\n\n" + L.md_table(ltab) + "\n")
