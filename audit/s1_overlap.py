"""s1_overlap.py - propensity-score overlap for the PSM and DML analyses.

For each audited analysis: PS distribution by arm (plot), share of rows outside [0.05, 0.95],
whether trimming to that band changes the estimate, PS discriminating power (AUC), and a
calendar-month positivity table. Also repeats the PS diagnostics with the clock read in US
Eastern time instead of UTC (see s3_leakage.py / REPORT.md for why).

Writes results/01_overlap.json, results/01_overlap.md, figures/overlap_ps_by_arm.png,
figures/overlap_month_positivity.png.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import audit_lib as L

L.utf8_stdout()
plt = L.mpl_style()
LO, HI = 0.05, 0.95


def band_stats(ps, treat):
    out = {"n": int(len(ps)), "n_treated": int(treat.sum()),
           "outside_total": int(((ps < LO) | (ps > HI)).sum()),
           "outside_pct": float(100 * ((ps < LO) | (ps > HI)).mean()),
           "below_lo": int((ps < LO).sum()), "above_hi": int((ps > HI).sum()),
           "auc": float(roc_auc_score(treat, ps))}
    for arm, name in [(1, "morning"), (0, "evening")]:
        p = ps[treat == arm]
        out[name] = {"n": int(len(p)), "min": float(p.min()), "p05": float(np.percentile(p, 5)),
                     "median": float(np.median(p)), "p95": float(np.percentile(p, 95)), "max": float(p.max()),
                     "outside_n": int(((p < LO) | (p > HI)).sum()), "outside_pct": float(100 * ((p < LO) | (p > HI)).mean())}
    return out


def month_table(frame, ps, label):
    """Per calendar month: morning/evening rows and the mean PS the model assigns there."""
    t = pd.DataFrame({"month": frame["ts"].dt.to_period("M").astype(str).to_numpy(),
                      "morning": frame["morning"].to_numpy(), "ps": ps})
    g = t.groupby("month").agg(n=("morning", "size"), morning=("morning", "sum"), mean_ps=("ps", "mean")).reset_index()
    g["evening"] = g["n"] - g["morning"]
    g["morning_share"] = g["morning"] / g["n"]
    g["spec"] = label
    return g


results = {}
ps_store = {}      # name -> (ps, treat, frame, label)

# ---------------------------------------------------------------- PSM v2 and Phase 3 (in-sample logistic PS, as audited)
rl_nb, cards, notes, decks, meta = L.prepare(L.CUTOFF_NOTEBOOKS)

for tag, builder, cov, tcol, outcome_kind in [
    ("PSM v2 (UTC hours, as audited)", L.build_v2, L.V2_COVARIATES, "time", "cont"),
    ("PSM Phase 3 (UTC hours, as audited)", L.build_phase3, L.P3_COVARIATES, "retained", "binary"),
]:
    built = builder(rl_nb)
    frame = built[1] if isinstance(built, tuple) else built
    frame = frame.reset_index(drop=True)
    X_sc = L.scale_all(frame[cov].to_numpy(float))
    treat = frame["morning"].to_numpy()
    ps, t_pos, c_pos, cal, _ = L.psm_fit_match(X_sc, treat)
    y = frame[tcol].to_numpy(float)
    est_full = L.pair_effects(y[t_pos], y[c_pos])
    stats_ = band_stats(ps, treat)
    # Trimming, two variants. Both drop the rows whose PS lies outside the band.
    #   (a) keep the original PS, re-match among the remaining rows
    #   (b) refit scaler + PS on the remaining rows, then re-match (the whole audited pipeline on the trimmed sample)
    keep = (ps >= LO) & (ps <= HI)
    y_t = y[keep]
    tp_a, cp_a, _ = L.match_on_ps(ps[keep], treat[keep])
    est_trim_keep_ps = L.pair_effects(y_t[tp_a], y_t[cp_a])
    X_t = L.scale_all(frame.loc[keep, cov].to_numpy(float))
    _, tp_b, cp_b, _, _ = L.psm_fit_match(X_t, treat[keep])
    est_trim = L.pair_effects(y_t[tp_b], y_t[cp_b])
    dropped = frame.loc[~keep]
    dropped_desc = {}
    if len(dropped):
        dropped_desc = {"n": int(len(dropped)), "n_morning": int(dropped["morning"].sum()),
                        "retained_mean": float(dropped["retained"].mean()) if "retained" in dropped else None,
                        "ease_factor_zero": int((dropped.get("ease_factor", pd.Series(dtype=float)) == 0).sum()),
                        "deck_enc_zero": int((dropped.get("deck_enc", pd.Series(dtype=float)) == 0).sum())}
    results[tag] = {"band": stats_, "pairs_full": int(len(t_pos)), "pairs_trimmed": int(len(tp_b)),
                    "rows_dropped_by_trim": int((~keep).sum()), "dropped_rows": dropped_desc,
                    "est_full": est_full, "est_trimmed_keep_ps": est_trim_keep_ps, "est_trimmed": est_trim,
                    "outcome": tcol, "month_table": month_table(frame, ps, tag).to_dict(orient="records")}
    ps_store[tag] = (ps, treat, frame)

# ---------------------------------------------------------------- DML (cross-fitted OOF propensity, both learners)
rl_d, _, _, _, _ = L.prepare(L.CUTOFF_DML)
dml = L.build_dml(rl_d)
Xd, Dd, Yd = dml[L.DML_FEATURES].to_numpy(float), dml["morning"].to_numpy(), dml["log_rt"].to_numpy()
folds = L.make_folds(len(Yd))
for kind in ["ridge", "gbm"]:
    tag = f"DML {kind} (UTC hours, as audited)"
    res, Y_res, D_res, g_hat = L.dml_fit(Yd, Dd, Xd, kind, folds=folds)
    stats_ = band_stats(g_hat, Dd)
    base_ms = L.theta_to_ms(res["theta"], res["se"], Yd)
    keep = (g_hat >= LO) & (g_hat <= HI)
    drop_res = L.dml_theta(Y_res[keep], D_res[keep])  # same cross-fitted residuals, rows outside the band removed
    drop_ms = L.theta_to_ms(drop_res["theta"], drop_res["se"], Yd[keep])
    D_clip = Dd - np.clip(g_hat, LO, HI)               # DoubleML-style clipping of the propensity
    clip_res = L.dml_theta(Y_res, D_clip)
    clip_ms = L.theta_to_ms(clip_res["theta"], clip_res["se"], Yd)
    results[tag] = {"band": stats_, "est_full": {**res, **base_ms},
                    "est_trim_drop": {**drop_res, **drop_ms, "n_dropped": int((~keep).sum())},
                    "est_trim_clip": {**clip_res, **clip_ms},
                    "month_table": month_table(dml, g_hat, tag).to_dict(orient="records")}
    ps_store[tag] = (g_hat, Dd, dml)

# ---------------------------------------------------------------- same diagnostics with Eastern-time hours (sensitivity)
rl_nb_local, *_ = L.prepare(L.CUTOFF_NOTEBOOKS)
frame_v2_local = L.build_v2(rl_nb_local, hour_col="hour_local")[1].reset_index(drop=True)
X_sc = L.scale_all(frame_v2_local[L.V2_COVARIATES].to_numpy(float))
t_loc = frame_v2_local["morning"].to_numpy()
ps_l, tp_l, cp_l, _, _ = L.psm_fit_match(X_sc, t_loc)
tag_l = "PSM v2 (Eastern hours, sensitivity)"
y_l = frame_v2_local["time"].to_numpy(float)
results[tag_l] = {"band": band_stats(ps_l, t_loc), "pairs_full": int(len(tp_l)),
                  "est_full": L.pair_effects(y_l[tp_l], y_l[cp_l]),
                  "month_table": month_table(frame_v2_local, ps_l, tag_l).to_dict(orient="records")}
ps_store[tag_l] = (ps_l, t_loc, frame_v2_local)

# ---------------------------------------------------------------- v1 (saved notebook output only; cards.factor has changed since)
results["PSM v1"] = {"note": "Not re-run: v1 selects cards by cards.factor (today's snapshot), and cards.factor in the collection has "
                              "changed since 2026-04-23, so the v1 sample cannot be rebuilt. Overlap read from the executed notebook "
                              "output only (psm_response_time_executed.ipynb); see REPORT.md."}

L.write_json("01_overlap.json", results)

# ---------------------------------------------------------------- figure 1: PS by arm
panels = [k for k in ps_store if "Eastern" not in k]
fig, axes = plt.subplots(len(panels), 2, figsize=(11, 2.55 * len(panels)))
for r, tag in enumerate(panels):
    ps, treat, _ = ps_store[tag]
    s = results[tag]["band"]
    ax = axes[r, 0]
    bins = np.linspace(0, 1, 51)
    ax.hist(ps[treat == 0], bins=bins, color=L.ORANGE, alpha=0.75, label="evening (control)")
    ax.hist(ps[treat == 1], bins=bins, color=L.BLUE, alpha=0.75, label="morning (treated)")
    ax.set_yscale("log")
    ax.axvline(LO, color=L.INK2, lw=0.9, ls="--")
    ax.axvline(HI, color=L.INK2, lw=0.9, ls="--")
    ax.set_xlim(0, 1)
    ax.set_title(tag, fontsize=9.5)
    ax.set_ylabel("rows (log)")
    ax.text(0.5, 0.93, f"outside [0.05, 0.95]: {s['outside_total']:,} of {s['n']:,} ({s['outside_pct']:.1f}%)   AUC {s['auc']:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=L.INK2)
    if r == 0:
        ax.legend(loc="center right", fontsize=8)
    ax2 = axes[r, 1]
    lo_x, hi_x = float(ps.min()), float(ps.max())
    bins2 = np.linspace(lo_x, hi_x, 45)
    ax2.hist(ps[treat == 0], bins=bins2, color=L.ORANGE, alpha=0.75, density=True)
    ax2.hist(ps[treat == 1], bins=bins2, color=L.BLUE, alpha=0.75, density=True)
    ax2.set_title("same scores, zoomed to the data range", fontsize=9)
    ax2.set_ylabel("density")
    ax2.text(0.98, 0.93, f"range {lo_x:.3f} to {hi_x:.3f}", transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=L.INK2)
axes[-1, 0].set_xlabel("propensity score (PSM: in-sample logistic; DML: cross-fitted out-of-fold)")
axes[-1, 1].set_xlabel("propensity score")
fig.tight_layout()
fig.savefig(L.FIGURES / "overlap_ps_by_arm.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 2: calendar-month positivity
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
for ax, tag in zip(axes, ["PSM v2 (UTC hours, as audited)", "PSM v2 (Eastern hours, sensitivity)"]):
    mt = pd.DataFrame(results[tag]["month_table"])
    x = np.arange(len(mt))
    ax.bar(x, mt["morning_share"] * 100, color=L.BLUE, width=0.7, label="share of rows that are morning (%)")
    ax.plot(x, mt["mean_ps"] * 100, color=L.ORANGE, lw=1.8, marker="o", ms=3.5, label="mean propensity score the model assigns (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(mt["month"], rotation=60, ha="right", fontsize=7)
    ax.set_title(tag, fontsize=9.5)
    ax.set_ylabel("percent")
    zero = int((mt["morning"] == 0).sum())
    ax.text(0.02, 0.95, f"{zero} of {len(mt)} months have zero morning rows", transform=ax.transAxes, va="top", fontsize=8, color=L.INK2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(L.FIGURES / "overlap_month_positivity.png")
plt.close(fig)

# ---------------------------------------------------------------- markdown
lines = []
rows = []
for tag, r in results.items():
    if "band" not in r:
        continue
    b = r["band"]
    rows.append({"analysis": tag, "n": b["n"], "morning": b["n_treated"],
                 "PS range morning": f"{b['morning']['min']:.3f}-{b['morning']['max']:.3f}",
                 "PS range evening": f"{b['evening']['min']:.3f}-{b['evening']['max']:.3f}",
                 "outside [0.05,0.95]": f"{b['outside_total']} ({b['outside_pct']:.1f}%)",
                 "below 0.05": b["below_lo"], "above 0.95": b["above_hi"], "PS AUC": round(b["auc"], 3)})
lines.append(L.md_table(pd.DataFrame(rows)))
lines.append("")
trim_rows = []
for tag, r in results.items():
    if tag.startswith("PSM v2 (UTC"):
        for key, label in [("median_diff", "median diff, ms"), ("mean_diff", "mean diff, ms")]:
            trim_rows.append({"analysis": tag, "estimate": label, "full": f"{r['est_full'][key]:+.0f}",
                              "trim, keep PS": f"{r['est_trimmed_keep_ps'][key]:+.0f}",
                              "trim, refit PS": f"{r['est_trimmed'][key]:+.0f}", "rows dropped": r["rows_dropped_by_trim"]})
    elif tag.startswith("PSM Phase 3"):
        trim_rows.append({"analysis": tag, "estimate": "ATE retention, pp", "full": f"{100 * r['est_full']['mean_diff']:+.3f}",
                          "trim, keep PS": f"{100 * r['est_trimmed_keep_ps']['mean_diff']:+.3f}",
                          "trim, refit PS": f"{100 * r['est_trimmed']['mean_diff']:+.3f}", "rows dropped": r["rows_dropped_by_trim"]})
    elif tag.startswith("DML"):
        trim_rows.append({"analysis": tag, "estimate": "ATE ms", "full": f"{r['est_full']['ate_ms']:+.0f}",
                          "trim, keep PS": f"{r['est_trim_drop']['ate_ms']:+.0f} (drop rows)",
                          "trim, refit PS": f"{r['est_trim_clip']['ate_ms']:+.0f} (clip PS)",
                          "rows dropped": r["est_trim_drop"]["n_dropped"]})
lines.append(L.md_table(pd.DataFrame(trim_rows)))
with open(L.RESULTS / "01_overlap.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))

for tag in ["PSM v2 (UTC hours, as audited)", "PSM v2 (Eastern hours, sensitivity)", "PSM Phase 3 (UTC hours, as audited)",
            "DML ridge (UTC hours, as audited)", "DML gbm (UTC hours, as audited)"]:
    mt = pd.DataFrame(results[tag]["month_table"])
    zero = mt[mt["morning"] == 0]
    print(f"\n[{tag}] months with zero morning rows: {len(zero)} of {len(mt)}; "
          f"mean PS in those months: {zero['mean_ps'].mean() if len(zero) else float('nan'):.4f}; "
          f"mean PS in months with morning rows: {mt[mt['morning'] > 0]['mean_ps'].mean():.4f}")
