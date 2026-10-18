"""s2_placebo.py - placebo and permutation checks for the PSM and DML analyses.

(a) Shuffle the morning/evening labels B=500 times (unrestricted, and within calendar month) and
    compare the null distribution of the SAME estimator with the real estimate.
(b) Re-estimate on a pre-treatment outcome: the same card's PREVIOUS review (response time, or
    retained for Phase 3). The current review's time of day cannot cause it.
(c) Re-estimate on outcomes the treatment cannot affect: kanji count of the card's Japanese field and
    the note's position in the deck (rank of note id). Both are fixed before any review.

Each permutation refits the whole estimator. For DML the outcome nuisance never sees the treatment, so
it is fitted once per outcome and only the propensity nuisance is refit per permutation; this is exactly
equivalent to rerunning the audited pipeline on shuffled labels.

Writes results/02_placebo.json, results/02_placebo.md, results/02_perm_nulls.npz,
figures/placebo_permutation_null.png, figures/placebo_outcomes.png, figures/placebo_row_order.png.

Usage (repo root):  ANKI_DB=<collection copy> python audit/s2_placebo.py [B]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import audit_lib as L

L.utf8_stdout()
warnings.filterwarnings("ignore")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N_JOBS = 16
SEED_U, SEED_S, SEED_O = 20260418, 20260419, 20260420


def summarize(null, real):
    """perm_p is the standard two-sided permutation p-value. Because a matching estimator can be off-centre under
    shuffled labels, the centred versions (null mean subtracted from both sides) are reported next to it."""
    null = np.asarray(null, float)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    return {"real": float(real), "null_mean": mu, "null_sd": sd,
            "null_q025": float(np.percentile(null, 2.5)), "null_q975": float(np.percentile(null, 97.5)),
            "perm_p": L.perm_pvalue(null, real), "z_vs_null": float(real / sd),
            "z_centered": float((real - mu) / sd), "perm_p_centered": L.perm_pvalue(null - mu, real - mu), "B": int(len(null))}


def matching_diagnostics(frame, cov):
    """What the audited 1-NN matcher does with the real labels: reuse of controls and which controls get used."""
    X_sc = L.scale_all(frame[cov].to_numpy(float))
    treat = frame["morning"].to_numpy().astype(np.int8)
    ps, t_pos, c_pos, _, _ = L.psm_fit_match(X_sc, treat)
    c_all = np.flatnonzero(treat == 0)
    uses = np.bincount(c_pos, minlength=len(treat))[c_all]
    rank_in_controls = np.searchsorted(c_all, c_pos) / len(c_all)  # 0 = first control row in frame order, 1 = last
    nid = frame["nid_rank_pct"].to_numpy(float)
    ps_round = np.round(ps, 12)
    _, counts = np.unique(ps_round, return_counts=True)
    top10 = np.sort(uses)[::-1][:10].sum()
    rho = stats.spearmanr(np.arange(len(c_all)), uses).correlation
    return {"n_rows": int(len(treat)), "n_treated": int(treat.sum()), "n_controls": int(len(c_all)), "n_pairs": int(len(t_pos)),
            "n_unique_controls_used": int(len(np.unique(c_pos))), "max_times_one_control_used": int(uses.max()),
            "share_of_pairs_on_10_most_reused_controls": float(top10 / len(t_pos)),
            "median_position_of_matched_controls_in_row_order": float(np.median(rank_in_controls)),
            "spearman_control_row_position_vs_times_used": float(rho),
            "share_of_rows_sharing_a_ps_value_with_another_row": float(counts[counts > 1].sum() / len(ps)),
            "frame_sorted_by_cid": bool(np.all(np.diff(frame["cid"].to_numpy()) >= 0)),
            "share_rows_with_missing_card": float(np.isnan(nid).mean()),
            "mean_deck_position_all_treated": float(np.nanmean(nid[treat == 1])), "mean_deck_position_matched_treated": float(np.nanmean(nid[t_pos])),
            "mean_deck_position_matched_controls": float(np.nanmean(nid[c_pos])), "mean_deck_position_all_controls": float(np.nanmean(nid[c_all]))}


def row_order_sensitivity(frame, cov, y, kind, label):
    """Same data, same PS model; only the order in which rows reach the matcher changes (B random orders)."""
    X_sc = L.scale_all(frame[cov].to_numpy(float))
    treat = frame["morning"].to_numpy().astype(np.int8)
    rng = np.random.default_rng(SEED_O)
    orders = [rng.permutation(len(treat)) for _ in range(B)]
    res = Parallel(n_jobs=N_JOBS)(delayed(L.psm_order_worker)(o, X_sc, treat, y, kind) for o in orders)
    arr = {k: np.array([r[k] for r in res], float) for k in ["mean_diff", "median_diff", "p", "n_pairs", "n_unique_controls"]}
    audited = L.psm_order_worker(np.arange(len(treat)), X_sc, treat, y, kind)
    summ = {"label": label, "B": B, "audited_order": audited}
    for k in ["mean_diff", "median_diff", "p", "n_unique_controls"]:
        a = arr[k]
        summ[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()), "q05": float(np.percentile(a, 5)),
                   "median": float(np.median(a)), "q95": float(np.percentile(a, 95)), "max": float(a.max())}
    key = "median_diff" if kind == "cont" else "mean_diff"
    summ["share_p_below_0.05"] = float(np.mean(arr["p"] < 0.05))
    summ["share_estimate_same_sign_as_audited"] = float(np.mean(np.sign(arr[key]) == np.sign(audited[key])))
    return summ, arr


def run_psm(name, frame, cov, y_cols, stat_for, real_outcome):
    """frame: audited analysis sample. y_cols: outcome name -> array aligned with frame rows."""
    t0 = time.time()
    X_sc = L.scale_all(frame[cov].to_numpy(float))
    treat = frame["morning"].to_numpy().astype(np.int8)
    _, t_pos, c_pos, _, _ = L.psm_fit_match(X_sc, treat)
    real = {k: L.pair_effects(v[t_pos], v[c_pos]) for k, v in y_cols.items()}
    perms_u = L.make_permutations(treat, B, SEED_U)
    nul_u = Parallel(n_jobs=N_JOBS)(delayed(L.psm_perm_worker)(perms_u[b], X_sc, y_cols) for b in range(B))
    perms_s = L.make_permutations(treat, B, SEED_S, strata=frame["month_idx"].to_numpy())
    nul_s = Parallel(n_jobs=N_JOBS)(delayed(L.psm_perm_worker)(perms_s[b], X_sc, y_cols) for b in range(B))
    out, nulls = {}, {}
    for k in y_cols:
        stat = stat_for[k]
        arr_u = np.array([d[k][stat] for d in nul_u])
        arr_s = np.array([d[k][stat] for d in nul_s])
        a, b_ = y_cols[k][t_pos], y_cols[k][c_pos]
        ok = ~(np.isnan(a) | np.isnan(b_))
        mw_p = float(stats.mannwhitneyu(a[ok], b_[ok], alternative="two-sided").pvalue) if len(np.unique(np.concatenate([a[ok], b_[ok]]))) > 2 else None
        out[k] = {"stat": stat, "n_pairs": real[k]["n_pairs"], "mean_diff": real[k]["mean_diff"], "median_diff": real[k]["median_diff"],
                  "unrestricted": summarize(arr_u, real[k][stat]), "within_month": summarize(arr_s, real[k][stat]),
                  "mann_whitney_p_on_pairs": mw_p, "is_real_outcome": k == real_outcome}
        nulls[f"{name}|{k}|unrestricted"] = arr_u
        nulls[f"{name}|{k}|within_month"] = arr_s
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
    return out, nulls


def run_dml(kind, frame, outcomes, real_outcome):
    t0 = time.time()
    X_all = frame[L.DML_FEATURES].to_numpy(float)
    D_all = frame["morning"].to_numpy().astype(np.int8)
    cid_all = frame["cid"].to_numpy()
    month_all = frame["month_idx"].to_numpy()
    out, nulls = {}, {}
    for oname, y_full in outcomes.items():
        ok = np.isfinite(y_full)
        Y, X, D, cid, month = y_full[ok], X_all[ok], D_all[ok], cid_all[ok], month_all[ok]
        folds = L.make_folds(len(Y))
        res, Y_res, D_res, _ = L.dml_fit(Y, D, X, kind, folds=folds, clusters=cid)
        Xs = L.scale_all(X)
        perms_u = L.make_permutations(D, B, SEED_U)
        arr_u = np.array(Parallel(n_jobs=N_JOBS)(delayed(L.dml_perm_worker)(perms_u[b], Y_res, Xs, folds, kind) for b in range(B)))
        rec = {"n": int(len(Y)), "theta": res["theta"], "se_iid": res["se"], "p_iid": res["p"],
               "se_cluster_card": res["se_cluster"], "p_cluster_card": res["p_cluster"], "n_cards": res["n_clusters"],
               "unrestricted": summarize(arr_u, res["theta"]), "is_real_outcome": oname == real_outcome}
        nulls[f"DML {kind}|{oname}|unrestricted"] = arr_u
        if oname == real_outcome:
            perms_s = L.make_permutations(D, B, SEED_S, strata=month)
            arr_s = np.array(Parallel(n_jobs=N_JOBS)(delayed(L.dml_perm_worker)(perms_s[b], Y_res, Xs, folds, kind) for b in range(B)))
            rec["within_month"] = summarize(arr_s, res["theta"])
            nulls[f"DML {kind}|{oname}|within_month"] = arr_s
        out[oname] = rec
        print(f"[DML {kind}|{oname}] theta={res['theta']:+.4f} perm_p={rec['unrestricted']['perm_p']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return out, nulls


def main():
    results, all_nulls = {}, {}

    # ------------------------------------------------------------ PSM v2
    rl_nb, *_ = L.prepare(L.CUTOFF_NOTEBOOKS)
    f2 = L.build_v2(rl_nb)[1].reset_index(drop=True)
    y2 = {"time": f2["time"].to_numpy(float), "prev_time": f2["prev_time_valid"].to_numpy(float),
          "kanji_count": f2["kanji_count"].to_numpy(float), "nid_rank_pct": f2["nid_rank_pct"].to_numpy(float)}
    st2 = {"time": "median_diff", "prev_time": "median_diff", "kanji_count": "mean_diff", "nid_rank_pct": "mean_diff"}
    results["PSM v2"], n_ = run_psm("PSM v2", f2, L.V2_COVARIATES, y2, st2, "time")
    all_nulls.update(n_)
    extras = {"matching_diagnostics": {}, "row_order": {}}
    extras["matching_diagnostics"]["PSM v2"] = matching_diagnostics(f2, L.V2_COVARIATES)
    extras["row_order"]["PSM v2"], arr = row_order_sensitivity(f2, L.V2_COVARIATES, y2["time"], "cont", "response time, ms")
    all_nulls["order|PSM v2|median_diff"], all_nulls["order|PSM v2|p"] = arr["median_diff"], arr["p"]

    # ------------------------------------------------------------ PSM Phase 3
    f3 = L.build_phase3(rl_nb).reset_index(drop=True)
    y3 = {"retained": f3["retained"].to_numpy(float), "prev_retained": f3["prev_retained"].to_numpy(float),
          "kanji_count": f3["kanji_count"].to_numpy(float), "nid_rank_pct": f3["nid_rank_pct"].to_numpy(float)}
    st3 = {k: "mean_diff" for k in y3}
    results["PSM Phase 3"], n_ = run_psm("PSM Phase 3", f3, L.P3_COVARIATES, y3, st3, "retained")
    all_nulls.update(n_)
    extras["matching_diagnostics"]["PSM Phase 3"] = matching_diagnostics(f3, L.P3_COVARIATES)
    extras["row_order"]["PSM Phase 3"], arr = row_order_sensitivity(f3, L.P3_COVARIATES, y3["retained"], "binary", "retention (0/1)")
    all_nulls["order|PSM Phase 3|mean_diff"], all_nulls["order|PSM Phase 3|p"] = arr["mean_diff"], arr["p"]

    # ------------------------------------------------------------ DML (both learners)
    rl_d, *_ = L.prepare(L.CUTOFF_DML)
    fd = L.build_dml(rl_d)
    outcomes = {"log_rt": fd["log_rt"].to_numpy(float),
                "prev_log_rt": np.log(fd["prev_time_valid"].where(fd["prev_time_valid"] > 0).to_numpy(float)),
                "kanji_count": fd["kanji_count"].to_numpy(float),
                "nid_rank_pct": fd["nid_rank_pct"].to_numpy(float)}
    for kind in ["ridge", "gbm"]:
        results[f"DML {kind}"], n_ = run_dml(kind, fd, outcomes, "log_rt")
        all_nulls.update(n_)

    L.write_json("02_placebo.json", {"B": B, "seed_unrestricted": SEED_U, "seed_within_month": SEED_S, "seed_row_order": SEED_O,
                                     "results": results, "extras": extras})
    np.savez_compressed(L.RESULTS / "02_perm_nulls.npz", **all_nulls)
    make_tables(results, extras)
    make_figures(results, all_nulls, extras)


LABEL = {"time": "response time (real outcome)", "retained": "retained (real outcome)", "log_rt": "log response time (real outcome)",
         "prev_time": "(b) previous review's response time", "prev_retained": "(b) previous review retained",
         "prev_log_rt": "(b) previous review's log response time",
         "kanji_count": "(c) kanji count of card", "nid_rank_pct": "(c) note position in deck"}


def make_tables(results, extras):
    rows_a, rows_b = [], []
    for method, res in results.items():
        for oname, r in res.items():
            if r["is_real_outcome"]:
                u, w = r["unrestricted"], r.get("within_month")
                rows_a.append({"analysis": method, "statistic": r.get("stat", "theta (log RT)"), "real estimate": u["real"],
                               "null sd": u["null_sd"], "null 2.5%": u["null_q025"], "null 97.5%": u["null_q975"],
                               "perm p (unrestricted)": u["perm_p"], "perm p (within month)": w["perm_p"] if w else np.nan, "B": u["B"]})
            else:
                u = r["unrestricted"]
                rows_b.append({"analysis": method, "outcome": LABEL[oname], "n": r.get("n", r.get("n_pairs")),
                               "estimate": u["real"], "null mean": u["null_mean"], "null sd": u["null_sd"],
                               "centred z": u["z_centered"], "perm p": u["perm_p"], "perm p (centred)": u["perm_p_centered"],
                               "analytic p": r.get("p_iid", r.get("mann_whitney_p_on_pairs")),
                               "cluster-by-card p": r.get("p_cluster_card", np.nan)})
    ta, tb = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    md = pd.DataFrame(extras["matching_diagnostics"]).T.reset_index().rename(columns={"index": "analysis"})
    ro = []
    for name, r in extras["row_order"].items():
        key = "median_diff" if name == "PSM v2" else "mean_diff"
        ro.append({"analysis": name, "estimate": f"{r['label']} ({key})", "audited order": r["audited_order"][key],
                   "across row orders: mean": r[key]["mean"], "sd": r[key]["sd"], "min": r[key]["min"], "max": r[key]["max"],
                   "share of orders with p<0.05": r["share_p_below_0.05"], "same sign as audited": r["share_estimate_same_sign_as_audited"],
                   "B": r["B"]})
    tro = pd.DataFrame(ro)
    with open(L.RESULTS / "02_placebo.md", "w", encoding="utf-8") as fh:
        fh.write("### (a) Real estimate against the shuffled-label null\n\n" + L.md_table(ta, "{:.4g}") + "\n\n")
        fh.write("### (b) and (c) Placebo outcomes\n\n" + L.md_table(tb, "{:.4g}") + "\n\n")
        fh.write("### Matching diagnostics (real labels)\n\n" + L.md_table(md, "{:.4g}") + "\n\n")
        fh.write("### Row-order sensitivity of the PSM estimate\n\n" + L.md_table(tro, "{:.4g}") + "\n")
    print()
    print(L.md_table(md, "{:.4g}"))
    print()
    print(L.md_table(tro, "{:.4g}"))
    print(L.md_table(ta, "{:.4g}"))
    print()
    print(L.md_table(tb, "{:.4g}"))


def make_figures(results, nulls, extras):
    plt = L.mpl_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    spec = [("PSM v2", "time", "median matched-pair difference (ms)", 1.0),
            ("PSM Phase 3", "retained", "matched ATE on retention (percentage points)", 100.0),
            ("DML ridge", "log_rt", "theta, effect on log response time", 1.0),
            ("DML gbm", "log_rt", "theta, effect on log response time", 1.0)]
    for ax, (method, oname, xlabel, scale) in zip(axes.ravel(), spec):
        u = nulls[f"{method}|{oname}|unrestricted"] * scale
        w = nulls[f"{method}|{oname}|within_month"] * scale
        r = results[method][oname]
        real = r["unrestricted"]["real"] * scale
        lo, hi = min(u.min(), w.min(), real), max(u.max(), w.max(), real)
        pad = (hi - lo) * 0.06
        bins = np.linspace(lo - pad, hi + pad, 46)
        ax.hist(u, bins=bins, color=L.GREY, alpha=0.9, label="shuffled labels, unrestricted")
        ax.hist(w, bins=bins, histtype="step", color=L.ORANGE, lw=1.6, label="shuffled within calendar month")
        ax.axvline(real, color=L.BLUE, lw=2.2, label="real estimate")
        ax.set_title(method, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("permutations")
        ax.text(0.98, 0.95, f"p = {r['unrestricted']['perm_p']:.3f} (unrestricted)\np = {r['within_month']['perm_p']:.3f} (within month)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=L.INK2)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(L.FIGURES / "placebo_permutation_null.png")
    plt.close(fig)

    # placebo outcomes: estimate in units of its own permutation-null sd
    rows = []
    for method, res in results.items():
        for oname, r in res.items():
            rows.append((method, oname, r["unrestricted"]["z_centered"], r["is_real_outcome"]))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ypos = np.arange(len(rows))[::-1]
    for y, (method, oname, z, is_real) in zip(ypos, rows):
        ax.scatter(z, y, s=46, color=L.BLUE if is_real else L.ORANGE, zorder=3, edgecolor=L.SURFACE, linewidth=1.0)
        ax.text(z + (0.35 if z >= 0 else -0.35), y, f"{z:+.1f}", va="center", ha="left" if z >= 0 else "right", fontsize=8, color=L.INK2)
    ax.axvspan(-1.96, 1.96, color="#e8e7e2", alpha=0.6, zorder=0)
    ax.axvline(0, color=L.INK2, lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{m} | {LABEL[o]}" for m, o, _, _ in rows], fontsize=8)
    ax.set_xlabel("(estimate - null mean) / null sd, from shuffled-label runs of the same estimator  (grey band = +/-1.96)")
    ax.scatter([], [], color=L.BLUE, label="real outcome")
    ax.scatter([], [], color=L.ORANGE, label="placebo outcome (true effect is zero)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(L.FIGURES / "placebo_outcomes.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, name, key, scale, xlabel in [(axes[0], "PSM v2", "median_diff", 1.0, "median matched-pair difference (ms)"),
                                         (axes[1], "PSM Phase 3", "mean_diff", 100.0, "matched ATE on retention (percentage points)")]:
        arr = nulls[f"order|{name}|{key}"] * scale
        aud = extras["row_order"][name]["audited_order"][key] * scale
        ax.hist(arr, bins=40, color=L.GREY, alpha=0.95, label=f"{len(arr)} random row orders")
        ax.axvline(aud, color=L.BLUE, lw=2.2, label="estimate in the audited row order")
        ax.axvline(0, color=L.INK2, lw=0.9, ls="--", label="zero")
        ax.set_title(name + ": same data and propensity model, rows reach the matcher in a different order", fontsize=8.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("runs")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(L.FIGURES / "placebo_row_order.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
