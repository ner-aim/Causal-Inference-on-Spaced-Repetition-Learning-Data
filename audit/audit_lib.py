"""audit_lib.py - shared loaders and line-for-line re-implementations of the audited analyses.

Nothing in this file changes any audited analysis. Each builder mirrors the audited code and
names its source. Options that deviate from the audited behaviour (``hour_col='hour_local'``,
``timing='pre'``, ``days_mode='true_prev'``) exist only for the audit's sensitivity checks and
default to the audited behaviour.

Audited sources (all paths relative to the repo root):
  PSM v2            psm_response_time_v2.ipynb  cells 2, 4, 6, 7, 12
  PSM Phase 3       anki_causal_analysis.ipynb  cells 4, 14, 15, 17, 19
  DML               dashboard.py                load_hard_reviews_dml() and run_dml()

The Anki collection is read through the ANKI_DB environment variable (default: the live path).
It is opened read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------------------------
AUDIT_DIR = Path(__file__).resolve().parent
RESULTS = AUDIT_DIR / "results"
FIGURES = AUDIT_DIR / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DB_PATH = Path(os.environ.get("ANKI_DB", r"C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2"))

ANKI_CAP_MS = 60_000
HARD_THRESH = 2.5  # psm_response_time_v2.ipynb cell 1
LOCAL_TZ = "America/New_York"  # machine tz and col.crt both indicate UTC-5 (see REPORT.md section 3.2)

# Snapshot cutoffs that reproduce the sample sizes printed by the audited runs exactly
# (verified in s0_replication.py): rows with ts < cutoff, ts being naive UTC.
CUTOFF_NOTEBOOKS = pd.Timestamp("2026-04-24")  # 52,119 revlog rows; v2 and Phase 3 notebooks
CUTOFF_DML = pd.Timestamp("2026-07-11")        # DML run of 2026-07-10

V2_COVARIATES = ["ef", "log_ivl", "log_days_since", "month_idx"]
P3_COVARIATES = ["ease_factor", "log_ivl", "deck_enc", "log_days_since"]
DML_FEATURES = ["ease_factor", "log_ivl", "month_idx"]
HUMAN_JAPANESE_DECK = "Human Japanese Intermediate Shared"

BLUE = "#2a78d6"    # morning / treated / real estimate  (validated, dataviz skill palette slot 1)
ORANGE = "#eb6834"  # evening / control                  (validated, dataviz skill palette slot 2)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GREY = "#b9b8b2"


def utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name: str, obj) -> Path:
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    """Render a DataFrame as a GitHub markdown table (no external dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False, name=None):  # itertuples keeps each column's own dtype (ints stay ints)
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    con.create_collation("unicase", lambda a, b: (a > b) - (a < b))
    return con


def load_tables():
    """Return revlog (ordered by id, i.e. the table's rowid order the audited code relied on), cards, notes, decks."""
    con = connect()
    revlog = pd.read_sql("SELECT id, cid, ease, ivl, lastIvl, factor, time, type FROM revlog ORDER BY id", con)
    cards = pd.read_sql("SELECT id, nid, did, factor, reps, lapses FROM cards", con)
    notes = pd.read_sql("SELECT id, flds, tags FROM notes", con)
    decks = pd.read_sql("SELECT id, name FROM decks", con)
    col = con.execute("SELECT crt FROM col").fetchone()
    con.close()
    return revlog, cards, notes, decks, int(col[0])


def count_kanji(text: str) -> int:
    """CJK Unified Ideographs U+4E00..U+9FFF (same range as the project's kanji analysis)."""
    return sum(1 for ch in str(text) if "\u4e00" <= ch <= "\u9fff")


def content_features(cards: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    """Per-card properties fixed before any review: nid, kanji count of the Japanese field, deck position."""
    n = notes[["id", "flds"]].copy()
    n["jp"] = n["flds"].map(lambda f: f.split("\x1f")[1] if "\x1f" in f else "")
    n["kanji_count"] = n["jp"].map(count_kanji)
    n["nid_rank_pct"] = n["id"].rank(pct=True)
    n = n.rename(columns={"id": "nid"})[["nid", "kanji_count", "nid_rank_pct"]]
    out = cards[["id", "nid", "did"]].rename(columns={"id": "cid"}).merge(n, on="nid", how="left")
    return out


def prepare(cutoff: pd.Timestamp | None):
    """Load everything, add time columns (UTC as audited, plus Eastern), same-card history and card content.

    History features are computed on the full revlog before the cutoff is applied; they only look
    backwards, so the cutoff cannot change them for rows that survive it.
    """
    revlog, cards, notes, decks, crt = load_tables()
    rl = revlog.copy()
    rl["ts"] = pd.to_datetime(rl["id"], unit="ms")  # naive UTC, exactly as in every audited analysis
    rl["hour_utc"] = rl["ts"].dt.hour
    local = pd.to_datetime(rl["id"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    rl["hour_local"] = local.dt.hour

    rl = rl.sort_values(["cid", "id"])
    g = rl.groupby("cid", sort=False)
    for src, dst in [("id", "prev_id"), ("time", "prev_time"), ("ease", "prev_ease"), ("factor", "prev_factor"), ("type", "prev_type")]:
        rl[dst] = g[src].shift(1)
    rl["prev_time_valid"] = rl["prev_time"].where(rl["prev_time"] != ANKI_CAP_MS)  # capped = not a valid RT
    rl["prev_retained"] = (rl["prev_ease"] > 1).astype(float).where(rl["prev_ease"].notna())
    rl = rl.sort_values("id").reset_index(drop=True)

    content = content_features(cards, notes)
    deck_names = decks.set_index("id")["name"].to_dict()
    content["deck"] = content["did"].map(deck_names)
    rl = rl.merge(content[["cid", "nid", "kanji_count", "nid_rank_pct", "deck"]], on="cid", how="left")

    if cutoff is not None:
        rl = rl[rl["ts"] < cutoff].reset_index(drop=True)
    meta = {"crt": crt, "n_revlog_total": int(len(revlog)), "n_after_cutoff": int(len(rl))}
    return rl, cards, notes, decks, meta


# ---------------------------------------------------------------------------------------------
# Builders (mirror the audited feature code)
# ---------------------------------------------------------------------------------------------
def build_v2(rl: pd.DataFrame, hour_col: str = "hour_utc", timing: str = "post", days_mode: str = "subset"):
    """psm_response_time_v2.ipynb cells 2 and 4. Returns (df, df_trim) as in the notebook.

    timing='post' (audited): ef from revlog.factor, interval from revlog.ivl - both written AFTER the answer.
    timing='pre'  (sensitivity): ef from the previous row of the same card, interval from lastIvl.
    days_mode='subset' (audited): previous review is the previous row *inside the filtered subset*.
    days_mode='true_prev' (sensitivity): previous review of the card in the full history.
    """
    d = rl.copy()
    d["hour"] = d[hour_col]
    if timing == "post":
        d["ef"] = d["factor"] / 1000
        ivl_feat = d["ivl"]
    elif timing == "pre":
        d["ef"] = d["prev_factor"] / 1000
        ivl_feat = d["lastIvl"]
    else:
        raise ValueError(timing)
    keep = (d["ef"] > 0) & (d["ef"] < HARD_THRESH) & (d["hour"].between(6, 11) | d["hour"].between(18, 23))
    d = d[keep].copy()
    d["morning"] = d["hour"].between(6, 11).astype(int)
    d["capped"] = (d["time"] == ANKI_CAP_MS).astype(int)
    d["abs_ivl"] = ivl_feat.loc[d.index].abs()
    d["log_ivl"] = np.log1p(d["abs_ivl"])
    d = d.sort_values(["cid", "ts"])
    if days_mode == "subset":
        d["prev_ts_feat"] = d.groupby("cid")["ts"].shift(1)
    elif days_mode == "true_prev":
        d["prev_ts_feat"] = pd.to_datetime(d["prev_id"], unit="ms")
    else:
        raise ValueError(days_mode)
    d["days_since"] = (d["ts"] - d["prev_ts_feat"]).dt.total_seconds() / 86400
    d["log_days_since"] = np.log1p(d["days_since"].fillna(d["days_since"].median()))
    d["month_idx"] = (d["ts"].dt.year - d["ts"].dt.year.min()) * 12 + d["ts"].dt.month - d["ts"].dt.month.min()
    df_trim = d[d["capped"] == 0].dropna(subset=V2_COVARIATES + ["time", "morning"]).copy()
    return d, df_trim


def build_phase3(rl: pd.DataFrame, hour_col: str = "hour_utc", timing: str = "post", days_mode: str = "subset"):
    """anki_causal_analysis.ipynb cells 4 and 14. Returns psm_clean.

    The notebook does not filter ease_factor > 0 and filters ivl != 0 (kept as audited).
    """
    r = rl.copy()
    r["hour"] = r[hour_col]
    r["retained"] = (r["ease"] > 1).astype(int)
    if timing == "post":
        r["ease_factor"] = r["factor"] / 1000
        ivl_feat = r["ivl"]
    elif timing == "pre":
        r["ease_factor"] = r["prev_factor"] / 1000
        ivl_feat = r["lastIvl"]
    else:
        raise ValueError(timing)
    keep = (r["hour"].between(6, 11) | r["hour"].between(18, 23)) & (r["ivl"] != 0)
    p = r[keep].copy()
    p["morning"] = p["hour"].between(6, 11).astype(int)
    p["abs_ivl"] = ivl_feat.loc[p.index].abs()
    p["log_ivl"] = np.log1p(p["abs_ivl"])
    p["deck_enc"] = (p["deck"] == HUMAN_JAPANESE_DECK).astype(int)
    p = p.sort_values(["cid", "ts"])
    if days_mode == "subset":
        p["prev_ts_feat"] = p.groupby("cid")["ts"].shift(1)
    else:
        p["prev_ts_feat"] = pd.to_datetime(p["prev_id"], unit="ms")
    p["days_since_last"] = (p["ts"] - p["prev_ts_feat"]).dt.total_seconds() / 86400
    p["log_days_since"] = np.log1p(p["days_since_last"].fillna(p["days_since_last"].median()))
    p["month_idx"] = (p["ts"].dt.year - p["ts"].dt.year.min()) * 12 + p["ts"].dt.month - p["ts"].dt.month.min()
    return p.dropna(subset=P3_COVARIATES + ["retained", "morning"]).copy()


def build_dml(rl: pd.DataFrame, hour_col: str = "hour_utc", timing: str = "post"):
    """dashboard.py load_hard_reviews_dml(), local-DB branch. Note the audited windows are 6-12 and 18-24 (inclusive),
    which differ from the 6-11 / 18-23 windows of the PSM notebooks."""
    d = rl.copy()
    d["hour"] = d[hour_col]
    if timing == "post":
        fac, ivl_feat = d["factor"], d["ivl"]
    elif timing == "pre":
        fac, ivl_feat = d["prev_factor"], d["lastIvl"]
    else:
        raise ValueError(timing)
    keep = (fac > 0) & (fac < 2500) & (d["time"] > 0) & (d["time"] < ANKI_CAP_MS)
    d = d[keep].copy()
    d["ease_factor"] = fac[keep] / 1000
    d["log_ivl"] = np.log(ivl_feat[keep].abs().clip(1))
    d["month_idx"] = (d["ts"].dt.year - d["ts"].dt.year.min()) * 12 + d["ts"].dt.month
    d["log_rt"] = np.log(d["time"])
    d = d[d["hour"].between(6, 12) | d["hour"].between(18, 24)].copy()
    d["morning"] = d["hour"].between(6, 12).astype(int)
    return d.dropna(subset=DML_FEATURES + ["log_rt"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# PSM machinery (notebook cells: StandardScaler -> LogisticRegression -> caliper 1-NN on log-odds)
# ---------------------------------------------------------------------------------------------
def scale_all(X: np.ndarray) -> np.ndarray:
    """Audited: scaler fit on ALL rows (psm v2 cell 6, Phase 3 cell 15)."""
    return StandardScaler().fit_transform(X)


def fit_ps(X_sc: np.ndarray, treat: np.ndarray):
    """Audited propensity model: LogisticRegression(max_iter=1000, C=1.0, random_state=42) on scaled covariates."""
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_sc, treat)
    return lr.predict_proba(X_sc)[:, 1], lr


def match_on_ps(ps: np.ndarray, treat: np.ndarray, caliper_mult: float = 0.2):
    """1-NN (ball_tree) on log-odds from treated to controls WITH replacement, keep pairs within the caliper.

    log_odds uses eps=1e-6 and caliper = 0.2 * pandas-std(log_odds), as in v2 cell 7 / Phase 3 cell 17.
    Returns (matched_treated_positions, matched_control_positions, caliper).
    """
    eps = 1e-6
    log_odds = np.log((ps + eps) / (1 - ps + eps))
    caliper = caliper_mult * np.std(log_odds, ddof=1)
    t_pos = np.flatnonzero(treat == 1)
    c_pos = np.flatnonzero(treat == 0)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nbrs.fit(log_odds[c_pos].reshape(-1, 1))
    dist, ind = nbrs.kneighbors(log_odds[t_pos].reshape(-1, 1))
    ok = dist.ravel() <= caliper
    return t_pos[ok], c_pos[ind.ravel()[ok]], float(caliper)


def psm_fit_match(X_sc: np.ndarray, treat: np.ndarray, caliper_mult: float = 0.2):
    """Return (ps, matched_treated_positions, matched_control_positions, caliper, lr) - the audited PSM steps in one call."""
    ps, lr = fit_ps(X_sc, treat)
    t_pos, c_pos, caliper = match_on_ps(ps, treat, caliper_mult)
    return ps, t_pos, c_pos, caliper, lr


def pair_effects(y_t: np.ndarray, y_c: np.ndarray) -> dict:
    """Matched-pair mean and median differences. Pairs with a missing outcome on either side are dropped
    (a no-op for the audited outcomes, which have no missing values)."""
    y_t = np.asarray(y_t, float)
    y_c = np.asarray(y_c, float)
    ok = ~(np.isnan(y_t) | np.isnan(y_c))
    y_t, y_c = y_t[ok], y_c[ok]
    return {"mean_diff": float(np.mean(y_t) - np.mean(y_c)), "median_diff": float(np.median(y_t) - np.median(y_c)),
            "n_pairs": int(ok.sum())}


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Notebook definition (v2 cell 8): |mean diff| / sqrt((sd_a^2 + sd_b^2)/2 ) with pandas ddof=1 and +1e-9."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(abs(a.mean() - b.mean()) / (np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) + 1e-9))


# ---------------------------------------------------------------------------------------------
# DML machinery (dashboard.py run_dml / _dml_fit)
# ---------------------------------------------------------------------------------------------
def make_learners(kind: str):
    if kind == "ridge":
        return (lambda: RidgeCV(alphas=np.logspace(-3, 3, 20)),
                lambda: LogisticRegressionCV(cv=5, max_iter=1000, random_state=42))
    if kind == "gbm":
        return (lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
                lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    raise ValueError(kind)


def make_folds(n: int, seed: int = 42, n_splits: int = 5):
    """Audited: KFold(n_splits=5, shuffle=True, random_state=42) over rows (row-level, ignores card, note and time)."""
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)))


def grouped_folds(groups: np.ndarray, seed: int = 42, n_splits: int = 5):
    """Sensitivity: every group (e.g. every note) lands wholly in one test fold."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {g: int(k % n_splits) for k, g in zip(order, uniq)}  # order[i] is the rank of uniq[i]
    f = np.array([fold_of[g] for g in groups])
    idx = np.arange(len(groups))
    return [(idx[f != k], idx[f == k]) for k in range(n_splits)]


def blocked_time_folds(n: int, n_splits: int = 5):
    """Sensitivity: contiguous blocks of the time-ordered rows (rows must already be in id order)."""
    idx = np.arange(n)
    blocks = np.array_split(idx, n_splits)
    return [(np.concatenate([b for j, b in enumerate(blocks) if j != k]), blocks[k]) for k in range(n_splits)]


def crossfit(Y: np.ndarray, D: np.ndarray, X: np.ndarray, folds, kind: str, scale: str = "global"):
    """Return (Y_res, D_res, g_hat). scale='global' is audited (scaler fit on all rows before splitting)."""
    n = len(Y)
    Xs_all = scale_all(X) if scale == "global" else None
    Y_res = np.zeros(n)
    D_res = np.zeros(n)
    g_hat = np.zeros(n)
    out_factory, prop_factory = make_learners(kind)
    for tr, te in folds:
        if scale == "global":
            Xtr, Xte = Xs_all[tr], Xs_all[te]
        elif scale == "in_fold":
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        else:
            raise ValueError(scale)
        m = out_factory()
        m.fit(Xtr, Y[tr])
        Y_res[te] = Y[te] - m.predict(Xte)
        g = prop_factory()
        g.fit(Xtr, D[tr])
        g_hat[te] = g.predict_proba(Xte)[:, 1]
        D_res[te] = D[te] - g_hat[te]
    return Y_res, D_res, g_hat


def dml_theta(Y_res: np.ndarray, D_res: np.ndarray, clusters=None) -> dict:
    """theta, iid SE (exactly the audited formula) and optionally a cluster-robust SE of the same score."""
    n = len(Y_res)
    theta = float(np.dot(D_res, Y_res) / np.dot(D_res, D_res))
    psi = D_res * (Y_res - theta * D_res)
    J = float(np.mean(D_res ** 2))
    se = float(np.sqrt(np.mean(psi ** 2) / n) / abs(J))
    out = {"theta": theta, "se": se, "z": theta / se, "p": float(2 * stats.norm.sf(abs(theta / se))), "n": int(n)}
    if clusters is not None:
        s = pd.Series(psi).groupby(np.asarray(clusters)).sum().to_numpy()
        se_c = float(np.sqrt(np.sum(s ** 2)) / (n * abs(J)))
        out.update(se_cluster=se_c, p_cluster=float(2 * stats.norm.sf(abs(theta / se_c))), n_clusters=int(len(s)))
    return out


def theta_to_ms(theta: float, se: float, log_y: np.ndarray) -> dict:
    """Audited conversion: (exp(theta) - 1) * geometric-mean RT, with theta +/- 1.96 se."""
    gm = float(np.exp(np.mean(log_y)))
    return {"gm_rt_ms": gm, "ate_ms": (np.exp(theta) - 1) * gm,
            "ci_lo_ms": (np.exp(theta - 1.96 * se) - 1) * gm, "ci_hi_ms": (np.exp(theta + 1.96 * se) - 1) * gm}


def dml_fit(Y, D, X, kind, folds=None, scale="global", clusters=None):
    folds = folds if folds is not None else make_folds(len(Y))
    Y_res, D_res, g_hat = crossfit(Y, D, X, folds, kind, scale=scale)
    res = dml_theta(Y_res, D_res, clusters=clusters)
    return res, Y_res, D_res, g_hat


# ---------------------------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------------------------
def make_permutations(labels: np.ndarray, n_perm: int, seed: int, strata=None) -> np.ndarray:
    """n_perm x n integer array of permuted labels. strata=None: unrestricted shuffle; otherwise shuffle within strata."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    out = np.empty((n_perm, len(labels)), dtype=labels.dtype)
    groups = None
    if strata is not None:
        groups = list(pd.Series(np.arange(len(labels))).groupby(np.asarray(strata)).indices.values())
    for b in range(n_perm):
        if groups is None:
            out[b] = rng.permutation(labels)
        else:
            row = labels.copy()
            for idx in groups:
                row[idx] = rng.permutation(labels[idx])
            out[b] = row
    return out


def perm_pvalue(null: np.ndarray, real: float) -> float:
    """Two-sided permutation p-value with the +1 correction."""
    null = np.asarray(null, float)
    return float((1 + np.sum(np.abs(null) >= abs(real))) / (1 + len(null)))


def psm_perm_worker(treat_perm: np.ndarray, X_sc: np.ndarray, y_cols: dict) -> dict:
    """One PSM replicate: refit PS on the shuffled labels, re-match, return the matched effects for each outcome."""
    _, t_pos, c_pos, _, _ = psm_fit_match(X_sc, treat_perm)
    return {k: pair_effects(v[t_pos], v[c_pos]) for k, v in y_cols.items()}


def psm_order_worker(order: np.ndarray, X_sc: np.ndarray, treat: np.ndarray, y: np.ndarray, kind: str) -> dict:
    """One row-order replicate: identical data, identical PS model, rows presented to the matcher in a different order.

    Only the tie-breaking of the nearest-neighbour search can change. kind='cont' reports a Mann-Whitney p on the
    matched pairs (as in PSM v2), kind='binary' a t-test p (as in Phase 3).
    """
    _, t_pos, c_pos, _, _ = psm_fit_match(X_sc[order], treat[order])
    yo = y[order]
    a, b = yo[t_pos], yo[c_pos]
    eff = pair_effects(a, b)
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if kind == "cont" else stats.ttest_ind(a, b).pvalue
    return {"mean_diff": eff["mean_diff"], "median_diff": eff["median_diff"], "p": float(p),
            "n_pairs": int(len(t_pos)), "n_unique_controls": int(len(np.unique(c_pos)))}


def dml_perm_worker(D_perm: np.ndarray, Y_res: np.ndarray, Xs: np.ndarray, folds, kind: str) -> float:
    """One DML replicate. The outcome nuisance never sees D, so Y_res is shared; only the propensity nuisance is refit."""
    _, prop_factory = make_learners(kind)
    D_res = np.zeros(len(D_perm))
    for tr, te in folds:
        g = prop_factory()
        g.fit(Xs[tr], D_perm[tr])
        D_res[te] = D_perm[te] - g.predict_proba(Xs[te])[:, 1]
    return dml_theta(Y_res, D_res)["theta"]


# ---------------------------------------------------------------------------------------------
# Plot style (one place; palette validated with the dataviz skill's validate_palette.js)
# ---------------------------------------------------------------------------------------------
def mpl_style():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GREY, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })
    return plt
