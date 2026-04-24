"""
dashboard.py — Anki Learning Analytics Dashboard
Run with:  streamlit run dashboard.py
"""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────
DB_PATH          = Path(r"C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2")
DATA_DIR         = Path(__file__).parent / "data"
EXPERIMENT_START = date(2026, 4, 24)
EXPERIMENT_DAYS  = 60
FIGURES_DIR      = Path(__file__).parent
ANKI_CAP_MS      = 60_000

# True when running on Streamlit Cloud (no local DB available)
CLOUD_MODE = not DB_PATH.exists()

st.set_page_config(
    page_title="Anki Learning Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ──────────────────────────────────────────────────────────
BLUE   = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
RED    = "#F44336"

# ── DB helpers ──────────────────────────────────────────────────────────────

def get_connection():
    con = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
    )
    con.create_collation("unicase", lambda a, b: (a > b) - (a < b))
    con.row_factory = sqlite3.Row
    return con


@st.cache_data(ttl=300, show_spinner=False)
def load_revlog() -> pd.DataFrame:
    if CLOUD_MODE:
        p = DATA_DIR / "revlog.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    try:
        con = get_connection()
        df = pd.read_sql(
            "SELECT id, cid, ease, ivl, factor, time, type FROM revlog", con
        )
        con.close()
    except Exception:
        return pd.DataFrame()
    df["ts"]           = pd.to_datetime(df["id"], unit="ms")
    df["date"]         = df["ts"].dt.date
    df["hour"]         = df["ts"].dt.hour
    df["dow"]          = df["ts"].dt.dayofweek
    df["retained"]     = (df["ease"] > 1).astype(int)
    df["ease_factor"]  = df["factor"] / 1000
    df["time_s"]       = df["time"] / 1000
    df["capped"]       = (df["time"] == ANKI_CAP_MS).astype(int)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_experiment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (treatment_reviews, control_reviews) since EXPERIMENT_START."""
    if CLOUD_MODE:
        p = DATA_DIR / "experiment_revlog.parquet"
        if not p.exists() or p.stat().st_size < 100:
            return pd.DataFrame(), pd.DataFrame()
        exp = pd.read_parquet(p)
        if exp.empty:
            return pd.DataFrame(), pd.DataFrame()
        exp["date"] = pd.to_datetime(exp["date"]).dt.date
        return exp[exp["group"] == "treatment"], exp[exp["group"] == "control"]

    try:
        con = get_connection()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    exp_start_ms = int(datetime.combine(EXPERIMENT_START, datetime.min.time()).timestamp() * 1000)
    revlog = pd.read_sql(
        """
        SELECT r.id, r.cid, r.ease, r.ivl, r.factor, r.time, r.type,
               CASE WHEN n.tags LIKE '%exp::treatment%' THEN 'treatment' ELSE 'control' END AS grp
        FROM revlog r
        JOIN cards c ON r.cid = c.id
        JOIN notes n ON c.nid = n.id
        WHERE n.tags LIKE '%exp::%'
          AND r.id >= ?
        """,
        con, params=[exp_start_ms]
    )
    con.close()

    if revlog.empty:
        return pd.DataFrame(), pd.DataFrame()

    revlog["ts"]          = pd.to_datetime(revlog["id"], unit="ms")
    revlog["date"]        = revlog["ts"].dt.date
    revlog["retained"]    = (revlog["ease"] > 1).astype(int)
    revlog["ease_factor"] = revlog["factor"] / 1000
    revlog["capped"]      = (revlog["time"] == ANKI_CAP_MS).astype(int)
    revlog = revlog.rename(columns={"grp": "group"})

    treatment = revlog[revlog["group"] == "treatment"]
    control   = revlog[revlog["group"] == "control"]
    return treatment, control


def fig_path(name: str) -> Path:
    return FIGURES_DIR / name


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Anki Analytics")
    st.caption("Human Japanese Intermediate Shared")
    st.divider()

    days_elapsed   = (date.today() - EXPERIMENT_START).days
    days_remaining = max(0, EXPERIMENT_DAYS - days_elapsed)
    pct            = min(100, int(days_elapsed / EXPERIMENT_DAYS * 100))

    st.markdown("**A/B Experiment**")
    st.progress(pct / 100, text=f"Day {days_elapsed} of {EXPERIMENT_DAYS}")
    col1, col2 = st.columns(2)
    col1.metric("Elapsed", f"{days_elapsed}d")
    col2.metric("Remaining", f"{days_remaining}d")
    st.caption(f"Started {EXPERIMENT_START.strftime('%b %d, %Y')}")
    st.divider()

    st.markdown("**Data**")
    if CLOUD_MODE:
        meta_p = DATA_DIR / "meta.parquet"
        if meta_p.exists():
            meta = pd.read_parquet(meta_p)
            exported_at = pd.to_datetime(meta["exported_at"].iloc[0])
            st.caption(f"Last export: {exported_at.strftime('%b %d %H:%M')}")
        st.caption("Cloud mode — run `export_data.py` to update")
    else:
        st.caption("Local mode — live DB")
        st.caption("Refreshes every 5 min")

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_overview, tab_eda, tab_its, tab_psm, tab_survival, tab_ab = st.tabs([
    "📊 Overview",
    "🔍 EDA",
    "📈 ITS",
    "⚖️ PSM",
    "📉 Survival",
    "🧪 A/B Experiment",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Project Overview")
    st.markdown(
        "Causal inference analysis of my personal Anki spaced-repetition data. "
        "**449 days · 52,119 reviews · Jan 2025 – Apr 2026**"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews",    "52,119")
    c2.metric("Unique Cards",     "9,037")
    c3.metric("Retention Rate",   "99.2%")
    c4.metric("Avg Reviews/Day",  "116")
    c5.metric("Median Response",  "4.2s")

    st.divider()

    st.subheader("Summary of Findings")
    results = {
        "ITS — Did Sep 2025 change retention?":         ("−1.4 pp level drop",     "Low",      "⚠️"),
        "PSM — Does morning review improve retention?": ("No effect (−0.07 pp)",   "High",     "✅"),
        "Survival — Do harder cards lapse sooner?":     ("Yes — 7.5% vs 0.4%",     "High",     "✅"),
        "PSM v2 — Are morning hard reviews slower?":    ("+1,084 ms (p=0.005)",    "Moderate", "⚠️"),
        "A/B — Does reading aloud reduce lapses?":       ("Ongoing",                "TBD",      "🧪"),
    }
    rows = [
        {"Analysis": k, "Result": v[0], "Confidence": v[1], "Status": v[2]}
        for k, v in results.items()
    ]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.divider()
    st.subheader("Methodology")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
| Phase | Method |
|-------|--------|
| 1 | Exploratory Data Analysis |
| 2 | Interrupted Time Series |
| 3 | Propensity Score Matching (retention) |
| 4 | Survival Analysis (Cox PH + KM) |
| 5 | PSM v1 — response time (flawed) |
| 6 | PSM v2 — response time (corrected) |
| 7 | Randomised A/B experiment (active) |
""")
    with col_r:
        st.markdown("""
**Key insight:**
My retention is 99.2% — almost no variation in whether I remember cards.
The meaningful signal is in *which* cards are structurally hard.
Cards with ease factor < 2.5 (degraded below Anki's default) account
for nearly all lapses. These cards also take ~1 second longer to answer
in the morning vs evening.
""")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")

    revlog = load_revlog()

    if revlog.empty:
        if CLOUD_MODE:
            st.error("No data file found. Run `export_data.py` locally and push `data/` to GitHub.")
        else:
            st.error("Cannot read database — is Anki open? Close Anki and refresh.")
        st.stop()

    # ── Live metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews",   f"{len(revlog):,}")
    c2.metric("Unique Cards",    f"{revlog['cid'].nunique():,}")
    c3.metric("Retention Rate",  f"{revlog['retained'].mean()*100:.1f}%")
    c4.metric("Median Response", f"{revlog['time_s'].median():.1f}s")

    st.divider()

    # ── Daily reviews + retention ────────────────────────────────────────────
    st.subheader("Daily Review Volume & Retention")
    daily = (
        revlog.groupby("date")
        .agg(reviews=("id", "count"), retention=("retained", "mean"))
        .reset_index()
    )
    daily["date"]        = pd.to_datetime(daily["date"])
    daily["reviews_7d"]  = daily["reviews"].rolling(7, center=True, min_periods=1).mean()
    daily["ret_7d"]      = daily["retention"].rolling(7, center=True, min_periods=1).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Reviews per day", "Retention rate (7-day rolling)"),
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["reviews"],
                             fill="tozeroy", fillcolor="rgba(33,150,243,0.15)",
                             line=dict(color="rgba(33,150,243,0)"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["reviews_7d"],
                             line=dict(color=BLUE, width=2), name="7-day avg"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["ret_7d"] * 100,
                             line=dict(color=GREEN, width=2), name="Retention %"), row=2, col=1)
    fig.add_hline(y=daily["retention"].mean() * 100, line_dash="dash",
                  line_color="grey", row=2, col=1)
    fig.update_yaxes(title_text="Reviews", row=1)
    fig.update_yaxes(title_text="Retention (%)", row=2)
    fig.update_layout(height=450, margin=dict(t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')

    # ── Heatmap + maturity ───────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Retention by Hour & Day")
        heat = revlog.groupby(["dow", "hour"])["retained"].agg(["mean", "count"]).reset_index()
        heat_piv  = heat.pivot(index="dow", columns="hour", values="mean")
        count_piv = heat.pivot(index="dow", columns="hour", values="count")
        heat_piv  = heat_piv.where(count_piv >= 10)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig2 = px.imshow(
            heat_piv * 100,
            color_continuous_scale="RdYlGn",
            zmin=70, zmax=100,
            labels=dict(x="Hour", y="Day", color="Retention (%)"),
            y=day_labels,
            aspect="auto",
        )
        fig2.update_layout(height=260, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width='stretch')

    with col_r:
        st.subheader("Retention by Card Maturity")
        mature = revlog[revlog["ivl"] > 0].copy()
        bins   = [0, 1, 7, 21, 60, 180, 9999]
        labels = ["<1d", "1-7d", "7-21d", "21-60d", "60-180d", "180d+"]
        mature["maturity"] = pd.cut(mature["ivl"], bins=bins, labels=labels)
        mat_ret = (mature.groupby("maturity", observed=True)["retained"]
                         .agg(["mean", "count", "sem"]).reset_index())
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=mat_ret["maturity"].astype(str),
            y=mat_ret["mean"] * 100,
            error_y=dict(type="data", array=mat_ret["sem"] * 196),
            marker_color=BLUE,
            text=[f"n={int(n):,}" for n in mat_ret["count"]],
            textposition="outside",
        ))
        fig3.add_hline(y=90, line_dash="dash", line_color="grey",
                       annotation_text="Anki target 90%")
        fig3.update_yaxes(title="Retention (%)", range=[50, 105])
        fig3.update_xaxes(title="Interval at review")
        fig3.update_layout(height=260, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ITS
# ═══════════════════════════════════════════════════════════════════════════
with tab_its:
    st.header("Interrupted Time Series")
    st.markdown(
        "Segmented regression on daily retention rate. "
        "Intervention auto-detected at **September 13, 2025** — largest single-day volume shift."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Level change (β₂)",    "−0.014",  help="Immediate drop in retention at intervention")
    c2.metric("Slope change (β₃)",    "−0.000136/day", help="Post-intervention deceleration")
    c3.metric("R²",                   "0.265")

    st.info(
        "**Caution:** the intervention point was detected algorithmically. "
        "Any life event coinciding with Sep 2025 could explain the level change.",
        icon="⚠️"
    )

    if fig_path("fig4_its.png").exists():
        st.image(str(fig_path("fig4_its.png")), width='stretch')
    else:
        st.warning("Run `anki_causal_analysis.ipynb` to generate fig4_its.png")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — PSM
# ═══════════════════════════════════════════════════════════════════════════
with tab_psm:
    st.header("Propensity Score Matching")

    psm_tab1, psm_tab2 = st.tabs(["Binary Retention", "Response Time (v2)"])

    with psm_tab1:
        st.subheader("Morning vs Evening — Does time of day affect retention?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Morning reviews", "1,375")
        c2.metric("Evening reviews", "20,428")
        c3.metric("ATE",             "−0.07 pp",  delta_color="off")
        c4.metric("p-value",         "0.827")
        st.success(
            "**Null result — and that's a real finding.** "
            "With 99.2% retention, time of day has no detectable effect on whether I remember a card. "
            "Rosenbaum bounds confirm this holds even under strong unmeasured confounding.",
            icon="✅"
        )
        col_l, col_r = st.columns(2)
        with col_l:
            if fig_path("fig5_psm_overlap.png").exists():
                st.image(str(fig_path("fig5_psm_overlap.png")), width='stretch')
        with col_r:
            if fig_path("fig6_balance.png").exists():
                st.image(str(fig_path("fig6_balance.png")), width='stretch')

    with psm_tab2:
        st.subheader("Morning vs Evening — Response time on hard reviews (ef < 2.5)")
        st.markdown("Hard reviews defined as `revlog.factor < 2.5` **at review time** (corrected from v1).")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matched pairs",       "309")
        c2.metric("ATE (median)",        "+1,084 ms",  help="Morning is slower")
        c3.metric("95% CI",              "[+209, +1,924]")
        c4.metric("Mann-Whitney p",      "0.0045")

        st.warning(
            "**Residual concern:** `month_idx` SMD after matching = 0.117 (above 0.1 threshold). "
            "The v1 definition (using current card state) was confounded by my learning curve — "
            "82% of those reviews were from 2025 when I was a beginner.",
            icon="⚠️"
        )

        col_l, col_r = st.columns(2)
        with col_l:
            if fig_path("v2_fig2_distributions.png").exists():
                st.image(str(fig_path("v2_fig2_distributions.png")), width='stretch')
        with col_r:
            if fig_path("v2_fig3_comparison.png").exists():
                st.image(str(fig_path("v2_fig3_comparison.png")), width='stretch')

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — SURVIVAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_survival:
    st.header("Survival Analysis — Time to First Lapse")
    st.markdown(
        "Event: card receives its first lapse (`ease = Again`). "
        "Time: number of reviews until event (right-censored). "
        "Stratified by ease factor quartile."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cards analysed",  "9,002")
    c2.metric("First-lapse events", "237 (2.6%)")
    c3.metric("Cox HR (ease factor)", "2.06", help="Per 1-unit increase in ease factor")
    c4.metric("Cox p-value",     "≈ 0.000")

    st.divider()

    lapse_data = pd.DataFrame({
        "Quartile":    ["Q1 — Hardest", "Q2", "Q3", "Q4 — Easiest"],
        "Cards":       [2742, 2445, 2890, 925],
        "Lapse rate":  [7.47, 0.61, 0.45, 0.43],
    })
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.dataframe(
            lapse_data.style.format({"Lapse rate": "{:.2f}%"})
                            .highlight_max(subset=["Lapse rate"], color="#ffcccc"),
            width='stretch', hide_index=True
        )
        st.caption(
            "Q1 cards (bottom ease-factor quartile) lapse at nearly 10× the rate of all other groups. "
            "These are the cards the A/B experiment targets."
        )
    with col_r:
        fig_bar = px.bar(
            lapse_data, x="Quartile", y="Lapse rate",
            color="Lapse rate",
            color_continuous_scale=["#4CAF50", "#FF9800", "#F44336"],
            text=lapse_data["Lapse rate"].apply(lambda x: f"{x:.2f}%"),
            labels={"Lapse rate": "Lapse rate (%)"},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=300, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig_bar, width='stretch')

    col_l, col_r = st.columns(2)
    with col_l:
        if fig_path("fig7_km_survival.png").exists():
            st.image(str(fig_path("fig7_km_survival.png")), width='stretch')
    with col_r:
        if fig_path("fig8_cox_hr.png").exists():
            st.image(str(fig_path("fig8_cox_hr.png")), width='stretch')

    st.warning(
        "PH assumption mildly violated (Schoenfeld p = 0.025) — "
        "the hazard ratio is not constant over time.",
        icon="⚠️"
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — A/B EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab_ab:
    st.header("🧪 A/B Experiment — Read Aloud")

    # ── Progress banner ──────────────────────────────────────────────────────
    days_elapsed   = (date.today() - EXPERIMENT_START).days
    days_remaining = max(0, EXPERIMENT_DAYS - days_elapsed)
    pct            = min(100, days_elapsed / EXPERIMENT_DAYS)
    end_date       = EXPERIMENT_START + timedelta(days=EXPERIMENT_DAYS)

    st.progress(pct, text=f"Day {days_elapsed} of {EXPERIMENT_DAYS} — ends {end_date.strftime('%B %d, %Y')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start date",   EXPERIMENT_START.strftime("%b %d, %Y"))
    c2.metric("End date",     end_date.strftime("%b %d, %Y"))
    c3.metric("Days elapsed", days_elapsed)
    c4.metric("Days remaining", days_remaining)

    st.divider()

    # ── Design ───────────────────────────────────────────────────────────────
    with st.expander("Experiment Design", expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
**Research question:**
Does reading the sentence out loud during review reduce lapse rate and response time over 60 days?

**Groups (randomised, seed=42):**
- **Treatment** — 1,745 notes — marked with ★, read sentence aloud before rating
- **Control** — 1,745 notes — reviewed silently as normal

**Primary outcomes:**
1. Lapse rate (proportion of reviews where ease = Again)
2. Median response time (ms)
""")
        with col_r:
            st.markdown("""
**Causal assumptions:**
- **SUTVA** — no interference between cards
- **No anticipation** — control cards unaffected by knowing treatment exists
- **Consistency** — treatment is applied uniformly
- **Randomisation** — verified by seed=42 assignment

**Analysis at day 60:**
Two-sample Mann-Whitney (response time) and proportion test (lapse rate).
PSM used as robustness check only.
""")

    st.divider()

    # ── Live results ──────────────────────────────────────────────────────────
    st.subheader("Live Results")

    with st.spinner("Loading experiment data..."):
        treatment_rev, control_rev = load_experiment_data()

    if treatment_rev.empty and control_rev.empty and not CLOUD_MODE:
        parquet_missing = not (DATA_DIR / "experiment_revlog.parquet").exists()
        if not parquet_missing:
            st.error("Cannot read database — is Anki open? Close Anki and refresh.")

    no_data = treatment_rev.empty and control_rev.empty

    if no_data:
        st.info(
            f"No reviews recorded yet for experiment cards since {EXPERIMENT_START}. "
            "Results will appear here after your first review session.",
            icon="⏳"
        )
    else:
        # ── Top-line metrics ─────────────────────────────────────────────────
        def group_stats(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"reviews": 0, "lapse_rate": None, "median_rt": None, "cards": 0}
            trimmed = df[df["capped"] == 0]
            return {
                "reviews":    len(df),
                "cards":      df["cid"].nunique(),
                "lapse_rate": (1 - df["retained"].mean()) * 100,
                "median_rt":  trimmed["time"].median() if len(trimmed) else None,
            }

        t_stats = group_stats(treatment_rev)
        c_stats = group_stats(control_rev)

        col_t, col_c = st.columns(2)
        with col_t:
            st.markdown(f"### ★ Treatment (read aloud)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews",    f"{t_stats['reviews']:,}")
            m2.metric("Lapse rate", f"{t_stats['lapse_rate']:.2f}%" if t_stats['lapse_rate'] is not None else "—")
            m3.metric("Median RT",  f"{t_stats['median_rt']/1000:.1f}s" if t_stats['median_rt'] else "—")

        with col_c:
            st.markdown(f"### 🔵 Control (no change)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews",    f"{c_stats['reviews']:,}")
            m2.metric("Lapse rate", f"{c_stats['lapse_rate']:.2f}%" if c_stats['lapse_rate'] is not None else "—")
            m3.metric("Median RT",  f"{c_stats['median_rt']/1000:.1f}s" if c_stats['median_rt'] else "—")

        # ── Delta metrics ─────────────────────────────────────────────────────
        if t_stats["lapse_rate"] is not None and c_stats["lapse_rate"] is not None:
            st.divider()
            d1, d2 = st.columns(2)
            lapse_delta = t_stats["lapse_rate"] - c_stats["lapse_rate"]
            d1.metric(
                "Lapse rate difference (Treatment − Control)",
                f"{lapse_delta:+.2f} pp",
                delta=f"{lapse_delta:+.2f} pp",
                delta_color="inverse",
                help="Negative = treatment has fewer lapses (desired direction)"
            )
            if t_stats["median_rt"] and c_stats["median_rt"]:
                rt_delta = t_stats["median_rt"] - c_stats["median_rt"]
                d2.metric(
                    "Median RT difference (Treatment − Control)",
                    f"{rt_delta:+.0f} ms",
                    delta=f"{rt_delta:+.0f} ms",
                    delta_color="inverse",
                    help="Negative = treatment is faster (desired direction)"
                )

        # ── Daily lapse rate over time ────────────────────────────────────────
        st.divider()
        st.subheader("Daily Lapse Rate Over Experiment Period")

        all_exp = pd.concat([
            treatment_rev.assign(group="Treatment"),
            control_rev.assign(group="Control")
        ])
        daily_exp = (
            all_exp.groupby(["date", "group"])
            .agg(reviews=("id", "count"), lapses=("retained", lambda x: (x == 0).sum()))
            .reset_index()
        )
        daily_exp["lapse_rate"] = daily_exp["lapses"] / daily_exp["reviews"] * 100
        daily_exp["date"] = pd.to_datetime(daily_exp["date"])

        if len(daily_exp) > 0:
            fig_daily = px.line(
                daily_exp, x="date", y="lapse_rate", color="group",
                color_discrete_map={"Treatment": BLUE, "Control": ORANGE},
                markers=True,
                labels={"lapse_rate": "Lapse rate (%)", "date": "Date", "group": "Group"},
            )
            fig_daily.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig_daily, width='stretch')

        # ── Response time distributions ────────────────────────────────────────
        st.subheader("Response Time Distribution")

        trimmed_all = all_exp[all_exp["capped"] == 0].copy()
        if len(trimmed_all) > 10:
            fig_rt = go.Figure()
            for grp, color in [("Treatment", BLUE), ("Control", ORANGE)]:
                vals = trimmed_all[trimmed_all["group"] == grp]["time"] / 1000
                if len(vals) > 0:
                    fig_rt.add_trace(go.Histogram(
                        x=vals, name=grp, opacity=0.6,
                        marker_color=color, nbinsx=40,
                        histnorm="probability density",
                    ))
                    fig_rt.add_vline(
                        x=vals.median(), line_dash="dash", line_color=color,
                        annotation_text=f"{grp} median: {vals.median():.1f}s",
                        annotation_position="top"
                    )
            fig_rt.update_xaxes(title="Response time (seconds)", range=[0, 30])
            fig_rt.update_yaxes(title="Density")
            fig_rt.update_layout(
                barmode="overlay", height=350, margin=dict(t=40, b=20),
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig_rt, width='stretch')
        else:
            st.info("Not enough reviews yet to plot distributions.", icon="⏳")

        # ── Running significance ──────────────────────────────────────────────
        st.divider()
        st.subheader("Significance (updated daily)")
        st.caption(
            "⚠️ Treat p-values here as exploratory only. "
            "Peeking at significance during a running experiment inflates Type I error. "
            "The pre-registered analysis runs at day 60."
        )

        from scipy import stats as scipy_stats

        sig_cols = st.columns(2)
        with sig_cols[0]:
            st.markdown("**Lapse rate — proportion test**")
            if not treatment_rev.empty and not control_rev.empty:
                t_lapses  = int((treatment_rev["retained"] == 0).sum())
                c_lapses  = int((control_rev["retained"]   == 0).sum())
                t_total   = len(treatment_rev)
                c_total   = len(control_rev)
                if t_total > 0 and c_total > 0:
                    if t_lapses + c_lapses == 0:
                        st.info("No lapses recorded yet — test not applicable.", icon="⏳")
                    else:
                        from statsmodels.stats.proportion import proportions_ztest
                        z, p = proportions_ztest(
                            [t_lapses, c_lapses], [t_total, c_total]
                        )
                        st.metric("z-statistic", f"{z:.3f}")
                        st.metric("p-value", f"{p:.4f}")
                        if p < 0.05:
                            st.success("Significant at α=0.05", icon="✅")
                        else:
                            st.info("Not yet significant", icon="⏳")

        with sig_cols[1]:
            st.markdown("**Response time — Mann-Whitney U**")
            t_trim = treatment_rev[treatment_rev["capped"] == 0]["time"].values
            c_trim = control_rev[control_rev["capped"]   == 0]["time"].values
            if len(t_trim) >= 5 and len(c_trim) >= 5:
                u, p_mw = scipy_stats.mannwhitneyu(t_trim, c_trim, alternative="two-sided")
                st.metric("U-statistic", f"{u:.0f}")
                st.metric("p-value", f"{p_mw:.4f}")
                if p_mw < 0.05:
                    st.success("Significant at α=0.05", icon="✅")
                else:
                    st.info("Not yet significant", icon="⏳")
            else:
                st.info("Need more reviews for this test.", icon="⏳")

    st.divider()
    st.caption(
        "Data refreshes automatically every 5 minutes. "
        "Pre-registered analysis runs on day 60 using `psm_response_time_v2.ipynb` methodology."
    )
