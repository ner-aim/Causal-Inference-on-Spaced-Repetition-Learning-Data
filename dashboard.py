"""
dashboard.py — Anki Learning Analytics Dashboard
Run with:  python -m streamlit run dashboard.py
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

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH          = Path(r"C:\Users\sid99\AppData\Roaming\Anki2\Pottapatri\collection.anki2")
DATA_DIR         = Path(__file__).parent / "data"
EXPERIMENT_START = date(2026, 4, 24)
EXPERIMENT_DAYS  = 60
FIGURES_DIR      = Path(__file__).parent
ANKI_CAP_MS      = 60_000

CLOUD_MODE = not DB_PATH.exists()

st.set_page_config(
    page_title="Anki Analytics",
    page_icon="火",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE   = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
RED    = "#F44336"

# ── DB helpers ───────────────────────────────────────────────────────────────

def get_connection():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
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
        df = pd.read_sql("SELECT id, cid, ease, ivl, factor, time, type FROM revlog", con)
        con.close()
    except Exception:
        return pd.DataFrame()
    df["ts"]          = pd.to_datetime(df["id"], unit="ms")
    df["date"]        = df["ts"].dt.date
    df["hour"]        = df["ts"].dt.hour
    df["dow"]         = df["ts"].dt.dayofweek
    df["retained"]    = (df["ease"] > 2).astype(int)
    df["ease_factor"] = df["factor"] / 1000
    df["time_s"]      = df["time"] / 1000
    df["capped"]      = (df["time"] == ANKI_CAP_MS).astype(int)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_experiment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    revlog["retained"]    = (revlog["ease"] > 2).astype(int)
    revlog["ease_factor"] = revlog["factor"] / 1000
    revlog["capped"]      = (revlog["time"] == ANKI_CAP_MS).astype(int)
    revlog = revlog.rename(columns={"grp": "group"})

    return revlog[revlog["group"] == "treatment"], revlog[revlog["group"] == "control"]


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("火 Anki Analytics")
    st.caption("Human Japanese Intermediate Shared")
    st.divider()

    days_elapsed   = (date.today() - EXPERIMENT_START).days
    days_remaining = max(0, EXPERIMENT_DAYS - days_elapsed)
    pct            = min(1.0, days_elapsed / EXPERIMENT_DAYS)

    st.markdown("**A/B Experiment**")
    st.progress(pct, text=f"Day {days_elapsed} of {EXPERIMENT_DAYS}")
    c1, c2 = st.columns(2)
    c1.metric("Elapsed",   f"{days_elapsed}d")
    c2.metric("Remaining", f"{days_remaining}d")
    st.caption(f"Started {EXPERIMENT_START.strftime('%b %d, %Y')}")
    st.divider()

    if CLOUD_MODE:
        meta_p = DATA_DIR / "meta.parquet"
        if meta_p.exists():
            meta = pd.read_parquet(meta_p)
            exported_at = pd.to_datetime(meta["exported_at"].iloc[0])
            st.caption(f"Last export: {exported_at.strftime('%b %d %H:%M')}")
        st.caption("Cloud mode — run `export_data.py` to update")
    else:
        st.caption("Live DB · refreshes every 5 min")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ab, tab_findings, tab_data = st.tabs([
    "🧪 A/B Experiment",
    "📊 Findings",
    "🔍 Raw Data",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — A/B EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════
with tab_ab:
    end_date = EXPERIMENT_START + timedelta(days=EXPERIMENT_DAYS)
    st.progress(pct, text=f"Day {days_elapsed} of {EXPERIMENT_DAYS} — ends {end_date.strftime('%B %d, %Y')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start",     EXPERIMENT_START.strftime("%b %d, %Y"))
    c2.metric("End",       end_date.strftime("%b %d, %Y"))
    c3.metric("Treatment", "1,745 cards — read aloud (★)")
    c4.metric("Control",   "1,745 cards — silent review")

    st.divider()

    with st.spinner("Loading experiment data..."):
        treatment_rev, control_rev = load_experiment_data()

    if treatment_rev.empty and control_rev.empty and not CLOUD_MODE:
        if (DATA_DIR / "experiment_revlog.parquet").exists():
            st.error("Cannot read database — is Anki open? Close Anki and refresh.")

    no_data = treatment_rev.empty and control_rev.empty

    if no_data:
        st.info(
            f"No reviews recorded yet since {EXPERIMENT_START}. "
            "Results will appear after your first review session.",
            icon="⏳"
        )
    else:
        # ── Top-line metrics ──────────────────────────────────────────────────
        def group_stats(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"reviews": 0, "lapse_rate": None, "median_rt": None}
            trimmed = df[df["capped"] == 0]
            return {
                "reviews":    len(df),
                "lapse_rate": (1 - df["retained"].mean()) * 100,
                "median_rt":  trimmed["time"].median() if len(trimmed) else None,
            }

        t = group_stats(treatment_rev)
        c = group_stats(control_rev)

        col_t, col_c = st.columns(2)
        with col_t:
            st.markdown("### ★ Treatment — read aloud")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews",    f"{t['reviews']:,}")
            m2.metric("Lapse rate", f"{t['lapse_rate']:.2f}%" if t['lapse_rate'] is not None else "—")
            m3.metric("Median RT",  f"{t['median_rt']/1000:.1f}s" if t['median_rt'] else "—")

        with col_c:
            st.markdown("### 🔵 Control — silent")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews",    f"{c['reviews']:,}")
            m2.metric("Lapse rate", f"{c['lapse_rate']:.2f}%" if c['lapse_rate'] is not None else "—")
            m3.metric("Median RT",  f"{c['median_rt']/1000:.1f}s" if c['median_rt'] else "—")

        # ── Deltas ────────────────────────────────────────────────────────────
        if t["lapse_rate"] is not None and c["lapse_rate"] is not None:
            st.divider()
            d1, d2 = st.columns(2)
            lapse_delta = t["lapse_rate"] - c["lapse_rate"]
            d1.metric(
                "Lapse rate Δ (Treatment − Control)",
                f"{lapse_delta:+.2f} pp",
                delta=f"{lapse_delta:+.2f} pp",
                delta_color="inverse",
            )
            if t["median_rt"] and c["median_rt"]:
                rt_delta = t["median_rt"] - c["median_rt"]
                d2.metric(
                    "Median RT Δ (Treatment − Control)",
                    f"{rt_delta:+.0f} ms",
                    delta=f"{rt_delta:+.0f} ms",
                    delta_color="inverse",
                )

        # ── Daily lapse rate ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Daily Lapse Rate")

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
        daily_exp["date"]       = pd.to_datetime(daily_exp["date"])

        fig_daily = px.line(
            daily_exp, x="date", y="lapse_rate", color="group",
            color_discrete_map={"Treatment": BLUE, "Control": ORANGE},
            markers=True,
            labels={"lapse_rate": "Lapse rate (%)", "date": "Date", "group": "Group"},
        )
        fig_daily.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig_daily, width='stretch')

        # ── Response time distribution ────────────────────────────────────────
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
                        annotation_text=f"{grp}: {vals.median():.1f}s",
                        annotation_position="top",
                    )
            fig_rt.update_xaxes(title="Response time (s)", range=[0, 30])
            fig_rt.update_yaxes(title="Density")
            fig_rt.update_layout(
                barmode="overlay", height=320,
                margin=dict(t=40, b=20), legend=dict(orientation="h")
            )
            st.plotly_chart(fig_rt, width='stretch')
        else:
            st.info("Not enough reviews yet to plot distributions.", icon="⏳")

        # ── Significance ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Significance")
        st.caption("⚠️ Exploratory only — pre-registered analysis runs at day 60.")

        from scipy import stats as scipy_stats
        sig_cols = st.columns(2)

        with sig_cols[0]:
            st.markdown("**Lapse rate — proportion test**")
            t_lapses = int((treatment_rev["retained"] == 0).sum())
            c_lapses = int((control_rev["retained"]   == 0).sum())
            if t_lapses + c_lapses == 0:
                st.info("No lapses yet.", icon="⏳")
            else:
                from statsmodels.stats.proportion import proportions_ztest
                z, p = proportions_ztest(
                    [t_lapses, c_lapses], [len(treatment_rev), len(control_rev)]
                )
                st.metric("p-value", f"{p:.4f}")
                if p < 0.05:
                    st.success("Significant at α=0.05", icon="✅")
                else:
                    st.info("Not yet significant", icon="⏳")

        with sig_cols[1]:
            st.markdown("**Response time — Mann-Whitney**")
            t_trim = treatment_rev[treatment_rev["capped"] == 0]["time"].values
            c_trim = control_rev[control_rev["capped"]   == 0]["time"].values
            if len(t_trim) >= 5 and len(c_trim) >= 5:
                _, p_mw = scipy_stats.mannwhitneyu(t_trim, c_trim, alternative="two-sided")
                st.metric("p-value", f"{p_mw:.4f}")
                if p_mw < 0.05:
                    st.success("Significant at α=0.05", icon="✅")
                else:
                    st.info("Not yet significant", icon="⏳")
            else:
                st.info("Need more reviews.", icon="⏳")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — FINDINGS
# ═════════════════════════════════════════════════════════════════════════════
with tab_findings:
    st.header("Key Findings")
    st.caption("52,119 reviews · 449 days · Jan 2025 – Apr 2026")

    # ── Headline numbers ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retention rate",    "82.8%", help="Good + Easy only; Hard counted as lapse")
    c2.metric("Avg reviews / day", "116")
    c3.metric("Unique cards",      "9,037")
    c4.metric("Median response",   "4.2s")

    st.divider()

    # ── Finding cards ─────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("#### 📉 Hard cards fail 20× more")
        st.markdown(
            "Bottom ease-factor quartile lapse at **7.5%** vs **0.4%** for all other cards "
            "(Cox HR = 2.06, p < 0.001). 30% of cards account for 86% of all failures."
        )
        if (FIGURES_DIR / "fig7_km_survival.png").exists():
            st.image(str(FIGURES_DIR / "fig7_km_survival.png"))

    with f2:
        st.markdown("#### ⏱ Morning hard reviews are slower")
        st.markdown(
            "After propensity score matching, hard cards reviewed in the morning take "
            "**+1,084 ms** longer than the same cards reviewed in the evening "
            "(p = 0.005, 309 matched pairs)."
        )
        if (FIGURES_DIR / "v2_fig3_comparison.png").exists():
            st.image(str(FIGURES_DIR / "v2_fig3_comparison.png"))

    with f3:
        st.markdown("#### ✅ Time of day doesn't affect retention")
        st.markdown(
            "Morning vs evening reviews show **no difference in whether a card is remembered** "
            "(ATE = −0.07 pp, p = 0.827). With 99.2% retention there is simply no signal to find."
        )
        if (FIGURES_DIR / "fig6_balance.png").exists():
            st.image(str(FIGURES_DIR / "fig6_balance.png"))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RAW DATA
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Raw Data")

    revlog = load_revlog()

    if revlog.empty:
        if CLOUD_MODE:
            st.error("No data file found. Run `export_data.py` locally and push `data/` to GitHub.")
        else:
            st.error("Cannot read database — is Anki open? Close Anki and refresh.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews",   f"{len(revlog):,}")
    c2.metric("Unique Cards",    f"{revlog['cid'].nunique():,}")
    c3.metric("Retention Rate",  f"{revlog['retained'].mean()*100:.1f}%")
    c4.metric("Median Response", f"{revlog['time_s'].median():.1f}s")

    st.divider()

    # ── Daily reviews + retention ─────────────────────────────────────────────
    daily = (
        revlog.groupby("date")
        .agg(reviews=("id", "count"), retention=("retained", "mean"))
        .reset_index()
    )
    daily["date"]       = pd.to_datetime(daily["date"])
    daily["reviews_7d"] = daily["reviews"].rolling(7, center=True, min_periods=1).mean()
    daily["ret_7d"]     = daily["retention"].rolling(7, center=True, min_periods=1).mean()

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
    fig.add_hline(y=daily["retention"].mean() * 100, line_dash="dash", line_color="grey", row=2, col=1)
    fig.update_yaxes(title_text="Reviews", row=1)
    fig.update_yaxes(title_text="Retention (%)", row=2)
    fig.update_layout(height=440, margin=dict(t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')

    # ── Heatmap + maturity ────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Retention by Hour & Day")
        heat      = revlog.groupby(["dow", "hour"])["retained"].agg(["mean", "count"]).reset_index()
        heat_piv  = heat.pivot(index="dow", columns="hour", values="mean")
        count_piv = heat.pivot(index="dow", columns="hour", values="count")
        heat_piv  = heat_piv.where(count_piv >= 10)
        fig2 = px.imshow(
            heat_piv * 100,
            color_continuous_scale="RdYlGn", zmin=70, zmax=100,
            labels=dict(x="Hour", y="Day", color="Retention (%)"),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            aspect="auto",
        )
        fig2.update_layout(height=260, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width='stretch')

    with col_r:
        st.subheader("Retention by Card Maturity")
        mature = revlog[revlog["ivl"] > 0].copy()
        mature["maturity"] = pd.cut(
            mature["ivl"],
            bins=[0, 1, 7, 21, 60, 180, 9999],
            labels=["<1d", "1-7d", "7-21d", "21-60d", "60-180d", "180d+"]
        )
        mat_ret = (mature.groupby("maturity", observed=True)["retained"]
                         .agg(["mean", "count", "sem"]).reset_index())
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=mat_ret["maturity"].astype(str), y=mat_ret["mean"] * 100,
            error_y=dict(type="data", array=mat_ret["sem"] * 196),
            marker_color=BLUE,
            text=[f"n={int(n):,}" for n in mat_ret["count"]],
            textposition="outside",
        ))
        fig3.add_hline(y=90, line_dash="dash", line_color="grey", annotation_text="Anki target 90%")
        fig3.update_yaxes(title="Retention (%)", range=[50, 105])
        fig3.update_xaxes(title="Interval at review")
        fig3.update_layout(height=260, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, width='stretch')
