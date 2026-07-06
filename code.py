"""
Integrated Steel Supply Chain — Single Source of Truth (Demo Dashboard)
=======================================================================
A teaching dashboard for a Tata Steel session on digital execution.

The point it lands: procurement (SAP), logistics (TMS), the plant (MES) and
sales (CRM) each hold *a* number. This governed layer stitches them into
*the* number — with an exceptions pane that turns a report into an
execution tool.

The data is synthetic but curated so a realistic exception is always present:
  -> Coking coal at Kalinganagar is below safety stock, with days-of-cover
     shorter than the ETA of the next inbound vessel (delayed by port
     congestion + cyclone). That gap is the "firefight" in the caselet.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config + light styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Steel SCM — Single Source of Truth",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      .source-badge {
        display:inline-block; padding:3px 10px; margin-right:6px;
        border-radius:12px; font-size:0.72rem; font-weight:600;
        background:#1f2a37; color:#cbd5e1; border:1px solid #334155;
      }
      .alert-card {
        border-radius:8px; padding:10px 14px; margin-bottom:8px;
        background:#111827; border-left:5px solid #64748b;
      }
      .alert-red    {border-left-color:#ef4444;}
      .alert-amber  {border-left-color:#f59e0b;}
      .alert-green  {border-left-color:#22c55e;}
      .alert-title  {font-weight:700; font-size:0.92rem; margin-bottom:2px;}
      .alert-body   {font-size:0.82rem; color:#94a3b8;}
      .tag {font-size:0.68rem; padding:1px 7px; border-radius:10px;
            background:#334155; color:#e2e8f0; margin-left:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Anchor "today" to run-time so inbound ETAs always read as near-future.
NOW = datetime.now()
TODAY = NOW.date()


def d(days):
    """Return a date `days` from today (used for ETAs)."""
    return (NOW + timedelta(days=days)).date()


# --------------------------------------------------------------------------- #
# Curated synthetic data (source system noted per table)
# --------------------------------------------------------------------------- #
@st.cache_data
def raw_materials():
    """Procurement (SAP) stock levels reconciled with plant (MES) consumption."""
    rows = [
        # plant, material, uom, stock, safety_stock, daily_consumption, source
        ("Jamshedpur",   "Coking Coal",   "t", 210_000, 168_000, 12_000),
        ("Jamshedpur",   "Iron Ore",      "t", 340_000, 220_000, 22_000),
        ("Jamshedpur",   "Limestone",     "t",  48_000,  40_000,  3_500),
        ("Jamshedpur",   "Ferro Silicon", "t",     900,   1_200,     60),
        ("Kalinganagar", "Coking Coal",   "t",  42_000, 114_000,  9_500),  # << exception
        ("Kalinganagar", "Iron Ore",      "t", 180_000, 130_000, 15_000),
        ("Kalinganagar", "Limestone",     "t",  30_000,  26_000,  2_600),
        ("Kalinganagar", "Dolomite",      "t",  14_000,  12_000,  1_100),
        ("Angul",        "Coking Coal",   "t",  95_000,  90_000,  8_000),  # watch
        ("Angul",        "Iron Ore",      "t", 120_000, 110_000, 12_500),
        ("Angul",        "Manganese Ore", "t",   5_000,   4_500,    380),
    ]
    df = pd.DataFrame(
        rows,
        columns=["plant", "material", "uom", "stock", "safety_stock", "daily_use"],
    )
    df["days_of_cover"] = (df["stock"] / df["daily_use"]).round(1)
    df["safety_days"] = (df["safety_stock"] / df["daily_use"]).round(1)
    df["below_safety"] = df["stock"] < df["safety_stock"]
    df["source"] = "SAP ↔ MES"
    return df


@st.cache_data
def inbound():
    """In-transit / port / rake visibility (TMS)."""
    rows = [
        # material, mode, id, qty, origin, destination, plant, status, eta, reason, position
        ("Coking Coal", "Vessel", "MV Cape Prosperity", 75_000, "Dalrymple Bay, AU",
         "Dhamra Port", "Kalinganagar", "Delayed", d(6),
         "Port congestion + cyclone diversion", "Anchored — Dhamra outer anchorage"),
        ("Coking Coal", "Vessel", "MV Ocean Vega", 80_000, "Hay Point, AU",
         "Paradip Port", "Angul", "On time", d(9),
         "—", "At sea — Bay of Bengal"),
        ("Iron Ore", "Rake", "RR-4471", 3_800, "Joda Mines",
         "Kalinganagar", "Kalinganagar", "On time", d(1),
         "—", "In transit — Keonjhar section"),
        ("Limestone", "Rake", "RR-2290", 3_600, "Birmitrapur",
         "Jamshedpur", "Jamshedpur", "On time", d(2),
         "—", "In transit"),
        ("Coking Coal", "Rake", "RR-5108", 3_700, "Dhamra Port",
         "Kalinganagar", "Kalinganagar", "Delayed", d(3),
         "Rake availability at port", "Awaiting placement — Dhamra sidings"),
        ("Ferro Silicon", "Road", "TRK-8821", 120, "Vendor — Raipur",
         "Jamshedpur", "Jamshedpur", "On time", d(1),
         "—", "In transit — NH-49"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["material", "mode", "shipment_id", "qty", "origin",
                 "destination", "plant", "status", "eta", "reason", "position"],
    )
    df["eta_days"] = (pd.to_datetime(df["eta"]) - pd.Timestamp(TODAY)).dt.days
    df["source"] = "TMS"
    return df


@st.cache_data
def finished_goods():
    """Finished-goods inventory & aging (MES stockyard ↔ CRM allocation)."""
    rows = [
        # grade, location, qty, age_days, threshold_days
        ("HR Coil",    "Jamshedpur SY",   9_400, 12, 30),
        ("HR Coil",    "Kolkata SY",      6_800, 34, 30),   # breach
        ("CR Coil",    "Kalinganagar SY", 5_100, 18, 30),
        ("CR Coil",    "Mumbai SY",       4_200, 47, 30),   # breach
        ("Galvanized", "Delhi SY",        1_500, 52, 45),   # breach
        ("Galvanized", "Jamshedpur SY",   2_300, 22, 45),
        ("TMT Rebar",  "Kalinganagar SY", 7_600,  9, 25),
        ("TMT Rebar",  "Delhi SY",        3_400, 15, 25),
        ("Wire Rod",   "Kolkata SY",      2_800, 20, 30),
    ]
    df = pd.DataFrame(
        rows, columns=["grade", "location", "qty", "age_days", "threshold_days"]
    )
    df["aging_breach"] = df["age_days"] > df["threshold_days"]
    df["source"] = "MES ↔ CRM"
    return df


@st.cache_data
def dispatch_otif():
    """Dispatch-vs-plan and OTIF by region (TMS ↔ CRM)."""
    rows = [
        # region, planned_t, actual_t, on_time_pct, in_full_pct, otif_pct
        ("North", 14_000, 13_400, 95, 96, 92),
        ("East",  20_000, 19_600, 97, 98, 95),
        ("West",  13_000,  9_800, 82, 88, 78),   # the gap
        ("South", 11_000, 10_400, 92, 94, 88),
    ]
    df = pd.DataFrame(
        rows,
        columns=["region", "planned_t", "actual_t",
                 "on_time_pct", "in_full_pct", "otif_pct"],
    )
    df["dispatch_pct"] = (df["actual_t"] / df["planned_t"] * 100).round(1)
    df["gap_t"] = df["planned_t"] - df["actual_t"]
    df["source"] = "TMS ↔ CRM"
    return df


@st.cache_data
def rake_turnaround():
    """Rake turnaround by loading point (TMS)."""
    rows = [
        ("Jamshedpur",    7.2, 8.0, 11),
        ("Kalinganagar",  9.8, 8.0,  9),
        ("Angul",         8.4, 8.0,  7),
        ("Kolkata SY",   14.2, 8.0,  4),   # breach
    ]
    df = pd.DataFrame(
        rows, columns=["loading_point", "turnaround_h", "target_h", "rakes"]
    )
    df["breach"] = df["turnaround_h"] > df["target_h"]
    df["source"] = "TMS"
    return df


rm = raw_materials()
inb = inbound()
fg = finished_goods()
otif = dispatch_otif()
rake = rake_turnaround()

# --------------------------------------------------------------------------- #
# Sidebar — the "single source of truth" framing + filters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🏭 Control Tower")
    st.caption(f"Data as of **{TODAY:%d %b %Y}**  ·  simulated near-real-time feed")

    st.markdown("**Unified from four systems**")
    st.markdown(
        '<span class="source-badge">SAP · Procurement</span>'
        '<span class="source-badge">TMS · Logistics</span><br>'
        '<span class="source-badge">MES · Plant</span>'
        '<span class="source-badge">CRM · Sales</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    plants = sorted(rm["plant"].unique())
    sel_plants = st.multiselect("Plant", plants, default=plants)
    only_exceptions = st.checkbox("Show only exceptions", value=False)
    st.divider()

    with st.expander("📋 Session task — answer by navigating", expanded=True):
        st.markdown(
            "1. **Which raw material is below safety stock, at which plant — "
            "and when does the next inbound arrive?**\n\n"
            "2. **Where is finished-goods inventory aging past threshold, "
            "by grade & location?**\n\n"
            "3. **What is today's OTIF / dispatch-vs-plan — and where's the gap?**"
        )
    st.caption("Facilitator: the answer key is in README.md")

if not sel_plants:
    sel_plants = plants  # never show an empty board

rm_f = rm[rm["plant"].isin(sel_plants)]
inb_f = inb[inb["plant"].isin(sel_plants)]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Integrated Steel Supply Chain — Single Source of Truth")
st.caption(
    "Everyone had *a* number. This is *the* number. "
    "Scroll the alerts, then use the tabs to work the three questions."
)

# --------------------------------------------------------------------------- #
# Build alerts from the data (this is what makes it an execution tool)
# --------------------------------------------------------------------------- #
def project_stockout(plant, material, stock, daily_use, horizon=30):
    """Day-by-day projection: burn consumption, add inbound on its ETA day.
    Returns (stockout_day or None, biggest_inbound(shipment,qty,eta) or None)."""
    events = (
        inb[(inb["material"] == material) & (inb["plant"] == plant)]
        .groupby("eta_days")["qty"].sum().to_dict()
    )
    s = stock
    stockout_day = None
    for day in range(1, horizon + 1):
        s -= daily_use
        s += events.get(day, 0)
        if s < 0 and stockout_day is None:
            stockout_day = day
    relief = (
        inb[(inb["material"] == material) & (inb["plant"] == plant)]
        .sort_values("qty", ascending=False)
    )
    biggest = None if relief.empty else (
        relief.iloc[0]["shipment_id"], int(relief.iloc[0]["qty"]),
        int(relief.iloc[0]["eta_days"]),
    )
    return stockout_day, biggest


def build_alerts():
    alerts = []  # (severity, title, body, tag)

    # Safety-stock breaches, cross-checked against a real inbound projection
    breaches = rm_f[rm_f["below_safety"]]
    for _, r in breaches.iterrows():
        so_day, biggest = project_stockout(
            r["plant"], r["material"], r["stock"], r["daily_use"]
        )
        if so_day is not None:
            sev = "red"
            if biggest:
                ship, qty, eta = biggest
                body = (
                    f"{r['days_of_cover']:.1f}d cover. Day-by-day projection: "
                    f"**stock-out ~day {so_day}**; largest relief ({ship}, "
                    f"{qty:,} t) not until day {eta}. Production risk — small "
                    f"in-transit rakes don't bridge it."
                )
            else:
                body = (
                    f"{r['days_of_cover']:.1f}d cover, below safety "
                    f"({r['safety_days']:.1f}d). Projected stock-out day "
                    f"{so_day}. **No inbound scheduled.**"
                )
        else:
            sev = "amber"
            body = (
                f"{r['days_of_cover']:.1f}d cover, below the "
                f"{r['safety_days']:.1f}d safety line, but scheduled inbound "
                f"keeps stock positive across the horizon. No buffer — watch it."
            )
        alerts.append(
            (sev, f"Safety-stock breach — {r['material']} @ {r['plant']}",
             body, "SAP↔MES↔TMS")
        )

    # Delayed inbound
    for _, r in inb_f[inb_f["status"] == "Delayed"].iterrows():
        alerts.append(
            ("amber", f"Inbound delayed — {r['shipment_id']} ({r['material']})",
             f"{r['qty']:,} t for {r['plant']} · {r['reason']} · "
             f"revised ETA {r['eta']:%d %b}. {r['position']}.", "TMS")
        )

    # FG aging breaches
    for _, r in fg[fg["aging_breach"]].iterrows():
        alerts.append(
            ("amber", f"FG aging — {r['grade']} @ {r['location']}",
             f"{r['qty']:,} t aged {r['age_days']}d vs {r['threshold_days']}d "
             f"threshold. Working-capital & quality risk.", "MES↔CRM")
        )

    # OTIF gap
    worst = otif.sort_values("otif_pct").iloc[0]
    if worst["otif_pct"] < 90:
        alerts.append(
            ("amber", f"OTIF gap — {worst['region']} region",
             f"OTIF {worst['otif_pct']}% (in-full {worst['in_full_pct']}% "
             f"is the drag) · dispatch {worst['dispatch_pct']}% of plan · "
             f"short {worst['gap_t']:,} t today.", "TMS↔CRM")
        )

    # Rake turnaround breach
    for _, r in rake[rake["breach"]].iterrows():
        alerts.append(
            ("amber", f"Rake turnaround — {r['loading_point']}",
             f"{r['turnaround_h']}h vs {r['target_h']}h target across "
             f"{r['rakes']} rakes.", "TMS")
        )

    order = {"red": 0, "amber": 1, "green": 2}
    return sorted(alerts, key=lambda a: order[a[0]])


alerts = build_alerts()
red_n = sum(1 for a in alerts if a[0] == "red")
amber_n = sum(1 for a in alerts if a[0] == "amber")

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
def cover(plant, material):
    row = rm[(rm["plant"] == plant) & (rm["material"] == material)]
    return float(row["days_of_cover"].iloc[0]) if not row.empty else np.nan


k1, k2, k3, k4, k5, k6 = st.columns(6)
cc = cover("Kalinganagar", "Coking Coal")
k1.metric("Coking coal cover · Kalinganagar", f"{cc:.1f} d",
          "below 12d safety", delta_color="inverse")
k2.metric("Iron ore cover · Kalinganagar",
          f"{cover('Kalinganagar', 'Iron Ore'):.1f} d", "healthy",
          delta_color="off")
overall_otif = round((otif["actual_t"] * otif["otif_pct"]).sum()
                     / otif["actual_t"].sum(), 1)
k3.metric("OTIF today", f"{overall_otif:.1f}%", "vs 95% target",
          delta_color="inverse")
k4.metric("Safety-stock breaches", int(rm["below_safety"].sum()),
          "action needed", delta_color="inverse")
k5.metric("FG aging breaches", int(fg["aging_breach"].sum()),
          "review", delta_color="inverse")
k6.metric("Avg rake turnaround",
          f"{rake['turnaround_h'].mean():.1f} h", "target 8.0 h",
          delta_color="off")

st.divider()

# --------------------------------------------------------------------------- #
# Exceptions & Alerts pane
# --------------------------------------------------------------------------- #
st.subheader(f"🚨 Exceptions & Alerts  ·  {red_n} critical · {amber_n} watch")
st.caption("Ranked by severity. This pane is what turns a report into an execution tool.")

acol1, acol2 = st.columns(2)
for i, (sev, title, body, tag) in enumerate(alerts):
    target = acol1 if i % 2 == 0 else acol2
    target.markdown(
        f'<div class="alert-card alert-{sev}">'
        f'<div class="alert-title">{title}<span class="tag">{tag}</span></div>'
        f'<div class="alert-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------------------------------- #
# Tabs — one per scenario question, plus logistics visibility
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4 = st.tabs(
    ["① Raw material & days of cover",
     "② Finished-goods aging",
     "③ OTIF / dispatch-vs-plan",
     "④ In-transit · port · rake"]
)

# ---- Tab 1 --------------------------------------------------------------- #
with tab1:
    st.markdown("**Q1 — Which raw material is below safety stock, and what's the inbound ETA?**")

    view = rm_f.copy()
    if only_exceptions:
        view = view[view["below_safety"]]

    fig = px.bar(
        view.sort_values("days_of_cover"),
        x="days_of_cover", y="material", color="plant", orientation="h",
        text="days_of_cover",
        labels={"days_of_cover": "Days of cover", "material": ""},
        title="Days of cover by material & plant (bar) vs safety threshold",
        height=420,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    show = view[["plant", "material", "stock", "safety_stock", "daily_use",
                 "days_of_cover", "safety_days", "below_safety"]].rename(
        columns={"stock": "stock (t)", "safety_stock": "safety (t)",
                 "daily_use": "daily use (t)", "days_of_cover": "cover (d)",
                 "safety_days": "safety (d)", "below_safety": "breach"}
    )
    st.dataframe(
        show.style.apply(
            lambda r: ["background-color:#3b1113" if r["breach"] else "" for _ in r],
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Matching inbound (from TMS):**")
    st.dataframe(
        inb_f[["material", "shipment_id", "mode", "qty", "origin",
               "plant", "status", "eta", "eta_days", "reason", "position"]],
        use_container_width=True, hide_index=True,
    )

# ---- Tab 2 --------------------------------------------------------------- #
with tab2:
    st.markdown("**Q2 — Where is finished-goods inventory aging past threshold, by grade/location?**")

    view = fg.copy()
    if only_exceptions:
        view = view[view["aging_breach"]]

    fig = px.scatter(
        view, x="age_days", y="grade", size="qty", color="location",
        labels={"age_days": "Avg age (days)", "grade": ""},
        title="FG aging — bubble size = tonnes; dots right of the line breach",
        height=420,
    )
    fig.add_vline(x=30, line_dash="dash", line_color="#f59e0b",
                  annotation_text="typical 30d threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view[["grade", "location", "qty", "age_days", "threshold_days",
              "aging_breach"]].rename(
            columns={"qty": "qty (t)", "age_days": "age (d)",
                     "threshold_days": "threshold (d)", "aging_breach": "breach"}
        ).style.apply(
            lambda r: ["background-color:#3a2a10" if r["breach"] else "" for _ in r],
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

# ---- Tab 3 --------------------------------------------------------------- #
with tab3:
    st.markdown("**Q3 — What's today's OTIF / dispatch-vs-plan, and where's the gap?**")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=otif["region"], y=otif["planned_t"], name="Planned",
                    marker_color="#334155")
        fig.add_bar(x=otif["region"], y=otif["actual_t"], name="Actual",
                    marker_color="#22c55e")
        fig.update_layout(barmode="group", title="Dispatch vs plan (t) by region",
                          height=380, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(otif, x="region", y="otif_pct", text="otif_pct",
                     title="OTIF % by region", height=380,
                     labels={"otif_pct": "OTIF %", "region": ""})
        fig.add_hline(y=95, line_dash="dash", line_color="#f59e0b",
                      annotation_text="95% target")
        fig.update_traces(textposition="outside", marker_color="#38bdf8")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        otif[["region", "planned_t", "actual_t", "gap_t", "dispatch_pct",
              "on_time_pct", "in_full_pct", "otif_pct"]].rename(
            columns={"planned_t": "planned (t)", "actual_t": "actual (t)",
                     "gap_t": "gap (t)", "dispatch_pct": "dispatch %",
                     "on_time_pct": "on-time %", "in_full_pct": "in-full %",
                     "otif_pct": "OTIF %"}
        ),
        use_container_width=True, hide_index=True,
    )
    st.info(
        "Read the split: West OTIF is dragged by **in-full**, not on-time — "
        "trucks left on schedule but short, pointing at an availability/allocation "
        "problem, not a transport one."
    )

# ---- Tab 4 --------------------------------------------------------------- #
with tab4:
    st.markdown("**In-transit, port & rake visibility (TMS)** — the physical picture behind the numbers.")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Inbound pipeline**")
        st.dataframe(
            inb[["material", "shipment_id", "mode", "qty", "origin",
                 "destination", "plant", "status", "eta", "eta_days", "position"]],
            use_container_width=True, hide_index=True,
        )
    with c2:
        fig = px.bar(
            rake.sort_values("turnaround_h"),
            x="turnaround_h", y="loading_point", orientation="h",
            text="turnaround_h", labels={"turnaround_h": "Hours", "loading_point": ""},
            title="Rake turnaround vs 8h target", height=340,
        )
        fig.add_vline(x=8, line_dash="dash", line_color="#f59e0b")
        fig.update_traces(textposition="outside", marker_color="#a78bfa")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Synthetic data for training. The coking-coal-at-Kalinganagar exception is "
    "seeded deliberately to mirror the caselet. Do not load confidential data "
    "into a shared demo."
)
