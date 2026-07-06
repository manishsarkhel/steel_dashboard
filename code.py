"""
Integrated Steel Digital Control Tower — Executive Training Dashboard
====================================================================

A realistic synthetic Streamlit dashboard for teaching digital supply chain
control towers to participants from the steel industry.

Core teaching point:
Procurement, logistics, production, inventory, sales, finance and ESG systems
all hold partial truths. A digital control tower integrates them into a single
operational picture, highlights exceptions, estimates business impact, and
supports scenario-based decision making.

Run:
    pip install streamlit pandas numpy plotly
    streamlit run steel_control_tower_app.py

Data:
    Synthetic but curated for training. Do not load confidential company data
    into this shared demo.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Steel Digital Control Tower",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

NOW = datetime.now()
TODAY = NOW.date()


def eta(days: int):
    return (NOW + timedelta(days=days)).date()


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 14px 16px;
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            min-height: 118px;
        }
        .kpi-label {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .kpi-value {
            color: #f8fafc;
            font-size: 1.72rem;
            font-weight: 800;
            margin-top: 4px;
        }
        .kpi-delta-good {color: #22c55e; font-size: 0.82rem; font-weight: 700;}
        .kpi-delta-bad {color: #ef4444; font-size: 0.82rem; font-weight: 700;}
        .kpi-delta-watch {color: #f59e0b; font-size: 0.82rem; font-weight: 700;}
        .risk-card {
            border-radius: 12px;
            padding: 13px 15px;
            background: #111827;
            border: 1px solid #334155;
            margin-bottom: 10px;
        }
        .risk-red {border-left: 6px solid #ef4444;}
        .risk-amber {border-left: 6px solid #f59e0b;}
        .risk-green {border-left: 6px solid #22c55e;}
        .risk-title {font-weight: 800; font-size: 0.98rem; color: #f8fafc;}
        .risk-body {font-size: 0.86rem; color: #cbd5e1; margin-top: 4px;}
        .risk-meta {font-size: 0.75rem; color: #94a3b8; margin-top: 6px;}
        .pill {
            display:inline-block;
            padding: 3px 9px;
            border-radius: 999px;
            background:#1e293b;
            color:#cbd5e1;
            border:1px solid #334155;
            font-size:0.72rem;
            margin-right:5px;
            margin-top:5px;
        }
        .section-note {
            color: #94a3b8;
            font-size: 0.86rem;
        }
        .action-card {
            border-radius: 12px;
            padding: 14px 16px;
            background:#0f172a;
            border:1px solid #334155;
            margin-bottom:10px;
        }
        .action-title {font-weight:800; color:#f8fafc;}
        .action-body {font-size:0.86rem; color:#cbd5e1; margin-top:4px;}
        .small-muted {font-size:0.78rem; color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Synthetic data layer
# -----------------------------------------------------------------------------
@st.cache_data
def raw_materials() -> pd.DataFrame:
    rows = [
        # plant, material, category, stock, safety, daily_use, unit_cost_lakh_per_t, supplier_risk
        ("Jamshedpur", "Coking Coal", "Fuel", 210000, 168000, 12000, 0.18, 0.36),
        ("Jamshedpur", "Iron Ore", "Ore", 340000, 220000, 22000, 0.055, 0.22),
        ("Jamshedpur", "Limestone", "Flux", 48000, 40000, 3500, 0.015, 0.18),
        ("Jamshedpur", "Ferro Silicon", "Alloy", 900, 1200, 60, 1.10, 0.52),
        ("Kalinganagar", "Coking Coal", "Fuel", 42000, 114000, 9500, 0.18, 0.81),
        ("Kalinganagar", "Iron Ore", "Ore", 180000, 130000, 15000, 0.055, 0.25),
        ("Kalinganagar", "Limestone", "Flux", 30000, 26000, 2600, 0.015, 0.21),
        ("Kalinganagar", "Dolomite", "Flux", 14000, 12000, 1100, 0.020, 0.20),
        ("Angul", "Coking Coal", "Fuel", 95000, 90000, 8000, 0.18, 0.58),
        ("Angul", "Iron Ore", "Ore", 120000, 110000, 12500, 0.055, 0.29),
        ("Angul", "Manganese Ore", "Alloy", 5000, 4500, 380, 0.42, 0.41),
        ("Dolvi", "Coking Coal", "Fuel", 76000, 82000, 7100, 0.19, 0.62),
        ("Dolvi", "Iron Ore", "Ore", 98000, 92000, 9300, 0.060, 0.31),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "plant",
            "material",
            "category",
            "stock_t",
            "safety_stock_t",
            "daily_use_t",
            "unit_cost_lakh_per_t",
            "supplier_risk",
        ],
    )
    df["days_cover"] = (df["stock_t"] / df["daily_use_t"]).round(1)
    df["safety_days"] = (df["safety_stock_t"] / df["daily_use_t"]).round(1)
    df["inventory_value_cr"] = (df["stock_t"] * df["unit_cost_lakh_per_t"] / 100).round(1)
    df["below_safety"] = df["stock_t"] < df["safety_stock_t"]
    df["source"] = "SAP MM ↔ MES"
    return df


@st.cache_data
def inbound_shipments() -> pd.DataFrame:
    rows = [
        ("Coking Coal", "Vessel", "MV Cape Prosperity", 75000, "Dalrymple Bay", "Dhamra Port", "Kalinganagar", "Delayed", eta(6), "Cyclone diversion + port congestion", "Anchored at outer anchorage", 0.85),
        ("Coking Coal", "Rake", "RR-5108", 3700, "Dhamra Port", "Kalinganagar", "Kalinganagar", "Delayed", eta(3), "Rake placement delay", "Awaiting placement", 0.72),
        ("Coking Coal", "Vessel", "MV Ocean Vega", 80000, "Hay Point", "Paradip Port", "Angul", "On time", eta(9), "—", "Bay of Bengal", 0.25),
        ("Iron Ore", "Rake", "RR-4471", 3800, "Joda Mines", "Kalinganagar", "Kalinganagar", "On time", eta(1), "—", "Keonjhar section", 0.18),
        ("Limestone", "Rake", "RR-2290", 3600, "Birmitrapur", "Jamshedpur", "Jamshedpur", "On time", eta(2), "—", "In transit", 0.15),
        ("Ferro Silicon", "Road", "TRK-8821", 120, "Vendor — Raipur", "Jamshedpur", "Jamshedpur", "On time", eta(1), "—", "NH-49", 0.20),
        ("Iron Ore", "Rake", "RR-1192", 4100, "Noamundi", "Jamshedpur", "Jamshedpur", "On time", eta(1), "—", "Tatanagar approach", 0.12),
        ("Coking Coal", "Vessel", "MV Black Diamond", 69000, "Newcastle", "Jaigarh Port", "Dolvi", "Delayed", eta(5), "Berth congestion", "At sea", 0.61),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "material",
            "mode",
            "shipment_id",
            "qty_t",
            "origin",
            "destination",
            "plant",
            "status",
            "eta",
            "delay_reason",
            "position",
            "delay_probability",
        ],
    )
    df["eta_days"] = (pd.to_datetime(df["eta"]) - pd.Timestamp(TODAY)).dt.days
    df["source"] = "TMS / Port Community System"
    return df


@st.cache_data
def production_units() -> pd.DataFrame:
    rows = [
        # plant, unit, planned_t, actual_t, utilization, availability, yield, energy_gcal_t, status, constraint
        ("Jamshedpur", "Blast Furnace", 13500, 12950, 96, 94, 98.2, 5.52, "Stable", "None"),
        ("Jamshedpur", "BOF Shop", 12800, 12500, 97, 96, 97.8, 0.82, "Stable", "None"),
        ("Jamshedpur", "Hot Strip Mill", 9800, 9200, 89, 91, 96.4, 0.44, "Watch", "Slab mix"),
        ("Kalinganagar", "Blast Furnace", 11800, 9800, 83, 87, 97.1, 5.88, "At risk", "Coking coal cover"),
        ("Kalinganagar", "BOF Shop", 11200, 10100, 90, 91, 97.0, 0.86, "Watch", "Hot metal availability"),
        ("Kalinganagar", "CRM", 7600, 6700, 84, 88, 94.9, 0.51, "Watch", "Input coil availability"),
        ("Angul", "Blast Furnace", 9200, 8700, 91, 92, 97.5, 5.76, "Watch", "Coal buffer"),
        ("Angul", "Plate Mill", 6100, 5900, 92, 93, 95.8, 0.49, "Stable", "None"),
        ("Dolvi", "Blast Furnace", 8600, 8050, 89, 90, 97.4, 5.81, "Watch", "Port delay"),
        ("Dolvi", "HSM", 7200, 6900, 91, 92, 96.1, 0.46, "Stable", "None"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "plant",
            "unit",
            "planned_t",
            "actual_t",
            "utilization_pct",
            "availability_pct",
            "yield_pct",
            "energy_gcal_t",
            "status",
            "constraint",
        ],
    )
    df["production_gap_t"] = df["planned_t"] - df["actual_t"]
    df["source"] = "MES / Level-3 Automation"
    return df


@st.cache_data
def finished_goods() -> pd.DataFrame:
    rows = [
        ("HR Coil", "Jamshedpur SY", "East", 9400, 12, 30, 54),
        ("HR Coil", "Kolkata SY", "East", 6800, 34, 30, 53),
        ("CR Coil", "Kalinganagar SY", "East", 5100, 18, 30, 67),
        ("CR Coil", "Mumbai SY", "West", 4200, 47, 30, 68),
        ("Galvanized", "Delhi SY", "North", 1500, 52, 45, 76),
        ("Galvanized", "Jamshedpur SY", "East", 2300, 22, 45, 75),
        ("TMT Rebar", "Kalinganagar SY", "East", 7600, 9, 25, 46),
        ("TMT Rebar", "Delhi SY", "North", 3400, 15, 25, 47),
        ("Wire Rod", "Kolkata SY", "East", 2800, 20, 30, 52),
        ("Plate", "Angul SY", "South", 3900, 39, 30, 62),
        ("HR Coil", "Dolvi SY", "West", 5800, 28, 30, 55),
    ]
    df = pd.DataFrame(
        rows,
        columns=["grade", "location", "region", "qty_t", "age_days", "threshold_days", "realization_k_per_t"],
    )
    df["aging_breach"] = df["age_days"] > df["threshold_days"]
    df["inventory_value_cr"] = (df["qty_t"] * df["realization_k_per_t"] / 1000).round(1)
    df["source"] = "MES Stockyard ↔ CRM Allocation"
    return df


@st.cache_data
def dispatch_otif() -> pd.DataFrame:
    rows = [
        ("North", 14000, 13400, 95, 96, 92, "Truck availability normal"),
        ("East", 20000, 19600, 97, 98, 95, "Stable"),
        ("West", 13000, 9800, 82, 88, 78, "Material short at Mumbai yard"),
        ("South", 11000, 10400, 92, 94, 88, "Rail ETA slippage"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["region", "planned_t", "actual_t", "on_time_pct", "in_full_pct", "otif_pct", "constraint"],
    )
    df["dispatch_pct"] = (df["actual_t"] / df["planned_t"] * 100).round(1)
    df["gap_t"] = df["planned_t"] - df["actual_t"]
    df["source"] = "TMS ↔ CRM"
    return df


@st.cache_data
def logistics_assets() -> pd.DataFrame:
    rows = [
        ("Jamshedpur", 7.2, 8.0, 11, 2, 94),
        ("Kalinganagar", 9.8, 8.0, 9, 5, 76),
        ("Angul", 8.4, 8.0, 7, 3, 82),
        ("Kolkata SY", 14.2, 8.0, 4, 6, 63),
        ("Dolvi", 10.8, 8.0, 6, 4, 70),
    ]
    df = pd.DataFrame(
        rows,
        columns=["node", "turnaround_h", "target_h", "active_rakes", "waiting_rakes", "asset_health_pct"],
    )
    df["breach"] = df["turnaround_h"] > df["target_h"]
    df["source"] = "TMS / Rail Yard System"
    return df


@st.cache_data
def supplier_scorecard() -> pd.DataFrame:
    rows = [
        ("CoalCo Australia", "Coking Coal", 320, 91, 7.2, 420, 0.69),
        ("Joda Mines", "Iron Ore", 180, 97, 1.1, 95, 0.22),
        ("Noamundi Mines", "Iron Ore", 160, 98, 0.8, 72, 0.18),
        ("Raipur Ferro Alloys", "Ferro Silicon", 42, 86, 4.5, 1200, 0.63),
        ("Birmitrapur Minerals", "Limestone", 38, 94, 2.8, 180, 0.30),
        ("Newcastle Coal Export", "Coking Coal", 210, 88, 8.1, 510, 0.74),
    ]
    df = pd.DataFrame(
        rows,
        columns=["supplier", "material", "monthly_spend_cr", "otif_pct", "avg_delay_days", "quality_ppm", "risk_score"],
    )
    df["source"] = "SAP Ariba / Supplier Portal"
    return df


@st.cache_data
def esg_metrics() -> pd.DataFrame:
    rows = [
        ("Jamshedpur", 2.12, 2.05, 5.52, 5.45, 18, 22),
        ("Kalinganagar", 2.28, 2.08, 5.88, 5.50, 14, 24),
        ("Angul", 2.22, 2.10, 5.76, 5.55, 16, 21),
        ("Dolvi", 2.19, 2.09, 5.81, 5.52, 15, 23),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "plant",
            "co2_t_per_tcs",
            "co2_target",
            "energy_gcal_tcs",
            "energy_target",
            "scrap_mix_pct",
            "scrap_target_pct",
        ],
    )
    df["co2_gap_pct"] = ((df["co2_t_per_tcs"] / df["co2_target"] - 1) * 100).round(1)
    df["source"] = "ESG Data Lake / Energy Meters"
    return df


rm = raw_materials()
inb = inbound_shipments()
prod = production_units()
fg = finished_goods()
otif = dispatch_otif()
logi = logistics_assets()
supp = supplier_scorecard()
esg = esg_metrics()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def kpi_card(label: str, value: str, delta: str, status: str = "watch") -> None:
    css = {
        "good": "kpi-delta-good",
        "bad": "kpi-delta-bad",
        "watch": "kpi-delta-watch",
        "neutral": "small-muted",
    }.get(status, "kpi-delta-watch")
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="{css}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_stockout(plant: str, material: str, extra_delay_days: int = 0, horizon: int = 30) -> Tuple[int | None, float]:
    row = rm[(rm["plant"] == plant) & (rm["material"] == material)]
    if row.empty:
        return None, np.nan
    r = row.iloc[0]
    events = (
        inb[(inb["plant"] == plant) & (inb["material"] == material)]
        .assign(adjusted_eta=lambda x: x["eta_days"] + extra_delay_days)
        .groupby("adjusted_eta")["qty_t"]
        .sum()
        .to_dict()
    )
    stock = float(r["stock_t"])
    min_stock = stock
    stockout_day = None
    for day in range(1, horizon + 1):
        stock -= float(r["daily_use_t"])
        stock += float(events.get(day, 0))
        min_stock = min(min_stock, stock)
        if stock < 0 and stockout_day is None:
            stockout_day = day
    return stockout_day, min_stock


def build_risks(extra_delay_days: int = 0) -> List[Dict[str, str]]:
    risks: List[Dict[str, str]] = []

    for _, r in rm[rm["below_safety"]].iterrows():
        so_day, min_stock = project_stockout(r["plant"], r["material"], extra_delay_days)
        if so_day is not None:
            severity = "red"
            probability = "High"
            impact_cr = round(max(1.5, (r["daily_use_t"] * r["unit_cost_lakh_per_t"] / 100) * 1.8), 1)
            body = (
                f"{r['material']} at {r['plant']} has {r['days_cover']:.1f} days cover "
                f"against {r['safety_days']:.1f} safety days. Scenario projects stock-out around day {so_day}."
            )
            action = "Expedite vessel/rake priority, rebalance coal allocation, and protect blast furnace schedule."
        else:
            severity = "amber"
            probability = "Medium"
            impact_cr = round(max(0.8, (r["daily_use_t"] * r["unit_cost_lakh_per_t"] / 100) * 0.7), 1)
            body = (
                f"{r['material']} at {r['plant']} is below safety stock but scheduled inbound prevents full stock-out "
                f"within 30 days under this scenario. Buffer remains thin."
            )
            action = "Monitor ETA, confirm unloading slot, and avoid diverting material to lower-priority demand."
        risks.append(
            {
                "severity": severity,
                "title": f"Raw material risk — {r['material']} @ {r['plant']}",
                "body": body,
                "meta": f"Probability: {probability} · Estimated impact: ₹{impact_cr} Cr/day · Owner: Procurement + Logistics",
                "action": action,
            }
        )

    for _, r in inb[inb["status"] == "Delayed"].iterrows():
        risks.append(
            {
                "severity": "amber",
                "title": f"Inbound disruption — {r['shipment_id']}",
                "body": f"{r['qty_t']:,} t {r['material']} for {r['plant']} delayed. Reason: {r['delay_reason']}. Revised ETA: {r['eta']:%d %b}.",
                "meta": f"Delay probability: {r['delay_probability']:.0%} · Source: {r['source']} · Owner: Logistics Control Tower",
                "action": "Escalate to port/rail desk, secure berth/rake slot, and test alternate supply lane.",
            }
        )

    worst_otif = otif.sort_values("otif_pct").iloc[0]
    risks.append(
        {
            "severity": "amber" if worst_otif["otif_pct"] < 90 else "green",
            "title": f"Customer service risk — {worst_otif['region']} region",
            "body": f"OTIF is {worst_otif['otif_pct']}%; in-full is {worst_otif['in_full_pct']}%. Gap is {worst_otif['gap_t']:,} t. Constraint: {worst_otif['constraint']}.",
            "meta": "Owner: Sales & Operations Planning · Source: TMS ↔ CRM",
            "action": "Reallocate aged finished goods, review customer priority, and protect high-margin dispatches.",
        }
    )

    for _, r in fg[fg["aging_breach"]].iterrows():
        risks.append(
            {
                "severity": "amber",
                "title": f"Working-capital risk — {r['grade']} @ {r['location']}",
                "body": f"{r['qty_t']:,} t aged {r['age_days']} days vs {r['threshold_days']} day threshold. Inventory value ₹{r['inventory_value_cr']} Cr.",
                "meta": "Owner: Sales Allocation + Stockyard · Source: MES ↔ CRM",
                "action": "Push liquidation plan, match to open orders, and prevent quality downgrading.",
            }
        )

    severity_order = {"red": 0, "amber": 1, "green": 2}
    risks = sorted(risks, key=lambda x: severity_order[x["severity"]])
    return risks


def render_risk_card(risk: Dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="risk-card risk-{risk['severity']}">
            <div class="risk-title">{risk['title']}</div>
            <div class="risk-body">{risk['body']}</div>
            <div class="risk-meta">{risk['meta']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_sankey(selected_plants: List[str]) -> go.Figure:
    rm_sel = rm[rm["plant"].isin(selected_plants)]
    prod_sel = prod[prod["plant"].isin(selected_plants)]
    fg_total = fg["qty_t"].sum()
    dispatch_total = otif["actual_t"].sum()

    labels = [
        "Mines & Vendors",
        "Ports",
        "Rail / Road",
        "Raw Material Yard",
        "Blast Furnace",
        "Steelmaking",
        "Rolling Mills",
        "Finished Goods Yard",
        "Customers",
    ]
    source = [0, 1, 2, 3, 4, 5, 6, 7]
    target = [1, 2, 3, 4, 5, 6, 7, 8]
    values = [
        float(inb[inb["plant"].isin(selected_plants)]["qty_t"].sum() / 10),
        float(inb[inb["plant"].isin(selected_plants)]["qty_t"].sum() / 12),
        float(rm_sel["daily_use_t"].sum()),
        float(prod_sel[prod_sel["unit"].str.contains("Blast Furnace", case=False)]["actual_t"].sum()),
        float(prod_sel[prod_sel["unit"].str.contains("BOF", case=False)]["actual_t"].sum()),
        float(prod_sel[prod_sel["unit"].str.contains("HSM|CRM|Mill", case=False, regex=True)]["actual_t"].sum()),
        float(fg_total / 4),
        float(dispatch_total),
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(pad=18, thickness=18, line=dict(width=0.5), label=labels),
                link=dict(source=source, target=target, value=values),
            )
        ]
    )
    fig.update_layout(title="Material-flow digital twin: mine → port → rail → plant → customer", height=430)
    return fig


def scenario_outputs(extra_delay_days: int, demand_shock_pct: int, bf_constraint_pct: int) -> Dict[str, float]:
    kg_coal = rm[(rm["plant"] == "Kalinganagar") & (rm["material"] == "Coking Coal")].iloc[0]
    so_day, _ = project_stockout("Kalinganagar", "Coking Coal", extra_delay_days)
    production_loss_t = max(0, extra_delay_days - 2) * 1800 + max(0, bf_constraint_pct) * 95
    revenue_at_risk_cr = production_loss_t * 0.055
    service_drop = max(0, extra_delay_days * 1.2 + demand_shock_pct * 0.35 + bf_constraint_pct * 0.08)
    projected_otif = max(55, round(otif["otif_pct"].mean() - service_drop, 1))
    coal_cover = max(0, round(kg_coal["days_cover"] - extra_delay_days * 0.9, 1))
    return {
        "stockout_day": so_day if so_day is not None else 30,
        "production_loss_t": round(production_loss_t, 0),
        "revenue_at_risk_cr": round(revenue_at_risk_cr, 1),
        "projected_otif": projected_otif,
        "coal_cover": coal_cover,
    }


# -----------------------------------------------------------------------------
# Sidebar filters and scenario controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🏭 Control Tower")
    st.caption(f"Data timestamp: **{TODAY:%d %b %Y}** · synthetic near-real-time feed")
    st.markdown(
        '<span class="pill">SAP MM</span><span class="pill">TMS</span>'
        '<span class="pill">MES</span><span class="pill">CRM</span>'
        '<span class="pill">ESG Lake</span><span class="pill">Finance</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    plants = sorted(rm["plant"].unique())
    selected_plants = st.multiselect("Plant", plants, default=plants)
    if not selected_plants:
        selected_plants = plants

    only_exceptions = st.checkbox("Show only exceptions", value=False)

    st.divider()
    st.subheader("Scenario Simulator")
    extra_delay = st.slider("Additional coal shipment delay", 0, 10, 2, help="Adds delay days to inbound coal shipments.")
    demand_shock = st.slider("Demand surge / shortfall (%)", -10, 20, 5)
    bf_constraint = st.slider("Blast furnace constraint (%)", 0, 25, 8)

    st.divider()
    with st.expander("Teaching tasks", expanded=True):
        st.markdown(
            "1. Identify the top material risk and its business impact.\n\n"
            "2. Explain whether the OTIF issue is a logistics, inventory, or allocation problem.\n\n"
            "3. Use the scenario sliders to test how delay changes production and revenue risk.\n\n"
            "4. Recommend three actions for the morning control-tower meeting."
        )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Integrated Steel Digital Control Tower</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Executive view of procurement, logistics, production, inventory, customer service, finance and ESG — converted into risks and actions.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------
rm_sel = rm[rm["plant"].isin(selected_plants)]
prod_sel = prod[prod["plant"].isin(selected_plants)]
esg_sel = esg[esg["plant"].isin(selected_plants)]

scenario = scenario_outputs(extra_delay, demand_shock, bf_constraint)
risks = build_risks(extra_delay)
critical_count = sum(1 for r in risks if r["severity"] == "red")
watch_count = sum(1 for r in risks if r["severity"] == "amber")

actual_prod = prod_sel["actual_t"].sum()
plan_prod = prod_sel["planned_t"].sum()
prod_attainment = actual_prod / plan_prod * 100 if plan_prod else 0
weighted_otif = round((otif["actual_t"] * otif["otif_pct"]).sum() / otif["actual_t"].sum(), 1)
inventory_value = rm_sel["inventory_value_cr"].sum() + fg["inventory_value_cr"].sum()
avg_co2 = round(esg_sel["co2_t_per_tcs"].mean(), 2)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Production attainment", f"{prod_attainment:.1f}%", f"{actual_prod:,.0f} t vs plan", "watch" if prod_attainment < 95 else "good")
with c2:
    kpi_card("OTIF today", f"{weighted_otif:.1f}%", "Target 95%", "bad" if weighted_otif < 90 else "watch")
with c3:
    kpi_card("Critical risks", str(critical_count), f"{watch_count} watch items", "bad" if critical_count else "watch")
with c4:
    kpi_card("Inventory value", f"₹{inventory_value:,.0f} Cr", "raw + finished goods", "neutral")
with c5:
    kpi_card("Revenue at risk", f"₹{scenario['revenue_at_risk_cr']:.1f} Cr", "scenario estimate", "bad" if scenario["revenue_at_risk_cr"] > 5 else "watch")
with c6:
    kpi_card("CO₂ intensity", f"{avg_co2:.2f}", "tCO₂/t crude steel", "bad" if avg_co2 > 2.15 else "watch")

st.divider()

# -----------------------------------------------------------------------------
# Executive summary and material flow
# -----------------------------------------------------------------------------
left, right = st.columns([1.6, 1])

with left:
    st.subheader("End-to-end material flow")
    st.plotly_chart(make_sankey(selected_plants), use_container_width=True)

with right:
    st.subheader("AI operations briefing")
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-title">Morning summary</div>
            <div class="action-body">
            Kalinganagar coking coal remains the highest-risk constraint. Under the current scenario,
            projected coal cover is <b>{scenario['coal_cover']:.1f} days</b>, projected OTIF is
            <b>{scenario['projected_otif']:.1f}%</b>, and revenue at risk is approximately
            <b>₹{scenario['revenue_at_risk_cr']:.1f} Cr</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="action-card">
            <div class="action-title">Recommended control-tower actions</div>
            <div class="action-body">
            1. Prioritize Dhamra → Kalinganagar rake placement for coking coal.<br>
            2. Rebalance finished goods from aged West/North yards to open orders.<br>
            3. Protect blast furnace schedule by reviewing coke blend and alternate procurement.<br>
            4. Escalate West region in-full gap in S&OP huddle.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Risks and actions
# -----------------------------------------------------------------------------
st.subheader(f"Exception and action centre · {critical_count} critical · {watch_count} watch")
st.caption("Ranked by severity. Each alert is linked to root cause, owner and action.")

r1, r2 = st.columns(2)
for i, risk in enumerate(risks[:8]):
    with r1 if i % 2 == 0 else r2:
        render_risk_card(risk)

st.divider()

# -----------------------------------------------------------------------------
# Detailed tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "① Raw materials",
        "② Production",
        "③ Finished goods",
        "④ Dispatch & OTIF",
        "⑤ Logistics",
        "⑥ ESG & energy",
        "⑦ Supplier risk",
    ]
)

with tab1:
    st.subheader("Raw material health and stockout projection")
    view = rm_sel.copy()
    if only_exceptions:
        view = view[view["below_safety"]]

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(
            view.sort_values("days_cover"),
            x="days_cover",
            y="material",
            color="plant",
            orientation="h",
            text="days_cover",
            labels={"days_cover": "Days of cover", "material": "Material"},
            title="Days of cover by material and plant",
            height=430,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(legend_title_text="Plant")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        heat = view.pivot_table(index="material", columns="plant", values="days_cover", aggfunc="mean")
        fig = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            title="Inventory heatmap: days of cover",
            labels=dict(color="Days"),
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Raw material table**")
    st.dataframe(
        view[[
            "plant", "material", "category", "stock_t", "safety_stock_t", "daily_use_t",
            "days_cover", "safety_days", "inventory_value_cr", "below_safety", "source"
        ]].rename(
            columns={
                "stock_t": "stock (t)",
                "safety_stock_t": "safety stock (t)",
                "daily_use_t": "daily use (t)",
                "days_cover": "cover (days)",
                "safety_days": "safety (days)",
                "inventory_value_cr": "inventory value (₹ Cr)",
                "below_safety": "below safety",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Inbound shipments linked to material risks**")
    st.dataframe(
        inb[inb["plant"].isin(selected_plants)][[
            "material", "mode", "shipment_id", "qty_t", "origin", "destination", "plant", "status",
            "eta", "eta_days", "delay_reason", "position", "source"
        ]].rename(columns={"qty_t": "qty (t)", "eta_days": "ETA days"}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Production performance and bottleneck visibility")
    view = prod_sel.copy()
    if only_exceptions:
        view = view[view["status"].isin(["At risk", "Watch"])]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            view,
            x="unit",
            y="utilization_pct",
            color="plant",
            text="utilization_pct",
            title="Unit utilization %",
            labels={"utilization_pct": "Utilization %", "unit": "Unit"},
            height=420,
        )
        fig.add_hline(y=95, line_dash="dash", annotation_text="95% target")
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            view,
            x="availability_pct",
            y="yield_pct",
            size="actual_t",
            color="status",
            hover_name="unit",
            title="Availability vs yield — bubble size = production",
            labels={"availability_pct": "Availability %", "yield_pct": "Yield %"},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view[[
            "plant", "unit", "planned_t", "actual_t", "production_gap_t", "utilization_pct",
            "availability_pct", "yield_pct", "energy_gcal_t", "status", "constraint", "source"
        ]].rename(
            columns={
                "planned_t": "planned (t)",
                "actual_t": "actual (t)",
                "production_gap_t": "gap (t)",
                "utilization_pct": "utilization %",
                "availability_pct": "availability %",
                "yield_pct": "yield %",
                "energy_gcal_t": "energy GCal/t",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("Finished-goods aging and working capital")
    view = fg.copy()
    if only_exceptions:
        view = view[view["aging_breach"]]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            view,
            x="age_days",
            y="grade",
            size="qty_t",
            color="region",
            hover_name="location",
            title="FG aging — bubble size = tonnes",
            labels={"age_days": "Age days", "grade": "Grade"},
            height=420,
        )
        fig.add_vline(x=30, line_dash="dash", annotation_text="30d threshold")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.treemap(
            view,
            path=["region", "location", "grade"],
            values="inventory_value_cr",
            color="age_days",
            title="Working capital locked in finished goods",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view[["grade", "location", "region", "qty_t", "age_days", "threshold_days", "inventory_value_cr", "aging_breach", "source"]].rename(
            columns={
                "qty_t": "qty (t)",
                "age_days": "age (days)",
                "threshold_days": "threshold (days)",
                "inventory_value_cr": "value (₹ Cr)",
                "aging_breach": "aging breach",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.subheader("Dispatch performance and customer service")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=otif["region"], y=otif["planned_t"], name="Planned")
        fig.add_bar(x=otif["region"], y=otif["actual_t"], name="Actual")
        fig.update_layout(barmode="group", title="Dispatch vs plan by region", height=410)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            otif,
            x="region",
            y="otif_pct",
            text="otif_pct",
            title="OTIF % by region",
            labels={"otif_pct": "OTIF %"},
            height=410,
        )
        fig.add_hline(y=95, line_dash="dash", annotation_text="Target")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Teaching note: West has a customer-service problem driven mainly by the in-full component. "
        "That points to inventory allocation/availability rather than pure transport delay."
    )
    st.dataframe(
        otif[["region", "planned_t", "actual_t", "gap_t", "dispatch_pct", "on_time_pct", "in_full_pct", "otif_pct", "constraint", "source"]].rename(
            columns={
                "planned_t": "planned (t)",
                "actual_t": "actual (t)",
                "gap_t": "gap (t)",
                "dispatch_pct": "dispatch %",
                "on_time_pct": "on-time %",
                "in_full_pct": "in-full %",
                "otif_pct": "OTIF %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab5:
    st.subheader("Logistics visibility: port, rail, road and yard assets")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            logi.sort_values("turnaround_h"),
            x="turnaround_h",
            y="node",
            orientation="h",
            text="turnaround_h",
            title="Rake turnaround vs 8h target",
            labels={"turnaround_h": "Hours", "node": "Node"},
            height=420,
        )
        fig.add_vline(x=8, line_dash="dash", annotation_text="Target")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            logi,
            x="waiting_rakes",
            y="asset_health_pct",
            size="active_rakes",
            color="breach",
            hover_name="node",
            title="Yard congestion and asset health",
            labels={"waiting_rakes": "Waiting rakes", "asset_health_pct": "Asset health %"},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        logi.rename(
            columns={
                "turnaround_h": "turnaround (h)",
                "target_h": "target (h)",
                "active_rakes": "active rakes",
                "waiting_rakes": "waiting rakes",
                "asset_health_pct": "asset health %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab6:
    st.subheader("ESG, energy and decarbonization signals")
    view = esg_sel.copy()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=view["plant"], y=view["co2_t_per_tcs"], name="Actual")
        fig.add_bar(x=view["plant"], y=view["co2_target"], name="Target")
        fig.update_layout(barmode="group", title="CO₂ intensity: actual vs target", height=410, yaxis_title="tCO₂/t crude steel")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            view,
            x="plant",
            y="energy_gcal_tcs",
            text="energy_gcal_tcs",
            title="Specific energy consumption",
            labels={"energy_gcal_tcs": "GCal/t crude steel"},
            height=410,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view.rename(
            columns={
                "co2_t_per_tcs": "CO₂ t/tcs",
                "co2_target": "CO₂ target",
                "energy_gcal_tcs": "energy GCal/tcs",
                "energy_target": "energy target",
                "scrap_mix_pct": "scrap mix %",
                "scrap_target_pct": "scrap target %",
                "co2_gap_pct": "CO₂ gap %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab7:
    st.subheader("Supplier risk and procurement performance")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            supp,
            x="otif_pct",
            y="avg_delay_days",
            size="monthly_spend_cr",
            color="risk_score",
            hover_name="supplier",
            title="Supplier performance: OTIF vs delay",
            labels={"otif_pct": "Supplier OTIF %", "avg_delay_days": "Avg delay days"},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            supp.sort_values("risk_score", ascending=False),
            x="supplier",
            y="risk_score",
            color="material",
            title="Supplier risk score",
            labels={"risk_score": "Risk score"},
            height=420,
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        supp.rename(
            columns={
                "monthly_spend_cr": "monthly spend (₹ Cr)",
                "otif_pct": "supplier OTIF %",
                "avg_delay_days": "avg delay days",
                "quality_ppm": "quality PPM",
                "risk_score": "risk score",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------------------------------------------------------
# Scenario panel
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Scenario impact summary")
s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    kpi_card("Coal cover", f"{scenario['coal_cover']:.1f} d", "Kalinganagar", "bad" if scenario["coal_cover"] < 5 else "watch")
with s2:
    kpi_card("Projected stockout", f"Day {scenario['stockout_day']}", "30-day horizon", "bad" if scenario["stockout_day"] < 10 else "watch")
with s3:
    kpi_card("Production loss", f"{scenario['production_loss_t']:,.0f} t", "scenario estimate", "bad" if scenario["production_loss_t"] > 5000 else "watch")
with s4:
    kpi_card("Projected OTIF", f"{scenario['projected_otif']:.1f}%", "after demand/delay shock", "bad" if scenario["projected_otif"] < 85 else "watch")
with s5:
    kpi_card("Revenue risk", f"₹{scenario['revenue_at_risk_cr']:.1f} Cr", "not accounting for recovery", "bad" if scenario["revenue_at_risk_cr"] > 5 else "watch")

st.caption(
    "Synthetic training dashboard. Figures are illustrative and designed for executive education, case discussion and hands-on dashboard teaching."
)
