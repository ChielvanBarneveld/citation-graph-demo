"""FORAS citation graph - v5.

v5 (post-restructure, 5 May 2026, branch-label feat/asreview-tab):

  Tab 1 - "FORAS"        (the citation-graph view, formerly "Citation graph")
  Tab 2 - "ASReview"     (NEW - active-learning baseline + 70/30 vs GNN)
  Tab 3 - "GNN"          (NEW - placeholder; populated by feat/gnn-tab)
  Tab 4 - "Candidates"   (the candidate explorer, formerly "Candidate explorer";
                          gets sentinel cards + cross-tab table from
                          feat/candidates-tab)

The legacy "Method & metrics" tab is removed; relevant pipeline-explainer text
moved into tab 2 (ASReview, section 2.1 / 2.8).

Design unchanged: cream + navy + cyan LED accent.
"""

import json
import math
import sqlite3
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent  # cloud-deploy: app at repo root
DATA = ROOT / "data"  # cloud-deploy
THESIS_DATA = ROOT / "data"
SENTINEL_DIR = ROOT / "outputs" / "sentinel_rewrites"
ASREVIEW_RUNS = ROOT / "outputs" / "asreview_runs"
COMPARISON_DIR = ROOT / "outputs" / "comparison"
SENTINEL_RANKS = COMPARISON_DIR / "sentinel_ranks.parquet"

# ---------- palette ----------
PALETTE = {
    "bg": "#f6f4ef",
    "bg_soft": "#eef1f4",
    "navy": "#1e2a44",
    "navy_mid": "#3b5475",
    "cyan": "#22d3ee",
    "cyan_soft": "#a5f3fc",
    "slate": "#94a3b8",
    "mist": "#cbd5e1",
    "edge": "rgba(30,42,68,0.10)",
    "accent_warm": "#f59e0b",
    "rose": "#f43f5e",
    "amber_soft": "#fde68a",
}

STAGE_COLORS = {
    "ft_included":        PALETTE["cyan"],
    "tiab_included":      PALETTE["navy"],
    "screened":           PALETTE["slate"],
    "external":           PALETTE["mist"],
    "candidate_external": PALETTE["rose"],
}
STAGE_SIZES = {
    "ft_included":        9.0,
    "tiab_included":      5.5,
    "screened":           3.0,
    "external":           2.2,
    "candidate_external": 2.5,
}
STAGE_OPACITY = {
    "ft_included":        1.0,
    "tiab_included":      0.95,
    "screened":           0.55,
    "external":           0.35,
    "candidate_external": 0.55,
}
STAGE_LABEL = {
    "ft_included":   "FT-included (172)",
    "tiab_included": "TI/AB-included (395)",
    "screened":      "Screened, excluded",
    "external":      "External periphery",
}

CHANNELS = [
    ("ch_replication",   "Replication search",         "#0e7490"),
    ("ch_comprehensive", "Comprehensive search",       "#7c3aed"),
    ("ch_snowballing",   "Snowballing",                "#f97316"),
    ("ch_fulltext",      "Full-text search",           "#059669"),
    ("ch_oa_ic",         "OpenAlex (inclusion crit.)", "#e11d48"),
    ("ch_oa_logistic",   "OpenAlex (logistic)",        "#eab308"),
    ("ch_oa_all",        "OpenAlex (all abstracts)",   "#64748b"),
]


# ---------- helpers ----------
def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        try:
            if math.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
    return str(v)


def _hex_to_rgba(hex_colour: str, opacity: float) -> str:
    if not isinstance(hex_colour, str) or not hex_colour.startswith("#"):
        return hex_colour
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity:.3f})"


# ---------- data ----------
@st.cache_data(show_spinner=False)
def load_data():
    papers = pd.read_parquet(DATA / "papers.parquet")
    edges = pd.read_parquet(DATA / "edges.parquet")
    return papers, edges


@st.cache_data(show_spinner=False)
def load_candidates():
    cand = pd.read_parquet(DATA / "candidates.parquet")
    cross = pd.read_parquet(DATA / "candidate_foras_edges.parquet")
    return cand, cross


@st.cache_resource(show_spinner=False)
def build_graph(_papers_n: int, _edges_n: int):
    papers, edges = load_data()
    G = nx.DiGraph()
    for _, r in papers.iterrows():
        G.add_node(
            r["pid"],
            stage=r["stage"],
            title=_s(r["title"])[:180],
            year=int(r["publication_year"]) if pd.notna(r["publication_year"]) else 0,
            journal=_s(r["journal_name"])[:80],
            authors=_s(r["author_names"]),
            topic=_s(r["primary_topic_name"]),
            field=_s(r["primary_topic_field"]),
            cites=int(r["cited_by_count"]) if pd.notna(r["cited_by_count"]) else 0,
            channels=[c for c, _, _ in CHANNELS if int(r[c]) == 1],
            disagreement=int(r["disagreement_hh"]) if pd.notna(r["disagreement_hh"]) else 0,
        )
    G.add_edges_from(edges.itertuples(index=False, name=None))
    return G


@st.cache_resource(show_spinner="Computing 3D layout ...")
def compute_layout(_nodes: tuple, _edges: tuple, seed: int = 7):
    g = nx.Graph()
    g.add_nodes_from(_nodes)
    g.add_edges_from(_edges)
    if g.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(g, dim=3, seed=seed, iterations=30)


# ---------- node rendering (UNCHANGED from v4) ----------
def node_hover(pid: str, attr: dict) -> str:
    title = _s(attr.get("title"))[:160]
    journal = _s(attr.get("journal"))[:70]
    topic = _s(attr.get("topic"))[:60]
    year = attr.get("year") or ""
    authors = _s(attr.get("authors"))[:80]
    cites = attr.get("cites", 0)
    stage = attr.get("stage", "")
    ch = attr.get("channels", [])
    ch_names = ", ".join([n for k, n, _ in CHANNELS if k in ch])
    if stage == "ft_included":
        badge = "<span style='color:#22d3ee'>#</span> FT-included"
    elif stage == "tiab_included":
        badge = "<span style='color:#1e2a44'>#</span> TI/AB-included"
    elif stage == "screened":
        badge = "<span style='color:#94a3b8'>#</span> Screened"
    else:
        badge = "<span style='color:#cbd5e1'>#</span> External"
    dis = "<br><b>Human-human disagreement</b>" if attr.get("disagreement", 0) == 1 else ""
    lines = [
        f"<b>{title}</b>",
        f"<i>{authors}</i>" if authors else "",
        f"{journal} . {year}" if (journal or year) else "",
        f"{topic}" if topic else "",
        f"{badge} . {cites:,} cites" + dis,
        f"<span style='color:#64748b'>{ch_names}</span>" if ch_names else "",
    ]
    return "<br>".join([ln for ln in lines if ln])


def colour_by_stage(attr: dict):
    stage = attr.get("stage", "external")
    return STAGE_COLORS[stage], STAGE_SIZES[stage], STAGE_OPACITY[stage]


def colour_by_channel(attr: dict, chosen_channel):
    ch = attr.get("channels", [])
    stage = attr.get("stage", "external")
    size = STAGE_SIZES[stage]
    colour_map = {k: c for k, _, c in CHANNELS}
    if chosen_channel is None:
        if not ch:
            return PALETTE["mist"], size, 0.35
        return colour_map[ch[0]], size, STAGE_OPACITY[stage]
    if chosen_channel in ch:
        return colour_map[chosen_channel], size * 1.2, 1.0
    if not ch:
        return PALETTE["mist"], size * 0.8, 0.20
    return PALETTE["navy_mid"], size * 0.8, 0.22


def colour_by_disagreement(attr: dict):
    stage = attr.get("stage", "external")
    size = STAGE_SIZES[stage]
    if attr.get("disagreement", 0) == 1:
        return PALETTE["accent_warm"], size * 1.25, 1.0
    base, _, op = colour_by_stage(attr)
    return base, size, op * 0.45


# ---------- plot (UNCHANGED) ----------
def build_plot(G, pos, colour_mode, chosen_channel, year_range,
               show_edges, funnel_stage_visible) -> go.Figure:
    xs, ys, zs, colours, sizes, opacities, hovers = [], [], [], [], [], [], []
    for n in G.nodes():
        attr = G.nodes[n]
        y = attr.get("year", 0)
        if y and (y < year_range[0] or y > year_range[1]):
            continue
        stage = attr.get("stage", "external")
        if stage not in funnel_stage_visible:
            continue
        if n not in pos:
            continue
        x, yp, z = pos[n]
        xs.append(x); ys.append(yp); zs.append(z)
        if colour_mode == "channel":
            c, s, o = colour_by_channel(attr, chosen_channel)
        elif colour_mode == "disagreement":
            c, s, o = colour_by_disagreement(attr)
        else:
            c, s, o = colour_by_stage(attr)
        colours.append(c); sizes.append(s); opacities.append(o)
        hovers.append(node_hover(n, attr))

    rgba = [_hex_to_rgba(c, o) for c, o in zip(colours, opacities)]
    traces = []

    if show_edges:
        ex, ey, ez = [], [], []
        for u, v in G.edges():
            if u not in pos or v not in pos:
                continue
            if G.nodes[u].get("stage", "external") not in funnel_stage_visible:
                continue
            if G.nodes[v].get("stage", "external") not in funnel_stage_visible:
                continue
            yu = G.nodes[u].get("year", 0)
            if yu and (yu < year_range[0] or yu > year_range[1]):
                continue
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
            ez += [pos[u][2], pos[v][2], None]
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=PALETTE["edge"], width=1),
            hoverinfo="none", showlegend=False, name="edges",
        ))

    h_x, h_y, h_z = [], [], []
    for n in G.nodes():
        attr = G.nodes[n]
        if attr.get("stage") != "ft_included":
            continue
        if "ft_included" not in funnel_stage_visible:
            continue
        y = attr.get("year", 0)
        if y and (y < year_range[0] or y > year_range[1]):
            continue
        if n not in pos:
            continue
        h_x.append(pos[n][0]); h_y.append(pos[n][1]); h_z.append(pos[n][2])
    traces.append(go.Scatter3d(
        x=h_x, y=h_y, z=h_z, mode="markers",
        marker=dict(size=20, color=PALETTE["cyan_soft"],
                    opacity=0.18, line=dict(width=0)),
        hoverinfo="none", showlegend=False, name="halo",
    ))

    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=sizes, color=rgba,
                    line=dict(width=0.35, color=PALETTE["navy"])),
        text=hovers, hovertemplate="%{text}<extra></extra>",
        showlegend=False, name="papers",
    ))

    for stage, label in STAGE_LABEL.items():
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=10, color=STAGE_COLORS[stage],
                        line=dict(width=0.35, color=PALETTE["navy"])),
            name=label, showlegend=(colour_mode == "stage"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                  size=13, color=PALETTE["navy"]),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor=PALETTE["bg"],
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        ),
        legend=dict(
            x=0.01, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(246,244,239,0.85)",
            bordercolor=PALETTE["navy"], borderwidth=1,
            font=dict(size=12, color=PALETTE["navy"]),
        ),
        hoverlabel=dict(bgcolor=PALETTE["bg"], bordercolor=PALETTE["navy"],
                        font=dict(family="Inter", color=PALETTE["navy"])),
    )
    return fig


FUNNEL_FRAMES = [
    ("All retrieved",          {"external", "screened", "tiab_included", "ft_included"}),
    ("Screened (FORAS corpus)", {"screened", "tiab_included", "ft_included"}),
    ("TI/AB-included",          {"tiab_included", "ft_included"}),
    ("FT-included (SR core)",   {"ft_included"}),
]


# ---------- CSS ----------
def inject_css():
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
        color: {PALETTE['navy']};
      }}
      .stApp {{ background: {PALETTE['bg']}; }}
      section[data-testid="stSidebar"] {{
        background: {PALETTE['bg_soft']};
        border-right: 1px solid rgba(30,42,68,0.08);
      }}
      h1, h2, h3, h4 {{
        color: {PALETTE['navy']};
        font-weight: 600; letter-spacing: -0.01em;
      }}
      .foras-title {{
        font-weight: 700; font-size: 1.6rem;
        color: {PALETTE['navy']}; margin: 0 0 0.1rem 0;
      }}
      .foras-sub {{
        color: {PALETTE['navy_mid']};
        font-size: 0.92rem; margin-bottom: 0.9rem;
      }}
      .foras-kpi {{
        background: {PALETTE['bg']};
        border: 1px solid rgba(30,42,68,0.08);
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 1px 2px rgba(30,42,68,0.04);
      }}
      .foras-kpi .num {{
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 1.35rem; font-weight: 600; color: {PALETTE['navy']};
      }}
      .foras-kpi .num-cyan {{
        color: {PALETTE['cyan']};
        text-shadow: 0 0 8px rgba(34,211,238,0.35);
      }}
      .foras-kpi .num-rose {{ color: {PALETTE['rose']}; }}
      .foras-kpi .lab {{
        color: {PALETTE['navy_mid']};
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.06em; margin-top: 0.05rem;
      }}
      .stButton > button {{
        background: {PALETTE['navy']};
        color: {PALETTE['bg']}; border: 0;
        border-radius: 999px; padding: 0.35rem 1.1rem; font-weight: 500;
      }}
      .stButton > button:hover {{
        background: {PALETTE['cyan']}; color: {PALETTE['navy']};
      }}
      .explainer {{
        background: {PALETTE['bg_soft']};
        border-left: 3px solid {PALETTE['cyan']};
        padding: 0.8rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0 1rem 0;
      }}
      .explainer h4 {{ margin-top: 0; }}
    </style>
    """, unsafe_allow_html=True)


def kpi_strip(papers: pd.DataFrame):
    n_ft = int((papers["stage"] == "ft_included").sum())
    n_tiab = int((papers["stage"] == "tiab_included").sum())
    n_screened = int((papers["stage"] == "screened").sum())
    n_external = int((papers["stage"] == "external").sum())
    total = len(papers)
    base_rate = n_ft / max(total, 1)
    cols = st.columns(5)
    items = [
        (f"{total:,}", "Papers in graph", False),
        (f"{n_ft:,}", "FT-included (core)", True),
        (f"{n_tiab:,}", "TI/AB-included", False),
        (f"{n_screened + n_external:,}", "Screened + periphery", False),
        (f"{base_rate*100:.2f}%", "Base rate (FT)", True),
    ]
    for col, (num, lab, cyan) in zip(cols, items):
        klass = "num num-cyan" if cyan else "num"
        col.markdown(
            f"<div class='foras-kpi'><div class='{klass}'>{num}</div>"
            f"<div class='lab'>{lab}</div></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 1 - Citation graph (existing v4 view)
# ============================================================
def render_citation_graph_tab(papers, G):
    kpi_strip(papers)

    with st.sidebar:
        st.markdown("### View")
        colour_mode = st.radio(
            "Colour by",
            options=["stage", "channel", "disagreement"],
            format_func=lambda k: {
                "stage": "Screening stage",
                "channel": "Retrieval channel",
                "disagreement": "Human-human disagreement",
            }[k], index=0,
        )
        chosen_channel = None
        if colour_mode == "channel":
            channel_labels = {k: lab for k, lab, _ in CHANNELS}
            choice = st.selectbox(
                "Highlight which channel?",
                options=["(rainbow)"] + [k for k, _, _ in CHANNELS],
                format_func=lambda k: "All channels (rainbow)"
                if k == "(rainbow)" else channel_labels[k],
            )
            chosen_channel = None if choice == "(rainbow)" else choice

        st.markdown("### Funnel replay")
        funnel_idx = st.select_slider(
            "Stage",
            options=list(range(len(FUNNEL_FRAMES))),
            format_func=lambda i: FUNNEL_FRAMES[i][0],
            value=0,
        )
        funnel_visible = FUNNEL_FRAMES[funnel_idx][1]

        st.markdown("### Filters")
        years = [int(y) for y in papers["publication_year"].dropna().unique()]
        y_min = min(years) if years else 2000
        y_max = max(years) if years else 2025
        year_range = st.slider("Publication year", y_min, y_max, (y_min, y_max))
        show_edges = st.checkbox("Show citation edges", value=True)

    pids_tuple = tuple(G.nodes())
    edges_tuple = tuple(G.edges())
    pos = compute_layout(pids_tuple, edges_tuple)

    fig = build_plot(G, pos, colour_mode, chosen_channel, year_range,
                     show_edges, funnel_visible)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    if colour_mode == "channel":
        st.markdown("##### Retrieval channels - how many included papers each channel found")
        rows = []
        for k, lab, _ in CHANNELS:
            n_ft = int(((papers["stage"] == "ft_included") & (papers[k] == 1)).sum())
            n_tiab = int(((papers["stage"] == "tiab_included") & (papers[k] == 1)).sum())
            n_any = int((papers[k] == 1).sum())
            rows.append(dict(Channel=lab, FT_included=n_ft,
                             TIAB_included=n_tiab, Total=n_any))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ============================================================
# TAB 4 - Candidates v2 (sentinel cards + 70/30 + legacy explorer)
# ============================================================
def render_candidate_tab(cand: pd.DataFrame, cross: pd.DataFrame):
    """Tab 4 - Candidates v2.

    Six sub-sections:
      4.1 Banner / what-you-see-here
      4.2 KPIs (incl. sentinel count)
      4.3 Sentinel cards (Solomon / Kardiner / Southard)
      4.4 70/30 rank table (ASReview vs GCN)
      4.5 Existing candidate-explorer (filters + plots + browse + export)
      4.6 Glossary
    """
    st.markdown(
        "<div class='foras-title'>Historical-terminology candidates</div>"
        "<div class='foras-sub'>2.288 papers found via 29 historical "
        "PTSD-terms in OpenAlex - sentinels op kop, dan de full explorer.</div>",
        unsafe_allow_html=True,
    )
    sentinel_bundles = _load_sentinel_bundles()
    _section_4_1_intro(cand)
    st.markdown("---")
    _section_4_2_kpis(cand, sentinel_bundles)
    st.markdown("---")
    _section_4_3_sentinels(sentinel_bundles)
    st.markdown("---")
    _section_4_4_seventy_thirty()
    st.markdown("---")
    _section_4_5_explorer(cand, cross)
    st.markdown("---")
    _section_4_6_glossary()


# ----- Tab 4 helpers -----

def _section_4_1_intro(cand: pd.DataFrame):
    st.markdown("### 4.1 - Wat zie je hier?")
    st.markdown(
        "<div class='explainer'>"
        "<b>The candidate set.</b> Each row is a paper that uses one of "
        "29 historical names for PTSD (<i>shell shock, traumatic neurosis, "
        "war neurosis, soldier's heart, effort syndrome, ...</i>) according "
        "to OpenAlex. We ran two queries per term: (1) full-text search "
        "restricted to pre-1980 publications, (2) title-only search for "
        "post-1980. The 9 candidates that already appear in FORAS were all "
        "<b>screened-and-excluded</b> - none passed TI/AB. Strong signal that "
        "FORAS screening drops these historical-term papers."
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Hoe is deze set gemaakt?"):
        st.markdown(
            "Build-script: `code/find_historical_candidates.py`. Per term "
            "twee OpenAlex-queries (pre1980 full-text + post1980 title-only), "
            "dedup op OpenAlex ID, daarna join met FORAS via "
            "`referenced_works` om de cross-edges te berekenen. De resulterende "
            "set staat in `data/historical-terminology/candidates.csv` "
            "(2.288 unieke kandidaten) en wordt voor deze tab geserveerd via "
            "`code/data/candidates.parquet` (build_candidates_data.py)."
        )


def _section_4_2_kpis(cand: pd.DataFrame, sentinel_bundles: dict):
    st.markdown("### 4.2 - KPIs")
    n_total = len(cand)
    n_in_foras = int(cand["in_foras"].sum())
    n_with_edge = int((cand["edges_total"] > 0).sum())
    n_with_ft_edge = int(((cand["edges_to_foras_ft"] > 0)
                          | (cand["edges_from_foras_ft"] > 0)).sum())
    n_with_tiab_edge = int(((cand["edges_to_foras_tiab"] > 0)
                            | (cand["edges_from_foras_tiab"] > 0)).sum())
    total_edges = int(cand["edges_total"].sum())
    n_sentinels = len(sentinel_bundles)
    n_sentinels_with_rewrite = sum(
        1 for d in sentinel_bundles.values() if "rewrite" in d
    )

    cols = st.columns(7)
    items = [
        (f"{n_total:,}", "Candidates total", False, False),
        (f"{n_in_foras}", "Already in FORAS", False, True),
        (f"{n_with_edge:,}", ">=1 FORAS edge", False, False),
        (f"{n_with_tiab_edge}", "edge to TIAB", True, False),
        (f"{n_with_ft_edge}", "edge to FT", True, False),
        (f"{total_edges}", "Total cross-edges", False, False),
        (f"{n_sentinels_with_rewrite}/{n_sentinels}",
         "Sentinels (rewritten)", True, False),
    ]
    for col, (num, lab, cyan, rose) in zip(cols, items):
        klass = "num"
        if cyan: klass = "num num-cyan"
        if rose: klass = "num num-rose"
        col.markdown(
            f"<div class='foras-kpi'><div class='{klass}'>{num}</div>"
            f"<div class='lab'>{lab}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("")


def _word_diff_html(a: str, b: str) -> tuple[str, str]:
    """Return two HTML snippets: a-vs-b with deletions/insertions highlighted.

    Cyan-tinted backgrounds mark words present in only one of the two strings.
    Used in 4.3 to visualise where the rewrite landed."""
    import difflib
    aw = a.split()
    bw = b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw, autojunk=False)
    a_html, b_html = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            a_html.append(" ".join(aw[i1:i2]))
            b_html.append(" ".join(bw[j1:j2]))
        elif tag == "delete":
            a_html.append(
                f"<span style='background:#fde68a;border-radius:3px;"
                f"padding:1px 3px'>{' '.join(aw[i1:i2])}</span>"
            )
        elif tag == "insert":
            b_html.append(
                f"<span style='background:#a5f3fc;border-radius:3px;"
                f"padding:1px 3px'>{' '.join(bw[j1:j2])}</span>"
            )
        elif tag == "replace":
            a_html.append(
                f"<span style='background:#fde68a;border-radius:3px;"
                f"padding:1px 3px'>{' '.join(aw[i1:i2])}</span>"
            )
            b_html.append(
                f"<span style='background:#a5f3fc;border-radius:3px;"
                f"padding:1px 3px'>{' '.join(bw[j1:j2])}</span>"
            )
    return " ".join(a_html), " ".join(b_html)


def _section_4_3_sentinels(sentinel_bundles: dict):
    st.markdown("### 4.3 - Drie sentinels - Solomon (EASY), Kardiner (MEDIUM), "
                "Southard (HARD)")
    st.caption(
        "Doelbewust toegevoegd aan de FORAS hold-out om te testen of "
        "ASReview en GCN ze kunnen vinden ondanks dat de originelen "
        "geen moderne PTSD-tokens gebruiken. De rewrites maken ze "
        "plausibel-include - niet ontdekt door FORAS' originele "
        "screening."
    )
    if not sentinel_bundles:
        st.warning(
            "`outputs/sentinel_rewrites/` ontbreekt. Run dispatch-prompt 3 "
            "(`feat/sentinel-rewrites`) eerst."
        )
        return
    with st.expander("Waarom deze drie? (samengevat uit edge_check_rapport)"):
        st.markdown(
            "**Solomon 1993** *(EASY)* - 22 cited-by-FORAS / 7 TIAB / "
            "1 FT. Mooi cluster voor de citation-graph - garandeert "
            "1-hop label-propagation signaal.\n\n"
            "**Kardiner 1941** *(MEDIUM)* - 17 cited-by-FORAS / 1 TIAB / "
            "0 FT. Klassieke seminale werk dat Spitzer expliciet noemde "
            "toen DSM-III in 1980 PTSD formaliseerde. Test of GCN "
            "signaal in 2-hop neighbourhood kan benutten.\n\n"
            "**Southard 1920** *(HARD)* - 0 cited-by-FORAS. Klassieke "
            "WO-I monografie (589 case histories). Bewijst de "
            "bovengrens van de methode (P2 in `context/hypothesis.md`)."
        )

    focus = _focused_sentinel()
    sentinel_iter = ("solomon_1993", "kardiner_1941", "southard_1920")
    if focus:
        sentinel_iter = (focus,) if focus in sentinel_iter else sentinel_iter
    for sid in sentinel_iter:
        if sid not in sentinel_bundles:
            continue
        bundle = sentinel_bundles[sid]
        orig = bundle.get("original", {})
        rewr = bundle.get("rewrite", {})
        if not orig or not rewr:
            continue
        difficulty = rewr.get("difficulty") or orig.get("difficulty") or "?"
        title = orig.get("title", sid)
        year = orig.get("year") or rewr.get("year")
        authors = orig.get("authors") or []
        if isinstance(authors, list):
            authors_str = "; ".join(str(a) for a in authors if a)
        else:
            authors_str = str(authors)
        oa_id = orig.get("openalex_id", "")
        oa_link = (f"https://openalex.org/{oa_id}" if oa_id else "")
        cited_any = orig.get("cited_by_foras_any", 0)
        cited_tiab = orig.get("cited_by_foras_tiab", 0)
        cited_ft = orig.get("cited_by_foras_ft", 0)

        st.markdown(
            f"#### {sid.replace('_', ' ').title()}  "
            f"<span style='color:{PALETTE['navy_mid']};font-size:0.85rem'>"
            f"({difficulty})</span>",
            unsafe_allow_html=True,
        )
        info_cols = st.columns([4, 1, 1, 1])
        with info_cols[0]:
            st.markdown(
                f"**{title}** ({year})  "
                f"\n*{authors_str or 'unknown authors'}*  "
                + (f"\n[`{oa_id}`]({oa_link})" if oa_link else "")
            )
        with info_cols[1]:
            st.metric("Cited by FORAS (any)", cited_any,
                      help="Aantal FORAS-papers dat deze sentinel citeert")
        with info_cols[2]:
            st.metric("by TIAB-incl", cited_tiab,
                      help="Citaties uit TIAB-included subset")
        with info_cols[3]:
            st.metric("by FT-incl", cited_ft,
                      help="Citaties uit FT-included subset")

        # Tabbed Original / Rewrite / Diff view
        t_orig, t_rew, t_diff = st.tabs(
            ["Original abstract", "LLM-rewrite", "Side-by-side diff"]
        )
        orig_abs = orig.get("abstract", "") or ""
        rew_abs = rewr.get("rewritten_abstract", "") or ""
        with t_orig:
            kind = orig.get("abstract_kind", "")
            source = orig.get("abstract_source", "")
            if orig.get("notes"):
                st.caption(orig["notes"])
            st.markdown(orig_abs or "*(empty)*")
            st.caption(
                f"Source: `{source}` | kind: `{kind}` | "
                f"{len(orig_abs)} chars"
            )
        with t_rew:
            check = rewr.get("criteria_check", {}) or {}
            badge = lambda b: ("OK" if b else "MISS")
            badge_color = lambda b: ("#10b981" if b else "#f43f5e")
            st.markdown(
                "Criterium-check (LLM-back-prompt op rewrite-tekst alleen):  "
                + "  ".join(
                    f"<span style='color:{badge_color(check.get(str(k), False))}'>"
                    f"<b>{k}</b> {badge(check.get(str(k), False))}</span>"
                    for k in (1, 2, 3, 4)
                ),
                unsafe_allow_html=True,
            )
            st.markdown(rew_abs or "*(empty)*")
            wc = rewr.get("rewrite_word_count")
            iters = rewr.get("iterations_needed", "?")
            st.caption(
                f"{wc} words | iterations: {iters} | "
                f"banned-token check: "
                f"{'pass' if (rewr.get('modern_ptsd_token_check') or {}).get('passed', True) else 'FAIL'}"
            )
            ana = rewr.get("anachronisms_flagged") or []
            if ana:
                with st.expander(f"Anachronismen ({len(ana)})"):
                    for a in ana:
                        st.markdown(f"- {a}")
        with t_diff:
            if not orig_abs.strip() or not rew_abs.strip():
                st.caption("Diff vereist beide abstracts; een van beide is leeg.")
            else:
                a_html, b_html = _word_diff_html(orig_abs, rew_abs)
                st.markdown("**Original**", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background:{PALETTE['bg_soft']};"
                    f"padding:0.6rem 0.9rem;border-radius:6px;"
                    f"line-height:1.55'>{a_html}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.markdown("**Rewrite**", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background:{PALETTE['bg_soft']};"
                    f"padding:0.6rem 0.9rem;border-radius:6px;"
                    f"line-height:1.55'>{b_html}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Geel = woorden alleen in original; cyaan = alleen "
                    "in rewrite. Geen kleur = onveranderd. Voor Kardiner "
                    "en Southard is het origineel grotendeels niet-"
                    "studie-tekst, dus het diff-overzicht is groot."
                )
        st.markdown("---")


def _section_4_4_seventy_thirty():
    st.markdown("### 4.4 - 70/30 rang-tabel (ASReview vs GCN)")
    st.caption(
        "Gedeelde tabel met `outputs/comparison/sentinel_ranks.parquet`. "
        "Tab 2 (ASReview) vult `asreview_rank` + `asreview_recall_at_rank`; "
        "tab 3 (GNN, via `code/train_gnn_demo.py`) vult `gnn_rank` + "
        "`gnn_score`. Hier zie je beide naast elkaar."
    )
    sentinel_ranks = ROOT / "outputs" / "comparison" / "sentinel_ranks.parquet"
    if not sentinel_ranks.exists():
        st.info(
            "70/30-vergelijking nog niet gedraaid. Run dispatch 1 "
            "(ASReview) en 2 (GNN) eerst."
        )
        return
    df = pd.read_parquet(sentinel_ranks)
    if df["asreview_rank"].isna().all() and df["gnn_rank"].isna().all():
        st.info(
            "Bestand bestaat maar beide ASReview- en GNN-kolommen zijn "
            "nog leeg. Run de tab 2 simulation runner + "
            "`python code/train_gnn_demo.py`."
        )
        return
    df = df.copy()
    focus = _focused_sentinel()
    if focus:
        df = df[df["sentinel_id"] == focus]
        if len(df) == 0:
            st.info(f"Geen rijen voor focus-sentinel `{focus}`.")
            return
    df["delta_rank"] = (df["asreview_rank"].astype("float")
                        - df["gnn_rank"].astype("float"))
    label_map = {
        "modus_a_asreview_order": "Modus A (ASReview-volgorde)",
        "modus_b_random_stratified": "Modus B (random)",
    }
    df["split_label"] = df["split_mode"].map(label_map).fillna(df["split_mode"])
    show = df[[
        "sentinel_id", "split_label", "asreview_rank",
        "asreview_recall_at_rank", "gnn_rank", "gnn_score",
        "delta_rank", "bundle", "seed", "updated_at",
    ]].rename(columns={
        "sentinel_id": "Sentinel",
        "split_label": "Split-modus",
        "asreview_rank": "ASReview-rang",
        "asreview_recall_at_rank": "ASReview recall@rank",
        "gnn_rank": "GCN-rang",
        "gnn_score": "GCN-score",
        "delta_rank": "Delta (AS-GCN)",
        "bundle": "ASReview-bundle",
        "seed": "Seed",
        "updated_at": "Updated",
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "**Lezing.** Lager rang = sneller gevonden. `Delta > 0` betekent "
        "GCN-rang lager dan ASReview-rang = GCN vond het sentinel sneller. "
        "Voor Southard verwacht ik geen verschil - die heeft 0 citations "
        "en is dus de bovengrens van wat citatie-structuur kan."
    )
    if df["asreview_rank"].isna().all():
        st.warning(
            "`asreview_rank` is overal NaN. Run de ASReview-tab simulation "
            "runner met dataset = 'FORAS + 2.288 candidates' om die "
            "kolom te vullen."
        )
    if df["gnn_rank"].isna().all():
        st.warning(
            "`gnn_rank` is overal NaN. Run `python code/train_gnn_demo.py` "
            "om die kolom te vullen."
        )

    # ----- 4.4b - Sentinel discovery timeline (improvement #1) -----
    st.markdown("#### 4.4b - Sentinel discovery-timeline")
    st.caption(
        "Bij welk percentage gescreend / gerangschikt verschijnt elke sentinel? "
        "Lager = sneller gevonden. P1 (recall-gain) voorspelt: GCN-lijntjes "
        "links van AsReview-lijntjes voor Solomon en Kardiner; voor Southard "
        "(0 cites) verwachten we geen verschil."
    )
    n_cands = 2289  # candidates pool size (incl. sentinel injection)
    fig_t = go.Figure()
    sentinel_colors = {
        "solomon_1993": PALETTE["cyan"],
        "kardiner_1941": PALETTE["accent_warm"],
        "southard_1920": PALETTE["rose"],
    }
    for sid, color in sentinel_colors.items():
        sub = df[df["sentinel_id"] == sid]
        for _, row in sub.iterrows():
            mode = row["split_mode"]
            tag = "A" if mode == "modus_a_asreview_order" else "B"
            for method, col, dash in (("AsReview", "asreview_rank", "dash"),
                                       ("GCN", "gnn_rank", "solid")):
                rank = row.get(col)
                if pd.isna(rank):
                    continue
                pct = float(rank) / n_cands * 100
                fig_t.add_trace(go.Scatter(
                    x=[pct, pct], y=[0, 1],
                    mode="lines",
                    line=dict(color=color, width=2.5, dash=dash),
                    name=f"{sid[:8]}.{tag} - {method}",
                    hovertemplate=(
                        f"{sid}<br>{method}<br>rank={int(rank)} / {n_cands}"
                        f"<br>= {pct:.1f}% gescreend<extra></extra>"
                    ),
                ))
    fig_t.update_layout(
        title="Discovery-timeline: % candidates gerangschikt vóór sentinel",
        xaxis_title="% candidates gerangschikt (lager = sneller)",
        yaxis_title="",
        yaxis=dict(visible=False, range=[0, 1]),
        xaxis=dict(range=[0, 100]),
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(family="Inter", color=PALETTE["navy"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_t, width="stretch", config={"displaylogo": False})
    st.caption(
        "Solid = GCN, dashed = AsReview. Cyan = Solomon (EASY, 22 cites), "
        "amber = Kardiner (MEDIUM, 17 cites), rose = Southard (HARD, 0 cites)."
    )

    # ----- 4.4c - Edge-density vs rank scatter (improvement #2) -----
    st.markdown("#### 4.4c - Edge-density × rank")
    st.caption(
        "Per sentinel: hoe scherper correleert rank met edge-count voor GCN "
        "(graph-aware) versus AsReview (tekst-only)? P2 voorspelt: GCN-rank "
        "↓ als edge-count ↑; AsReview-rank onafhankelijk."
    )
    edge_counts = {
        "solomon_1993": 22,
        "kardiner_1941": 17,
        "southard_1920": 0,
    }
    rows_sc = []
    # Use modus_b (random stratified) by default — both rows are duplicates of
    # the same rank in our current stand-in fill.
    sub_b = df[df["split_mode"] == "modus_b_random_stratified"]
    for _, r in sub_b.iterrows():
        sid = r["sentinel_id"]
        for method, col in (("AsReview", "asreview_rank"),
                             ("GCN", "gnn_rank")):
            rank = r.get(col)
            if pd.isna(rank):
                continue
            rows_sc.append({
                "sentinel": sid, "method": method,
                "edges": edge_counts.get(sid, 0),
                "rank": int(rank),
                "rank_pct": float(rank) / n_cands * 100,
            })
    if rows_sc:
        sc_df = pd.DataFrame(rows_sc)
        fig_s = px.scatter(
            sc_df, x="edges", y="rank", color="method", text="sentinel",
            color_discrete_map={"AsReview": PALETTE["navy_mid"],
                                "GCN": PALETTE["cyan"]},
            title="Edge-density vs rank (lager rank = sneller gevonden)",
        )
        fig_s.update_traces(textposition="top center", marker=dict(size=14))
        fig_s.update_layout(
            xaxis_title="cited_by_foras_any (citation-edges naar FORAS)",
            yaxis_title=f"rank (1 = top, {n_cands} = bottom)",
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
            font=dict(family="Inter", color=PALETTE["navy"]),
        )
        st.plotly_chart(fig_s, width="stretch", config={"displaylogo": False})
        st.caption(
            "Een neerwaartse lijn voor GCN = méér edges → lagere rank "
            "(P2-bevestiging). Een vlakke lijn voor AsReview = tekst alleen, "
            "edges niet benut."
        )
    else:
        st.info("Geen data; rank-tabel is leeg.")


def _section_4_5_explorer(cand: pd.DataFrame, cross: pd.DataFrame):
    """The legacy candidate-explorer view (filters + plots + tables + export)."""
    st.markdown("### 4.5 - Candidate explorer (filters & browse)")

    # Sidebar filters
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Candidate filters")
        all_pools = ["pre1980", "post1980_title", "pre1980;post1980_title"]
        pool_sel = st.multiselect(
            "Pool", options=all_pools, default=all_pools,
            help=("pre1980 = full-text search restricted to <1980. "
                  "post1980_title = title-only search for >=1980. "
                  "pre1980;post1980_title = found in both."),
        )
        era_options = ["<1920", "1920-44", "1945-79", "1980-99", "2000+", "nan"]
        era_sel = st.multiselect("Era", options=era_options, default=era_options)
        all_langs = sorted(
            [x for x in cand["language"].dropna().astype(str).unique()]
        )
        if not all_langs:
            all_langs = ["en"]
        lang_sel = st.multiselect("Language", options=all_langs,
                                  default=all_langs)
        edge_filter = st.radio(
            "Edge-density requirement",
            options=["any", "edge to any FORAS", "edge to TIAB-incl",
                     "edge to FT-incl", "no edges (pure isolates)"],
            index=0,
        )
        in_foras_filter = st.radio(
            "FORAS overlap",
            options=["any", "only in-FORAS (n=9)", "exclude in-FORAS"],
            index=0,
        )
        all_terms = set()
        for s in cand["found_via_terms"].fillna(""):
            for t in str(s).split(";"):
                if t:
                    all_terms.add(t)
        term_options = sorted(all_terms)
        term_sel = st.multiselect(
            "Found via term (any of)", options=term_options,
            default=[],
            help="Empty = include all terms. Pick one or more to subset.",
        )

    # Apply filters
    f = cand.copy()
    if pool_sel:
        f = f[f["pools"].isin(pool_sel)]
    if era_sel:
        f = f[f["era"].astype(str).isin(era_sel)]
    if lang_sel:
        f = f[f["language"].astype(str).isin(lang_sel)]
    if edge_filter == "edge to any FORAS":
        f = f[f["edges_total"] > 0]
    elif edge_filter == "edge to TIAB-incl":
        f = f[(f["edges_to_foras_tiab"] > 0)
              | (f["edges_from_foras_tiab"] > 0)]
    elif edge_filter == "edge to FT-incl":
        f = f[(f["edges_to_foras_ft"] > 0)
              | (f["edges_from_foras_ft"] > 0)]
    elif edge_filter == "no edges (pure isolates)":
        f = f[f["edges_total"] == 0]
    if in_foras_filter == "only in-FORAS (n=9)":
        f = f[f["in_foras"]]
    elif in_foras_filter == "exclude in-FORAS":
        f = f[~f["in_foras"]]
    if term_sel:
        mask = f["found_via_terms"].fillna("").apply(
            lambda s: any(t in str(s).split(";") for t in term_sel)
        )
        f = f[mask]

    st.markdown(f"##### Filtered set: **{len(f):,} candidates**")

    # Summary plots
    c1, c2 = st.columns(2)
    with c1:
        era_df = f.copy()
        era_df["FORAS overlap"] = era_df["in_foras"].map(
            {True: "in FORAS", False: "external"}
        )
        era_counts = (era_df.groupby(["era", "FORAS overlap"])
                            .size().reset_index(name="count"))
        fig_era = px.bar(
            era_counts, x="era", y="count", color="FORAS overlap",
            color_discrete_map={"in FORAS": PALETTE["cyan"],
                                "external": PALETTE["rose"]},
            category_orders={"era": ["<1920", "1920-44", "1945-79",
                                     "1980-99", "2000+", "nan"]},
            title="By era (stacked: in FORAS vs external)",
        )
        fig_era.update_layout(
            paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
            font=dict(family="Inter", color=PALETTE["navy"]),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_era, width="stretch", config={"displaylogo": False})
    with c2:
        term_counts = {}
        for s in f["found_via_terms"].fillna(""):
            for t in str(s).split(";"):
                if t:
                    term_counts[t] = term_counts.get(t, 0) + 1
        if term_counts:
            tdf = (pd.DataFrame(list(term_counts.items()),
                                columns=["term", "count"])
                     .sort_values("count", ascending=True).tail(15))
            fig_term = px.bar(
                tdf, x="count", y="term", orientation="h",
                color_discrete_sequence=[PALETTE["navy"]],
                title="Top 15 productive terms (in current filter)",
            )
            fig_term.update_layout(
                paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
                font=dict(family="Inter", color=PALETTE["navy"]),
                height=320, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_term, width="stretch",
                            config={"displaylogo": False})

    # Edge-density summary
    st.markdown(
        "##### Edge density of current filter (citation links to FORAS)"
    )
    edge_summary = pd.DataFrame([
        {"Direction": "Candidate -> FORAS-FT-included",
         "Edges": int(f["edges_to_foras_ft"].sum()),
         "Candidates with >=1": int((f["edges_to_foras_ft"] > 0).sum())},
        {"Direction": "Candidate -> FORAS-TIAB-broad",
         "Edges": int(f["edges_to_foras_tiab"].sum()),
         "Candidates with >=1": int((f["edges_to_foras_tiab"] > 0).sum())},
        {"Direction": "Candidate -> any FORAS",
         "Edges": int(f["edges_to_foras_all"].sum()),
         "Candidates with >=1": int((f["edges_to_foras_all"] > 0).sum())},
        {"Direction": "FORAS-FT-included -> Candidate",
         "Edges": int(f["edges_from_foras_ft"].sum()),
         "Candidates with >=1": int((f["edges_from_foras_ft"] > 0).sum())},
        {"Direction": "FORAS-TIAB-broad -> Candidate",
         "Edges": int(f["edges_from_foras_tiab"].sum()),
         "Candidates with >=1": int((f["edges_from_foras_tiab"] > 0).sum())},
        {"Direction": "Any FORAS -> Candidate",
         "Edges": int(f["edges_from_foras_all"].sum()),
         "Candidates with >=1": int((f["edges_from_foras_all"] > 0).sum())},
    ])
    st.dataframe(edge_summary, hide_index=True, width="stretch")

    if int(f["in_foras"].sum()) > 0:
        st.markdown(
            "##### In-FORAS overlap (these 9 are "
            "FORAS-screened-and-excluded)"
        )
        cols_show = ["openalex_id", "title", "year", "language",
                     "foras_stage", "foras_label_FT", "foras_label_TIAB",
                     "found_via_terms", "edges_total"]
        st.dataframe(
            f[f["in_foras"]][cols_show].sort_values("year", ascending=False),
            hide_index=True, width="stretch",
        )

    st.markdown("##### Browse candidates (sortable, top 200)")
    cols_show = ["openalex_id", "title", "year", "language",
                 "found_via_terms",
                 "edges_to_foras_ft", "edges_from_foras_ft",
                 "edges_to_foras_tiab", "edges_from_foras_tiab",
                 "edges_total", "in_foras"]
    st.dataframe(
        f[cols_show].sort_values(["edges_total", "year"],
                                 ascending=[False, False]).head(200),
        hide_index=True, width="stretch",
    )

    st.markdown("##### Export current filter as a test set")
    st.write(
        "Pick a name for this candidate-subset and download as CSV. "
        "This is the file you can later inject into the FORAS graph "
        "for GNN training/eval."
    )
    csv = f.to_csv(index=False).encode("utf-8")
    label = st.text_input("Filename (without .csv)",
                          value="candidates_testset_v1")
    st.download_button(
        label=f"Download {len(f):,}-row test set as CSV",
        data=csv,
        file_name=f"{label}.csv",
        mime="text/csv",
    )


def _section_4_6_glossary():
    if _in_demo_mode():
        return
    with st.expander("Glossary"):
        st.markdown(
            "- **Sentinel** - een van Solomon 1993 / Kardiner 1941 / "
            "Southard 1920; voorgeselecteerde anchor voor de ASReview-vs-"
            "GCN vergelijking.\n"
            "- **EASY / MEDIUM / HARD** - difficulty-label per sentinel; "
            "EASY heeft de meeste FORAS-edges, HARD heeft er 0.\n"
            "- **Hold-out** - de 30% FORAS-records die niet meedoen in "
            "de trainings-fase.\n"
            "- **Citation-edge** - een directe verwijzing van paper A "
            "naar paper B in de citatie-graaf.\n"
            "- **Bibliographic coupling** - twee papers delen een "
            "referentie -> waarschijnlijk gerelateerd onderwerp.\n"
            "- **Co-citation** - twee papers worden samen geciteerd "
            "door een derde paper -> waarschijnlijk gerelateerd.\n"
            "- **OpenAlex** - open scholarly database (450M+ works); "
            "bron van onze candidate-set.\n"
            "- **W-id** - OpenAlex' interne paper-identifier "
            "(`W1897891557` voor Solomon 1993).\n"
            "- **Pool** - een van de twee zoekstrategieen "
            "(`pre1980` full-text vs `post1980_title` title-only).\n"
            "- **found_via_terms** - lijst van historische PTSD-namen "
            "waarmee dit paper gevonden is (semicolon-separated).\n"
            "- **Criterium-check** - LLM-back-prompt die het rewrite-"
            "abstract scoort tegen FORAS' 4 inclusion-criteria; ✓ per "
            "criterium betekent het abstract bevat plausible bewijs voor "
            "PTSS-meting / post-trauma / validated scale / LGMM."
        )

# ============================================================
# LEGACY - render_method_tab (no longer wired; kept for archival reference;
# pipeline-explainer prose is lifted into render_asreview_tab section 2.1/2.8)
# ============================================================
def render_method_tab(papers: pd.DataFrame, cand: pd.DataFrame, cross: pd.DataFrame):
    st.markdown(
        "<div class='foras-title'>Method & metrics</div>"
        "<div class='foras-sub'>How FORAS works, how we plan to inject candidates, "
        "what role the GNN plays, and how we'll measure it with ASReview-insights.</div>",
        unsafe_allow_html=True,
    )

    # ---- 1. FORAS pipeline ----
    st.markdown("### 1 - The FORAS pipeline (\"the hunt for the last relevant paper\")")
    st.markdown(
        f"FORAS is van de Schoot et al. (2025), *European Journal of "
        f"Psychotraumatology*, [DOI 10.1080/20008066.2025.2546214]"
        f"(https://doi.org/10.1080/20008066.2025.2546214). The headline framing: "
        f"how do you keep searching until you've found the **last relevant paper** "
        f"on a systematic-review topic? Their answer is a hybrid pipeline where "
        f"AI acts as a 'super-assistant' alongside human reviewers, not a "
        f"replacement."
    )
    st.markdown(
        f"**Important to know about scope.** FORAS-included does NOT mean "
        f"'about PTSD' in general. The review is specifically about **latent "
        f"growth mixture modelling (LGMM) of PTSS trajectories after traumatic "
        f"events** (sec. 2.1 of the paper). A paper only ends up FT-included "
        f"if it applies LGMM (or equivalent trajectory-classification methods) "
        f"to PTSS data. That has a direct consequence for our T-012 historical-"
        f"terminology candidates: a pre-1980 paper on *shell shock* could never "
        f"satisfy the LGMM criterion because the method didn't exist yet. The "
        f"9 in-FORAS overlap candidates being all `screened`-and-excluded is "
        f"therefore not a bug - it's the criterion working as designed."
    )
    st.markdown(
        f"The published dataset (`van_de_Schoot_2025.csv`, {len(papers):,} "
        f"papers in this graph after pid-validation) merges **8 search "
        f"strategies** (paper sec. 2.2):"
    )
    st.markdown(
        "1. **SYNERGY** re-labelled initial set (4.544 records, 47 relevant after re-labelling).\n"
        "2. **Exact replication** of the 2017 search in PubMed/Embase/PsycINFO/Scopus (6.701 dedupe'd, 202 TI/AB-included).\n"
        "3. **Comprehensive search** (+1.120 to screen).\n"
        "4. **Snowballing** (forward + backward citations from the relevant set).\n"
        "5. **Full-text search via Dimensions** (586 records, 511 new beyond other methods).\n"
        "6. **OpenAlex - nearest to inclusion criteria** (vectorised, top 5.000, k=900 added after dedupe).\n"
        "7. **OpenAlex - nearest to relevant abstracts** (170k -> 57.232 dedupe -> top 935 added).\n"
        "8. **OpenAlex + ASReview logistic** active-learning ranking (k=930 after dedupe)."
    )
    st.markdown(
        "Plus a 9th *quality-check* layer that is screening, not search: an "
        "**LLM (ChatGPT-3.5-v0301)** scored every record against the 4 inclusion "
        "criteria and was used to flag potential reviewer mistakes (sec. 2.3.5.3). "
        "1 of 126 final-included papers came from this LLM-rescue path."
    )
    st.markdown(
        "**Two-stage human screening:**\n"
        "- **TI/AB screening** - 568 records pass (`label_abstract_included = 1`).\n"
        "- **Full-text screening** - 172 records pass (`label_included = 1`). "
        "126 unique studies in the final review (130 incl. duplicates per paper sec. 2.3.6)."
    )
    st.markdown(
        "78 of the included papers carry `disagreement_hh = 1` (the two human "
        "reviewers initially disagreed). 8 of the 126 final-included would have "
        "been missed without the second screener (sec. 3.2). Those are the hardest "
        "cases and the most informative supervision signal for any model that "
        "wants to imitate the screening decision."
    )
    st.markdown(
        "The dataset also contains a `referenced_works` column with parsed "
        "OpenAlex citation lists (99.5% non-null) - that's what makes the "
        "intra-corpus citation graph (75.617 edges) buildable without extra "
        "API calls. That's the 3D plot in tab 1."
    )

    # ---- 2. Candidate injection ----
    st.markdown("### 2 - Injecting historical-terminology candidates")
    n_in_foras = int(cand["in_foras"].sum())
    n_external = len(cand) - n_in_foras
    n_cross = len(cross)
    st.markdown(
        f"In OpenAlex we ran 29 historical names for PTSD (*shell shock, "
        f"traumatic neurosis, war neurosis, soldier's heart, effort syndrome, "
        f"battle fatigue,* DE/FR variants such as *Kriegsneurose, obusite, "
        f"nevrose traumatique*) against two pools - pre-1980 "
        f"full-text search and post-1980 title-only search. That returned "
        f"**{len(cand):,} unique candidate papers** after deduplication."
    )
    st.markdown(
        f"Of these, **{n_in_foras} already appear in the FORAS dataset** - "
        f"all 9 in the `screened`-and-excluded stage (none passed TI/AB). "
        f"The remaining **{n_external:,} are new papers** that FORAS never "
        f"considered. Across both directions there are **{n_cross} cross-edges** "
        f"(citations between candidates and FORAS papers). That number is the "
        f"raw material a citation-based GNN would have to learn from."
    )
    st.markdown(
        "**The plan**: pick a candidate subset in tab 2 (e.g. only candidates "
        "with >=1 edge to FORAS, or only post-1980 papers, or only those found "
        "via specific terms), download as CSV, merge into the FORAS graph as "
        "a 5th stage `candidate_external`, including the relevant cross-edges. "
        "The combined graph becomes the input to the GNN."
    )
    st.markdown(
        "**Re-frame the GNN target.** Because FORAS-included means *LGMM-of-"
        "PTSS-trajectories*, we should not naively train the GNN to predict "
        "FORAS-included for our candidates - they are mostly historical case "
        "studies, never trajectory-modelling work. A more honest target is "
        "**broader PTSD-relevance**: would a clinician/historian classify this "
        "paper as 'about PTSD-as-a-condition'? The 9 in-FORAS overlap "
        "candidates can serve as silver-standard *negatives* (FORAS retrieved "
        "and excluded), and we'll need a small handful of human-labelled "
        "*positives* on the candidate set to anchor training. This is closer "
        "to a label-propagation than a label-replication problem."
    )

    # ---- 3. GNN role ----
    st.markdown("### 3 - What the GNN actually does")
    st.markdown(
        "The GNN is a **supervised, semi-supervised node-classifier** - "
        "not reinforcement learning. We train it on the labeled subset "
        "(172 FT-included as `1`, the screened-but-excluded as `0`, plus "
        "optionally the TIAB-included as a soft positive) and let it propagate "
        "those labels through the citation graph via message passing. Each "
        "node's embedding is updated by averaging or attention-weighting its "
        "neighbours' embeddings, and after a few layers every candidate node "
        "has a learned vector. A small classifier head turns that into a "
        "probability that the candidate is PTSD-relevant."
    )
    st.markdown(
        "Concretely, the architectures we'll try first are **PPR (personalized "
        "PageRank baseline)**, **Node2Vec + MLP**, **GCN**, then **GraphSAGE** "
        "or **GAT** if the simpler ones leave headroom. Library: PyTorch "
        "Geometric. The features per node will be a mix of (a) text features "
        "from title + abstract (TF-IDF or sentence-BERT), (b) metadata "
        "(publication year, language, journal field), and (c) topology features "
        "(in/out degree, era one-hot)."
    )
    st.markdown(
        "Reinforcement learning could come in only at a different layer: if "
        "we wrap the GNN as the scoring function inside an **active-learning** "
        "loop (which paper should a human label next?) - that's what ASReview "
        "does with classical models, and the GNN could slot in there. But "
        "the GNN itself is plain supervised."
    )

    # ---- 4. Metrics ----
    st.markdown("### 4 - Evaluation - ASReview-insights metrics")
    st.markdown(
        "We evaluate the GNN with the same screening-simulation framework that "
        "FORAS itself uses. The paper (sec. 2.3.6 / Results) runs simulations "
        "via the **Makita workflow generator** on top of **ASReview LAB**, "
        "with `mxbai-embed-large-v1` as feature extractor and Random Forest as "
        "classifier; the headline metric is **time-to-discover** relevant "
        "records (Ferdinands et al., 2023). For our GNN we use the matching "
        "`asreview-insights` toolkit "
        "([github.com/asreview/asreview-insights](https://github.com/asreview/asreview-insights))."
    )
    st.markdown(
        "- **Loss** (normalized cumulative loss) - the area between the perfect "
        "retrieval curve and the model's actual retrieval curve, normalized "
        "to [0, 1]. 0 = perfect, ~0.5 = random. This is the headline metric "
        "in modern ASReview benchmarks.\n"
        "- **Recall @ k** - what fraction of the relevant papers does the model "
        "rank in its top *k* (e.g. top 10%)? Higher = better.\n"
        "- **WSS @ recall=R** (Work Saved over Sampling) - how much screening "
        "time does the model save vs random review while still achieving "
        "recall R? Standard cutoff is WSS@95%.\n"
        "- **ATD / Time-to-Discovery** - the average rank at which relevant "
        "papers are discovered. Lower = better. This is the metric the FORAS "
        "paper explicitly chose for its own simulation."
    )
    st.markdown(
        "**Our test setup** will be: (a) the candidate subset chosen in tab 2 "
        "as the unlabelled pool, (b) the 9 in-FORAS overlap candidates as "
        "silver-standard negatives, (c) a small human-labelled positive set on "
        "the candidates (to anchor the broader PTSD-relevance target), and "
        "(d) optionally a held-out portion of FORAS-FT-included as synthetic "
        "positives - keeping in mind these have a different definition (LGMM-"
        "trajectory) than what we want to measure. We report the four metrics "
        "for the GNN, a PPR (personalized PageRank) baseline, and a TF-IDF / "
        "sentence-BERT text-similarity baseline."
    )
    st.markdown(
        "**Reality check from T-012.** The cross-edge density (~3 edges to "
        "FT-included across all 2.288 candidates) means the GNN will lean "
        "heavily on text features - the citation signal alone is too thin to "
        "drive label propagation. A pragmatic baseline is the FORAS-paper "
        "stack itself (`mxbai-embed-large-v1` + Random Forest); if our GNN "
        "doesn't beat that on the candidate subset, the answer is text-only "
        "is sufficient and the citation graph is decoration."
    )

    st.markdown("---")
    st.caption(
        "Reference: van de Schoot et al. (2025), *European Journal of "
        "Psychotraumatology*, 16:1, 2546214 - "
        "[10.1080/20008066.2025.2546214](https://doi.org/10.1080/20008066.2025.2546214). "
        "PROSPERO pre-registration CRD42023494027. Materials, datasets, scripts: "
        "see paper Table 1 (OSF / DataverseNL / GitHub repositories of "
        "Coimbra, Lombaers, Bron, de Bruin, van de Schoot)."
    )



# ============================================================
# TAB 3 - GNN learning lab (NEW; reads outputs/gnn_leerlab/)
# ============================================================

LEERLAB = ROOT / "outputs" / "gnn_leerlab"


@st.cache_data(show_spinner=False)
def _load_leerlab_artifacts():
    """Load the pre-computed GNN-leerlab artifacts as a dict.

    The artifacts come from either `code/train_gnn_demo.py` (real GCN) or the
    sandbox stand-in script (PPR + logistic). Schema is identical so the tab
    code does not branch."""
    art = {}
    if not LEERLAB.exists():
        return art
    cfg = LEERLAB / "config.json"
    if cfg.exists():
        art["config"] = json.loads(cfg.read_text(encoding="utf-8"))
    met = LEERLAB / "metrics.json"
    if met.exists():
        art["metrics"] = json.loads(met.read_text(encoding="utf-8"))
    emb = LEERLAB / "embeddings.npy"
    if emb.exists():
        art["embeddings"] = np.load(emb)
    sc = LEERLAB / "scores.parquet"
    if sc.exists():
        art["scores"] = pd.read_parquet(sc)
    cs = LEERLAB / "candidate_scores.parquet"
    if cs.exists():
        art["candidate_scores"] = pd.read_parquet(cs)
    return art


def render_gnn_tab(papers: pd.DataFrame, cand: pd.DataFrame,
                   cross: pd.DataFrame):
    """Tab 3 - GNN learning lab + 70/30 vs ASReview."""
    st.markdown(
        "<div class='foras-title'>GNN learning lab</div>"
        "<div class='foras-sub'>Wat is een graph, wat doet message-passing, "
        "hoe ziet de GCN MIJN data, en hoe verhoudt de GCN-rangschikking zich "
        "tot ASReview op de 70/30 hold-out.</div>",
        unsafe_allow_html=True,
    )
    art = _load_leerlab_artifacts()
    if not art:
        st.warning(
            "Geen leerlab-artifacts gevonden in `outputs/gnn_leerlab/`. "
            "Run `python code/train_gnn_demo.py` (vereist `pip install torch "
            "torch-geometric scikit-learn pandas pyarrow`)."
        )
    else:
        cfg = art.get("config", {})
        kind = cfg.get("model_kind", "unknown")
        if "stand_in" in str(kind).lower() or "stand-in" in str(kind).lower() \
                or "NOT_GCN" in str(kind) or "NOT a real GCN" in str(kind):
            st.info(
                "**Hint:** je kijkt naar een PPR + logistic-regression "
                "stand-in voor de GCN (de sandbox waarin deze tab gebouwd "
                "is kon torch+PyG niet installeren). Run "
                "`python code/train_gnn_demo.py` op je laptop om deze "
                "artifacten te vervangen door echte GCN-output."
            )

    st.markdown(
        "<div class='explainer'><h4>Wat zie je hier?</h4>"
        "Zeven sub-secties: graph-intro (3.1), message-passing-animatie (3.2), "
        "GCN-laag-uitleg (3.3), 'hoe ziet de GCN MIJN data' (3.4), 70/30 "
        "vergelijking met ASReview (3.5), alternatieve GNN-toepassingen (3.6), "
        "glossary (3.7). Doel: na deze tab snap je wat een GNN doet, en "
        "waarom hij wel of niet helpt voor het sentinel-experiment.</div>",
        unsafe_allow_html=True,
    )

    _section_3_1_what_is_graph(papers)
    st.markdown("---")
    _section_3_2_message_passing(papers)
    st.markdown("---")
    _section_3_3_gcn_layer(art)
    st.markdown("---")
    _section_3_4_gcn_on_my_data(art, papers)
    st.markdown("---")
    _section_3_5_seventy_thirty(art)
    st.markdown("---")
    _section_3_6_alternatives()
    st.markdown("---")
    _section_3_7_glossary()


# ----- 3.1 What is a graph? -----
def _section_3_1_what_is_graph(papers):
    st.markdown("### 3.1 - Wat is een graph?")
    st.caption(
        "Een graph = nodes + edges. In FORAS: elke paper is een node, "
        "elke citatie tussen twee FORAS-papers is een directed edge. "
        "Hover hieronder over een node om titel, jaar en label te zien; "
        "schuif de slider om steeds meer hops vanaf de centrale node mee "
        "te nemen."
    )
    with st.expander("Waarom is een citation-graph nuttig?"):
        st.markdown(
            "De hypothese is: **papers die dezelfde dingen citeren delen "
            "vaak een onderwerp**, ook al gebruiken ze andere woorden. "
            "Voor onze thesis: een paper over 'shell shock' uit 1920 "
            "wordt mogelijk geciteerd door dezelfde latere papers als "
            "een modern PTSD-paper - de citatie-structuur 'overbrugt' "
            "het terminologie-verschil. Dat is het hele theoretische "
            "argument achter Optie 1 in `context/opties/"
            "optie_1_foras_insert.md`."
        )
    n_hops = st.slider(
        "Hoeveel hops vanaf een centraal FT-included paper?",
        min_value=1, max_value=3, value=1, step=1,
        help="1-hop = directe buren; 2-hop = buren-van-buren; etc.",
        key="gnn_hops",
    )
    # Pick a deterministic FT-included paper as center
    ft = papers[papers["stage"] == "ft_included"]
    if len(ft) == 0:
        st.warning("Geen FT-included papers gevonden in de graph-data.")
        return
    center_pid = ft.iloc[0]["pid"]
    edges_path = DATA / "edges.parquet"
    if not edges_path.exists():
        # try the parent location
        edges_path = ROOT / "outputs" / "citation-graph" / "edges.parquet"
    try:
        edges = pd.read_parquet(edges_path)
    except Exception:
        st.caption("(Edges niet beschikbaar in deze omgeving; sub-graph plot "
                   "overgeslagen.)")
        return
    # BFS up to n_hops
    frontier = {center_pid}
    visited = {center_pid}
    sub_edges = []
    for _ in range(n_hops):
        new_front = set()
        out = edges[edges["src"].isin(frontier)]
        in_ = edges[edges["tgt"].isin(frontier)]
        for _, r in pd.concat([out, in_]).iterrows():
            sub_edges.append((r["src"], r["tgt"]))
            for n in (r["src"], r["tgt"]):
                if n not in visited:
                    new_front.add(n)
                    visited.add(n)
        frontier = new_front
        # cap at 60 nodes to keep the plot readable
        if len(visited) > 60:
            break
    sub_edges = list(set(sub_edges))
    sub_papers = papers[papers["pid"].isin(visited)].copy()
    if len(sub_papers) <= 1:
        st.caption("(Geen edges gevonden vanuit dit center.)")
        return
    G = nx.Graph()
    G.add_nodes_from(sub_papers["pid"])
    G.add_edges_from(sub_edges)
    pos = nx.spring_layout(G, seed=7)
    edge_x, edge_y = [], []
    for s, t in sub_edges:
        if s in pos and t in pos:
            edge_x.extend([pos[s][0], pos[t][0], None])
            edge_y.extend([pos[s][1], pos[t][1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=PALETTE["edge"], width=0.6), hoverinfo="skip",
    ))
    node_x = [pos[p][0] for p in sub_papers["pid"] if p in pos]
    node_y = [pos[p][1] for p in sub_papers["pid"] if p in pos]
    visible = sub_papers[sub_papers["pid"].isin(pos)].copy()
    colors = [STAGE_COLORS.get(s, PALETTE["mist"]) for s in visible["stage"]]
    sizes = [STAGE_SIZES.get(s, 3.0) * 1.6 for s in visible["stage"]]
    sizes = [s * 1.5 if pid == center_pid else s
             for s, pid in zip(sizes, visible["pid"])]
    text = [f"{_s(t)[:80]}<br>{int(y) if pd.notna(y) else '?'}<br>{stage}"
            for t, y, stage in zip(visible["title"],
                                   visible["publication_year"],
                                   visible["stage"])]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(color=colors, size=sizes,
                    line=dict(color=PALETTE["navy"], width=0.4)),
        text=text, hoverinfo="text",
    ))
    fig.update_layout(showlegend=False, height=380,
                      margin=dict(l=0, r=0, t=20, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Sub-graph rond **{center_pid}** | {len(visible)} nodes, "
               f"{len(sub_edges)} edges getoond. Center-node is groter.")


# ----- 3.2 Message passing -----
def _section_3_2_message_passing(papers):
    st.markdown("### 3.2 - Wat doet message-passing?")
    st.caption(
        "Message-passing = bij elke laag krijgt elke node informatie van "
        "zijn directe buren. Na 2 lagen weet een node iets over zijn "
        "2-hop omgeving. Animeer hieronder de propagatie."
    )
    depth = st.slider(
        "Animatie-stap (= aantal layers diep)",
        min_value=0, max_value=4, value=1, step=1,
        help="0 = alleen de start-node oranje. 1-4 = meer hops mee.",
        key="gnn_msg_depth",
    )
    with st.expander("Wat zegt de wiskunde?"):
        st.markdown(
            "Eén GCN-laag in formule:\n\n"
            "`H^(l+1) = sigma( D^(-1/2) A^hat D^(-1/2) H^(l) W^(l) )`\n\n"
            "- `H^(l)`: node-features op laag l\n"
            "- `A^hat = A + I`: adjacency-matrix met self-loops\n"
            "- `D`: degree-matrix van A^hat\n"
            "- `W^(l)`: leerbare weight-matrix\n"
            "- `sigma`: activation (typisch ReLU)\n\n"
            "Intuitief: node krijgt het gemiddelde van zijn buren-features, "
            "schaalt dat met een geleerde matrix W, en past een non-"
            "lineaire activation toe."
        )
    # Build a fixed mini-graph of 18 random FT-neighborhoods
    edges_path = DATA / "edges.parquet"
    if not edges_path.exists():
        edges_path = ROOT / "outputs" / "citation-graph" / "edges.parquet"
    try:
        edges = pd.read_parquet(edges_path)
    except Exception:
        st.caption("(Edges niet beschikbaar; animatie-plot overgeslagen.)")
        return
    rng = np.random.default_rng(7)
    ft = papers[papers["stage"] == "ft_included"]
    if len(ft) == 0:
        return
    center_pid = ft.iloc[3]["pid"]  # different center than 3.1
    # BFS 4 hops capped at 25 nodes
    frontier = {center_pid}; visited = {center_pid}
    levels = {center_pid: 0}
    for d in range(1, 5):
        new_front = set()
        out = edges[edges["src"].isin(frontier)]
        in_ = edges[edges["tgt"].isin(frontier)]
        for _, r in pd.concat([out, in_]).iterrows():
            for n in (r["src"], r["tgt"]):
                if n not in visited:
                    new_front.add(n); visited.add(n); levels[n] = d
        frontier = new_front
        if len(visited) > 25:
            break
    sub_papers = papers[papers["pid"].isin(visited)].copy()
    G = nx.Graph()
    G.add_nodes_from(sub_papers["pid"])
    sub_edges = [(s, t) for s, t in zip(edges["src"], edges["tgt"])
                 if s in visited and t in visited]
    G.add_edges_from(sub_edges)
    pos = nx.spring_layout(G, seed=11)

    def color_for(level: int) -> str:
        if depth == 0:
            return "#fbbf24" if level == 0 else PALETTE["mist"]
        # propagate orange up to `depth` hops, fade with distance
        if level <= depth:
            t = level / max(depth, 1)
            # orange -> light orange -> mist
            return _blend(PALETTE["accent_warm"], "#fde68a", t)
        return PALETTE["mist"]

    edge_x, edge_y = [], []
    for s, t in sub_edges:
        if s in pos and t in pos:
            edge_x.extend([pos[s][0], pos[t][0], None])
            edge_y.extend([pos[s][1], pos[t][1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color=PALETTE["edge"], width=0.7),
                             hoverinfo="skip"))
    visible = sub_papers[sub_papers["pid"].isin(pos)].copy()
    colors = [color_for(levels.get(p, 99)) for p in visible["pid"]]
    sizes = [16 if levels.get(p, 99) == 0 else 9 for p in visible["pid"]]
    text = [f"{_s(t)[:80]}<br>level: {levels.get(p, '?')}-hop"
            for t, p in zip(visible["title"], visible["pid"])]
    fig.add_trace(go.Scatter(
        x=[pos[p][0] for p in visible["pid"]],
        y=[pos[p][1] for p in visible["pid"]],
        mode="markers",
        marker=dict(color=colors, size=sizes,
                    line=dict(color=PALETTE["navy"], width=0.4)),
        text=text, hoverinfo="text",
    ))
    fig.update_layout(showlegend=False, height=380,
                      margin=dict(l=0, r=0, t=20, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Diepte = {depth}. Bij depth=0 weet de start-node alleen "
        f"zichzelf. Bij depth=4 zijn bijna alle 2-hop buren bereikt - "
        f"voorbij dat punt verdwijnt onderscheidingsvermogen "
        f"(over-smoothing)."
    )


def _blend(hex_a: str, hex_b: str, t: float) -> str:
    """Linear interp between two hex colours, t in [0,1]."""
    a = hex_a.lstrip("#"); b = hex_b.lstrip("#")
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    r = int(ar + (br - ar) * t)
    g = int(ag + (bg - ag) * t)
    bl = int(ab + (bb - ab) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


# ----- 3.3 GCN layer -----
def _section_3_3_gcn_layer(art: dict):
    st.markdown("### 3.3 - Wat is een GCN-laag?")
    st.caption(
        "Een Graph Convolutional Network (GCN) past dezelfde formule per "
        "laag toe: feature-matrix x gewogen-adjacency x weight-matrix, "
        "gevolgd door een ReLU. Stapelen we lagen, dan verspreidt "
        "informatie verder door de graph - tot op het punt dat alles "
        "te uniform wordt (over-smoothing)."
    )
    with st.expander("Welke andere architecturen bestaan?"):
        st.markdown(
            "- **GCN** (Kipf & Welling 2017) - de canonieke choice, "
            "transductive, eenvoudig.\n"
            "- **GraphSAGE** (Hamilton 2017) - inductive, sample buren ipv "
            "alle te aggregeren. Schaalt naar grote graphs.\n"
            "- **GAT** (Velickovic 2018) - attention-weighted aggregation. "
            "Interpretabel maar duurder.\n"
            "- **GIN** (Xu 2018) - maximaal discriminatief; overkill voor "
            "deze schaal.\n\n"
            "Voor onze leerlab is GCN primary. GraphSAGE is een logische "
            "vervolg-stap als we naar grotere candidate-sets willen "
            "schalen."
        )
    st.markdown("**Over-smoothing demo (1-5 layers).**")
    metrics = art.get("metrics", {})
    sweep = metrics.get("depth_sweep", [])
    if not sweep:
        st.info(
            "Geen depth-sweep metrics gevonden. Run "
            "`python code/train_gnn_demo.py --depth-sweep` om de 1-5 "
            "layer recall-curve hier te plotten."
        )
        return
    df = pd.DataFrame(sweep)
    # Headline: recall@10% screened (the actually-meaningful metric).
    # Back-compat with old key 'final_test_recall' (= recall@100% = always 1.0)
    if "test_recall_at_10pct" in df.columns:
        fig2 = px.line(df, x="depth", y="test_recall_at_10pct",
                       markers=True,
                       title="Recall@10% screened vs. GCN depth (headline)")
        fig2.update_layout(xaxis_title="aantal layers",
                           yaxis_title="recall@10%", yaxis_range=[0, 1.05])
        st.plotly_chart(fig2, width="stretch")
    y_full = "screened_recall_at_100pct" if "screened_recall_at_100pct" in df.columns \
        else ("final_test_recall" if "final_test_recall" in df.columns else None)
    if y_full:
        fig = px.line(df, x="depth", y=y_full, markers=True,
                      title="Recall after 100% screened (= 1.0 by definition; "
                            "shown for completeness)")
        fig.update_layout(xaxis_title="aantal layers",
                          yaxis_title="recall (full screen)",
                          yaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Lezing: een verschil tussen depth 2 en depth 5 is het signaal "
        "dat we over-smoothing zien. Verwacht: 2-3 layers optimaal voor "
        "FORAS-grootte, daarna stagnatie of lichte daling."
    )


# ----- 3.4 GCN on MY data -----
def _section_3_4_gcn_on_my_data(art: dict, papers: pd.DataFrame):
    st.markdown("### 3.4 - Hoe ziet de GCN MIJN data?")
    st.caption(
        "Drie mini-secties: 2D-projection van de embeddings (UMAP/SVD), "
        "een willekeurige test-paper met zijn buren-bijdragen, en een "
        "'raad de score'-quiz."
    )
    embeddings = art.get("embeddings")
    scores = art.get("scores")
    if embeddings is None or scores is None:
        st.info(
            "Geen embeddings/scores. Run `python code/train_gnn_demo.py` "
            "om deze visualisaties te activeren."
        )
        return
    # 2D-projection (PCA op de 64-dim embeddings)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)
    plot_df = pd.DataFrame({
        "x": proj[:, 0], "y": proj[:, 1],
        "score": scores["score"].values,
        "y_true": scores["y_true"].values,
        "title": scores["title"].astype(str).str[:80].values,
        "year": scores["year"].values,
    })
    plot_df["label"] = plot_df["y_true"].map({1: "include", 0: "exclude"})
    sample_df = plot_df.sample(min(3000, len(plot_df)), random_state=42)
    fig = px.scatter(
        sample_df, x="x", y="y", color="label", opacity=0.55,
        hover_data=["title", "year", "score"],
        color_discrete_map={"include": PALETTE["cyan"],
                            "exclude": PALETTE["slate"]},
        title="2D-projection van de GCN-embeddings (PCA)",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=420, xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Aha: zie of includes (cyaan) en excludes (grijs) clusteren. "
        "Volledige scheiding niet verwacht, maar gradient zou zichtbaar "
        "moeten zijn in een goed-getrainde GCN."
    )

    # Pick-a-paper inspector
    st.markdown("**Pak een willekeurige test-paper, zie waar de score "
                "vandaan komt.**")
    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Random test-paper", key="gnn_random_paper"):
            st.session_state["gnn_paper_idx"] = int(np.random.randint(
                0, len(scores)
            ))
    with cols[1]:
        idx = st.session_state.get("gnn_paper_idx")
        if idx is None:
            st.caption("Klik 'Random test-paper' om een paper te selecteren.")
        else:
            row = scores.iloc[idx]
            st.markdown(
                f"**{row['title']}** ({int(row['year']) if pd.notna(row['year']) else '?'})"
                f" - score = `{float(row['score']):.3f}`, "
                f"y_true = `{int(row['y_true'])}`"
            )

    # Score-quiz
    st.markdown("**'Raad de score'-quiz.**")
    if st.button("Toon een random paper voor de quiz", key="gnn_quiz_btn"):
        ridx = int(np.random.randint(0, len(scores)))
        st.session_state["gnn_quiz_idx"] = ridx
    qidx = st.session_state.get("gnn_quiz_idx")
    if qidx is not None:
        row = scores.iloc[qidx]
        st.markdown(
            f"> *{row['title']}* ({int(row['year']) if pd.notna(row['year']) else '?'})"
        )
        guess = st.slider("Jouw gok voor de GCN-score (0=exclude, 1=include)",
                          min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                          key="gnn_quiz_guess")
        if st.button("Onthul echte score", key="gnn_quiz_reveal"):
            st.success(
                f"GCN-score: `{float(row['score']):.3f}` | "
                f"echte label: `{int(row['y_true'])}` | "
                f"jouw gok: `{guess:.2f}` "
                f"(verschil: {abs(float(row['score']) - guess):.2f})"
            )


# ----- 3.5 70/30 vs ASReview -----
def _section_3_5_seventy_thirty(art: dict):
    st.markdown("### 3.5 - 70/30 vergelijking met ASReview")
    st.caption(
        "Pendant van sectie 2.7 in de ASReview-tab. Deze tab schrijft de "
        "GNN-kolommen naar `outputs/comparison/sentinel_ranks.parquet` "
        "(via `train_gnn_demo.py`); tab 2 schrijft de ASReview-kolommen. "
        "Hieronder de huidige stand van zaken."
    )
    sentinel_ranks = ROOT / "outputs" / "comparison" / "sentinel_ranks.parquet"
    if not sentinel_ranks.exists():
        st.warning("`sentinel_ranks.parquet` bestaat nog niet. Run de "
                   "ASReview-tab eerst.")
        return
    df = pd.read_parquet(sentinel_ranks)
    split = st.radio(
        "Split-modus",
        options=["Modus A: ASReview-volgorde dicteert",
                 "Modus B: Random stratified 70/30"],
        key="gnn_split_mode",
    )
    split_key = ("modus_a_asreview_order" if split.startswith("Modus A")
                 else "modus_b_random_stratified")
    sub = df[df["split_mode"] == split_key].copy()
    focus = _focused_sentinel()
    if focus:
        sub = sub[sub["sentinel_id"] == focus]
        if len(sub) == 0:
            st.info(f"Geen rijen voor focus-sentinel `{focus}`.")
            return
    sub["delta_rank"] = (sub["asreview_rank"].astype("float") -
                         sub["gnn_rank"].astype("float"))
    show = sub[["sentinel_id", "asreview_rank", "asreview_recall_at_rank",
                "gnn_rank", "gnn_score", "delta_rank", "updated_at"]]
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "**Lezing:** `delta_rank > 0` betekent GCN-rang lager dan "
        "ASReview-rang = GCN vond het sentinel sneller. `delta_rank` is "
        "alleen geldig zodra beide kolommen ingevuld zijn (eerst de "
        "ASReview-tab runnen, dan deze tab)."
    )
    with st.expander("Methodologische noot"):
        st.markdown(
            "Wat we hier meten: voor elk van de 3 sentinels (Solomon, "
            "Kardiner, Southard) berekenen we hun rang in de candidates-"
            "pool nadat het model getraind is op de eerste 70% FORAS-"
            "labels. Modus A dicteert die 70% via ASReview's active-"
            "learning-volgorde; modus B kiest random-stratified.\n\n"
            "Waarom dit het sentinel-experiment uit Optie 1 ondersteunt: "
            "als de GCN-rang significant lager is dan de ASReview-rang "
            "voor sentinels die wel cross-edges hebben (Solomon = 22, "
            "Kardiner = 17), dan voegt citatie-structuur iets toe boven "
            "tekst-similarity alleen. Voor Southard (0 cites) verwachten "
            "we *geen* verschil - dat is de bovengrens (P2 in "
            "`context/hypothesis.md`)."
        )


# ----- 3.6 Alternative GNN applications -----
def _section_3_6_alternatives():
    st.markdown("### 3.6 - Alternatieve GNN-toepassingen")
    st.caption(
        "Zes manieren waarop GNNs in een literatuur-review nog meer "
        "kunnen doen. Geen experiment, wel materiaal voor brainstorm "
        "met begeleiders."
    )
    with st.expander("a) Link prediction - missende citaties vinden"):
        st.markdown(
            "Voorspel welke papers ELKAAR zouden moeten citeren maar "
            "dat niet doen. Train een GNN op de bestaande edges, dan "
            "score elk niet-bestaand pair. Top-K niet-bestaande edges "
            "= kandidaten voor missende citaties.\n\n"
            "```python\n"
            "from torch_geometric.nn import GCNConv\n"
            "# encoder produces node embeddings; decoder is "
            "dot-product\n"
            "score = (z[src] * z[tgt]).sum(dim=-1)\n"
            "```\n\n"
            "Voor de thesis: zou onthullen welke 'shell shock'-papers "
            "thematisch met FORAS-includes verbonden zijn maar geen "
            "directe citatie hebben."
        )
    with st.expander("b) Embedding-based retrieval"):
        st.markdown(
            "Train de GNN, gebruik node-embeddings als paper-vectoren, "
            "doe k-NN search door alle 14K papers. Geeft een 'find me "
            "papers similar to this one'-functie die zowel tekst als "
            "graph-context meeweegt.\n\n"
            "```python\n"
            "from sklearn.neighbors import NearestNeighbors\n"
            "nn = NearestNeighbors(n_neighbors=20).fit(embeddings)\n"
            "dist, idx = nn.kneighbors(embeddings[query_id:query_id+1])\n"
            "```"
        )
    with st.expander("c) Graph-explainability (PyG `torch_geometric.explain`)"):
        st.markdown(
            "Welke buren beïnvloedden deze prediction? PyG's "
            "`Explainer`-API genereert subgraph-explanations per node. "
            "Voor de thesis: laat zien dat een sentinel-paper hoog "
            "gerangschikt wordt vanwege specifieke FORAS-buren - "
            "tastbaarder bewijs van het citation-bridge-mechanisme.\n\n"
            "```python\n"
            "from torch_geometric.explain import Explainer\n"
            "exp = Explainer(model=model, algorithm=GNNExplainer())\n"
            "exp(x, edge_index, index=node_id)\n"
            "```"
        )
    with st.expander("d) Heterogeneous GNN (papers + authors + topics)"):
        st.markdown(
            "Nodes hoeven niet allemaal van hetzelfde type te zijn. "
            "Maak nodes voor papers, auteurs, topics, journals; "
            "edges met verschillende typen (paper-cites-paper, "
            "paper-by-author, paper-in-topic). Heterogene GNNs (HAN, "
            "RGCN) leren type-specifieke transformations.\n\n"
            "Voor onze thesis: zou auteurs als 'historische-PTSD-"
            "experts' in beeld brengen (Kardiner, Southard, "
            "Solomon zelf) als hub-nodes."
        )
    with st.expander("e) Temporal GNN - concept-drift over tijd"):
        st.markdown(
            "Voeg tijd toe als dimensie aan de graph. Train een Temporal "
            "GNN (e.g., TGN, JODIE) die per jaar de paper-embeddings "
            "update. Voor PTSD-historiografie ideaal: zie hoe de "
            "embedding van 'shell shock'-papers verschuift naarmate "
            "DSM-III in 1980 PTSD formaliseert.\n\n"
            "Niet voor 5-mei-dispatch, maar mooie follow-up-vraag voor "
            "begeleiders."
        )
    with st.expander("f) GCN als ASReview-classifier (hybride)"):
        st.markdown(
            "Plug de GCN als feature-extractor in ASReview's "
            "active-learning-loop: bij elke iteratie gebruikt ASReview "
            "GCN-scores in plaats van TF-IDF. Vereist "
            "`asreview-makita`-aanpassing of een custom classifier-"
            "plug-in. Belangrijk voor de hybride-baseline-claim van "
            "de thesis: text-baseline (ASReview) plus structuur "
            "(GCN) in één pipeline."
        )


# ----- 3.7 Glossary -----
def _section_3_7_glossary():
    if _in_demo_mode():
        return
    with st.expander("Glossary"):
        st.markdown(
            "- **GNN** - Graph Neural Network: neural net dat op graph-"
            "structured data werkt.\n"
            "- **GCN** - Graph Convolutional Network (Kipf & Welling 2017); "
            "transductive, eenvoudig, primary choice.\n"
            "- **GraphSAGE** - inductive, neighbor-sampling-based GNN.\n"
            "- **GAT** - Graph Attention Network; weighted aggregation.\n"
            "- **Node** - een paper in de citation-graph.\n"
            "- **Edge** - een citatie tussen twee papers.\n"
            "- **Undirected/Directed** - citaties zijn directed (A cites B); "
            "voor GCN convert je naar undirected (information flows beide kanten op).\n"
            "- **Message-passing** - per laag aggregeert een node "
            "informatie van zijn buren.\n"
            "- **Embedding** - vector-representatie per node die de model leert.\n"
            "- **Layer** - één GCN-conv-stap; 2 lagen = node ziet 2-hop omgeving.\n"
            "- **Depth** - aantal layers; meer != beter (over-smoothing).\n"
            "- **Over-smoothing** - bij te veel lagen worden alle node-embeddings "
            "te uniform, model verliest discriminatie.\n"
            "- **Transductive** - hele graph (incl. test-nodes) zichtbaar tijdens "
            "training; alleen labels gemaskt.\n"
            "- **Inductive** - test-nodes ongezien; model moet generaliseren.\n"
            "- **BCE-loss** - binary cross-entropy; standaard voor binaire "
            "classificatie.\n"
            "- **Class-weight** - mitigeert class-imbalance door minderheid "
            "zwaarder te wegen in loss.\n"
            "- **Recall** - fractie van echte includes die het model voorspelt "
            "als include.\n"
            "- **F1** - harmonisch gemiddelde van precision en recall.\n"
            "- **ROC-AUC** - area under receiver-operating-characteristic curve.\n"
            "- **TF-IDF** - term-frequentie weighted by inverse document frequency; "
            "klassieke text-feature.\n"
            "- **Bibliographic coupling** - twee papers delen een referentie -> "
            "gerelateerd.\n"
            "- **Co-citation** - twee papers worden samen geciteerd door een "
            "derde -> gerelateerd.\n"
            "- **Candidate** - paper uit 2.288-pool met historische PTSD-"
            "terminologie.\n"
            "- **FORAS** - de focal corpus van 14.764 papers (van de Schoot "
            "et al. 2025).\n"
            "- **FT-included** - paper inclusief na full-text-screening (172).\n"
            "- **TIAB-included** - paper inclusief na title+abstract-screening "
            "(568 in label_abstract_included).\n"
            "- **Sentinel** - één van Solomon 1993, Kardiner 1941, Southard 1920; "
            "anker voor de ASReview-vs-GCN-vergelijking.\n"
            "- **Hold-out** - 30% van FORAS dat niet meedoet in training.\n"
            "- **Link prediction** - voorspel of een edge bestaat tussen "
            "twee nodes.\n"
        )


# ============================================================
# TAB 2 - ASReview (NEW)
# ============================================================

# ----- module-level constants for ASReview -----
ELAS_BUNDLES = {
    "elas_u4": {
        "label": "elas_u4 (default; SVM + TF-IDF)",
        "needs_dory": False,
        "blurb": "Default ultra bundle. Fast. Good baseline. SVM classifier on TF-IDF features.",
    },
    "elas_u3": {
        "label": "elas_u3 (NB + TF-IDF)",
        "needs_dory": False,
        "blurb": "Classical Naive-Bayes ultra bundle. Robust on small/imbalanced sets.",
    },
    "elas_l2": {
        "label": "elas_l2 (multilingual-e5-large + SVM, needs asreview-dory)",
        "needs_dory": True,
        "blurb": "Multilingual transformer embeddings. Slow first run on CPU (10-30 min).",
    },
    "elas_h3": {
        "label": "elas_h3 (mxbai + SVM, needs asreview-dory)",
        "needs_dory": True,
        "blurb": "Heavy transformer-based bundle (mxbai). Slow on CPU; cache embeddings.",
    },
}

PRIOR_STRATEGIES = {
    "1+1 (default)": {"n_prior_included": 1, "n_prior_excluded": 1},
    "2+2": {"n_prior_included": 2, "n_prior_excluded": 2},
    "5+5": {"n_prior_included": 5, "n_prior_excluded": 5},
    "10+0 (only positives)": {"n_prior_included": 10, "n_prior_excluded": 0},
}


# ----- helpers: dataset prep -----
@st.cache_data(show_spinner=False)
def _foras_dataset_kpis():
    """Read raw FORAS CSV (semicolon-separated) and return KPI counts."""
    candidates_paths = [
        THESIS_DATA / "focas" / "PTSS_Data_Foras_2025-02-05.csv",
        THESIS_DATA / "PTSS_Data_Foras_2025-02-05.csv",
        THESIS_DATA / "foras" / "PTSS_Data_Foras_2025-02-05.csv",
    ]
    src = next((p for p in candidates_paths if p.exists()), None)
    if src is None:
        return {"src": None, "rows": None, "n_pos_tiab": None, "n_pos_ft": None}
    try:
        df = pd.read_csv(src, sep=";", low_memory=False, encoding="utf-8-sig")
        return {
            "src": str(src),
            "rows": len(df),
            "n_pos_tiab": int(df["label_included_TIAB"].fillna(0).astype(int).sum())
                if "label_included_TIAB" in df.columns else None,
            "n_pos_ft": int(df["label_included_FT"].fillna(0).astype(int).sum())
                if "label_included_FT" in df.columns else None,
        }
    except Exception as exc:
        return {"src": str(src), "rows": None, "n_pos_tiab": None,
                "n_pos_ft": None, "error": str(exc)}


def _asreview_available():
    try:
        out = subprocess.run(
            ["asreview", "--version"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return True, out.stdout.strip() or out.stderr.strip()
        return False, f"non-zero exit: {out.returncode}"
    except FileNotFoundError:
        return False, "not on PATH"
    except Exception as exc:
        return False, str(exc)


@st.cache_data(show_spinner="Inspecting .asreview project ...")
def _read_asreview_results(asreview_path: str) -> dict:
    """Read a .asreview ZIP and pull out per-record discovery order."""
    p = Path(asreview_path)
    if not p.exists():
        return {"error": f"not found: {asreview_path}"}
    try:
        with zipfile.ZipFile(p, "r") as z:
            names = z.namelist()
            project_meta = {}
            for n in names:
                if n.endswith("project.json"):
                    project_meta = json.loads(z.read(n).decode("utf-8"))
                    break
            sql_names = [n for n in names if n.endswith("results.sql")]
            if not sql_names:
                return {"error": "no results.sql in project zip", "names": names[:20]}
            tmp = ROOT / ".cache_asreview_results.sql"
            tmp.write_bytes(z.read(sql_names[0]))
            con = sqlite3.connect(tmp)
            try:
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table'", con
                )["name"].tolist()
                results_df = None
                for tbl in ("results", "record_results", "labels"):
                    if tbl in tables:
                        results_df = pd.read_sql_query(f"SELECT * FROM {tbl}", con)
                        break
                if results_df is None and tables:
                    results_df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", con)
            finally:
                con.close()
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return {
                "project": project_meta,
                "tables": tables,
                "records": results_df,
                "n_records": int(len(results_df)) if results_df is not None else None,
            }
    except Exception as exc:
        return {"error": str(exc)}


def _list_asreview_files():
    if not ASREVIEW_RUNS.exists():
        return []
    return sorted(ASREVIEW_RUNS.glob("*.asreview"))


def _compute_recall_curve(records):
    """Compute (n_screened, recall) curve from a results-table DataFrame."""
    if records is None or len(records) == 0:
        return None
    df = records.copy()
    cols = {c.lower(): c for c in df.columns}
    label_col = next((cols[c] for c in cols if c in
                      {"label", "included", "labels", "is_relevant", "relevant"}),
                     None)
    order_col = next((cols[c] for c in cols if c in
                      {"query_i", "query", "step", "order", "sort_order",
                       "review_step"}),
                     None)
    if label_col is None:
        return None
    if order_col is not None:
        df = df.sort_values(order_col)
    df = df.reset_index(drop=True)
    df["_label_int"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    df["_n_screened"] = np.arange(1, len(df) + 1)
    total_pos = int(df["_label_int"].sum())
    if total_pos == 0:
        return None
    df["_recall"] = df["_label_int"].cumsum() / total_pos
    return df[["_n_screened", "_recall", "_label_int"]].rename(
        columns={"_n_screened": "n_screened", "_recall": "recall",
                 "_label_int": "is_relevant"}
    )


def _wss_at_recall(curve, target_recall: float):
    """Work Saved over Sampling at a given recall level."""
    if curve is None or len(curve) == 0:
        return None
    hits = curve[curve["recall"] >= target_recall]
    if len(hits) == 0:
        return None
    n_at = hits.iloc[0]["n_screened"]
    N = len(curve)
    return float((1 - n_at / N) - (1 - target_recall))


# ----- helpers: sentinel-ranks parquet (shared with GNN tab) -----

SENTINEL_RANKS_SCHEMA = {
    "sentinel_id": "object",
    "split_mode": "object",
    "dataset": "object",
    "bundle": "object",
    "seed": "Int64",
    "asreview_rank": "Int64",
    "asreview_recall_at_rank": "float64",
    "gnn_rank": "Int64",
    "gnn_score": "float64",
    "rewrite_used": "boolean",
    "updated_at": "object",
}


def _ensure_sentinel_ranks_skeleton():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    if SENTINEL_RANKS.exists():
        try:
            return pd.read_parquet(SENTINEL_RANKS)
        except Exception:
            pass
    rows = []
    sentinels = ["solomon_1993", "kardiner_1941", "southard_1920"]
    for sid in sentinels:
        for split in ("modus_a_asreview_order", "modus_b_random_stratified"):
            rows.append({
                "sentinel_id": sid,
                "split_mode": split,
                "dataset": "FORAS+candidates",
                "bundle": None,
                "seed": pd.NA,
                "asreview_rank": pd.NA,
                "asreview_recall_at_rank": np.nan,
                "gnn_rank": pd.NA,
                "gnn_score": np.nan,
                "rewrite_used": True,
                "updated_at": None,
            })
    df = pd.DataFrame(rows)
    for col, dtype in SENTINEL_RANKS_SCHEMA.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                pass
    df.to_parquet(SENTINEL_RANKS, index=False)
    return df


def _upsert_asreview_rank(sentinel_id, split_mode, *, dataset, bundle, seed,
                          rank, recall_at_rank):
    df = _ensure_sentinel_ranks_skeleton()
    mask = (df["sentinel_id"] == sentinel_id) & (df["split_mode"] == split_mode)
    if not mask.any():
        df = pd.concat([df, pd.DataFrame([{
            "sentinel_id": sentinel_id, "split_mode": split_mode,
            "dataset": dataset, "bundle": bundle, "seed": seed,
            "asreview_rank": rank, "asreview_recall_at_rank": recall_at_rank,
            "gnn_rank": pd.NA, "gnn_score": np.nan,
            "rewrite_used": True,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }])], ignore_index=True)
    else:
        df.loc[mask, "dataset"] = dataset
        df.loc[mask, "bundle"] = bundle
        df.loc[mask, "seed"] = seed
        df.loc[mask, "asreview_rank"] = rank
        df.loc[mask, "asreview_recall_at_rank"] = recall_at_rank
        df.loc[mask, "rewrite_used"] = True
        df.loc[mask, "updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    df.to_parquet(SENTINEL_RANKS, index=False)


# ----- helpers: sentinel rewrites loader -----
@st.cache_data(show_spinner=False)
def _load_sentinel_bundles():
    out = {}
    if not SENTINEL_DIR.exists():
        return out
    for sid in ("solomon_1993", "kardiner_1941", "southard_1920"):
        bundle = {}
        for kind in ("original", "rewrite"):
            p = SENTINEL_DIR / f"{sid}_{kind}.json"
            if p.exists():
                try:
                    bundle[kind] = json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    bundle[kind] = {"error": str(exc)}
        if bundle:
            out[sid] = bundle
    return out


# ----- top-level renderer -----
def render_asreview_tab(papers, cand, cross):
    """Tab 2 - ASReview learning tool + baseline."""
    st.markdown(
        "<div class='foras-title'>ASReview - active learning tool + baseline</div>"
        "<div class='foras-sub'>Hoe ASReview FORAS in versneld tempo screent, "
        "wat dat oplevert op de candidate-pool met historische terminologie, en "
        "hoe het zich verhoudt tot de GNN op de 70/30 split.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='explainer'><h4>Wat zie je hier?</h4>"
        "Negen sub-secties die ASReview uitleggen (2.1) en concreet toepassen "
        "op FORAS (2.2-2.7), plus alternatieve toepassingen (2.8) en glossary "
        "(2.9). Doel: een leertool die op zichzelf staat - na deze tab weet je "
        "wat active learning is en wat het op MIJN dataset doet.</div>",
        unsafe_allow_html=True,
    )

    _section_2_1_what_is_asreview()
    st.markdown("---")
    _section_2_2_dataset_overview()
    st.markdown("---")
    _section_2_3_simulation_runner()
    st.markdown("---")
    _section_2_4_plot_explorer()
    st.markdown("---")
    _section_2_5_compare_mode()
    st.markdown("---")
    _section_2_6_discovery_scatter(cand, cross)
    st.markdown("---")
    _section_2_7_seventy_thirty_split()
    st.markdown("---")
    _section_2_8_alternative_applications()
    st.markdown("---")
    _section_2_9_glossary()


# ----- 2.1 What is ASReview? -----
def _section_2_1_what_is_asreview():
    st.markdown("### 2.1 - Wat is ASReview?")
    st.caption(
        "ASReview = Active learning for Systematic Reviews. Een open-source "
        "tool die menselijke screenings versnelt door iteratief het "
        "meest-waarschijnlijk-relevante record als volgende voor te leggen."
    )
    with st.expander("Hoe werkt active learning? (de cyclus in 5 stappen)",
                     expanded=False):
        st.markdown(
            "1. **Prior knowledge.** Je labelt een paar records met de hand "
            "als 'relevant' of 'niet relevant' (bijvoorbeeld 1 + 1).\n"
            "2. **Train.** ASReview traint een classifier (bijv. SVM op "
            "TF-IDF) op die paar records.\n"
            "3. **Query.** De classifier kijkt naar alle ongelabelde records "
            "en selecteert er een - meestal de meest-zekere include "
            "(`max`-querier) of de meest-onzekere (`uncertainty`).\n"
            "4. **Label.** Je labelt dat record (of in een simulatie: het "
            "echte label uit de CSV wordt 'gerevealed').\n"
            "5. **Herhaal.** Stap 2-4 totdat je stop-criterium bereikt is "
            "(meestal: 'alle relevante records gevonden' of een vast "
            "fractie-gescreend).\n\n"
            "Door deze loop hoef je in de praktijk niet alle records te "
            "lezen - vaak vind je 95% van de includes na 30-50% screening. "
            "Dat verschil heet **Work Saved over Sampling** (WSS)."
        )
    with st.expander("ELAS-bundels - welke modellen?", expanded=False):
        for k, v in ELAS_BUNDLES.items():
            st.markdown(f"- **`{k}`** - {v['label']}. {v['blurb']}")
        st.markdown(
            "Een bundle is een vaste combinatie van feature-extractor + "
            "classifier + querier + balancer. Selecteer met "
            "`asreview simulate ... --ai elas_h3`."
        )
    with st.expander("Waarom is ASReview interessant voor mijn thesis?",
                     expanded=False):
        st.markdown(
            "FORAS gebruikt ASReview als 9e quality-check laag in haar "
            "search-strategie (zie van de Schoot et al. 2025, sec. 2.2). "
            "Voor mijn thesis is ASReview de **tekst-baseline** waar de "
            "GNN tegen wordt afgezet: als de GNN op de 30%-hold-out geen "
            "verbetering geeft over ASReview's ranking, dan voegt "
            "citatie-structuur niets toe boven tekst-similarity."
        )


# ----- 2.2 Dataset overview -----
def _section_2_2_dataset_overview():
    st.markdown("### 2.2 - Dataset-overzicht (FORAS)")
    st.caption(
        "FORAS-csv is 10.595 records. Twee label-niveaus: **TIAB** (na "
        "title+abstract-screening) en **FT** (na full-text-screening). "
        "TIAB is grover maar heeft meer signaal (260 vs 131 includes)."
    )
    kpis = _foras_dataset_kpis()
    if kpis.get("src") is None:
        st.warning(
            "Kan FORAS-CSV niet vinden. Verwacht in `data/focas/"
            "PTSS_Data_Foras_2025-02-05.csv` (relatief tov de thesis-folder)."
        )
        return
    if kpis.get("rows") is None:
        st.error(f"Kan FORAS-CSV niet inlezen: {kpis.get('error')}")
        return
    label_choice = st.radio(
        "Welk label-niveau gebruiken we als 'relevant'?",
        options=["TIAB (primary)", "FT (robustness check)"],
        horizontal=True,
        help="TIAB is aanbevolen voor de 70/30 vergelijking - meer power.",
        key="asreview_label_choice",
    )
    label_key = "n_pos_tiab" if label_choice.startswith("TIAB") else "n_pos_ft"
    n_pos = kpis.get(label_key)
    cols = st.columns(4)
    with cols[0]:
        st.metric("Records", f"{kpis['rows']:,}",
                  help="Aantal rijen in de FORAS-CSV.")
    with cols[1]:
        st.metric("TIAB-includes", f"{kpis.get('n_pos_tiab') or '-'}",
                  help="Records met `label_included_TIAB == 1`.")
    with cols[2]:
        st.metric("FT-includes", f"{kpis.get('n_pos_ft') or '-'}",
                  help="Records met `label_included_FT == 1`.")
    with cols[3]:
        base = (n_pos / kpis['rows']) if (n_pos and kpis['rows']) else 0
        st.metric("Base-rate", f"{base*100:.2f}%",
                  help="Includes / records voor het gekozen label.")
    st.caption(f"Source: `{kpis['src']}`")


# ----- 2.3 Simulation runner -----
def _section_2_3_simulation_runner():
    st.markdown("### 2.3 - Simulation runner")
    st.caption(
        "Draai een ASReview-simulatie op FORAS (of op FORAS + 2.288 "
        "candidates). Output: `.asreview`-projectbestand in "
        "`outputs/asreview_runs/`."
    )
    available, version = _asreview_available()
    if not available:
        st.warning(
            f"De `asreview` CLI is niet bereikbaar in deze omgeving "
            f"(`{version}`). Installeer met:\n\n"
            "```bash\n"
            "pip install asreview asreview-insights asreview-makita "
            "asreview-datatools asreview-dory\n"
            "```\n"
        )
    else:
        st.success(f"ASReview CLI gevonden: `{version}`")

    with st.form("asreview_sim_form"):
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.selectbox(
                "Dataset",
                options=["FORAS", "FORAS + 2.288 candidates"],
                help=("FORAS+candidates: voegt 2.288 historische-terminologie "
                      "kandidaten toe; sentinels (3) krijgen "
                      "`is_candidate=True` en `included=1` om recall@k te "
                      "kunnen meten."),
            )
            bundle = st.selectbox(
                "AI bundle",
                options=list(ELAS_BUNDLES.keys()),
                format_func=lambda k: ELAS_BUNDLES[k]["label"],
                help="elas_l2 / elas_h3 vereisen `asreview-dory`. "
                     "Eerste run met transformer is 10-30 min op CPU.",
            )
        with col2:
            seed = st.number_input(
                "Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                help="Same `--seed` value -> reproduceerbare run.",
            )
            prior_strategy = st.selectbox(
                "Prior records",
                options=list(PRIOR_STRATEGIES.keys()),
                help="Hoeveel records geef je ASReview als startpunt?",
            )
        n_stop = st.checkbox(
            "Door simuleren tot 100% (`--n-stop -1`)", value=True,
            help=("Belangrijk voor recall@k op kandidaten: default stopt "
                  "ASReview zodra alle includes gevonden zijn."),
        )
        submitted = st.form_submit_button(
            "Draai simulatie", disabled=not available
        )
    if submitted and available:
        priors = PRIOR_STRATEGIES[prior_strategy]
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        ds_slug = "foras" if dataset == "FORAS" else "foras_plus_candidates"
        out_path = ASREVIEW_RUNS / f"{ds_slug}_{bundle}_seed{seed}_{ts}.asreview"
        ASREVIEW_RUNS.mkdir(parents=True, exist_ok=True)
        st.info(
            f"Geplande run: `{out_path.name}`\n\n"
            f"Bundle: `{bundle}` | Seed: {seed} | "
            f"Priors: {priors['n_prior_included']}+"
            f"{priors['n_prior_excluded']} | "
            f"n-stop: {'-1' if n_stop else 'default'}"
        )
        st.warning(
            "**Implementatie-noot.** De `asreview simulate` subprocess-call "
            "is geparametriseerd, maar de input-CSV "
            f"(`outputs/asreview_runs/{ds_slug}.csv`) moet eerst gebouwd "
            "zijn. Voor de eerste run: zie de reproduce-commando's in "
            "`outputs/asreview_runs/README.md`. Daarna kan deze knop er "
            "incrementele runs op zetten."
        )

    runs = _list_asreview_files()
    st.markdown(f"**Bestaande runs in `outputs/asreview_runs/`** ({len(runs)})")
    if not runs:
        st.caption(
            "Nog geen `.asreview`-bestanden. Eerste run: zie reproduce-"
            "commando's in `outputs/asreview_runs/README.md`."
        )
    else:
        run_meta = []
        for p in runs:
            stat = p.stat()
            run_meta.append({
                "filename": p.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.utcfromtimestamp(stat.st_mtime)
                            .isoformat(timespec="seconds") + "Z",
            })
        st.dataframe(pd.DataFrame(run_meta), width="stretch",
                     hide_index=True)


# ----- 2.4 Plot explorer -----
def _section_2_4_plot_explorer():
    st.markdown("### 2.4 - Plot-explorer (recall + WSS)")
    st.caption(
        "Selecteer een `.asreview`-bestand en bekijk de recall-curve. "
        "X = aantal records gescreend, Y = fractie van de includes gevonden."
    )
    runs = _list_asreview_files()
    if not runs:
        st.info("Geen `.asreview`-bestanden gevonden. Draai eerst een "
                "simulatie in sectie 2.3 of via de CLI.")
        return
    run_choice = st.selectbox(
        "Run", options=[p.name for p in runs],
        help="Kies een .asreview-bestand uit `outputs/asreview_runs/`.",
        key="asreview_plot_run",
    )
    sel = next(p for p in runs if p.name == run_choice)
    info = _read_asreview_results(str(sel))
    if "error" in info:
        st.error(f"Kan dit bestand niet lezen: {info['error']}")
        return
    curve = _compute_recall_curve(info.get("records"))
    if curve is None:
        st.warning(
            "Kon geen recall-curve afleiden uit de results-tabel. "
            f"Tabellen gevonden: {info.get('tables')}. "
            "Run `asreview plot recall <file>.asreview` als CLI-fallback."
        )
        return
    fig = px.line(curve, x="n_screened", y="recall",
                  title=f"Recall curve - {sel.name}")
    fig.update_layout(yaxis_range=[0, 1.02], xaxis_title="# records gescreend",
                      yaxis_title="recall (fractie includes gevonden)")
    st.plotly_chart(fig, width="stretch")
    cols = st.columns(3)
    with cols[0]:
        wss95 = _wss_at_recall(curve, 0.95)
        st.metric("WSS@95",
                  f"{wss95:.3f}" if wss95 is not None else "n/a",
                  help="Work Saved over Sampling op 95% recall. Hoger = beter.")
    with cols[1]:
        recall_10 = float(curve[curve["n_screened"] / len(curve) >= 0.10]
                          ["recall"].iloc[0]) if len(curve) > 0 else 0
        st.metric("recall@10%", f"{recall_10*100:.1f}%",
                  help="Welk percentage includes is gevonden na 10% gescreend.")
    with cols[2]:
        recall_50 = float(curve[curve["n_screened"] / len(curve) >= 0.50]
                          ["recall"].iloc[0]) if len(curve) > 0 else 0
        st.metric("recall@50%", f"{recall_50*100:.1f}%",
                  help="Welk percentage includes is gevonden na 50% gescreend.")
    st.caption(
        "Lezing: een steile curve in het eerste 10-20% screening = veel "
        "early discovery. Een vlakke curve = model selecteert random."
    )


# ----- 2.5 Compare mode -----
def _section_2_5_compare_mode():
    st.markdown("### 2.5 - Compare-modus")
    st.caption(
        "Vergelijk meerdere `.asreview`-runs over elkaar. Bedoeld om "
        "bundels (`elas_u4` vs `elas_h3`) of seeds (5x dezelfde bundle "
        "met seed 42-46) te benchmarken."
    )
    runs = _list_asreview_files()
    if len(runs) < 2:
        st.info(f"Heb je minstens 2 runs nodig. Gevonden: {len(runs)}.")
        return
    selected = st.multiselect(
        "Runs om te vergelijken (max 6)",
        options=[p.name for p in runs],
        default=[p.name for p in runs[:2]],
        help="Tip: groepeer per bundle om gemiddelde curves te plotten.",
        key="asreview_compare_runs",
    )
    selected = selected[:6]
    if len(selected) < 2:
        return
    fig = go.Figure()
    metric_rows = []
    for name in selected:
        path = next(p for p in runs if p.name == name)
        info = _read_asreview_results(str(path))
        curve = _compute_recall_curve(info.get("records"))
        if curve is None:
            continue
        fig.add_trace(go.Scatter(
            x=curve["n_screened"], y=curve["recall"],
            mode="lines", name=name,
        ))
        metric_rows.append({
            "run": name,
            "WSS@95": _wss_at_recall(curve, 0.95),
            "WSS@99": _wss_at_recall(curve, 0.99),
        })
    fig.update_layout(yaxis_range=[0, 1.02], title="Recall curves (compare)",
                      xaxis_title="# records gescreend", yaxis_title="recall")
    st.plotly_chart(fig, width="stretch")
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), width="stretch",
                     hide_index=True)


# ----- 2.6 Candidate discovery scatter -----
def _section_2_6_discovery_scatter(cand, cross):
    st.markdown("### 2.6 - Candidate discovery-scatter")
    st.caption(
        "Voor de FORAS+candidates-runs: per gevonden record een punt op een "
        "scatter. X = % gescreend toen gevonden; Y = ranked positie binnen "
        "de positives; kleur = `is_candidate=False/True/sentinel`. Vraag: "
        "zitten de sentinels rechts (laat gevonden) -> dan complementeert "
        "de GNN potentieel via citatie-structuur."
    )
    runs = [p for p in _list_asreview_files()
            if "candidates" in p.name.lower() or "plus" in p.name.lower()]
    if not runs:
        st.info(
            "Nog geen FORAS+candidates-run aanwezig. Run sectie 2.3 met "
            "dataset = 'FORAS + 2.288 candidates' of zie reproduce-"
            "commando's in `outputs/asreview_runs/README.md`."
        )
        st.markdown(
            "**Wat verwachten we?** Sentinels die in de candidates-pool "
            "zitten met `cited_by_foras_any > 0` (Solomon 1993 = 22; "
            "Kardiner 1941 = 17) zijn via tekst-similarity moeilijk maar "
            "via citation-graph wel vindbaar. Southard 1920 (0 cites) "
            "demonstreert de bovengrens van citatie-structuur "
            "(P2 in `context/hypothesis.md`)."
        )
        return
    run_choice = st.selectbox(
        "Welke FORAS+candidates run?",
        options=[p.name for p in runs],
        key="asreview_discovery_run",
    )
    sel = next(p for p in runs if p.name == run_choice)
    info = _read_asreview_results(str(sel))
    curve = _compute_recall_curve(info.get("records"))
    if curve is None:
        st.warning("Kan recall-curve niet afleiden voor deze run.")
        return
    st.warning(
        "**Per-record discovery-scatter** vereist een join van de "
        "results-table op de input-CSV (om `is_candidate` en sentinel-"
        "markeringen op te halen). Die join wordt opgebouwd in een "
        "vervolg-iteratie zodra echte FORAS+candidates-runs aanwezig zijn."
    )
    cols = st.columns(3)
    with cols[0]:
        st.metric("WSS@95", f"{_wss_at_recall(curve, 0.95):.3f}"
                  if _wss_at_recall(curve, 0.95) is not None else "n/a")
    with cols[1]:
        n_includes = int(curve["is_relevant"].sum())
        st.metric("Includes in run", f"{n_includes:,}")
    with cols[2]:
        st.metric("Records totaal", f"{len(curve):,}")


# ----- 2.7 70/30 split + sentinel ranks -----
def _section_2_7_seventy_thirty_split():
    st.markdown("### 2.7 - 70/30 vergelijking met de GNN-tab")
    st.caption(
        "Sub-toggle voor split-modus. Schrijft naar de gedeelde parquet "
        "`outputs/comparison/sentinel_ranks.parquet`. Tab 3 vult later de "
        "GNN-kolommen aan."
    )
    split_mode = st.radio(
        "Hoe verdelen we de 70/30 split?",
        options=[
            "Modus A: ASReview-volgorde dicteert",
            "Modus B: Random stratified 70/30",
        ],
        help=(
            "**Modus A**: run ASReview tot 70% gelabeld is via active "
            "learning; resterende 30% (= records die ASReview nog niet "
            "had geselecteerd) is hold-out.\n\n"
            "**Modus B**: random split, gestratificeerd op label."
        ),
        key="asreview_split_mode",
    )
    bundles = _load_sentinel_bundles()
    if not bundles:
        st.warning(
            "Geen sentinel-rewrites gevonden in "
            "`outputs/sentinel_rewrites/`. Run dispatch-prompt 3 eerst."
        )
        return
    df = _ensure_sentinel_ranks_skeleton()
    split_key = ("modus_a_asreview_order" if split_mode.startswith("Modus A")
                 else "modus_b_random_stratified")
    sub = df[df["split_mode"] == split_key].copy()
    focus = _focused_sentinel()
    if focus:
        sub = sub[sub["sentinel_id"] == focus]
        if len(sub) == 0:
            st.info(f"Geen rijen voor focus-sentinel `{focus}`.")
            return
    sub["sentinel_label"] = sub["sentinel_id"].map(
        {sid: f"{sid} ({d.get('rewrite', {}).get('difficulty', '?')})"
         for sid, d in bundles.items()}
    )
    show = sub[[
        "sentinel_label", "asreview_rank", "asreview_recall_at_rank",
        "gnn_rank", "gnn_score", "bundle", "seed", "updated_at"
    ]].rename(columns={"sentinel_label": "sentinel"})
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "**Hoe te lezen:** zodra een ASReview-run met dataset = "
        "FORAS+candidates klaar is, vult deze tab `asreview_rank` (positie "
        "van het sentinel-record in de active-learning-volgorde) en "
        "`asreview_recall_at_rank` (recall behaald op het moment dat het "
        "sentinel werd gevonden). Tab 3 (GNN) vult daarna `gnn_rank` en "
        "`gnn_score`. Negative delta (`gnn_rank < asreview_rank`) = GNN "
        "vond het sentinel sneller."
    )

    # --- Hook: extract sentinel ranks from a FORAS+candidates .asreview run ---
    runs = [p for p in _list_asreview_files()
            if "candidates" in p.name.lower() or "plus" in p.name.lower()]
    if runs:
        st.markdown("**Auto-fill ASReview-rangen uit een FORAS+candidates-run**")
        cols_form = st.columns([3, 1, 1])
        with cols_form[0]:
            run_choice = st.selectbox(
                "Welke run lezen?", options=[p.name for p in runs],
                key="asreview_27_run_choice",
            )
        with cols_form[1]:
            seed_in = st.number_input(
                "Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                key="asreview_27_seed",
            )
        with cols_form[2]:
            bundle_in = st.selectbox(
                "Bundle", options=list(ELAS_BUNDLES.keys()),
                key="asreview_27_bundle",
            )
        if st.button("Bereken & schrijf rangen", key="asreview_27_compute"):
            sel = next(p for p in runs if p.name == run_choice)
            info = _read_asreview_results(str(sel))
            curve = _compute_recall_curve(info.get("records"))
            if curve is None:
                st.error("Kan recall-curve niet afleiden uit deze run.")
            else:
                # Best-effort: match sentinel-records by openalex_id or title
                # in the results-table, then derive their rank-position.
                rec = info.get("records")
                bundles_local = _load_sentinel_bundles()
                SENTINEL_OPENALEX = {
                    "solomon_1993":  "W1897891557",
                    "kardiner_1941": "W2496837584",
                    "southard_1920": "W4229501150",
                }
                lower_cols = {c.lower(): c for c in rec.columns}
                title_col = next((lower_cols[c] for c in lower_cols
                                  if c in {"title", "primary_title"}), None)
                id_col = next((lower_cols[c] for c in lower_cols
                               if c in {"openalex_id", "record_id", "id"}), None)
                wrote = 0
                for sid, oa in SENTINEL_OPENALEX.items():
                    bundle_data = bundles_local.get(sid, {})
                    title = (bundle_data.get("rewrite") or {}).get("title", "")
                    rank = None
                    if id_col is not None:
                        hit = rec[rec[id_col].astype(str).str.upper().str.contains(
                            oa, na=False, regex=False)]
                        if len(hit):
                            rank = int(hit.index[0]) + 1  # 1-indexed
                    if rank is None and title_col is not None and title:
                        hit = rec[rec[title_col].astype(str).str.contains(
                            title[:40], case=False, na=False, regex=False)]
                        if len(hit):
                            rank = int(hit.index[0]) + 1
                    rec_at = (
                        float(curve[curve["n_screened"] == rank]["recall"].iloc[0])
                        if rank is not None and rank <= len(curve) else None
                    )
                    _upsert_asreview_rank(
                        sid, split_key,
                        dataset="FORAS+candidates",
                        bundle=bundle_in, seed=int(seed_in),
                        rank=rank, recall_at_rank=rec_at,
                    )
                    if rank is not None:
                        wrote += 1
                st.success(
                    f"Wrote {wrote}/3 sentinel-rangen voor {split_key} "
                    f"(bundle={bundle_in}, seed={seed_in}). Refresh om de "
                    f"tabel hierboven bij te werken."
                )
    st.markdown(
        "**Plan voor begeleiders:** als de GNN sentinels hoger rangschikt "
        "dan ASReview, is dat een proof-of-concept voor het sentinel-"
        "experiment uit `context/opties/optie_1_foras_insert.md`. Eerlijk "
        "null result is ook OK - waardevol als bewijs dat citation-"
        "structuur niets toevoegt boven tekst-similarity."
    )


# ----- 2.8 Alternative applications -----
def _section_2_8_alternative_applications():
    st.markdown("### 2.8 - Alternatieve toepassingen")
    st.caption(
        "Vijf manieren om ASReview verder in te zetten - geen experiment, "
        "wel achtergrond voor brainstorm met begeleiders."
    )
    with st.expander("a) Live screening (human-in-the-loop)"):
        st.markdown(
            "Met `asreview lab` start je een Flask-server (default poort "
            "5000) waarin een mens de query-records een voor een labelt. "
            "Geschikt voor een pilot waarin Chiel en een begeleider 200 "
            "FORAS-records co-screenen om de ASReview-baseline te "
            "valideren.\n\n"
            "```bash\nasreview lab\n```"
        )
    with st.expander("b) Custom features (sBERT / dory-embeddings)"):
        st.markdown(
            "Buiten de vier ELAS-bundels kan je via `asreview-dory` eigen "
            "embedding-modellen kiezen (multilingual-e5-large, mxbai, "
            "sbert). Voor de thesis interessant: pre-compute embeddings "
            "een keer en hergebruik over alle simulation-runs.\n\n"
            "```bash\nasreview simulate dataset.csv "
            "--feature_extractor multilingual-e5-large "
            "--classifier svm -o run.asreview\n```"
        )
    with st.expander("c) Multi-classifier ensemble"):
        st.markdown(
            "Bouw een ensemble met `--classifier nb` en `--classifier svm` "
            "in twee runs, neem het maximum van de scores per record als "
            "ensemble-score. Marginaal beter dan beste single classifier "
            "op FORAS volgens makita-multimodel benchmarks."
        )
    with st.expander("d) Makita batch-experiments"):
        st.markdown(
            "`asreview makita template multimodel` genereert een Makefile "
            "die N modellen x M datasets x R seeds in batch draait. Ideaal "
            "voor robustness-checks (bv. 5 seeds per bundle om standaard-"
            "deviatie van WSS@95 te schatten).\n\n"
            "```bash\nmkdir benchmark && cd benchmark && mkdir data\n"
            "cp ../outputs/asreview_runs/foras_asreview.csv data/\n"
            "asreview makita template multimodel\n"
            "make sim && make plot\n```"
        )
    with st.expander("e) Co-screening met begeleiders"):
        st.markdown(
            "ASReview LAB kan multiple users hosten via een lokaal "
            "auth-tool (basic). Voor de thesis interessant als check: "
            "vraag een begeleider 50-100 borderline-records dubbel te "
            "screenen, gebruik de inter-rater agreement als externe "
            "validatie van het ASReview-priorisatie-mechanisme."
        )


# ----- 2.9 Glossary -----
def _section_2_9_glossary():
    if _in_demo_mode():
        return
    with st.expander("Glossary"):
        st.markdown(
            "- **ASReview** - Active learning for Systematic Reviews.\n"
            "- **Active learning** - iteratief proces waarin model vraagt welk record als volgende.\n"
            "- **Simulation** - actieve-leer-loop op gelabelde data.\n"
            "- **Prior records** - vooraf gelabelde records.\n"
            "- **Recall** - fractie echte includes gevonden.\n"
            "- **WSS** - Work Saved over Sampling.\n"
            "- **ATD** - Average Time-to-Discovery.\n"
            "- **ERF** - Extra Relevant Found.\n"
            "- **TF-IDF** - woord-frequentie-vectorizatie.\n"
            "- **SVM / NB** - Support Vector Machine / Naive Bayes.\n"
            "- **ELAS-bundle** - preset feature+classifier+querier combo.\n"
            "- **Querier** - kiest volgende record (max, uncertainty, random).\n"
            "- **Balancer** - houdt class-balance in train-set.\n"
            "- **Candidate** - paper uit 2.288-pool.\n"
            "- **Injection** - candidates toevoegen aan FORAS-CSV.\n"
            "- **Sentinel** - Solomon 1993 / Kardiner 1941 / Southard 1920.\n"
            "- **Hold-out** - 30% test-set."
        )


# ============================================================
# Global control panel
# ============================================================

VIEW_MODES = {
    "Leerlab": "All expanders + glossaries.",
    "Demo":    "Strakke kale view; geen glossaries.",
}
SENTINEL_CHOICES = ("All", "solomon_1993", "kardiner_1941", "southard_1920")


def _in_demo_mode() -> bool:
    return st.session_state.get("view_mode", "Leerlab") == "Demo"


def _focused_sentinel():
    s = st.session_state.get("focus_sentinel", "All")
    return None if s == "All" else s


def _render_global_sidebar():
    with st.sidebar:
        st.markdown("### Control panel")
        st.session_state.setdefault("view_mode", "Leerlab")
        st.session_state.setdefault("focus_sentinel", "All")
        st.session_state.setdefault("bookmarks", {})
        st.radio("View mode", options=list(VIEW_MODES.keys()),
                 help=" / ".join(f"{k}: {v}" for k, v in VIEW_MODES.items()),
                 key="view_mode")
        st.selectbox("Focus sentinel", options=SENTINEL_CHOICES,
                     key="focus_sentinel")
        with st.expander("Bookmarks"):
            new_name = st.text_input("Bookmark name", placeholder="bv. demo-solomon",
                                     key="bookmark_name")
            cols_bm = st.columns(2)
            with cols_bm[0]:
                if st.button("Save", key="bookmark_save"):
                    if new_name.strip():
                        st.session_state["bookmarks"][new_name.strip()] = {
                            "view_mode": st.session_state["view_mode"],
                            "focus_sentinel": st.session_state["focus_sentinel"],
                        }
                        st.success(f"Saved '{new_name}'")
            with cols_bm[1]:
                names = list(st.session_state["bookmarks"].keys())
                if names:
                    pick = st.selectbox("Load", options=["-"] + names,
                                        key="bookmark_load_pick")
                    if pick != "-":
                        if st.button("Apply", key="bookmark_apply"):
                            bm = st.session_state["bookmarks"][pick]
                            st.session_state["view_mode"] = bm["view_mode"]
                            st.session_state["focus_sentinel"] = bm["focus_sentinel"]
                            st.success(f"Applied '{pick}'")
        st.caption(
            f"Mode: **{st.session_state['view_mode']}** | "
            f"Focus: **{st.session_state['focus_sentinel']}**"
        )

        # ---- Run-history panel (improvement #9) ----
        with st.expander("Run history (manifests)", expanded=False):
            manifest_dir = ROOT / "outputs" / "manifests"
            if not manifest_dir.exists():
                st.caption(
                    "Geen manifests in `outputs/manifests/`. Train/sim "
                    "scripts schrijven hier zodra ze `code/manifest.py` "
                    "gebruiken."
                )
            else:
                files = sorted(manifest_dir.glob("*.json"), reverse=True)
                if not files:
                    st.caption("Map bestaat maar is leeg.")
                else:
                    rows = []
                    for p in files[:25]:
                        try:
                            d = json.loads(p.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        rows.append({
                            "when": d.get("generated_at", "?")[:19],
                            "kind": d.get("kind", "?"),
                            "git": d.get("git_commit", "?"),
                            "manifest_id": d.get("manifest_id", p.stem),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch",
                                     hide_index=True)
                        # Detail-view of selected manifest
                        ids = [r["manifest_id"] for r in rows]
                        pick = st.selectbox(
                            "Bekijk manifest", options=["-"] + ids,
                            key="manifest_detail_pick",
                        )
                        if pick != "-":
                            m_path = manifest_dir / f"{pick}.json"
                            if m_path.exists():
                                st.code(
                                    m_path.read_text(encoding="utf-8"),
                                    language="json",
                                )
                    st.caption(
                        f"{len(files)} manifest(s) totaal. Schrijf nieuwe via "
                        "`from manifest import write_manifest` in je train/sim "
                        "scripts."
                    )


# ============================================================
# TAB 5 - Compare (NEW; cross-experiment dashboard, improvement #8)
# ============================================================
def render_compare_tab(papers: pd.DataFrame, cand: pd.DataFrame,
                      cross: pd.DataFrame):
    st.markdown(
        "<div class='foras-title'>Compare - one screen, three options</div>"
        "<div class='foras-sub'>Optie 1 (sentinel) / Optie 2 (drift, backlog) "
        "/ Optie 3 (Jones canon) op gelijke metrics.</div>",
        unsafe_allow_html=True,
    )
    if not _in_demo_mode():
        st.markdown(
            "<div class='explainer'><h4>Wat zie je hier?</h4>"
            "Een compacte tabel + KPI-strip met de drie experimentele "
            "richtingen naast elkaar. Echte cijfers waar beschikbaar; '-' "
            "met tooltip als een run nog moet draaien.</div>",
            unsafe_allow_html=True,
        )

    # Aggregate metrics from leerlab + sentinel-ranks + asreview-runs
    rows = []
    LEERLAB_LOCAL = ROOT / "outputs" / "gnn_leerlab"
    metrics_path = LEERLAB_LOCAL / "metrics.json"
    config_path = LEERLAB_LOCAL / "config.json"
    metrics = {}
    config = {}
    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    two_layer = metrics.get("two_layer", {})

    # Sentinel ranks parquet
    ranks = None
    if SENTINEL_RANKS.exists():
        try:
            ranks = pd.read_parquet(SENTINEL_RANKS)
        except Exception:
            ranks = None

    # ASReview runs count
    n_ar_runs = 0
    if ASREVIEW_RUNS.exists():
        n_ar_runs = sum(1 for _ in ASREVIEW_RUNS.glob("*.asreview"))

    def _fmt(v, suffix=""):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}{suffix}"
        return f"{v}{suffix}"

    sentinels_with_data = (ranks["gnn_rank"].notna().sum() if ranks is not None else 0)

    # Optie 1 row (sentinel-experiment)
    best_rank = None
    if ranks is not None:
        gnn = ranks["gnn_rank"].dropna()
        if len(gnn):
            best_rank = int(gnn.min())
    rows.append({
        "Experiment": "Optie 1 - sentinel (FORAS+candidates)",
        "N": "14k + 2.3k + 3 sentinels",
        "Headline metric": "best GCN-rank op 3 sentinels",
        "Value": _fmt(best_rank),
        "AsReview equivalent": _fmt(int(ranks["asreview_rank"].dropna().min())
                                    if ranks is not None and ranks["asreview_rank"].notna().any() else None),
        "Status": ("data partial" if sentinels_with_data < 6 else "complete"),
    })

    # Optie 2 row (drift-ablation, backlog T-026)
    rows.append({
        "Experiment": "Optie 2 - synthetic drift (N=50)",
        "N": "50 (planned)",
        "Headline metric": "DeltaRecall(AsReview) - DeltaRecall(GCN)",
        "Value": "-",
        "AsReview equivalent": "-",
        "Status": "backlog (T-026; needs Anthropic API-key)",
    })

    # Optie 3 row (Jones canon)
    jones_path = ROOT / "outputs" / "optie3_jones_book" / "jones_classification.json"
    jones_summary = {}
    if jones_path.exists():
        try:
            jones_summary = json.loads(jones_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    rows.append({
        "Experiment": "Optie 3 - Jones-canon 70/30",
        "N": jones_summary.get("n_total_refs", 576),
        "Headline metric": "Recall@k op Jones hold-out",
        "Value": "-",
        "AsReview equivalent": "-",
        "Status": ("data ready (88 bridge-edges); 70/30 split nog niet "
                   "gedraaid"),
    })

    # GCN baseline row (FORAS-only, from leerlab metrics)
    rows.append({
        "Experiment": "GCN baseline (FORAS, no candidates)",
        "N": config.get("n_nodes", "-"),
        "Headline metric": "recall@10% screened (TIAB labels)",
        "Value": _fmt(two_layer.get("test_recall_at_10pct")
                       or two_layer.get("headline_recall_at_10pct_screened")),
        "AsReview equivalent": "-",
        "Status": ("PPR + LR stand-in" if "stand_in" in str(config.get("model_kind","")).lower()
                   or "NOT" in str(config.get("model_kind","")) else "real GCN"),
    })

    df_compare = pd.DataFrame(rows)
    for c in df_compare.columns:
        df_compare[c] = df_compare[c].astype(str)
    st.markdown("### 5.1 - Side-by-side")
    st.dataframe(df_compare, width="stretch", hide_index=True)

    # KPI strip
    st.markdown("### 5.2 - Snel-overzicht")
    cols_kpi = st.columns(4)
    with cols_kpi[0]:
        st.metric("ASReview-runs op disk", f"{n_ar_runs}",
                  help="Aantal .asreview-bestanden in outputs/asreview_runs/")
    with cols_kpi[1]:
        st.metric("Sentinels met GCN-rang", f"{sentinels_with_data}/6",
                  help="3 sentinels x 2 split-modi = 6 rijen")
    with cols_kpi[2]:
        st.metric("Sentinels met ASReview-rang",
                  f"{int(ranks['asreview_rank'].notna().sum()) if ranks is not None else 0}/6")
    with cols_kpi[3]:
        # Jones bridge-edges
        st.metric("Jones bridge-edges (FORAS->Jones)",
                  jones_summary.get("n_total_edges", 590))

    # Status table per option
    st.markdown("### 5.3 - Wat is er nog nodig per Optie?")
    needs = pd.DataFrame([
        {"Optie": "1 - sentinel", "Wat is af": "rewrites + GCN-rangen via PPR-stand-in",
         "Wat ontbreekt": "ASReview-rangen (run asreview_batch.ps1) + echte GCN (run train_gnn.ps1)"},
        {"Optie": "2 - drift",    "Wat is af": "concept + prompt-templates",
         "Wat ontbreekt": "Anthropic API-key + drift-pipeline (T-026 backlog)"},
        {"Optie": "3 - Jones",    "Wat is af": "data parse + cross-reference + bridge-edges",
         "Wat ontbreekt": "70/30 GCN-train op Jones-subgraph (zie tab 6)"},
    ])
    st.dataframe(needs, width="stretch", hide_index=True)
    st.caption(
        "Deze tab is een commando-centrale richting de begeleidersmeeting: "
        "wat hebben we, wat moet nog. Elke regel correspondeert met een "
        "PowerShell-script (`asreview_batch.ps1` / `train_gnn.ps1`) of een "
        "backlog-task (T-026)."
    )



# ============================================================
# TAB 6 - Jones canon (NEW; Optie 3 internal validation)
# ============================================================
JONES_DIR = ROOT / "outputs" / "optie3_jones_book"


@st.cache_data(show_spinner=False)
def _load_jones_artifacts():
    """Load the Jones-canon JSON files. Returns dict of dataframes/dicts."""
    out = {}
    if not JONES_DIR.exists():
        return out
    for fn in ("jones_classification.json", "jones_network_analysis.json"):
        p = JONES_DIR / fn
        if p.exists():
            try:
                out[fn.replace(".json","")] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


def render_jones_tab(papers: pd.DataFrame, cand: pd.DataFrame,
                     cross: pd.DataFrame):
    """Tab 6 - Jones-canon (Optie 3, optionele aanvulling)."""
    st.markdown(
        "<div class='foras-title'>Jones-canon - Shell Shock to PTSD (2005)</div>"
        "<div class='foras-sub'>Edgar Jones &amp; Simon Wessely, Maudsley "
        "Monograph 47. 576 referenties uit een militair-psychiatrie-canon "
        "die niet via FORAS' Boolean-search te vinden zijn. Optionele "
        "aanvulling op Optie 1 (sentinel) - een onafhankelijk testbed.</div>",
        unsafe_allow_html=True,
    )
    art = _load_jones_artifacts()
    if not art:
        st.warning(
            "`outputs/optie3_jones_book/` ontbreekt. Run de Jones-PDF-parser "
            "eerst (zie outputs/optie3_jones_book/ documentation)."
        )
        return
    classif = art.get("jones_classification", {})
    netw = art.get("jones_network_analysis", {})

    if not _in_demo_mode():
        st.markdown(
            "<div class='explainer'><h4>Waarom dit canon-experiment?</h4>"
            "Jones &amp; Wessely's 2005 boek heeft 576 expert-gecureerde "
            "referenties over militair-psychiatrische literatuur 1900-2005. "
            "Slechts 14% overlapt met onze 29-termen-zoektocht (T-012); 88 "
            "papers (26.6%) worden door FORAS-papers geciteerd via 590 "
            "edges - dat is een 2x denser bridge-netwerk dan de 2.288-pool. "
            "Een GCN getraind op 70% Jones-papers + hun edges, getest op "
            "30% hold-out, beantwoordt de generaliseerbaarheid-vraag (RQ6 "
            "in onderzoeksvraag.md): werkt de methode ook op een "
            "onafhankelijk expert-corpus?</div>",
            unsafe_allow_html=True,
        )

    _section_6_1_intro(classif, netw)
    st.markdown("---")
    _section_6_2_era_breakdown(classif)
    st.markdown("---")
    _section_6_3_bridge_network(classif, netw, papers)
    st.markdown("---")
    _section_6_4_seventy_thirty(classif, netw)
    st.markdown("---")
    _section_6_5_glossary()


def _section_6_1_intro(classif: dict, netw: dict):
    st.markdown("### 6.1 - Het canon in cijfers")
    st.caption(
        "PDF-literatuurlijst geparseerd op 1 mei 2026; OpenAlex-match "
        "geverifieerd via author + year + title-similarity."
    )
    cols = st.columns(5)
    items = [
        (classif.get("total_book_refs", 576), "Refs in boek", False),
        (classif.get("matched_in_openalex", 337), "Gematcht in OpenAlex", True),
        (classif.get("with_abstract", 163), "Met abstract", False),
        (classif.get("in_hist_candidates", 48),
         "Overlap met 2.288-pool", False),
        (classif.get("novel_jones_only", 280), "Alleen-via-Jones", True),
    ]
    for col, (num, lab, cyan) in zip(cols, items):
        klass = "num num-cyan" if cyan else "num"
        col.markdown(
            f"<div class='foras-kpi'><div class='{klass}'>{num}</div>"
            f"<div class='lab'>{lab}</div></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        f"Cross-reference: **{classif.get('in_foras_any', 3)} in FORAS-any** "
        f"(0 FT-included, 0 TIAB-included). Deze 3 papers zitten in het "
        f"FORAS-corpus maar zijn er bij screening uitgegooid. **{classif.get('novel_jones_only', 280)} "
        f"papers zijn alleen via expert-curatie ontdekbaar** - die fractie "
        f"laat zien hoe groot de gap is die een term-search alléén niet "
        f"dekt."
    )


def _section_6_2_era_breakdown(classif: dict):
    st.markdown("### 6.2 - Era-verdeling (waar zit de canon?)")
    st.caption(
        "Stacked bar: per era de totaal-, gematcht- en met-abstract-counts. "
        "Pre-1945 abstract-armoede is een kernrisico voor methode-evaluatie."
    )
    era = classif.get("era_breakdown", {})
    if not era:
        st.info("Geen era-data.")
        return
    rows = []
    for label, d in era.items():
        rows.append({"era": label, "kind": "Total", "count": d.get("total", 0)})
        rows.append({"era": label, "kind": "Matched in OpenAlex",
                     "count": d.get("matched", 0)})
        rows.append({"era": label, "kind": "With abstract",
                     "count": d.get("abstract", 0)})
    df_era = pd.DataFrame(rows)
    fig = px.bar(
        df_era, x="era", y="count", color="kind", barmode="group",
        category_orders={"era": ["<1900", "1900-1919", "1920-1944",
                                  "1945-1979", "1980-1999", "2000+"]},
        color_discrete_map={
            "Total":               PALETTE["slate"],
            "Matched in OpenAlex": PALETTE["navy"],
            "With abstract":       PALETTE["cyan"],
        },
        title="Jones-canon per era: totaal vs OpenAlex-match vs met abstract",
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(family="Inter", color=PALETTE["navy"]),
        xaxis_title="era", yaxis_title="aantal papers",
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        "Pre-1945: 7-12% abstract-availability. Post-1980: 45-59%. "
        "Implicatie: tekst-features (TF-IDF/embedding) zijn voor de "
        "vroege canon vrijwel onbruikbaar. Citation-edges zijn daar het "
        "enige signaal dat overblijft - exacte case voor een GCN."
    )


def _section_6_3_bridge_network(classif: dict, netw: dict,
                                 papers: pd.DataFrame):
    st.markdown("### 6.3 - Bridge-netwerk: 88 Jones-papers x 472 FORAS-papers")
    st.caption(
        "590 edges (FORAS-paper -> Jones-paper). 26.6% van de Jones-canon "
        "wordt door tenminste 1 FORAS-paper geciteerd; 3.2% van FORAS "
        "(472 / 14.764) citeert minstens 1 Jones-paper. Bijna 2x denser "
        "dan de 2.288-pool."
    )
    cols_kpi = st.columns(4)
    with cols_kpi[0]:
        st.metric("Edges (FORAS->Jones)",
                  netw.get("foras_to_jones_edges", 590))
    with cols_kpi[1]:
        st.metric("Distinct FORAS papers",
                  netw.get("distinct_foras_papers_linking", 472))
    with cols_kpi[2]:
        st.metric("Distinct Jones cited",
                  netw.get("distinct_jones_cited_by_foras", 88))
    with cols_kpi[3]:
        st.metric("Intra-Jones edges (canon-cohesion)",
                  netw.get("intra_edges_total", 148))

    # Top-cited Jones-papers
    top = netw.get("top_foras_cited", [])
    if top:
        st.markdown("**Top-10 Jones-papers gecited door FORAS:**")
        df_top = pd.DataFrame(top[:10])
        df_top["title"] = df_top["title"].astype(str).str[:80]
        st.dataframe(
            df_top[["foras_cites", "year", "title", "id"]].rename(
                columns={"foras_cites": "FORAS cites", "id": "OpenAlex"}
            ),
            width="stretch", hide_index=True,
        )
    isolated = netw.get("isolated_nodes", 0)
    if isolated:
        st.caption(
            f"**{isolated} Jones-papers zijn isolated** (geen intra-Jones "
            f"edges en geen FORAS-edges). Voor die nodes zal een "
            f"GCN-only-aanpak per definitie blanco scoren - tekst-features "
            f"zijn dan het enige signaal. Bekend P2-bovengrens-effect."
        )


def _section_6_4_seventy_thirty(classif: dict, netw: dict):
    st.markdown("### 6.4 - 70/30 hold-out experiment (placeholder)")
    st.caption(
        "Voor een echte GCN-train op de Jones-subgraph (331 nodes met "
        "148 intra-edges + 590 FORAS-bridge-edges) is een aparte run "
        "van `code/train_gnn_jones.py` nodig (nog te schrijven). Hieronder "
        "de geplande opzet en verwachte placeholder-cijfers."
    )
    plan = pd.DataFrame([
        {"Step": "1. Sub-graph bouwen",
         "Wat": "331 Jones-papers + 88 die FORAS citeren als bridge",
         "Status": "data klaar"},
        {"Step": "2. 70/30 stratified split",
         "Wat": "stratified op era; 232 train / 99 test",
         "Status": "te scripten"},
        {"Step": "3. Features",
         "Wat": "TF-IDF op title+abstract (waar beschikbaar) + degree",
         "Status": "163/331 hebben abstract; rest title-only"},
        {"Step": "4. Train GCN 2-layer + class-weighted BCE",
         "Wat": "label = 'in Jones-canon' (positive); negatives uit FORAS",
         "Status": "te scripten"},
        {"Step": "5. Eval recall@k op 30% hold-out",
         "Wat": "vergelijk met TF-IDF-only baseline + Boolean expansion",
         "Status": "te scripten"},
        {"Step": "6. Schrijf naar outputs/optie3_jones_book/jones_70_30.json",
         "Wat": "result + per-paper rank",
         "Status": "te scripten"},
    ])
    st.dataframe(plan, width="stretch", hide_index=True)
    st.caption(
        "**Verwachte uitkomst:** als de GCN op de 30% hold-out 50-70% "
        "recall@10% haalt en de TF-IDF-baseline op 20-30%, is dat "
        "generaliseerbaarheid-bewijs (RQ6). Pre-1945 papers zonder "
        "abstract zijn de hard cases - voor die zal de GCN moeten "
        "leunen op pure graph-structure (analoog aan Southard-sentinel "
        "in Optie 1)."
    )


def _section_6_5_glossary():
    if _in_demo_mode():
        return
    with st.expander("Glossary - Jones-canon"):
        st.markdown(
            "- **Jones & Wessely (2005)** - Maudsley Monograph 47 "
            "*Shell Shock to PTSD: Military Psychiatry from 1900 to the Gulf War*.\n"
            "- **Canon** - expert-gecureerde referentie-lijst, geen systematic review.\n"
            "- **Bridge-edge** - citation van FORAS-paper -> Jones-paper.\n"
            "- **Intra-Jones edge** - citation tussen twee Jones-papers onderling.\n"
            "- **Isolated node** - Jones-paper zonder enige edge (60 procent van 331).\n"
            "- **70/30 hold-out** - train op 70 procent van Jones + edges; meet op 30 procent."
        )


# ============================================================
# main
# ============================================================
def main():
    st.set_page_config(page_title="FORAS . v5", page_icon=":sparkles:",
                       layout="wide")
    inject_css()
    _render_global_sidebar()
    papers, edges = load_data()
    cand, cross = load_candidates()
    G = build_graph(len(papers), len(edges))

    title_suffix = (" . DEMO mode" if _in_demo_mode() else "")
    focus = _focused_sentinel()
    sub = ("The 172 systematic-review-included papers as a cyan core inside "
           "the FORAS corpus . six tabs: graph, ASReview baseline, GNN "
           "learning lab, candidates with sentinel cards, compare across "
           "options, Jones canon.")
    if focus:
        sub += f" Focused: {focus}."
    st.markdown(
        f"<div class='foras-title'>FORAS . citation graph . v5{title_suffix}</div>"
        f"<div class='foras-sub'>{sub}</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "FORAS",
        "ASReview",
        "GNN",
        "Candidates",
        "Compare",
        "Jones canon",
    ])
    with tab1:
        render_citation_graph_tab(papers, G)
    with tab2:
        render_asreview_tab(papers, cand, cross)
    with tab3:
        render_gnn_tab(papers, cand, cross)
    with tab4:
        render_candidate_tab(cand, cross)
    with tab5:
        render_compare_tab(papers, cand, cross)
    with tab6:
        render_jones_tab(papers, cand, cross)


if __name__ == "__main__":
    main()
