"""FORAS citation graph - v5.

v5 adds two new tabs on top of the v4 graph view:

  Tab 1 - "Citation graph"      (unchanged: 11.873 nodes, FT/TIAB/screened/external)
  Tab 2 - "Candidate explorer"  (new: 2.288 historical-terminology candidates,
                                 their cross-edges to FORAS, filters to build a
                                 test set for GNN evaluation)
  Tab 3 - "Method & metrics"    (new: compact FORAS pipeline explainer,
                                 candidate-injection logic, GNN role,
                                 ASReview-insights metrics)

Design unchanged: cream + navy + cyan LED accent.
"""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

DATA = Path(__file__).parent / "data"

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
# TAB 2 - Candidate explorer (NEW)
# ============================================================
def render_candidate_tab(cand: pd.DataFrame, cross: pd.DataFrame):
    st.markdown(
        "<div class='foras-title'>Historical-terminology candidates</div>"
        "<div class='foras-sub'>2.288 papers found via 29 historical PTSD-terms "
        "in OpenAlex - explore them, then build a test set for GNN evaluation.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='explainer'>"
        "<b>What you're looking at.</b> Each row is a paper that uses one of "
        "29 historical names for PTSD (<i>shell shock, traumatic neurosis, war "
        "neurosis, soldier's heart, effort syndrome, ...</i>) according to OpenAlex. "
        "We ran two queries per term: (1) full-text search restricted to pre-1980 "
        "publications, (2) title-only search for post-1980 publications. The 9 "
        "candidates that already appear in FORAS were all <b>screened-and-excluded</b> "
        "by FORAS - none of them passed TI/AB. That's a strong signal: FORAS "
        "screening tends to drop these historical-term papers."
        "</div>",
        unsafe_allow_html=True,
    )

    # KPIs
    n_total = len(cand)
    n_in_foras = int(cand["in_foras"].sum())
    n_with_edge = int((cand["edges_total"] > 0).sum())
    n_with_ft_edge = int(((cand["edges_to_foras_ft"] > 0) | (cand["edges_from_foras_ft"] > 0)).sum())
    n_with_tiab_edge = int(((cand["edges_to_foras_tiab"] > 0) | (cand["edges_from_foras_tiab"] > 0)).sum())
    total_edges = int(cand["edges_total"].sum())

    cols = st.columns(6)
    items = [
        (f"{n_total:,}", "Candidates total", False, False),
        (f"{n_in_foras}", "Already in FORAS", False, True),
        (f"{n_with_edge:,}", "with >=1 FORAS edge", False, False),
        (f"{n_with_tiab_edge}", "edge to TIAB-incl", True, False),
        (f"{n_with_ft_edge}", "edge to FT-incl", True, False),
        (f"{total_edges}", "Total cross-edges", False, False),
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

    # Filters in sidebar (when this tab is active)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Candidate filters")

        # Pool filter
        all_pools = ["pre1980", "post1980_title", "pre1980;post1980_title"]
        pool_sel = st.multiselect(
            "Pool", options=all_pools, default=all_pools,
            help="pre1980 = full-text search restricted to <1980. "
                 "post1980_title = title-only search for >=1980. "
                 "pre1980;post1980_title = found in both."
        )
        # Era
        era_options = ["<1920", "1920-44", "1945-79", "1980-99", "2000+", "nan"]
        era_sel = st.multiselect("Era", options=era_options, default=era_options)
        # Language
        all_langs = sorted([x for x in cand["language"].dropna().astype(str).unique()])
        if not all_langs:
            all_langs = ["en"]
        lang_sel = st.multiselect("Language", options=all_langs, default=all_langs)
        # Edge requirement
        edge_filter = st.radio(
            "Edge-density requirement",
            options=["any", "edge to any FORAS", "edge to TIAB-incl",
                     "edge to FT-incl", "no edges (pure isolates)"],
            index=0,
        )
        # In-FORAS toggle
        in_foras_filter = st.radio(
            "FORAS overlap",
            options=["any", "only in-FORAS (n=9)", "exclude in-FORAS"],
            index=0,
        )
        # Term filter (top-N)
        all_terms = set()
        for s in cand["found_via_terms"].fillna(""):
            for t in str(s).split(";"):
                if t:
                    all_terms.add(t)
        term_options = sorted(all_terms)
        term_sel = st.multiselect(
            "Found via term (any of)", options=term_options,
            default=[],
            help="Empty = include all terms. Pick one or more to subset."
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
        f = f[(f["edges_to_foras_tiab"] > 0) | (f["edges_from_foras_tiab"] > 0)]
    elif edge_filter == "edge to FT-incl":
        f = f[(f["edges_to_foras_ft"] > 0) | (f["edges_from_foras_ft"] > 0)]
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
        # Era distribution stacked by FORAS-overlap
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
        # Per-term breakdown (top 15)
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

    # Edge-density summary table
    st.markdown("##### Edge density of current filter (citation links to FORAS)")
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

    # In-FORAS overlap detail
    if int(f["in_foras"].sum()) > 0:
        st.markdown("##### In-FORAS overlap (these 9 are FORAS-screened-and-excluded)")
        cols_show = ["openalex_id", "title", "year", "language", "foras_stage",
                      "foras_label_FT", "foras_label_TIAB", "found_via_terms",
                      "edges_total"]
        st.dataframe(
            f[f["in_foras"]][cols_show].sort_values("year", ascending=False),
            hide_index=True, width="stretch",
        )

    # Candidate browsing table
    st.markdown("##### Browse candidates (sortable, top 200)")
    cols_show = ["openalex_id", "title", "year", "language", "found_via_terms",
                 "edges_to_foras_ft", "edges_from_foras_ft",
                 "edges_to_foras_tiab", "edges_from_foras_tiab",
                 "edges_total", "in_foras"]
    st.dataframe(
        f[cols_show].sort_values(["edges_total", "year"], ascending=[False, False]).head(200),
        hide_index=True, width="stretch",
    )

    # Export current filter as test set
    st.markdown("##### Export current filter as a test set")
    st.write(
        "Pick a name for this candidate-subset and download as CSV. "
        "This is the file you can later inject into the FORAS graph for GNN training/eval."
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


# ============================================================
# TAB 3 - Method & metrics (NEW)
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
# main
# ============================================================
def main():
    st.set_page_config(page_title="FORAS citation graph . v5",
                       page_icon=":sparkles:", layout="wide")
    inject_css()
    papers, edges = load_data()
    cand, cross = load_candidates()
    G = build_graph(len(papers), len(edges))

    st.markdown(
        "<div class='foras-title'>FORAS . citation graph</div>"
        "<div class='foras-sub'>The 172 systematic-review-included papers "
        "as a cyan core inside the FORAS corpus . v5 with candidate explorer</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "Citation graph",
        "Candidate explorer",
        "Method & metrics",
    ])
    with tab1:
        render_citation_graph_tab(papers, G)
    with tab2:
        render_candidate_tab(cand, cross)
    with tab3:
        render_method_tab(papers, cand, cross)


if __name__ == "__main__":
    main()
