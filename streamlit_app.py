"""FORAS citation graph — v4.

Focused on what makes FORAS unique:
1. Core-vs-periphery: 172 FT-included (cyan LED glow) -> 395 TI/AB-included
   (navy) -> 7.6k screened (muted) -> 3.7k external periphery (faint).
2. Funnel replay: retrieved -> TI/AB-included -> FT-included, dimming non-stage
   nodes progressively.
3. Search-channel facet: colour/filter by the 7 FORAS retrieval strategies,
   making retrieval-channel complementarity visible.

Design: light cream background, deep navy text, cyan LED-accent for the
FT-included core. Smooth/sculptural, not futuristic-dark.
"""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

# ---------- palette ----------
PALETTE = {
    "bg": "#f6f4ef",          # cream off-white
    "bg_soft": "#eef1f4",     # soft slate
    "navy": "#1e2a44",        # deep navy (primary text + edges)
    "navy_mid": "#3b5475",    # mid navy
    "cyan": "#22d3ee",        # LED cyan
    "cyan_soft": "#a5f3fc",   # cyan halo
    "slate": "#94a3b8",       # muted slate (screened nodes)
    "mist": "#cbd5e1",        # very light slate (external)
    "edge": "rgba(30,42,68,0.10)",
    "accent_warm": "#f59e0b",
}

STAGE_COLORS = {
    "ft_included":    PALETTE["cyan"],
    "tiab_included":  PALETTE["navy"],
    "screened":       PALETTE["slate"],
    "external":       PALETTE["mist"],
}
STAGE_SIZES = {
    "ft_included":    9.0,
    "tiab_included":  5.5,
    "screened":       3.0,
    "external":       2.2,
}
STAGE_OPACITY = {
    "ft_included":    1.0,
    "tiab_included":  0.95,
    "screened":       0.55,
    "external":       0.35,
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
    """Safe string: survives NaN floats, None, and anything else."""
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


# ---------- node rendering ----------
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


# ---------- plot ----------
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

    # edges first (muted)
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

    # halo behind FT-included
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

    # main nodes
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=sizes, color=rgba,
                    line=dict(width=0.35, color=PALETTE["navy"])),
        text=hovers, hovertemplate="%{text}<extra></extra>",
        showlegend=False, name="papers",
    ))

    # stage-legend proxies (only in stage mode)
    for stage, label in STAGE_LABEL.items():
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=10, color=STAGE_COLORS[stage],
                        line=dict(width=0.35, color=PALETTE["navy"])),
            name=label, showlegend=(colour_mode == "stage"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
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


# ---------- funnel ----------
FUNNEL_FRAMES = [
    ("All retrieved",          {"external", "screened", "tiab_included", "ft_included"}),
    ("Screened (FORAS corpus)", {"screened", "tiab_included", "ft_included"}),
    ("TI/AB-included",          {"tiab_included", "ft_included"}),
    ("FT-included (SR core)",   {"ft_included"}),
]


# ---------- app ----------
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
        font-weight: 600;
        letter-spacing: -0.01em;
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


def main():
    st.set_page_config(page_title="FORAS citation graph . v4",
                       page_icon=":sparkles:", layout="wide")
    inject_css()
    papers, edges = load_data()
    G = build_graph(len(papers), len(edges))

    st.markdown(
        "<div class='foras-title'>FORAS . citation graph</div>"
        "<div class='foras-sub'>The 172 systematic-review-included papers "
        "as a cyan core inside the FORAS corpus . v4</div>",
        unsafe_allow_html=True,
    )
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
            }[k],
            index=0,
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
        year_range = st.slider(
            "Publication year",
            min_value=y_min, max_value=y_max, value=(y_min, y_max),
        )
        show_edges = st.checkbox("Show citation edges", value=True)

        st.markdown("---")
        with st.expander("About v4"):
            n_ft = int((papers['stage'] == 'ft_included').sum())
            n_tiab = int((papers['stage'] == 'tiab_included').sum())
            st.markdown(
                f"- **{n_ft}** papers passed full-text screening and made it "
                f"into the systematic review (cyan core).\n"
                f"- **{n_tiab}** passed title/abstract screening but were "
                f"excluded at full-text.\n"
                f"- Edges are intra-corpus OpenAlex `referenced_works`.\n"
                f"- 7 retrieval channels from the FORAS screening trajectory "
                f"-- try *Colour by -> Retrieval channel* to see complementarity."
            )

    pids_tuple = tuple(G.nodes())
    edges_tuple = tuple(G.edges())
    pos = compute_layout(pids_tuple, edges_tuple)

    fig = build_plot(
        G, pos,
        colour_mode=colour_mode,
        chosen_channel=chosen_channel,
        year_range=year_range,
        show_edges=show_edges,
        funnel_stage_visible=funnel_visible,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    if colour_mode == "channel":
        st.markdown("##### Retrieval channels -- how many included papers each channel found")
        rows = []
        for k, lab, _ in CHANNELS:
            n_ft = int(((papers["stage"] == "ft_included") & (papers[k] == 1)).sum())
            n_tiab = int(((papers["stage"] == "tiab_included") & (papers[k] == 1)).sum())
            n_any = int((papers[k] == 1).sum())
            rows.append(dict(Channel=lab, FT_included=n_ft,
                             TIAB_included=n_tiab, Total=n_any))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
