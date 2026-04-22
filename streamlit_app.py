"""FORAS Citation Graph — v3 (clean futuristic green)."""
from __future__ import annotations

import ast
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from networkx.algorithms.community import louvain_communities


DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------- #
# palette                                                                     #
# --------------------------------------------------------------------------- #
COLORS = {
    "bg":           "#030a0a",          # near-black, slight green hint
    "bg_panel":     "#071411",
    "grid":         "#0e2a21",
    "text":         "#d1fae5",
    "text_muted":   "#6b8478",
    "accent":       "#10b981",          # emerald — brand primary
    "accent_glow":  "#34d399",          # mint — SR-included marker
    "sr":           "#34d399",          # mint / signature green
    "abs":          "#2dd4bf",          # teal / abstract-only
    "other":        "#2b3946",          # muted slate
    "seed":         "#fbbf24",          # amber — hot contrast vs green
    "highlight":    "#f472b6",          # hot pink — query match
    "path":         "#fbbf24",          # amber — shortest-path edges
    "edge":         "rgba(52, 211, 153, 0.10)",
    "edge_ego":     "rgba(52, 211, 153, 0.38)",
    "edge_path":    "rgba(251, 191, 36, 0.85)",
}

# 10-colour Louvain palette — greens / teals / cools with one amber
COMMUNITY_PALETTE = [
    "#34d399", "#2dd4bf", "#a7f3d0", "#6ee7b7", "#22d3ee",
    "#0ea5e9", "#818cf8", "#f472b6", "#fbbf24", "#fb7185",
]

st.set_page_config(
    page_title="FORAS Citation Graph",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# css — clean futuristic green skin on top of streamlit                       #
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, .stCodeBlock, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* page background — subtle vertical gradient */
.stApp {
    background:
        radial-gradient(1200px 800px at 20% -10%, rgba(16,185,129,0.08), transparent 60%),
        radial-gradient(1000px 700px at 110% 10%, rgba(34,211,238,0.05), transparent 55%),
        #030a0a;
}

/* title chrome */
h1.foras-title {
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0 0 0.1em 0;
    background: linear-gradient(90deg, #34d399 0%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
p.foras-sub {
    color: #6b8478;
    margin-top: 0;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

/* kpi strip */
[data-testid="stMetricLabel"] { color: #6b8478 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #34d399 !important; font-weight: 500; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #040d0b !important;
    border-right: 1px solid #0e2a21;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #34d399 !important;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.78rem !important;
}

/* sliders — green track */
[data-testid="stSlider"] [role="slider"] {
    background-color: #34d399 !important;
    box-shadow: 0 0 8px rgba(52,211,153,0.6);
}

/* radio / checkbox accents handled by primaryColor, but override text colour */
.stRadio label p, .stCheckbox label p { color: #d1fae5 !important; }

/* expander */
.streamlit-expanderHeader { color: #6ee7b7 !important; }

/* subtle glow on main plot container */
[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(16,185,129,0.18);
    border-radius: 8px;
    box-shadow:
        0 0 24px rgba(16,185,129,0.06),
        inset 0 0 40px rgba(16,185,129,0.02);
    background: #030a0a;
}

/* buttons */
.stDownloadButton button, .stButton button {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(52,211,153,0.35) !important;
    color: #34d399 !important;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: all 0.18s ease;
}
.stDownloadButton button:hover, .stButton button:hover {
    background: rgba(16,185,129,0.18) !important;
    border-color: #34d399 !important;
    box-shadow: 0 0 16px rgba(52,211,153,0.35);
}

/* hide streamlit footer / menu bloat */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading FORAS corpus …")
def load_data():
    papers = pd.read_parquet(DATA_DIR / "papers.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    papers["publication_year"] = pd.to_numeric(
        papers["publication_year"], errors="coerce"
    ).fillna(0).astype(int)
    for col in ("cited_by_count", "in_degree_corpus", "out_degree_corpus",
                "label_included", "label_abstract_included"):
        if col in papers.columns:
            papers[col] = pd.to_numeric(papers[col], errors="coerce").fillna(0).astype(int)
    return papers, edges


@st.cache_resource(show_spinner="Building graph index …")
def build_full_graph(papers, edges):
    g = nx.DiGraph()
    attrs = papers.set_index("pid").to_dict("index")
    for nid, a in attrs.items():
        g.add_node(nid, **a)
    g.add_edges_from(zip(edges["src"].tolist(), edges["tgt"].tolist()))
    return g


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def node_label_class(attr):
    if int(attr.get("label_included", 0) or 0) == 1:
        return "SR-included"
    if int(attr.get("label_abstract_included", 0) or 0) == 1:
        return "Abstract only"
    return "Other"


def _authors_str(raw):
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    if isinstance(raw, str):
        try:
            lst = ast.literal_eval(raw)
        except Exception:
            return raw[:90]
    else:
        lst = list(raw)
    if not lst:
        return ""
    if len(lst) > 3:
        return ", ".join(lst[:3]) + f" +{len(lst) - 3}"
    return ", ".join(lst)


def node_hover(nid, attr, in_deg, out_deg):
    title = (attr.get("title") or "").strip().replace("\n", " ")[:170]
    year = attr.get("publication_year") or ""
    journal = (attr.get("journal_name") or "").strip()[:70]
    topic = attr.get("primary_topic_field") or ""
    authors = _authors_str(attr.get("authors_short"))
    cls = node_label_class(attr)
    badge = {
        "SR-included":  "<span style='color:#34d399'>● SR-included</span>",
        "Abstract only": "<span style='color:#2dd4bf'>● abstract-only</span>",
        "Other":        "<span style='color:#6b8478'>● unlabeled</span>",
    }[cls]
    cites = int(attr.get("cited_by_count", 0) or 0)

    lines = [f"<b>{title}</b>"]
    if authors:
        lines.append(f"<span style='color:#6b8478'>{authors}</span>")
    meta = " · ".join(str(x) for x in [year, journal, topic] if x)
    if meta:
        lines.append(f"<span style='color:#a7f3d0'>{meta}</span>")
    lines.append(badge)
    lines.append(
        f"<span style='color:#6b8478'>in {in_deg} · out {out_deg} · OpenAlex {cites}</span>"
    )
    lines.append(f"<span style='color:#475569'>{nid}</span>")
    return "<br>".join(lines)


def included_nodes(g, include_abstract):
    out = []
    for nid, a in g.nodes(data=True):
        if int(a.get("label_included", 0) or 0) == 1:
            out.append(nid)
        elif include_abstract and int(a.get("label_abstract_included", 0) or 0) == 1:
            out.append(nid)
    return out


# --------------------------------------------------------------------------- #
# subgraph selectors                                                          #
# --------------------------------------------------------------------------- #
def smart_subgraph(g, cap, include_abstract):
    core = set(included_nodes(g, include_abstract))
    candidates = {}
    for n in core:
        for nbr in list(g.successors(n)) + list(g.predecessors(n)):
            if nbr in core:
                continue
            candidates[nbr] = max(candidates.get(nbr, 0), g.in_degree(nbr))
    remaining = cap - len(core)
    if remaining > 0 and candidates:
        extra = sorted(candidates.items(), key=lambda kv: -kv[1])[:remaining]
        core.update(nid for nid, _ in extra)
    if len(core) > cap:
        ranked = sorted(
            core,
            key=lambda n: (
                -int(g.nodes[n].get("label_included", 0) or 0),
                -int(g.nodes[n].get("label_abstract_included", 0) or 0),
                -g.in_degree(n),
            ),
        )
        core = set(ranked[:cap])
    return g.subgraph(core).copy()


def ego_subgraph(g, seed, hops, cap):
    if seed not in g:
        return g.subgraph([]).copy()
    und = g.to_undirected(as_view=True)
    nodes = {seed}
    frontier = {seed}
    for _ in range(hops):
        nf = set()
        for n in frontier:
            nf.update(und.neighbors(n))
        frontier = nf - nodes
        nodes.update(frontier)
        if len(nodes) >= cap:
            break
    if len(nodes) > cap:
        ranked = sorted((n for n in nodes if n != seed), key=lambda n: -g.in_degree(n))
        nodes = {seed, *ranked[: cap - 1]}
    return g.subgraph(nodes).copy()


def topn_subgraph(g, cap, include_abstract):
    core = set(included_nodes(g, include_abstract))
    remaining = cap - len(core)
    if remaining > 0:
        ranked = sorted(g.nodes(), key=lambda n: -g.in_degree(n))
        for n in ranked:
            if n in core:
                continue
            core.add(n)
            if len(core) >= cap:
                break
    return g.subgraph(core).copy()


def filter_by_year(g, min_year):
    if min_year <= 0:
        return g
    keep = [n for n, a in g.nodes(data=True)
            if int(a.get("publication_year", 0) or 0) >= min_year]
    return g.subgraph(keep).copy()


def filter_by_topic(g, topics):
    if not topics:
        return g
    keep = [n for n, a in g.nodes(data=True)
            if (a.get("primary_topic_field") or "") in topics]
    return g.subgraph(keep).copy()


# --------------------------------------------------------------------------- #
# layouts                                                                     #
# --------------------------------------------------------------------------- #
def subgraph_key(sg):
    return tuple(sorted(sg.nodes())), tuple(sorted(sg.edges()))


@st.cache_data(show_spinner="Computing layout …")
def compute_layout(node_key, edge_key, mode, seeds_key=None, hops_hint=1):
    h = nx.DiGraph()
    h.add_nodes_from(node_key)
    h.add_edges_from(edge_key)
    und = h.to_undirected()
    n = len(node_key)
    if n == 0:
        return {}

    if mode == "BFS ring" and seeds_key:
        return _bfs_ring_layout(und, list(node_key), list(seeds_key))

    if mode == "Ego smooth" and seeds_key:
        return _ego_smooth_layout(und, list(node_key), list(seeds_key), hops_hint)

    # default spring 3D
    iters = 130 if n < 400 else (70 if n < 1200 else 45)
    pos = nx.spring_layout(
        und, dim=3, seed=7, iterations=iters,
        k=1.2 / math.sqrt(max(n, 1)),
    )
    return {nid: tuple(map(float, p)) for nid, p in pos.items()}


def _bfs_ring_layout(und, nodes, seeds):
    """Concentric Fibonacci-sphere shells by BFS depth from seed set."""
    depths = {}
    for s in seeds:
        if s not in und:
            continue
        for n, d in nx.single_source_shortest_path_length(und, s).items():
            if d < depths.get(n, 10 ** 9):
                depths[n] = d
    if not depths:
        # no seeds reachable — fall back to spring
        pos = nx.spring_layout(und, dim=3, seed=7)
        return {n: tuple(map(float, p)) for n, p in pos.items()}
    max_d = max(depths.values())
    unreachable_d = max_d + 2
    shells = defaultdict(list)
    for n in nodes:
        shells[depths.get(n, unreachable_d)].append(n)
    pos = {}
    phi_golden = math.pi * (1 + math.sqrt(5))
    for d, bucket in shells.items():
        bucket.sort()  # stable ordering
        r = 0.9 + d * 1.3
        k = len(bucket)
        for i, nid in enumerate(bucket):
            t = (i + 0.5) / max(k, 1)
            phi = math.acos(1 - 2 * t)
            theta = phi_golden * i
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            pos[nid] = (x, y, z)
    return pos


def _ego_smooth_layout(und, nodes, seeds, hops):
    """Spring layout with seed pinned to origin — smoother 2-hop result."""
    seed = seeds[0] if seeds else None
    n = len(nodes)
    if n == 0 or seed is None or seed not in und:
        pos = nx.spring_layout(und, dim=3, seed=7)
        return {m: tuple(map(float, p)) for m, p in pos.items()}
    # initial positions: seed at origin, rest on unit sphere by BFS depth
    init = {seed: (0.0, 0.0, 0.0)}
    phi_golden = math.pi * (1 + math.sqrt(5))
    depth_map = nx.single_source_shortest_path_length(und, seed, cutoff=hops + 1)
    bucket = [m for m in nodes if m != seed]
    for i, m in enumerate(sorted(bucket)):
        d = depth_map.get(m, hops + 2)
        r = 0.6 + (d - 1) * 1.1
        t = (i + 0.5) / max(len(bucket), 1)
        phi = math.acos(1 - 2 * t)
        theta = phi_golden * i
        init[m] = (
            r * math.sin(phi) * math.cos(theta),
            r * math.sin(phi) * math.sin(theta),
            r * math.cos(phi),
        )
    iters = 140 if n < 400 else 90
    pos = nx.spring_layout(
        und, pos=init, fixed=[seed], dim=3, seed=7,
        iterations=iters, k=1.0 / math.sqrt(max(n, 1)),
    )
    return {m: tuple(map(float, p)) for m, p in pos.items()}


# --------------------------------------------------------------------------- #
# community detection (for backbone coloring)                                 #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Detecting communities …")
def community_map(node_key, edge_key):
    h = nx.DiGraph()
    h.add_nodes_from(node_key)
    h.add_edges_from(edge_key)
    und = h.to_undirected()
    try:
        comms = louvain_communities(und, seed=7, resolution=1.0)
    except Exception:
        return {}
    out = {}
    for idx, c in enumerate(sorted(comms, key=len, reverse=True)):
        colour = COMMUNITY_PALETTE[idx % len(COMMUNITY_PALETTE)]
        for n in c:
            out[n] = colour
    return out


# --------------------------------------------------------------------------- #
# trace builders                                                              #
# --------------------------------------------------------------------------- #
def edge_trace_3d(sg, pos, seed):
    xs, ys, zs = [], [], []
    for u, v in sg.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0, z0 = pos[u]; x1, y1, z1 = pos[v]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        zs.extend([z0, z1, None])
    color = COLORS["edge_ego"] if seed else COLORS["edge"]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=1.2),
        hoverinfo="skip",
        name="citations",
        showlegend=False,
    )


def path_trace_3d(path_nodes, pos):
    xs, ys, zs = [], [], []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if u not in pos or v not in pos:
            continue
        x0, y0, z0 = pos[u]; x1, y1, z1 = pos[v]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        zs.extend([z0, z1, None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=COLORS["edge_path"], width=4.5),
        hoverinfo="skip",
        name="path",
        showlegend=True,
    )


def node_traces_3d(sg, pos, seed, size_metric, community_colors=None, highlight_nodes=None,
                   path_endpoints=None):
    groups = {
        "SR-included":  dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["sr"]),
        "Abstract only": dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["abs"]),
        "Other":        dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["other"]),
        "Seed":         dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["seed"]),
    }
    # community mode — one trace per community colour
    community_groups = defaultdict(lambda: dict(x=[], y=[], z=[], text=[], size=[], color=None))
    highlight = dict(x=[], y=[], z=[], text=[], size=[])
    path_markers = dict(x=[], y=[], z=[], text=[], size=[])

    highlight_nodes = highlight_nodes or set()
    path_endpoints = set(path_endpoints or [])

    for nid, attr in sg.nodes(data=True):
        if nid not in pos:
            continue
        cls = "Seed" if nid == seed else node_label_class(attr)
        in_deg = sg.in_degree(nid); out_deg = sg.out_degree(nid)
        if size_metric == "OpenAlex cites":
            base = int(attr.get("cited_by_count", 0) or 0)
            size = 4 + min(18, math.log1p(base) * 1.6)
        else:
            size = 4 + min(18, math.sqrt(in_deg) * 2.2)
        x, y, z = pos[nid]
        hover = node_hover(nid, attr, in_deg, out_deg)

        if community_colors and nid in community_colors and cls != "Seed":
            d = community_groups[community_colors[nid]]
            d["color"] = community_colors[nid]
        else:
            d = groups[cls]

        d["x"].append(x); d["y"].append(y); d["z"].append(z)
        d["text"].append(hover); d["size"].append(size)

        if nid in highlight_nodes:
            highlight["x"].append(x); highlight["y"].append(y); highlight["z"].append(z)
            highlight["text"].append(hover); highlight["size"].append(size + 6)
        if nid in path_endpoints:
            path_markers["x"].append(x); path_markers["y"].append(y); path_markers["z"].append(z)
            path_markers["text"].append(hover); path_markers["size"].append(size + 8)

    traces = []
    if community_colors:
        # hide legend for communities (too many)
        for colour, d in community_groups.items():
            if not d["x"]:
                continue
            traces.append(go.Scatter3d(
                x=d["x"], y=d["y"], z=d["z"],
                mode="markers",
                marker=dict(
                    size=d["size"],
                    color=colour,
                    opacity=0.88,
                    line=dict(color="rgba(3,10,10,0.7)", width=0.4),
                ),
                hovertemplate="%{text}<extra></extra>",
                text=d["text"],
                name=f"community",
                showlegend=False,
            ))

    for name, d in groups.items():
        if not d["x"]:
            continue
        is_seed = name == "Seed"
        traces.append(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"],
            mode="markers",
            marker=dict(
                size=d["size"],
                color=d["color"],
                opacity=0.95 if is_seed else 0.88,
                line=dict(
                    color=COLORS["accent_glow"] if name == "SR-included" else "rgba(3,10,10,0.7)",
                    width=1.1 if name == "SR-included" else 0.4,
                ),
            ),
            hovertemplate="%{text}<extra></extra>",
            text=d["text"],
            name=name,
        ))

    if highlight["x"]:
        traces.append(go.Scatter3d(
            x=highlight["x"], y=highlight["y"], z=highlight["z"],
            mode="markers",
            marker=dict(
                size=highlight["size"],
                color="rgba(244,114,182,0.0)",  # transparent fill — rely on line
                line=dict(color=COLORS["highlight"], width=2.2),
                opacity=1.0,
            ),
            hovertemplate="%{text}<extra></extra>",
            text=highlight["text"],
            name="query match",
        ))

    if path_markers["x"]:
        traces.append(go.Scatter3d(
            x=path_markers["x"], y=path_markers["y"], z=path_markers["z"],
            mode="markers",
            marker=dict(
                size=path_markers["size"],
                color="rgba(251,191,36,0.0)",
                line=dict(color=COLORS["path"], width=2.4),
                opacity=1.0,
            ),
            hovertemplate="%{text}<extra></extra>",
            text=path_markers["text"],
            name="path endpoint",
        ))

    return traces


# --------------------------------------------------------------------------- #
# figure assembly                                                             #
# --------------------------------------------------------------------------- #
def _apply_scene(fig):
    axis_style = dict(
        visible=False, showbackground=False,
        gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Space Grotesk, Inter, sans-serif"),
        margin=dict(l=0, r=0, t=4, b=4),
        height=680,
        scene=dict(
            xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
            bgcolor=COLORS["bg"],
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            dragmode="orbit",
        ),
        legend=dict(
            orientation="h",
            y=1.02, x=0.5,
            xanchor="center",
            bgcolor="rgba(7,20,17,0.7)",
            bordercolor="rgba(52,211,153,0.25)",
            borderwidth=1,
            font=dict(color=COLORS["text"]),
        ),
    )


def build_static_3d(sg, pos, seed, size_metric, community_colors=None,
                     highlight_nodes=None, path=None):
    data = [edge_trace_3d(sg, pos, seed)]
    if path and len(path) > 1:
        data.append(path_trace_3d(path, pos))
    data.extend(node_traces_3d(
        sg, pos, seed, size_metric,
        community_colors=community_colors,
        highlight_nodes=highlight_nodes,
        path_endpoints=(path[0], path[-1]) if path else None,
    ))
    fig = go.Figure(data=data)
    _apply_scene(fig)
    return fig


def build_animated_3d(sg, pos, seed, size_metric, year_step, rotation_amount,
                      community_colors=None, highlight_nodes=None, path=None):
    years = [int(a.get("publication_year", 0) or 0) for _, a in sg.nodes(data=True)]
    years = [y for y in years if y > 0]
    if not years:
        return build_static_3d(sg, pos, seed, size_metric, community_colors,
                               highlight_nodes, path)

    y_min = max(1950, min(years))
    y_max = min(2026, max(years))
    if y_max - y_min < year_step:
        return build_static_3d(sg, pos, seed, size_metric, community_colors,
                               highlight_nodes, path)
    frame_years = list(range(y_min, y_max + 1, max(year_step, 1)))
    if frame_years[-1] != y_max:
        frame_years.append(y_max)

    path_endpoints = (path[0], path[-1]) if path else None

    def frame_traces(cutoff):
        keep = {n for n, a in sg.nodes(data=True)
                if 0 < int(a.get("publication_year", 0) or 0) <= cutoff}
        if seed:
            keep.add(seed)
        if path:
            keep.update(path)
        sub = sg.subgraph(keep)
        data = [edge_trace_3d(sub, pos, seed)]
        if path and len(path) > 1:
            data.append(path_trace_3d(path, pos))
        data.extend(node_traces_3d(
            sub, pos, seed, size_metric,
            community_colors=community_colors,
            highlight_nodes=highlight_nodes,
            path_endpoints=path_endpoints,
        ))
        return data

    def cam_for(i, total):
        theta = 2 * math.pi * (i / max(total - 1, 1)) * rotation_amount
        r = 1.9
        return dict(
            eye=dict(x=r * math.cos(theta), y=r * math.sin(theta), z=0.9),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

    n_frames = len(frame_years)
    frames = [
        go.Frame(
            data=frame_traces(y),
            name=str(y),
            layout=go.Layout(scene_camera=cam_for(i, n_frames)),
        )
        for i, y in enumerate(frame_years)
    ]
    fig = go.Figure(data=frame_traces(frame_years[0]), frames=frames)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="left",
            showactive=False,
            x=0.02, y=0.08, xanchor="left", yanchor="bottom",
            pad=dict(r=8, t=4),
            bgcolor="rgba(7,20,17,0.8)",
            bordercolor="rgba(52,211,153,0.35)",
            font=dict(color=COLORS["text"]),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {
                         "frame": {"duration": 520, "redraw": True},
                         "fromcurrent": True,
                         "transition": {"duration": 380, "easing": "cubic-in-out"},
                         "mode": "immediate",
                     }]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {
                         "frame": {"duration": 0, "redraw": False},
                         "mode": "immediate",
                         "transition": {"duration": 0},
                     }]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.14, y=0.06, len=0.82,
            currentvalue=dict(prefix="year ", font=dict(color=COLORS["text"])),
            font=dict(color=COLORS["text_muted"]),
            bgcolor="rgba(7,20,17,0.5)",
            activebgcolor="#34d399",
            bordercolor="rgba(52,211,153,0.25)",
            steps=[dict(method="animate", label=str(y),
                         args=[[str(y)], {
                             "mode": "immediate",
                             "frame": {"duration": 0, "redraw": True},
                             "transition": {"duration": 260, "easing": "cubic-in-out"},
                         }])
                   for y in frame_years],
        )],
    )
    _apply_scene(fig)
    return fig


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    papers, edges = load_data()
    g = build_full_graph(papers, edges)

    # ------ header ---------------------------------------------------------- #
    st.markdown(
        "<h1 class='foras-title'>FORAS · Citation Graph</h1>"
        "<p class='foras-sub'>"
        "cold-start exploration for GNN-based systematic review retrieval · "
        f"{g.number_of_nodes():,} papers · {g.number_of_edges():,} intra-corpus citations"
        "</p>",
        unsafe_allow_html=True,
    )

    # ------ sidebar --------------------------------------------------------- #
    with st.sidebar:
        st.header("View")
        view = st.radio(
            "Mode",
            ["Smart (labeled + neighbors)", "Included backbone",
             "Ego of seed", "Top-N by degree"],
            index=0, label_visibility="collapsed",
        )
        include_abs = st.checkbox("Include abstract-only in 'labeled'", value=True)
        max_nodes = st.slider("Max nodes", 300, 3500, 800, step=100,
                              help="Density. 3D stays smooth up to ~1500 in Chrome.")
        size_metric = st.radio(
            "Node size",
            ["Intra-corpus in-degree", "OpenAlex cites"],
            index=0,
        )

        seed = None
        hops = 1
        if view == "Ego of seed":
            sr_options = papers[papers["label_included"] == 1].copy()
            sr_options["label"] = sr_options["title"].fillna("(no title)").str.slice(0, 90)
            seed = st.selectbox(
                "Seed paper (SR-included)",
                options=sr_options["pid"].tolist(),
                format_func=lambda s: sr_options.loc[sr_options["pid"] == s, "label"].iloc[0],
            )
            hops = st.slider("Hops", 1, 2, 1)

        st.header("Filter")
        min_year = st.slider("Min publication year", 1990, 2025, 2000, step=1)
        topic_options = sorted([t for t in papers["primary_topic_field"].dropna().unique() if t])
        topic_filter = st.multiselect(
            "Topic field", options=topic_options, default=[],
            help="Leave empty to include all fields.",
        )
        query = st.text_input(
            "Highlight (title contains …)",
            value="",
            placeholder="e.g. trauma, veteran, meta-analysis",
        ).strip().lower()

        st.header("Layout")
        layout_mode = st.radio(
            "Algorithm",
            ["Spring 3D", "BFS ring"],
            index=0,
            help="Spring = force-directed. BFS ring = concentric shells by distance "
                 "from SR-core (or seed in Ego mode).",
        )
        color_by_community = False
        if view == "Included backbone":
            color_by_community = st.checkbox(
                "Colour by community (Louvain)", value=False,
                help="Detects communities inside the backbone and colours nodes accordingly.",
            )

        st.header("Animation")
        animate = st.checkbox("Animate chronologically", value=True)
        if animate:
            year_step = st.slider("Year step", 1, 5, 2)
            rotation_amount = st.slider(
                "Camera orbit", 0.0, 1.5, 0.45, step=0.05,
                help="How much the camera rotates across the animation. 0 = static.",
            )
        else:
            year_step, rotation_amount = 2, 0.0

        st.header("Path")
        path_mode = st.checkbox("Shortest path between two papers", value=False)
        path_nodes_selected = None
        if path_mode:
            sr_pool = papers[(papers["label_included"] == 1) |
                             (papers["label_abstract_included"] == 1)].copy()
            sr_pool["label"] = sr_pool["title"].fillna("(no title)").str.slice(0, 80)
            pa = st.selectbox("From", options=sr_pool["pid"].tolist(),
                              format_func=lambda s: sr_pool.loc[sr_pool["pid"] == s, "label"].iloc[0],
                              key="path_a")
            pb = st.selectbox("To", options=sr_pool["pid"].tolist(), index=1,
                              format_func=lambda s: sr_pool.loc[sr_pool["pid"] == s, "label"].iloc[0],
                              key="path_b")
            path_nodes_selected = (pa, pb)

        st.divider()
        st.caption("Drag to orbit · scroll to zoom · double-click to reset view")

    # ------ subgraph selection --------------------------------------------- #
    if view == "Smart (labeled + neighbors)":
        sg = smart_subgraph(g, max_nodes, include_abs)
    elif view == "Included backbone":
        nodes = included_nodes(g, include_abs)
        sg = g.subgraph(nodes).copy()
    elif view == "Ego of seed":
        sg = ego_subgraph(g, seed, hops, max_nodes)
    else:
        sg = topn_subgraph(g, max_nodes, include_abs)

    sg = filter_by_year(sg, min_year)
    sg = filter_by_topic(sg, topic_filter)

    # ensure path endpoints stay in graph if path mode is on
    path = None
    if path_mode and path_nodes_selected:
        pa, pb = path_nodes_selected
        if pa in g and pb in g:
            try:
                p = nx.shortest_path(g.to_undirected(as_view=True), source=pa, target=pb)
                # overlay path nodes onto current subgraph
                extra = [n for n in p if n not in sg]
                if extra:
                    union = set(sg.nodes()) | set(p)
                    sg = g.subgraph(union).copy()
                path = p
            except nx.NetworkXNoPath:
                path = None

    # resolve highlight nodes (query)
    highlight_nodes = set()
    if query:
        for nid, a in sg.nodes(data=True):
            title = (a.get("title") or "").lower()
            if query in title:
                highlight_nodes.add(nid)

    # ------ KPI strip ------------------------------------------------------ #
    total_sr = int((papers["label_included"] == 1).sum())
    visible_sr = sum(1 for n, a in sg.nodes(data=True)
                     if int(a.get("label_included", 0) or 0) == 1)
    years_visible = [int(a.get("publication_year", 0) or 0) for _, a in sg.nodes(data=True)]
    years_visible = [y for y in years_visible if y > 0]
    avg_year = f"{sum(years_visible) / len(years_visible):.0f}" if years_visible else "—"
    coverage = f"{visible_sr}/{total_sr} ({100 * visible_sr / total_sr:.0f}%)" if total_sr else "—"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nodes", f"{sg.number_of_nodes():,}")
    k2.metric("Edges", f"{sg.number_of_edges():,}")
    k3.metric("SR coverage", coverage)
    k4.metric("Avg year", avg_year)

    n_nodes = sg.number_of_nodes()
    if n_nodes == 0:
        st.warning("No nodes match the current filters. Lower the min year, clear topic filter, or switch view.")
        return

    # ------ layout --------------------------------------------------------- #
    if layout_mode == "BFS ring":
        if view == "Ego of seed" and seed:
            seeds_for_layout = (seed,)
        else:
            seeds_for_layout = tuple(sorted(included_nodes(sg, include_abs))[:1] or [])
        mode = "BFS ring"
    elif view == "Ego of seed" and seed and hops >= 2:
        mode = "Ego smooth"
        seeds_for_layout = (seed,)
    else:
        mode = "Spring 3D"
        seeds_for_layout = None

    nkey, ekey = subgraph_key(sg)
    pos = compute_layout(nkey, ekey, mode, seeds_for_layout, hops)

    # ------ community coloring (backbone only) ----------------------------- #
    community_colors = None
    if color_by_community and view == "Included backbone":
        community_colors = community_map(nkey, ekey)

    # ------ figure --------------------------------------------------------- #
    if animate and n_nodes > 12:
        fig = build_animated_3d(
            sg, pos, seed, size_metric, year_step, rotation_amount,
            community_colors=community_colors,
            highlight_nodes=highlight_nodes,
            path=path,
        )
    else:
        fig = build_static_3d(
            sg, pos, seed, size_metric,
            community_colors=community_colors,
            highlight_nodes=highlight_nodes,
            path=path,
        )

    st.plotly_chart(
        fig, width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

    # ------ action row ----------------------------------------------------- #
    col_a, col_b = st.columns([1, 3])
    with col_a:
        snapshot = fig.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            "Download snapshot (HTML)",
            data=snapshot,
            file_name=f"foras_graph_{view.split()[0].lower()}.html",
            mime="text/html",
            width="stretch",
        )
    with col_b:
        if query and highlight_nodes:
            st.caption(f"**{len(highlight_nodes)}** node(s) match `{query}` — outlined in pink")
        if path:
            st.caption(f"**Path** ({len(path)} hops): highlighted in amber")

    with st.expander("About this view"):
        st.markdown(
            "- **Nodes** — papers from the FORAS corpus (`van_de_Schoot_2025`).\n"
            "- **Edges** — intra-corpus citations (`referenced_works`, both endpoints in corpus).\n"
            "- **Smart** = all labeled (SR + abstract) plus their highest-degree 1-hop neighbors. "
            "**Backbone** = induced subgraph over labeled set. **Ego** = neighborhood of one SR paper. "
            "**Top-N** = SR core + highest-degree papers corpus-wide.\n"
            "- **Layout — Spring 3D** is force-directed. "
            "**BFS ring** places nodes on Fibonacci spheres by BFS distance from the SR core (or seed).\n"
            "- **Community color** on backbone uses Louvain to reveal clusters.\n"
            "- **Path** finds the shortest undirected citation path between two labeled papers.\n"
            "- **Hover** for authors, journal, topic, in/out-degree, OpenAlex cites.\n"
            "- **v3** — clean futuristic pass. Built for cold-start GNN thesis work."
        )


if __name__ == "__main__":
    main()
