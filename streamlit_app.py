"""FORAS Citation Graph Explorer — v2 (3D + animated).

Master-thesis citation-graph explorer for the FORAS corpus
(`van_de_Schoot_2025` — 14,764 papers, 75k+ intra-corpus citations).

v2 highlights:
* **3D Plotly** scene with native touch pan / zoom / rotate.
* **Chronological animation** — the citation network builds up year by
  year (Play button + slider), with a subtle auto-rotation on the camera.
* **Clean navigation** — all controls in a collapsible sidebar; the main
  canvas stays uncluttered for demos.
* **Performance-aware defaults** — a "smart" view (the ~740 labeled
  papers + their 1-hop neighborhood, capped) renders fast even on a
  phone; a slider lets you expand up to the full corpus.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
PAPERS_PATH = DATA_DIR / "papers.parquet"
EDGES_PATH = DATA_DIR / "edges.parquet"

# ---------- Colours ----------
COLORS = {
    "included": "#22c55e",      # green
    "abstract_only": "#f59e0b", # amber
    "excluded": "#64748b",      # slate
    "seed_ring": "#ef4444",     # rose — seed outline
    "bg": "#0b1220",
    "grid": "#1e293b",
    "edge": "rgba(148, 163, 184, 0.22)",
}
KIND_DISPLAY = {
    "included": "SR-included (172)",
    "abstract_only": "Abstract-only (568)",
    "excluded": "Excluded / unlabeled",
}


# ---------- Page config ----------
st.set_page_config(
    page_title="FORAS Citation Graph — 3D",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle global styling tweaks (kept conservative so phones still render fine).
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #080d19; }
    h1, h2, h3 { color: #e5e7eb !important; }
    [data-testid="stSidebar"] { background: #0b1220; }
    .small-caption { color: #94a3b8; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    papers = pd.read_parquet(PAPERS_PATH)
    edges = pd.read_parquet(EDGES_PATH)
    papers["publication_year"] = (
        pd.to_numeric(papers["publication_year"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return papers, edges


@st.cache_resource(show_spinner=False)
def build_graph(papers: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in papers.itertuples(index=False):
        g.add_node(
            row.pid,
            title=row.title or "",
            year=int(row.publication_year or 0),
            journal=row.journal_name or "",
            topic=row.primary_topic_name or "",
            field=row.primary_topic_field or "",
            authors=row.authors_short or "",
            cited_by=int(row.cited_by_count or 0),
            in_deg=int(row.in_degree_corpus or 0),
            out_deg=int(row.out_degree_corpus or 0),
            label_included=int(row.label_included or 0),
            label_abs=int(row.label_abstract_included or 0),
        )
    g.add_edges_from(edges[["src", "tgt"]].itertuples(index=False, name=None))
    return g


def node_kind(attrs: dict) -> str:
    if attrs.get("label_included") == 1:
        return "included"
    if attrs.get("label_abs") == 1:
        return "abstract_only"
    return "excluded"


# ---------- Subgraph selection ----------
def ego_subgraph(g: nx.DiGraph, seed: str, depth: int, cap: int) -> nx.DiGraph:
    und = g.to_undirected(as_view=True)
    nodes: set[str] = {seed}
    frontier = {seed}
    for _ in range(depth):
        nxt: set[str] = set()
        for n in frontier:
            nxt.update(und.neighbors(n))
        nxt -= nodes
        nodes |= nxt
        frontier = nxt
        if not frontier:
            break
    sg = g.subgraph(nodes).copy()
    return trim(sg, seed, cap)


def smart_subgraph(g: nx.DiGraph, cap: int, include_abs: bool) -> nx.DiGraph:
    """Default view: labeled papers + their 1-hop neighbors, capped at ``cap``."""
    core = [
        n for n, d in g.nodes(data=True)
        if d["label_included"] == 1 or (include_abs and d["label_abs"] == 1)
    ]
    und = g.to_undirected(as_view=True)
    keep = set(core)
    # Add 1-hop neighbors (prioritised by in-degree).
    candidates: dict[str, int] = {}
    for n in core:
        for nb in und.neighbors(n):
            if nb not in keep:
                candidates[nb] = max(candidates.get(nb, 0), g.nodes[nb]["in_deg"])
    for nb, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True):
        if len(keep) >= cap:
            break
        keep.add(nb)
    sg = g.subgraph(keep).copy()
    return trim(sg, None, cap)


def top_by_degree(g: nx.DiGraph, cap: int) -> nx.DiGraph:
    nodes = sorted(g.nodes(data=True), key=lambda nd: nd[1]["in_deg"], reverse=True)
    keep = [n for n, _ in nodes[:cap]]
    return g.subgraph(keep).copy()


def trim(sg: nx.DiGraph, protect: str | None, cap: int) -> nx.DiGraph:
    if sg.number_of_nodes() <= cap:
        return sg
    ordered = sorted(sg.nodes(data=True), key=lambda nd: nd[1]["in_deg"], reverse=True)
    keep: list[str] = []
    if protect is not None and protect in sg:
        keep.append(protect)
    for n, _ in ordered:
        if n not in keep:
            keep.append(n)
        if len(keep) >= cap:
            break
    return sg.subgraph(keep).copy()


# ---------- 3D layout ----------
@st.cache_data(show_spinner=False)
def compute_3d_layout(node_key: str, edge_key: str) -> dict[str, tuple[float, float, float]]:
    """Stable 3D spring layout (cached by frozenset hash of nodes+edges)."""
    nodes = node_key.split("|")
    edges = [tuple(e.split(">")) for e in edge_key.split("|")] if edge_key else []
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    try:
        pos = nx.spring_layout(g, dim=3, seed=42, k=None, iterations=60)
    except Exception:
        # Fallback: circular on a sphere.
        pos = {}
        for i, n in enumerate(nodes):
            a = 2 * math.pi * i / max(1, len(nodes))
            pos[n] = (math.cos(a), math.sin(a), 0.0)
    return {n: tuple(float(x) for x in p) for n, p in pos.items()}


def layout_key(sg: nx.DiGraph) -> tuple[str, str]:
    """Produce cache keys that encode node & edge structure."""
    ns = sorted(sg.nodes())
    es = sorted(f"{u}>{v}" for u, v in sg.edges())
    return "|".join(ns), "|".join(es)


# ---------- Figure ----------
def _hover(n: str, d: dict) -> str:
    title = d["title"] if len(d["title"]) <= 110 else d["title"][:108] + "…"
    return (
        f"<b>{title}</b><br>"
        f"{d['authors']} · {d['year']} · {d['journal'] or d['field'] or '—'}<br>"
        f"cited_by: {d['cited_by']} · in-corpus cites: {d['in_deg']}<br>"
        f"openalex.org/{n}"
    )


def _sizes(sg: nx.DiGraph, pids: list[str], metric: str, scale: float = 1.0) -> list[float]:
    if metric == "Intra-corpus in-degree":
        raw = [sg.nodes[n]["in_deg"] for n in pids]
    else:
        raw = [sg.nodes[n]["cited_by"] for n in pids]
    mx = max(raw) if raw else 1
    mx = max(mx, 1)
    return [scale * (4 + 18 * (v / mx) ** 0.55) for v in raw]


def build_static_3d(
    sg: nx.DiGraph,
    pos: dict[str, tuple[float, float, float]],
    seed: str | None,
    size_metric: str,
) -> go.Figure:
    # Edge trace.
    ex: list[float] = []
    ey: list[float] = []
    ez: list[float] = []
    for u, v in sg.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
        ez += [z0, z1, None]
    traces: list = [
        go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines",
            line=dict(width=1.4, color=COLORS["edge"]),
            hoverinfo="none",
            showlegend=False,
            name="edges",
        )
    ]

    for kind in ("excluded", "abstract_only", "included"):
        pids = [n for n, d in sg.nodes(data=True) if node_kind(d) == kind]
        if not pids:
            continue
        xs = [pos[n][0] for n in pids]
        ys = [pos[n][1] for n in pids]
        zs = [pos[n][2] for n in pids]
        sizes = _sizes(sg, pids, size_metric)
        line_c = [COLORS["seed_ring"] if n == seed else "rgba(15,23,42,0.85)" for n in pids]
        line_w = [3.5 if n == seed else 0.6 for n in pids]
        hovers = [_hover(n, sg.nodes[n]) for n in pids]
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=COLORS[kind],
                    opacity=0.92,
                    line=dict(color=line_c, width=line_w),
                ),
                hoverinfo="text",
                hovertext=hovers,
                name=KIND_DISPLAY[kind],
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)
    _apply_scene(fig)
    return fig


def build_animated_3d(
    sg: nx.DiGraph,
    pos: dict[str, tuple[float, float, float]],
    seed: str | None,
    size_metric: str,
    step: int = 2,
) -> go.Figure:
    """Chronological build-up animation. Each frame adds the nodes
    first published in that year (cumulative). Camera rotates slightly."""
    years = sorted({d["year"] for _, d in sg.nodes(data=True) if d["year"] > 0})
    if not years:
        return build_static_3d(sg, pos, seed, size_metric)

    y_min, y_max = years[0], years[-1]
    frame_years = list(range(max(y_min, 1990), y_max + 1, step))
    if not frame_years or frame_years[-1] != y_max:
        frame_years.append(y_max)

    # Pre-group nodes & edges for fast frame building.
    by_kind_pid = {"excluded": [], "abstract_only": [], "included": []}
    for n, d in sg.nodes(data=True):
        by_kind_pid[node_kind(d)].append(n)
    years_of = {n: sg.nodes[n].get("year", 0) for n in sg.nodes()}

    # Full edge list with max year (latest endpoint).
    edge_latest = []
    for u, v in sg.edges():
        edge_latest.append((u, v, max(years_of[u], years_of[v])))

    def frame_traces(year: int):
        # Visible set up to this year.
        visible = set(n for n, d in sg.nodes(data=True) if d.get("year", 0) <= year)

        ex: list[float] = []
        ey: list[float] = []
        ez: list[float] = []
        for u, v, ey_year in edge_latest:
            if ey_year > year:
                continue
            if u not in visible or v not in visible:
                continue
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
            ez += [z0, z1, None]

        traces: list = [
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(width=1.3, color=COLORS["edge"]),
                hoverinfo="none",
                showlegend=False,
            )
        ]
        for kind in ("excluded", "abstract_only", "included"):
            pids = [n for n in by_kind_pid[kind] if n in visible]
            if not pids:
                traces.append(
                    go.Scatter3d(
                        x=[], y=[], z=[],
                        mode="markers",
                        marker=dict(size=1, color=COLORS[kind]),
                        name=KIND_DISPLAY[kind],
                        showlegend=True,
                    )
                )
                continue
            xs = [pos[n][0] for n in pids]
            ys = [pos[n][1] for n in pids]
            zs = [pos[n][2] for n in pids]
            sizes = _sizes(sg, pids, size_metric)
            line_c = [COLORS["seed_ring"] if n == seed else "rgba(15,23,42,0.85)" for n in pids]
            line_w = [3.5 if n == seed else 0.6 for n in pids]
            hovers = [_hover(n, sg.nodes[n]) for n in pids]
            traces.append(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="markers",
                    marker=dict(
                        size=sizes, color=COLORS[kind], opacity=0.92,
                        line=dict(color=line_c, width=line_w),
                    ),
                    hoverinfo="text", hovertext=hovers,
                    name=KIND_DISPLAY[kind], showlegend=True,
                )
            )
        return traces

    # Camera path — gentle rotation around z-axis while frames advance.
    def cam_for(i: int, n_frames: int):
        theta = 2 * math.pi * (i / max(1, n_frames)) * 0.35  # ~126° total sweep
        r = 1.8
        return dict(
            eye=dict(x=r * math.cos(theta), y=r * math.sin(theta), z=0.9),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

    n_frames = len(frame_years)
    frames: list[go.Frame] = []
    for i, y in enumerate(frame_years):
        frames.append(
            go.Frame(
                data=frame_traces(y),
                name=str(y),
                layout=go.Layout(scene_camera=cam_for(i, n_frames)),
            )
        )

    # Initial = first frame.
    initial = frame_traces(frame_years[0])
    fig = go.Figure(data=initial, frames=frames)

    # Play / pause controls + year slider.
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.02,
                y=0.08,
                xanchor="left",
                yanchor="bottom",
                pad=dict(r=8, t=4),
                bgcolor="rgba(15,23,42,0.75)",
                bordercolor="rgba(148,163,184,0.3)",
                font=dict(color="#e5e7eb"),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 420, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 250, "easing": "cubic-in-out"},
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 