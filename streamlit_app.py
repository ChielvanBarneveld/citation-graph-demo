"""
Citation Graph Explorer — a tiny interactive demo.

Theme: automated systematic reviews / FORAS-style literature discovery.
Pick a seed paper, traversal depth, and a filter metric; the app renders a
small citation network built from synthetic sample data.

Mobile-friendly: compact layout, Plotly for touch interaction, few widgets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Citation Graph Explorer",
    page_icon="🔗",
    layout="centered",  # centered layout reads better on phones
    initial_sidebar_state="collapsed",
)

# ---------- Synthetic data ----------
@dataclass
class Paper:
    pid: str
    title: str
    year: int
    venue: str
    topic: str


TOPICS = ["GNN", "Systematic Review", "FORAS", "NLP", "Active Learning", "Embeddings"]
VENUES = ["NeurIPS", "ACL", "JASIST", "SIGIR", "ICML", "EMNLP"]

SEEDS = {
    "FORAS: GNNs for systematic review screening": "p000",
    "Semi-supervised relevance ranking for SR": "p001",
    "Active learning meets citation networks": "p002",
}


@st.cache_data(show_spinner=False)
def build_corpus(n: int = 80, seed: int = 42) -> tuple[dict[str, Paper], nx.DiGraph]:
    """Generate a small synthetic citation graph."""
    rng = random.Random(seed)
    papers: dict[str, Paper] = {}
    for i in range(n):
        pid = f"p{i:03d}"
        topic = rng.choice(TOPICS)
        venue = rng.choice(VENUES)
        year = rng.randint(2014, 2025)
        title = f"{topic}-based approach to {rng.choice(['retrieval', 'screening', 'ranking', 'clustering', 'classification'])} (#{i})"
        papers[pid] = Paper(pid, title, year, venue, topic)

    # Override the titles for known seeds so the dropdown makes sense
    papers["p000"] = Paper("p000", "FORAS: GNNs for systematic review screening", 2024, "JASIST", "FORAS")
    papers["p001"] = Paper("p001", "Semi-supervised relevance ranking for SR", 2023, "SIGIR", "Systematic Review")
    papers["p002"] = Paper("p002", "Active learning meets citation networks", 2022, "ACL", "Active Learning")

    g = nx.DiGraph()
    for pid, p in papers.items():
        g.add_node(pid, title=p.title, year=p.year, venue=p.venue, topic=p.topic)

    # Build citations: a paper cites earlier papers, with topic affinity
    for pid, p in papers.items():
        candidates = [q for q in papers.values() if q.year < p.year]
        if not candidates:
            continue
        k = rng.randint(1, min(6, len(candidates)))
        # Bias: prefer same-topic citations
        same_topic = [q for q in candidates if q.topic == p.topic]
        others = [q for q in candidates if q.topic != p.topic]
        weighted = same_topic * 3 + others
        chosen = rng.sample(weighted, k=min(k, len(weighted))) if weighted else []
        for q in chosen:
            g.add_edge(pid, q.pid)

    # Ensure seed papers have non-trivial neighborhoods
    for seed_pid in ["p000", "p001", "p002"]:
        if g.out_degree(seed_pid) < 3:
            older = [q for q in papers.values() if q.year < papers[seed_pid].year]
            extras = rng.sample(older, k=min(4, len(older)))
            for q in extras:
                g.add_edge(seed_pid, q.pid)

    return papers, g


def bfs_subgraph(g: nx.DiGraph, seed: str, depth: int) -> nx.DiGraph:
    """Return the undirected-BFS subgraph around seed up to given depth."""
    undirected = g.to_undirected()
    nodes: set[str] = {seed}
    frontier = {seed}
    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(undirected.neighbors(n))
        next_frontier -= nodes
        nodes |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    return g.subgraph(nodes).copy()


def filter_by_year(sg: nx.DiGraph, min_year: int) -> nx.DiGraph:
    keep = [n for n, d in sg.nodes(data=True) if d["year"] >= min_year]
    return sg.subgraph(keep).copy()


def layout_circular_by_year(sg: nx.DiGraph, seed: str) -> dict[str, tuple[float, float]]:
    """Simple deterministic layout: seed at center, others on rings by BFS distance."""
    undirected = sg.to_undirected()
    # BFS distance from seed
    dist = nx.single_source_shortest_path_length(undirected, seed)
    rings: dict[int, list[str]] = {}
    for n, d in dist.items():
        rings.setdefault(d, []).append(n)

    pos: dict[str, tuple[float, float]] = {}
    for d, members in rings.items():
        if d == 0:
            pos[seed] = (0.0, 0.0)
            continue
        members = sorted(members)
        for i, n in enumerate(members):
            angle = 2 * math.pi * i / max(1, len(members))
            r = d * 1.0
            pos[n] = (r * math.cos(angle), r * math.sin(angle))
    # Orphans (shouldn't happen in BFS subgraph but be safe)
    for n in sg.nodes():
        if n not in pos:
            pos[n] = (random.random(), random.random())
    return pos


# ---------- Metric selection ----------
METRICS = {
    "Degree": lambda sg: dict(sg.degree()),
    "In-degree (times cited)": lambda sg: dict(sg.in_degree()),
    "Out-degree (# references)": lambda sg: dict(sg.out_degree()),
    "PageRank": lambda sg: nx.pagerank(sg, alpha=0.85) if len(sg) else {},
    "Betweenness": lambda sg: nx.betweenness_centrality(sg) if len(sg) > 2 else {n: 0 for n in sg.nodes()},
}


def build_figure(sg: nx.DiGraph, seed: str, metric_name: str) -> go.Figure:
    pos = layout_circular_by_year(sg, seed)
    metric_values = METRICS[metric_name](sg)
    if not metric_values:
        metric_values = {n: 0 for n in sg.nodes()}
    max_m = max(metric_values.values()) or 1.0

    # Edges
    edge_x, edge_y = [], []
    for u, v in sg.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(140,140,160,0.55)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Nodes
    node_x, node_y, sizes, colors, texts, hover = [], [], [], [], [], []
    for n in sg.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        m = metric_values.get(n, 0)
        sizes.append(10 + 30 * (m / max_m) ** 0.6)
        colors.append(m)
        d = sg.nodes[n]
        is_seed = n == seed
        label = d["title"]
        if len(label) > 32:
            label = label[:30] + "…"
        texts.append("★ " if is_seed else "")
        hover.append(
            f"<b>{d['title']}</b><br>"
            f"{d['venue']} · {d['year']} · {d['topic']}<br>"
            f"{metric_name}: {m:.3f}" + ("<br><i>seed</i>" if is_seed else "")
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=texts,
        textposition="middle center",
        textfont=dict(size=12, color="white"),
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text=metric_name, side="right"),
                thickness=10,
                len=0.6,
            ),
            line=dict(width=1.2, color="rgba(20,20,20,0.9)"),
        ),
        hoverinfo="text",
        hovertext=hover,
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(14,17,23,1)",
        paper_bgcolor="rgba(14,17,23,1)",
        font=dict(color="#e6edf3"),
        height=520,
        dragmode="pan",
    )
    return fig


# ---------- UI ----------
st.markdown(
    "<h2 style='margin-bottom:0.2rem'>🔗 Citation Graph Explorer</h2>"
    "<p style='color:#8b949e;margin-top:0;font-size:0.95rem'>"
    "Interactive demo for ADS thesis work on automated systematic reviews. "
    "Synthetic data — proof-of-concept that deployment works."
    "</p>",
    unsafe_allow_html=True,
)

papers, g = build_corpus()

with st.container():
    col_a, col_b = st.columns(2)
    with col_a:
        seed_label = st.selectbox("Seed paper", list(SEEDS.keys()), index=0)
    with col_b:
        depth = st.selectbox("Traversal depth", [1, 2, 3], index=1)

    col_c, col_d = st.columns(2)
    with col_c:
        metric_name = st.selectbox("Node metric", list(METRICS.keys()), index=3)
    with col_d:
        min_year = st.selectbox("Min. year", [2014, 2018, 2020, 2022, 2024], index=1)

seed_pid = SEEDS[seed_label]
sg = bfs_subgraph(g, seed_pid, int(depth))
sg = filter_by_year(sg, int(min_year))

# Guarantee the seed is still there (in case year filter removed it)
if seed_pid not in sg:
    sg.add_node(seed_pid, **g.nodes[seed_pid])

colm1, colm2, colm3 = st.columns(3)
colm1.metric("Nodes", sg.number_of_nodes())
colm2.metric("Edges", sg.number_of_edges())
try:
    density = nx.density(sg)
except Exception:
    density = 0.0
colm3.metric("Density", f"{density:.3f}")

fig = build_figure(sg, seed_pid, metric_name)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("📄 Papers in view", expanded=False):
    rows = []
    metric_values = METRICS[metric_name](sg) or {}
    for n in sg.nodes():
        d = sg.nodes[n]
        rows.append(
            {
                "id": n,
                "title": d["title"],
                "year": d["year"],
                "venue": d["venue"],
                "topic": d["topic"],
                metric_name: round(metric_values.get(n, 0), 4),
            }
        )
    df = pd.DataFrame(rows).sort_values(metric_name, ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Built with Streamlit + NetworkX + Plotly. All data is synthetic. "
    "Tap a node for details; pinch to zoom."
)
