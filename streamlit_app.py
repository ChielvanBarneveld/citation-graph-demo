"""FORAS Citation Graph Explorer — v1 (real data).

First iteration of the citation graph for the ADS thesis on GNN-based
cold-start literature retrieval (FORAS).

Data: the van_de_Schoot_2025 corpus (14,764 papers, 75k+ intra-corpus
citation edges). Each paper is labeled as included in the systematic
review (``label_included``) or as passing abstract screening
(``label_abstract_included``).

What the app shows:

* An **ego graph** around a seed paper (default: most-cited SR-included
  paper), up to 1–2 citation hops.
* OR the **"included backbone"** — the induced subgraph over all papers
  that passed the systematic review, plus their direct neighbors.

Nodes are colored by SR label, sized by intra-corpus in-degree (how often
they're cited *inside the FORAS corpus*). Plotly makes this pan/pinch
friendly on a phone.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
PAPERS_PATH = DATA_DIR / "papers.parquet"
EDGES_PATH = DATA_DIR / "edges.parquet"

# ---------- Page config ----------
st.set_page_config(
    page_title="FORAS Citation Graph",
    page_icon="🔗",
    layout="centered",
    initial_sidebar_state="collapsed",
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
    """Build a DiGraph: src → tgt means src cites tgt."""
    g = nx.DiGraph()
    # Add nodes with attributes.
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
    # Edges.
    g.add_edges_from(edges[["src", "tgt"]].itertuples(index=False, name=None))
    return g


def node_label_kind(g: nx.DiGraph, n: str) -> str:
    if g.nodes[n].get("label_included") == 1:
        return "included"
    if g.nodes[n].get("label_abs") == 1:
        return "abstract_only"
    return "excluded"


COLORS = {
    "included": "#22c55e",      # green  — passed full-text SR
    "abstract_only": "#f59e0b", # amber  — passed abstract screening only
    "excluded": "#64748b",      # slate  — excluded / unlabeled
    "seed_ring": "#e11d48",     # rose   — outline for seed node
}


# ---------- Subgraph selection ----------
def ego_subgraph(g: nx.DiGraph, seed: str, depth: int) -> nx.DiGraph:
    """BFS on the undirected citation network (cites OR cited-by)."""
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
    return g.subgraph(nodes).copy()


def included_backbone(g: nx.DiGraph, include_abstract: bool) -> nx.DiGraph:
    """Return the induced subgraph of included papers (+ abstract-only optional)."""
    if include_abstract:
        core = [n for n, d in g.nodes(data=True) if d["label_included"] == 1 or d["label_abs"] == 1]
    else:
        core = [n for n, d in g.nodes(data=True) if d["label_included"] == 1]
    return g.subgraph(core).copy()


def trim_for_display(sg: nx.DiGraph, seed: str | None, max_nodes: int) -> nx.DiGraph:
    """Cap the subgraph at max_nodes, keeping the seed + highest in-degree nodes."""
    if sg.number_of_nodes() <= max_nodes:
        return sg
    ordered = sorted(sg.nodes(data=True), key=lambda nd: nd[1].get("in_deg", 0), reverse=True)
    keep: list[str] = []
    if seed is not None and seed in sg:
        keep.append(seed)
    for n, _ in ordered:
        if n not in keep:
            keep.append(n)
        if len(keep) >= max_nodes:
            break
    return sg.subgraph(keep).copy()


# ---------- Layout ----------
def compute_layout(sg: nx.DiGraph, seed: str | None) -> dict[str, tuple[float, float]]:
    """Spring layout, seeded from BFS rings when a seed is set."""
    if sg.number_of_nodes() == 0:
        return {}
    if seed is not None and seed in sg:
        und = sg.to_undirected(as_view=True)
        dist = nx.single_source_shortest_path_length(und, seed)
        rings: dict[int, list[str]] = {}
        for n, d in dist.items():
            rings.setdefault(d, []).append(n)
        init: dict[str, tuple[float, float]] = {}
        for d, members in rings.items():
            members = sorted(members)
            for i, n in enumerate(members):
                if d == 0:
                    init[n] = (0.0, 0.0)
                else:
                    a = 2 * math.pi * i / max(1, len(members))
                    init[n] = (d * math.cos(a), d * math.sin(a))
        for n in sg.nodes():
            if n not in init:
                init[n] = (0.0, 0.0)
        try:
            return nx.spring_layout(sg, pos=init, seed=42, iterations=40, k=None)
        except Exception:
            return init
    try:
        return nx.spring_layout(sg, seed=42, iterations=40)
    except Exception:
        return {n: (0.0, 0.0) for n in sg.nodes()}


# ---------- Figure ----------
def build_figure(sg: nx.DiGraph, seed: str | None, size_metric: str) -> go.Figure:
    pos = compute_layout(sg, seed)

    # Edges
    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in sg.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="rgba(140,150,170,0.35)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Nodes: one trace per label kind so legend works cleanly.
    traces: list[go.Scatter] = [edge_trace]
    kinds = ["excluded", "abstract_only", "included"]
    kind_display = {
        "included": "SR-included (172)",
        "abstract_only": "Abstract-only (568)",
        "excluded": "Excluded / unlabeled",
    }

    # Size scale setup.
    if size_metric == "Intra-corpus in-degree":
        sizes_raw = {n: d.get("in_deg", 0) for n, d in sg.nodes(data=True)}
    else:
        sizes_raw = {n: d.get("cited_by", 0) for n, d in sg.nodes(data=True)}
    max_s = max(sizes_raw.values()) if sizes_raw else 1
    max_s = max(max_s, 1)

    for kind in kinds:
        xs, ys, sizes, texts, hovers, line_colors, line_widths = [], [], [], [], [], [], []
        for n in sg.nodes():
            if node_label_kind(sg, n) != kind:
                continue
            x, y = pos[n]
            d = sg.nodes[n]
            xs.append(x)
            ys.append(y)
            s_raw = sizes_raw[n]
            sizes.append(8 + 22 * (s_raw / max_s) ** 0.55)
            title = d["title"] if len(d["title"]) <= 120 else d["title"][:117] + "…"
            label = "★" if n == seed else ""
            texts.append(label)
            hovers.append(
                f"<b>{title}</b><br>"
                f"{d['authors']} · {d['year']} · {d['journal'] or d['topic'] or '—'}<br>"
                f"cited_by: {d['cited_by']} · in-corpus cites: {d['in_deg']}<br>"
                f"<a href='https://openalex.org/{n}'>openalex.org/{n}</a>"
                + ("<br><i>seed paper</i>" if n == seed else "")
            )
            if n == seed:
                line_colors.append(COLORS["seed_ring"])
                line_widths.append(3.0)
            else:
                line_colors.append("rgba(15,23,42,0.85)")
                line_widths.append(0.8)

        if not xs:
            continue
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=texts,
                textposition="middle center",
                textfont=dict(color="#ffffff", size=14),
                marker=dict(
                    size=sizes,
                    color=COLORS[kind],
                    line=dict(color=line_colors, width=line_widths),
                    opacity=0.9,
                ),
                hoverinfo="text",
                hovertext=hovers,
                name=kind_display[kind],
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=6, b=0),
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        plot_bgcolor="#0b1220",
        paper_bgcolor="#0b1220",
        font=dict(color="#e5e7eb"),
        height=560,
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ---------- UI ----------
def main() -> None:
    st.markdown(
        "<h2 style='margin-bottom:0.1rem'>🔗 FORAS Citation Graph — v1</h2>"
        "<p style='color:#94a3b8;margin-top:0;font-size:0.9rem'>"
        "Real data from the <code>van_de_Schoot_2025</code> corpus "
        "(14,764 papers, 75k+ intra-corpus citations). Nodes colored by systematic-review label. "
        "Part of thesis work on GNN-based cold-start literature retrieval."
        "</p>",
        unsafe_allow_html=True,
    )

    papers, edges = load_data()
    g = build_graph(papers, edges)

    # Shortlist of seed options: SR-included papers, sorted by in-corpus in-degree.
    included = papers[papers["label_included"] == 1].sort_values(
        ["in_degree_corpus", "cited_by_count"], ascending=[False, False]
    )
    # Map "short display label" -> pid
    def seed_label(row: pd.Series) -> str:
        t = (row["title"] or "").strip()
        if len(t) > 90:
            t = t[:88] + "…"
        year = int(row["publication_year"]) if row["publication_year"] else "?"
        return f"[{row['in_degree_corpus']}× in-corp, {year}] {t}"

    seed_options = {seed_label(r): r["pid"] for _, r in included.iterrows()}
    seed_options = {"— Included backbone (no seed) —": None, **seed_options}

    # Controls
    col1, col2 = st.columns([3, 2])
    with col1:
        chosen_label = st.selectbox("Seed paper (SR-included)", list(seed_options.keys()), index=1)
        seed_pid: str | None = seed_options[chosen_label]
    with col2:
        view_kind = st.radio(
            "View",
            ["Ego graph (BFS)", "Included backbone"],
            index=(1 if seed_pid is None else 0),
            horizontal=True,
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        depth = st.selectbox("Hops", [1, 2], index=0, disabled=(view_kind != "Ego graph (BFS)"))
    with col4:
        size_metric = st.selectbox("Size by", ["Intra-corpus in-degree", "Total cited_by"], index=0)
    with col5:
        include_abstract_backbone = st.toggle(
            "Include abstract-only", value=False,
            disabled=(view_kind != "Included backbone"),
            help="Include the 568 papers that passed abstract screening (but not full-text).",
        )

    # Optional year filter.
    min_year_default = 1990
    min_year = st.slider(
        "Minimum year",
        min_value=1950,
        max_value=2025,
        value=min_year_default,
        step=1,
    )

    # Build subgraph.
    if view_kind == "Included backbone" or seed_pid is None:
        sg = included_backbone(g, include_abstract=include_abstract_backbone)
        # For backbone view, add 1-hop out to show context (only included→* edges inside corpus).
        active_seed = None
    else:
        sg = ego_subgraph(g, seed_pid, int(depth))
        active_seed = seed_pid

    # Year filter (but always keep seed).
    kept = [n for n, d in sg.nodes(data=True) if d.get("year", 0) >= min_year]
    if active_seed is not None and active_seed not in kept and active_seed in sg:
        kept.append(active_seed)
    sg = sg.subgraph(kept).copy()

    # Cap for rendering performance.
    sg = trim_for_display(sg, active_seed, max_nodes=350)

    # Metrics row.
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Nodes", sg.number_of_nodes())
    with m2:
        st.metric("Edges", sg.number_of_edges())
    with m3:
        n_inc = sum(1 for _, d in sg.nodes(data=True) if d["label_included"] == 1)
        st.metric("Included", n_inc)
    with m4:
        n_abs = sum(1 for _, d in sg.nodes(data=True) if d["label_abs"] == 1)
        st.metric("Abs-only", n_abs)

    if sg.number_of_nodes() == 0:
        st.info("No nodes match the current filters.")
        return

    fig = build_figure(sg, active_seed, size_metric)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("📄 Papers in view", expanded=False):
        rows = []
        for n, d in sg.nodes(data=True):
            rows.append(
                {
                    "pid": n,
                    "title": d["title"],
                    "year": d["year"],
                    "field": d["field"],
                    "cited_by": d["cited_by"],
                    "in_corpus_cites": d["in_deg"],
                    "SR": "included" if d["label_included"] == 1 else (
                        "abstract-only" if d["label_abs"] == 1 else "excluded"
                    ),
                }
            )
        tdf = pd.DataFrame(rows).sort_values("in_corpus_cites", ascending=False).reset_index(drop=True)
        st.dataframe(tdf, use_container_width=True, hide_index=True)

    st.caption(
        "Data: `van_de_Schoot_2025` FORAS corpus. Edges = paper A cites paper B (both in corpus). "
        "Node size = intra-corpus in-degree (or total citations). "
        "Seed paper outlined in red ★. v1 — iteration; feedback welcome."
    )


if __name__ == "__main__":
    main()
