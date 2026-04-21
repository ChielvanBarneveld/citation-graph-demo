"""FORAS Citation Graph — v2 (3D + animation)."""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_DIR = Path(__file__).parent / "data"

COLORS = {
    "bg": "#0b1220",
    "grid": "#1f2937",
    "edge": "rgba(148, 163, 184, 0.18)",
    "edge_ego": "rgba(96, 165, 250, 0.55)",
    "seed": "#f97316",
    "sr": "#22c55e",
    "abs": "#60a5fa",
    "other": "#475569",
}

st.set_page_config(
    page_title="FORAS Citation Graph",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading FORAS corpus …")
def load_data():
    papers = pd.read_parquet(DATA_DIR / "papers.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    papers["publication_year"] = papers["publication_year"].fillna(0).astype(int)
    return papers, edges


@st.cache_resource(show_spinner="Building graph index …")
def build_full_graph(papers, edges):
    g = nx.DiGraph()
    attrs = papers.set_index("pid").to_dict("index")
    for nid, a in attrs.items():
        g.add_node(nid, **a)
    g.add_edges_from(zip(edges["src"].tolist(), edges["tgt"].tolist()))
    return g


def node_label_class(attr):
    if int(attr.get("label_included", 0) or 0) == 1:
        return "SR-included"
    if int(attr.get("label_abstract_included", 0) or 0) == 1:
        return "Abstract only"
    return "Other"


def node_hover(nid, attr, in_deg):
    title = (attr.get("title") or "")[:160]
    year = attr.get("publication_year") or ""
    topic = attr.get("primary_topic_field") or ""
    cls = node_label_class(attr)
    return (
        f"<b>{title}</b><br>"
        f"{year} · {topic}<br>"
        f"class: {cls}<br>"
        f"in-degree (corpus): {in_deg}<br>"
        f"<i>{nid}</i>"
    )


def included_nodes(g, include_abstract):
    out = []
    for nid, a in g.nodes(data=True):
        if int(a.get("label_included", 0) or 0) == 1:
            out.append(nid)
        elif include_abstract and int(a.get("label_abstract_included", 0) or 0) == 1:
            out.append(nid)
    return out


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
    keep = [n for n, a in g.nodes(data=True) if int(a.get("publication_year", 0) or 0) >= min_year]
    return g.subgraph(keep).copy()


@st.cache_data(show_spinner="Computing 3D layout …")
def compute_3d_layout(node_key, edge_key):
    h = nx.DiGraph()
    h.add_nodes_from(node_key)
    h.add_edges_from(edge_key)
    n = len(node_key)
    iters = 120 if n < 400 else (70 if n < 1200 else 45)
    pos = nx.spring_layout(h.to_undirected(), dim=3, seed=7, iterations=iters,
                           k=1.2 / math.sqrt(max(n, 1)))
    return {nid: tuple(map(float, p)) for nid, p in pos.items()}


def subgraph_key(sg):
    return tuple(sorted(sg.nodes())), tuple(sorted(sg.edges()))


def edge_trace_3d(sg, pos, seed):
    xs, ys, zs = [], [], []
    for u, v in sg.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        zs.extend([z0, z1, None])
    color = COLORS["edge_ego"] if seed else COLORS["edge"]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=1.3),
        hoverinfo="skip",
        name="citations",
        showlegend=False,
    )


def node_traces_3d(sg, pos, seed, size_metric):
    groups = {
        "SR-included": dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["sr"]),
        "Abstract only": dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["abs"]),
        "Other": dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["other"]),
        "Seed": dict(x=[], y=[], z=[], text=[], size=[], color=COLORS["seed"]),
    }
    for nid, attr in sg.nodes(data=True):
        if nid not in pos:
            continue
        cls = "Seed" if nid == seed else node_label_class(attr)
        in_deg = sg.in_degree(nid)
        if size_metric == "OpenAlex cites":
            base = int(attr.get("cited_by_count", 0) or 0)
            size = 4 + min(18, math.log1p(base) * 1.6)
        else:
            size = 4 + min(18, math.sqrt(in_deg) * 2.2)
        x, y, z = pos[nid]
        d = groups[cls]
        d["x"].append(x); d["y"].append(y); d["z"].append(z)
        d["text"].append(node_hover(nid, attr, in_deg))
        d["size"].append(size)

    traces = []
    for name, d in groups.items():
        if not d["x"]:
            continue
        traces.append(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"],
            mode="markers",
            marker=dict(
                size=d["size"],
                color=d["color"],
                opacity=0.92 if name == "Seed" else 0.85,
                line=dict(color="rgba(17,24,39,0.6)", width=0.4),
            ),
            hovertemplate="%{text}<extra></extra>",
            text=d["text"],
            name=name,
        ))
    return traces


def build_static_3d(sg, pos, seed, size_metric):
    fig = go.Figure(data=[edge_trace_3d(sg, pos, seed), *node_traces_3d(sg, pos, seed, size_metric)])
    _apply_scene(fig)
    return fig


def build_animated_3d(sg, pos, seed, size_metric, year_step):
    years = [int(a.get("publication_year", 0) or 0) for _, a in sg.nodes(data=True)]
    years = [y for y in years if y > 0]
    if not years:
        return build_static_3d(sg, pos, seed, size_metric)

    y_min = max(1950, min(years))
    y_max = min(2026, max(years))
    if y_max - y_min < year_step:
        return build_static_3d(sg, pos, seed, size_metric)
    frame_years = list(range(y_min, y_max + 1, max(year_step, 1)))
    if frame_years[-1] != y_max:
        frame_years.append(y_max)

    def frame_traces(cutoff):
        keep = {n for n, a in sg.nodes(data=True)
                if 0 < int(a.get("publication_year", 0) or 0) <= cutoff}
        if seed:
            keep.add(seed)
        sub = sg.subgraph(keep)
        return [edge_trace_3d(sub, pos, seed), *node_traces_3d(sub, pos, seed, size_metric)]

    def cam_for(i, total):
        theta = 2 * math.pi * (i / max(total - 1, 1)) * 0.45
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
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.02, y=0.08,
                xanchor="left", yanchor="bottom",
                pad=dict(r=8, t=4),
                bgcolor="rgba(15,23,42,0.75)",
                bordercolor="rgba(148,163,184,0.3)",
                font=dict(color="#e5e7eb"),
                buttons=[
                    dict(
                        label="\u25B6 Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 420, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 250, "easing": "cubic-in-out"},
                            "mode": "immediate",
                        }],
                    ),
                    dict(
                        label="\u23F8 Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.14, y=0.06, len=0.82,
                currentvalue=dict(prefix="year ", font=dict(color="#e5e7eb")),
                font=dict(color="#94a3b8"),
                bgcolor="rgba(15,23,42,0.4)",
                steps=[
                    dict(
                        method="animate",
                        args=[[str(y)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                        label=str(y),
                    )
                    for y in frame_years
                ],
            )
        ],
    )
    _apply_scene(fig)
    return fig


def _apply_scene(fig):
    axis_style = dict(
        visible=False,
        showbackground=False,
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="#e5e7eb", family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        margin=dict(l=0, r=0, t=4, b=4),
        height=660,
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            bgcolor=COLORS["bg"],
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            dragmode="orbit",
        ),
        legend=dict(
            orientation="h",
            y=1.02, x=0.5,
            xanchor="center",
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(148,163,184,0.25)",
            borderwidth=1,
            font=dict(color="#e5e7eb"),
        ),
    )


def main():
    papers, edges = load_data()
    g = build_full_graph(papers, edges)

    st.markdown(
        "<h1 style='margin-bottom:0.1em'>FORAS Citation Graph</h1>"
        "<p style='color:#94a3b8; margin-top:0'>"
        f"{g.number_of_nodes():,} papers · {g.number_of_edges():,} intra-corpus citations · "
        f"{int((papers['label_included'] == 1).sum())} SR-included · "
        f"{int((papers['label_abstract_included'] == 1).sum())} abstract-only"
        "</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("View")
        view = st.radio(
            "Mode",
            ["Smart (labeled + neighbors)", "Included backbone", "Ego of seed", "Top-N by degree"],
            index=0,
        )
        include_abs = st.checkbox("Include abstract-only in 'labeled'", value=True)
        max_nodes = st.slider("Max nodes", 300, 3500, 800, step=100,
                              help="Density. 3D stays smooth up to ~1500 in Chrome.")
        min_year = st.slider("Min publication year", 1990, 2025, 2000, step=1)
        size_metric = st.radio("Node size", ["Intra-corpus in-degree", "OpenAlex cites"], index=0)

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

        animate = st.checkbox("Animate chronologically", value=True)
        year_step = st.slider("Animation step (years)", 1, 5, 2) if animate else 2

        st.divider()
        st.caption("Drag to orbit · scroll to zoom · double-click to reset.")
        st.caption("Green = SR-included · Blue = abstract-only · Orange = seed · Slate = other")

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

    n_nodes = sg.number_of_nodes()
    n_edges = sg.number_of_edges()
    st.caption(f"Rendering **{n_nodes:,} nodes** · **{n_edges:,} edges** · view: *{view}*")

    if n_nodes == 0:
        st.warning("No nodes match the current filters. Lower the min year or switch view.")
        return

    pos = compute_3d_layout(*subgraph_key(sg))

    if animate and n_nodes > 12:
        fig = build_animated_3d(sg, pos, seed, size_metric, year_step)
    else:
        fig = build_static_3d(sg, pos, seed, size_metric)

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    with st.expander("About this view"):
        st.markdown(
            "- **Nodes**: papers from the FORAS corpus (`van_de_Schoot_2025`).\n"
            "- **Edges**: intra-corpus citations (`referenced_works`, both endpoints in corpus).\n"
            "- **Animation** reveals papers chronologically by publication year while the camera slowly orbits.\n"
            "- **Smart view** = all labeled (SR + abstract) plus their highest-degree 1-hop neighbors.\n"
            "- **Size** scales with in-degree inside the corpus or total OpenAlex citations.\n"
            "- This is **v2** — 3D + animation pass. Feedback welcome."
        )


if __name__ == "__main__":
    main()
