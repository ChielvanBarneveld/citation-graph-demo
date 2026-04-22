# FORAS Citation Graph — v3

Interactive 3D citation-graph explorer for the **FORAS** systematic-review
dataset (`van_de_Schoot_2025`). Part of the thesis work on GNN-based cold-start
literature retrieval.

Live demo: https://chielvanbarneveld-citation-graph-demo-streamlit-app-wfx5qe.streamlit.app/

## What's in the graph

- **14,764 papers** from the FORAS corpus.
- **75,617 intra-corpus citation edges** (paper → paper, both in corpus).
- Every paper is labeled:
  - `label_included` (172 papers) — included after full-text screening.
  - `label_abstract_included` (568 papers) — passed abstract screening.
  - others — excluded / unlabeled.

## Views

- **Smart** — all labeled papers plus their highest-degree 1-hop neighbours.
- **Included backbone** — induced subgraph over the labeled set. Optional
  Louvain community colouring.
- **Ego of seed** — 1 or 2 hops around a chosen SR-included paper. A dedicated
  *Ego smooth* layout makes 2-hop neighborhoods readable.
- **Top-N by degree** — SR core plus the highest-degree papers corpus-wide.

## Layouts

- **Spring 3D** — force-directed.
- **BFS ring** — concentric Fibonacci-sphere shells by BFS distance from the
  SR core (or from the ego seed).

## Filters & interactions

- Year slider, **topic-field multi-select**, and a **highlight-by-title** box
  that outlines matching nodes in pink.
- **Shortest citation path** between any two labeled papers, rendered as an
  amber overlay.
- **Camera orbit** and **year step** sliders control the chronological
  animation (Play/Pause + year slider, cubic-in-out easing).
- Hover for title, authors, journal, topic, in/out-degree, and OpenAlex
  citation count.
- **Download snapshot** button exports the current view as a standalone HTML.

## Styling

Clean futuristic green theme: near-black background with mint/emerald accents,
JetBrains Mono for numeric metrics, Space Grotesk for everything else.
`.streamlit/config.toml` pins the palette so widgets match.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Data is bundled in `data/` as compact parquet files (~3 MB total). Rebuild from
the thesis folder via `python code/build_citation_graph.py`.

## Status

**v3** — clean futuristic pass. v1 was the initial 2D wire-up; v2 added 3D +
chronological animation; v3 adds topic/highlight/path/community/BFS-ring and a
top-to-bottom restyle.
