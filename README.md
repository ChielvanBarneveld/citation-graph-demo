# FORAS Citation Graph — v1

Interactive citation-graph explorer for the **FORAS** systematic-review
dataset (`van_de_Schoot_2025`). First iteration of the thesis work on
GNN-based cold-start literature retrieval.

Live demo: https://chielvanbarneveld-citation-graph-demo-streamlit-app-wfx5qe.streamlit.app/

## What's in the graph

- **14,764 papers** from the FORAS corpus.
- **75,617 intra-corpus citation edges** (paper → paper, both in corpus).
- Every paper is labeled:
  - `label_included` (172 papers) — included after full-text systematic-review screening.
  - `label_abstract_included` (568 papers) — passed abstract screening.
  - others — excluded / unlabeled.

## What you can do

- Pick a **seed paper** from the SR-included shortlist and render its 1- or 2-hop
  neighborhood.
- Switch to the **"Included backbone"** view — the induced citation subgraph over all
  SR-included papers (toggle to include the 568 abstract-only papers too).
- Scale node size by **intra-corpus in-degree** or **total OpenAlex citations**.
- Filter by **minimum publication year**.
- Hover for metadata, tap for OpenAlex links.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Data is included in `data/` as compact parquet files (~3 MB total). The
original csv lives in the thesis OneDrive folder and is not in this repo.

## Rebuild data

From the thesis folder (not in this repo), run:

```bash
python code/build_citation_graph.py
```

which writes `papers.parquet` + `edges.parquet` — copy them into `data/` here.

## Status

**v1.** Visible > perfect. Feedback welcome — this is iteration, not the final UI.
