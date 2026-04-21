# Citation Graph Explorer

A tiny interactive Streamlit demo for exploring citation neighborhoods around a
seed paper. Built as a deployment proof-of-concept for ADS thesis work on
automated systematic reviews (FORAS / GNN-style relevance discovery).

All data is **synthetic** — the goal is to prove the pipeline
(local → GitHub → Streamlit Community Cloud) works end to end.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What you can do

- Pick a seed paper
- Set traversal depth (1–3 hops)
- Choose a node-importance metric (degree, PageRank, betweenness, …)
- Filter by minimum publication year

The graph renders with Plotly so it's pan/zoom-friendly on a phone.

## Deployed version

Deployed via [Streamlit Community Cloud](https://share.streamlit.io).
