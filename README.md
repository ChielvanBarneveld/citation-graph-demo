# FORAS Citation Graph — v4

Interactive 3D citation-graph explorer for the **FORAS** systematic-review
dataset (`van_de_Schoot_2025` + PTSS screening trajectory). Part of the thesis
work on GNN-based cold-start literature retrieval.

Live demo: https://chielvanbarneveld-citation-graph-demo-streamlit-app-wfx5qe.streamlit.app/

## The FORAS story, at a glance

- **11,873 papers** in the graph (FORAS corpus + near periphery).
- **172 FT-included** — passed full-text screening, core of the systematic
  review (cyan LED core).
- **395 TI/AB-included** — passed title/abstract screening, excluded at
  full-text (navy mid-layer).
- **7,620 screened + excluded** · **3,686 external periphery** (muted).
- **~1.4 %** base-rate — the extreme class-imbalance cold-start problem this
  thesis is about.

## Views

- **Colour by screening stage** (default) — core-vs-periphery. The cyan core
  with its halo makes the "which papers made it into the review" question
  immediately visible.
- **Colour by retrieval channel** — 7 FORAS search strategies (replication,
  comprehensive, snowballing, full-text, three OpenAlex strategies). Pick one
  to see which included papers that channel alone would have caught — the
  direct ablation question from the van de Schoot 2025 paper.
- **Colour by human-human disagreement** — the 78 "last relevant paper"
  judgements highlighted, the hardest cases in the screening trajectory.

## Funnel replay

A sidebar slider steps through the four-stage funnel: `All retrieved →
Screened → TI/AB-included → FT-included`. Non-stage nodes fade out of view.
Visualises the 1.4 % base rate in four clicks.

## Data & edges

Intra-corpus OpenAlex `referenced_works` edges only — no API calls required.
Labels merged from two FORAS sources: `van_de_Schoot_2025.csv` (OpenAlex
metadata + 172 `label_included`) and `PTSS_Data_Foras_2025-02-05.csv` (7
search channels, per-screener labels, 4 inclusion criteria,
human-human/human-LLM disagreement).

## Styling

Light cream background, deep navy text, cyan LED accent for the FT-included
core. Inter + JetBrains Mono. `.streamlit/config.toml` pins the palette.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Data is bundled in `data/` as compact parquet files (~1.8 MB total). Rebuild
from the thesis folder via `python code/build_citation_graph.py` (or the v4
build script `build_v4.py`).

## Status

**v4** — FORAS-focused pivot. v1/v2/v3 were general citation-graph explorers
with many interaction modes; v4 strips that down to what makes FORAS unique:
the included core, the four-stage funnel, and the seven retrieval channels.
