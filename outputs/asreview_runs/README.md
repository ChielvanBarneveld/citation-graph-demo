# asreview_runs - reproduce-commando's & samenvatting

Branch-label: `feat/asreview-tab` (logical only - repo is not under git on this
machine on 5 May 2026; same convention as `feat/sentinel-rewrites`).
Generated: 2026-05-05 dispatch-werkstand modus 3 (5 mei festival, Wageningen).
Source dispatch prompt: `outputs/dispatch_prompts_5mei.md` § "Dispatch-prompt 1".

This folder is the on-disk landing-zone for ASReview simulation projects.
Each completed simulation is a `.asreview` ZIP file containing the per-record
discovery order, classifier scores, and metadata. The Streamlit app
(`code/streamlit_app_v5.py`, tab 2) reads these files; this README is the
manual for producing them.

## TL;DR - hoe vul je deze map?

Run `make_dataset.py` om FORAS naar ASReview-formaat te zetten, dan
`asreview simulate` om de eerste run te draaien. Beide commando's staan in
"Reproduce-commando's" hieronder. De Streamlit-tab pikt nieuwe
`.asreview`-bestanden automatisch op.

## Eenmalige install

```bash
pip install asreview asreview-insights asreview-makita asreview-datatools \
    asreview-dory pyarrow
```

`pyarrow` is nodig voor de gedeelde parquet
`outputs/comparison/sentinel_ranks.parquet`.

## Reproduce-commando's

Vanaf de thesis-folder root:

```bash
# 1) Convert FORAS to ASReview-friendly CSV (semicolon -> comma, label rename)
python - <<'PY'
import pandas as pd
src = "data/focas/PTSS_Data_Foras_2025-02-05.csv"
df = pd.read_csv(src, sep=";", encoding="utf-8-sig", low_memory=False)
out = (df[["title","abstract","label_included_TIAB"]]
       .rename(columns={"label_included_TIAB":"included"})
       .dropna(subset=["title"]))
out["abstract"] = out["abstract"].fillna("")
out["included"] = out["included"].fillna(0).astype(int)
out.to_csv("outputs/asreview_runs/foras_asreview.csv", index=False)
print(f"wrote {len(out)} rows; positives = {out['included'].sum()}")
PY

# 2) Baseline simulation, default ELAS bundle (elas_u4 = SVM + TF-IDF)
asreview simulate outputs/asreview_runs/foras_asreview.csv \
    --n-prior-included 1 --n-prior-excluded 1 --seed 42 \
    --n-stop -1 \
    -o outputs/asreview_runs/foras_elas_u4_seed42.asreview

# 3) NB classifier
asreview simulate outputs/asreview_runs/foras_asreview.csv \
    --ai elas_u3 --n-prior-included 1 --n-prior-excluded 1 --seed 42 \
    --n-stop -1 \
    -o outputs/asreview_runs/foras_elas_u3_seed42.asreview

# 4) Transformer bundles (slow on CPU; first call downloads model)
asreview simulate outputs/asreview_runs/foras_asreview.csv \
    --ai elas_l2 --n-prior-included 1 --n-prior-excluded 1 --seed 42 \
    --n-stop -1 \
    -o outputs/asreview_runs/foras_elas_l2_seed42.asreview

asreview simulate outputs/asreview_runs/foras_asreview.csv \
    --ai elas_h3 --n-prior-included 1 --n-prior-excluded 1 --seed 42 \
    --n-stop -1 \
    -o outputs/asreview_runs/foras_elas_h3_seed42.asreview

# 5) Compare metrics across the four bundles
asreview metrics \
    outputs/asreview_runs/foras_elas_u4_seed42.asreview \
    outputs/asreview_runs/foras_elas_u3_seed42.asreview \
    outputs/asreview_runs/foras_elas_l2_seed42.asreview \
    outputs/asreview_runs/foras_elas_h3_seed42.asreview \
    --recall 0.8 0.9 0.95 --wss 0.95 \
    -o outputs/asreview_runs/foras_metrics.json

# 6) Plot recall curves over each other (PNG fallback)
asreview plot recall \
    outputs/asreview_runs/foras_elas_u4_seed42.asreview \
    outputs/asreview_runs/foras_elas_h3_seed42.asreview \
    -o outputs/asreview_runs/compare_u4_vs_h3.png
```

## FORAS + 2.288 candidates (sentinel-injection)

```bash
# 1) Build FORAS+candidates CSV with is_candidate flag and sentinel positives.
#    Sentinels are added with included=1 to make recall@k measurable.
python - <<'PY'
import pandas as pd, json
from pathlib import Path

foras = pd.read_csv("data/focas/PTSS_Data_Foras_2025-02-05.csv",
                    sep=";", encoding="utf-8-sig", low_memory=False)
foras_min = (foras[["title","abstract","label_included_TIAB"]]
             .rename(columns={"label_included_TIAB":"included"})
             .dropna(subset=["title"]))
foras_min["abstract"] = foras_min["abstract"].fillna("")
foras_min["included"] = foras_min["included"].fillna(0).astype(int)
foras_min["is_candidate"] = False
foras_min["is_sentinel"]  = False
foras_min["sentinel_id"]  = ""

cand = pd.read_csv("data/historical-terminology/candidates.csv",
                   dtype=str, low_memory=False)
# Skip candidates that are already in FORAS (in_foras == "True")
cand = cand[cand["in_foras"].fillna("False") != "True"].copy()
cand_min = pd.DataFrame({
    "title": cand["title"].fillna(""),
    "abstract": cand["abstract"].fillna(""),
    "included": 0,
    "is_candidate": True,
    "is_sentinel":  False,
    "sentinel_id":  "",
})

# Sentinel rewrites (LLM-rewrite of Solomon / Kardiner / Southard)
SENTINEL_DIR = Path("outputs/sentinel_rewrites")
sent_rows = []
for sid in ("solomon_1993","kardiner_1941","southard_1920"):
    rw = json.loads((SENTINEL_DIR / f"{sid}_rewrite.json").read_text(encoding="utf-8"))
    sent_rows.append({
        "title":    rw["title"],
        "abstract": rw["rewritten_abstract"],
        "included": 1,
        "is_candidate": True,
        "is_sentinel":  True,
        "sentinel_id":  sid,
    })
sentinels_df = pd.DataFrame(sent_rows)

merged = pd.concat([foras_min, cand_min, sentinels_df], ignore_index=True)
merged.to_csv("outputs/asreview_runs/foras_plus_candidates.csv", index=False)
print(f"wrote {len(merged)} rows; positives={merged['included'].sum()}; "
      f"candidates={merged['is_candidate'].sum()}; sentinels={merged['is_sentinel'].sum()}")
PY

# 2) Run sentinel-injected simulation (use --n-stop -1 to discover sentinels)
asreview simulate outputs/asreview_runs/foras_plus_candidates.csv \
    --n-prior-included 1 --n-prior-excluded 1 --seed 42 --n-stop -1 \
    -o outputs/asreview_runs/foras_plus_candidates_elas_u4_seed42.asreview

# 3) Repeat across the 4 bundles (5 seeds each = 20 runs total via makita)
mkdir -p benchmark/data && cd benchmark
cp ../outputs/asreview_runs/foras_asreview.csv data/
cp ../outputs/asreview_runs/foras_plus_candidates.csv data/
asreview makita template multimodel \
    --classifiers logistic nb svm \
    --feature_extractors tfidf onehot \
    --n_runs 5
make sim
# resulting .asreview files land in benchmark/output/sim/<dataset>/
```

## 70/30 split for the GNN-vs-ASReview comparison

The Streamlit tab sectie 2.7 ondersteunt twee modi:

- **Modus A (`modus_a_asreview_order`)** - lees uit een
  `.asreview`-bestand welke records ASReview als eerste 70% labelde via
  active learning. Dat zijn de train-records voor de GNN. De resterende
  30% (records die ASReview later of niet selecteerde) is hold-out. Voor
  elke sentinel: rang in die 30% via ASReview's score-volgorde.

- **Modus B (`modus_b_random_stratified`)** - random 70/30 split,
  gestratificeerd op `included`. Reproduceerbaar via fixed seed.

Beide modi schrijven naar `outputs/comparison/sentinel_ranks.parquet`. De
parquet wordt geinitialiseerd met 6 lege rijen (3 sentinels x 2 modi). De
ASReview-tab vult `asreview_rank` + `asreview_recall_at_rank` per rij; de
GNN-tab vult `gnn_rank` + `gnn_score`.

## Verwachte WSS-getallen (placeholder)

Zodra de 4 ELAS-bundels op FORAS gedraaid zijn, vul de tabel hieronder in
met `asreview metrics --wss 0.95 -o ...` output:

| Bundle  | Seed | Recall@10% | WSS@95 | Time-to-discovery |
|---------|-----:|-----------:|-------:|------------------:|
| elas_u4 |   42 |        TBD |    TBD |               TBD |
| elas_u3 |   42 |        TBD |    TBD |               TBD |
| elas_l2 |   42 |        TBD |    TBD |               TBD |
| elas_h3 |   42 |        TBD |    TBD |               TBD |

## Sandbox-caveat

Deze README is geschreven in een dispatch-sandbox waar `asreview` zelf
niet geinstalleerd kon worden binnen de sessie-tijd (10K records x
transformer-bundle = >30 min). De code in `code/streamlit_app_v5.py` is
geschreven om de boven-staande `.asreview`-bestanden te lezen zodra ze
bestaan; ze zijn nog niet geproduceerd in deze sandbox-run. Eerste actie
voor Chiel bij terugkomst:

1. `pip install` regel uitvoeren (`asreview asreview-insights ... pyarrow`)
2. Stap 1 + 2 van "Reproduce-commando's" draaien voor de FORAS-baseline
3. Streamlit-app openen, naar tab 2, scroll naar sectie 2.4 - de
   recall-curve van die eerste run zou direct moeten verschijnen.

## Open punten

- **Implementatie-noot tab 2.3 simulation-runner.** De knop in de
  Streamlit-tab is geparametriseerd maar voert nog niet zelf
  `subprocess.run(["asreview", "simulate", ...])` uit. Reden: de input-CSV
  moet eerst gebouwd zijn (stap 1 uit "Reproduce-commando's"). Voeg de
  subprocess-aanroep toe zodra `foras_asreview.csv` op disk staat.
- **Per-record discovery-scatter (tab 2.6).** Vereist een join van de
  results-table uit het `.asreview`-bestand op de input-CSV om
  `is_candidate` en `sentinel_id` op te halen. Code-skelet staat in
  `_compute_recall_curve`; de join wordt opgebouwd zodra een echte
  FORAS+candidates-run aanwezig is.
- **70/30 modus A automation (tab 2.7).** Vereist een script dat een
  `.asreview`-run leest tot 70% gelabeld is, daarna voor elke sentinel
  zijn rang in de hold-out berekent en `_upsert_asreview_rank` aanroept.
  Deze logic kan grotendeels in `_compute_recall_curve` re-used worden;
  TODO voor Chiel of een vervolg-dispatch.
