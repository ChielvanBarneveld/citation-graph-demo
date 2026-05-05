# gnn_leerlab - GNN-tab artifacts

Branch-label: `feat/gnn-tab` (logical only).
Generated: 2026-05-05 in dispatch-werkstand modus 3.
Source dispatch prompt: `outputs/dispatch_prompts_5mei.md` § "Dispatch-prompt 2".

This folder is the on-disk source of truth for **tab 3: GNN learning lab** in
`code/streamlit_app_v5.py`. The Streamlit tab reads these files; this README
is the manual for producing/refreshing them.

## Wat zit er in deze map?

| File | Wat | Geproduceerd door |
|---|---|---|
| `config.json` | hyperparameters + branch-label + generator-name | training-script |
| `metrics.json` | `two_layer.final_test_recall` + `depth_sweep[1..5]` voor over-smoothing-demo | training-script |
| `embeddings.npy` | `(n_nodes, 64)` float32 - input voor de 2D-projection in tab 3.4 | training-script |
| `scores.parquet` | per-paper `score` + `y_true` + `title` + `year` - input voor 3.4 quiz/inspector | training-script |
| `candidate_scores.parquet` | per-candidate `score` + `is_sentinel` flag - input voor 3.5 ranking | training-script |

Deze map is **leeg** als je `train_gnn_demo.py` nog niet hebt gerund.
De Streamlit-tab toont in dat geval een vriendelijke "run het script
eerst"-melding.

## Reproduce - de echte GCN

```bash
# Install (CPU is fine; transformer-gewichten staan op pypi)
pip install torch torch-geometric scikit-learn pandas pyarrow

# Train + populate the leerlab folder
python code/train_gnn_demo.py
# Variants:
python code/train_gnn_demo.py --quick           # 30 epochs
python code/train_gnn_demo.py --depth-sweep     # adds 1-5 layer sweep
python code/train_gnn_demo.py --no-candidates   # skip transductive inference
python code/train_gnn_demo.py --label FT        # train on FT-included instead of TIAB
```

Verwacht resultaat (uit `outputs/dispatch_research/gnn_research.md` §L):
- 2-layer test-recall: 0.4-0.6 op TIAB-labels (klein, "bewijs van leven")
- Top-50 candidate scores: enkele herkenbaar PTSD-historisch
- 2155/2288 candidates hebben geen graph-edges - hun score is feature-only

## Sandbox-stand-in (als je deze README leest na een dispatch-run)

Tijdens de 5-mei dispatch kon de sandbox `torch` + `torch_geometric` niet
installeren (530 MB CUDA-deps + geblokkeerde proxy). Als plaatsvervanger
draaide een **PPR (Personalized PageRank) + logistic-regression** baseline
op dezelfde features (TF-IDF op title+abstract, 256 dim) en dezelfde
graph (75.617 undirected edges over 14.764 nodes). Output-schema is
identiek.

`config.json` gemarkeerd met `model_kind: "PPR + logistic-regression
baseline (NOT a real GCN)"` zodat de tab het verschil kan tonen via een
kleine info-banner.

`embeddings.npy` is in deze stand-in een TruncatedSVD-projectie van de
TF-IDF-matrix (64 dim). Die plot er in de 2D-projection net zo uit als
echte GCN-embeddings, maar bevat alleen tekst-informatie.

`metrics.json.depth_sweep` is een PPR-alpha-sweep (1-5) als surrogaat
voor de over-smoothing curve - laat een vergelijkbaar **plateau** zien
maar geen scherpe daling, omdat PPR uberhaupt geen over-smoothing heeft
zoals een diepe GCN. Trek hier geen conclusies uit; de echte curve komt
van `train_gnn_demo.py`.

## Sentinel-rangen (5 mei stand-in)

De stand-in gaf:

| Sentinel | gnn_rank | gnn_score | n_cited_by_foras |
|---|---:|---:|---:|
| Solomon 1993 (EASY) | 158 / 2289 | 0.166 | 22 |
| Kardiner 1941 (MEDIUM) | 29 / 2289 | 0.285 | 17 |
| Southard 1920 (HARD) | 1857 / 2289 | 0.011 | 0 |

Deze gaan in `outputs/comparison/sentinel_ranks.parquet` (kolommen
`gnn_rank`, `gnn_score`). De ASReview-tab vult de andere kolommen op
zijn moment.

**Lezing:**

- Kardiner ranket hoog (29/2289) ondanks `cited_by_foras_any = 17`.
  Reden: de **rewrite-abstract** bevat heel veel PTSS-trajectory-
  vocabulaire (LGMM, validated scale, war neurosis, longitudinal),
  dus de TF-IDF-LR-component scoort hem hoog. Goed teken voor de
  rewrite-strategy.
- Solomon scoort middelmatig (158/2289). Cited-by-FORAS is 22 maar de
  PPR-stand-in koppelt dat alleen door referenced_works van de
  candidate naar FORAS - en de Solomon-record in `candidates.csv` heeft
  geen `referenced_works`. Een echte GCN ziet de inkomende edges via
  `edges.parquet` en scoort hoger. Verwacht een grote sprong omhoog
  zodra Chiel `train_gnn_demo.py` runt.
- Southard ranket laag (1857/2289), zoals verwacht: 0 citations + sparse
  vocabulaire = de bovengrens van de methode (P2 in
  `context/hypothesis.md`). Bevestigt het sentinel-design.

## Open punten

- **Echte GCN runnen** is de belangrijkste vervolg-stap. De stand-in is
  nuttig om de tab te debug-renderen, niet om begeleiders bewijs te tonen.
- **`outputs/citation-graph/papers.parquet`** wordt door de
  Streamlit-app verwacht in `code/data/`. In sandbox staat het in
  `outputs/citation-graph/` (waar de build-script het zet). Lokaal heeft
  Chiel waarschijnlijk een copy/symlink naar `code/data/`. Confirm en
  documenteer in een follow-up.
- **PyG-version dependency**. `torch-geometric` 2.6.x werkt op torch 2.8.x;
  als torch 2.11.x op pypi staat moet PyG misschien expliciet pinned
  worden naar de matching wheel. Check de install-uitvoer voor warnings.
- **Kardiner is niet in `candidates.csv`** (zijn OpenAlex W2496837584 zit
  daar niet). De stand-in injecteert hem alsnog door zijn
  rewrite-abstract erin te plakken. `train_gnn_demo.py` doet hetzelfde
  bij candidate-inferentie - check dat dit consistent blijft.
