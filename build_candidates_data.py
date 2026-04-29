"""Build candidates.parquet + candidate_foras_edges.parquet for the v5 dashboard."""
from __future__ import annotations
import json
import re
import pandas as pd
import numpy as np

ID_RE = re.compile(r"w\d+", re.IGNORECASE)


def short_id(s):
    if not isinstance(s, str):
        return None
    m = ID_RE.search(s)
    return m.group(0).upper() if m else None


def parse_refs(s):
    if not isinstance(s, str) or not s.strip():
        return []
    s = s.strip()
    try:
        x = json.loads(s)
        if isinstance(x, list):
            return [short_id(v) for v in x if short_id(v)]
    except Exception:
        pass
    return [short_id(p) for p in re.split(r'[;,\s]+', s) if short_id(p)]


THESIS = "/sessions/ecstatic-sleepy-cori/mnt/Thesis"
OUT_DIR = "/tmp/citation-graph-demo/data"

# Load FORAS with proper id parsing
foras = pd.read_csv(f"{THESIS}/data/van_de_Schoot_2025.csv", low_memory=False)
foras["pid"] = foras["openalex_id"].fillna("").apply(short_id)
foras = foras[foras["pid"].notna() & (foras["pid"].str.len() > 0)].copy()
foras_pids = set(foras["pid"])
ft_pids = set(foras[foras["label_included"] == 1]["pid"])
ti_pids = set(foras[foras["label_abstract_included"] == 1]["pid"])
print(f"FORAS: {len(foras_pids)} papers, FT={len(ft_pids)}, TIAB-broad={len(ti_pids)}")


def stage_for_row(r):
    if r.get("label_included") == 1:
        return "ft_included"
    if r.get("label_abstract_included") == 1:
        return "tiab_included"
    return "screened"


foras["stage"] = foras.apply(stage_for_row, axis=1)
pid_to_stage = dict(zip(foras["pid"], foras["stage"]))

print("Parsing FORAS referenced_works...")
foras_refs = {}
for _, r in foras.iterrows():
    foras_refs[r["pid"]] = set(parse_refs(r.get("referenced_works")))
print(f"  papers with refs: {sum(1 for v in foras_refs.values() if v)}")

# Candidates
cand = pd.read_csv(f"{THESIS}/data/historical-terminology/candidates.csv", low_memory=False)
print(f"Candidates: {len(cand)}")

# Forward edges (candidate -> FORAS via candidate's own referenced_works)
fwd_ft, fwd_tiab, fwd_all = [], [], []
for _, r in cand.iterrows():
    rw = str(r.get("referenced_works") or "").strip()
    refs = [short_id(p) for p in re.split(r'[;,\s]+', rw) if short_id(p)] if rw else []
    fwd_ft.append(sum(1 for x in refs if x in ft_pids))
    fwd_tiab.append(sum(1 for x in refs if x in ti_pids))
    fwd_all.append(sum(1 for x in refs if x in foras_pids))

# Reverse edges (FORAS -> candidate)
cand_set = set(cand["openalex_id"].astype(str))
rev_ft = {pid: 0 for pid in cand_set}
rev_tiab = {pid: 0 for pid in cand_set}
rev_all = {pid: 0 for pid in cand_set}
for fpid, refs in foras_refs.items():
    is_ft = fpid in ft_pids
    is_tiab = fpid in ti_pids  # broad: includes FT
    for cid in (refs & cand_set):
        rev_all[cid] += 1
        if is_ft:
            rev_ft[cid] += 1
        if is_tiab:
            rev_tiab[cid] += 1

cand["edges_to_foras_ft"] = fwd_ft
cand["edges_to_foras_tiab"] = fwd_tiab
cand["edges_to_foras_all"] = fwd_all
cand["edges_from_foras_ft"] = cand["openalex_id"].map(rev_ft).fillna(0).astype(int)
cand["edges_from_foras_tiab"] = cand["openalex_id"].map(rev_tiab).fillna(0).astype(int)
cand["edges_from_foras_all"] = cand["openalex_id"].map(rev_all).fillna(0).astype(int)
cand["edges_total"] = cand["edges_to_foras_all"] + cand["edges_from_foras_all"]


def cand_stage(row):
    if not row["in_foras"]:
        return "candidate_external"
    return pid_to_stage.get(row["openalex_id"], "screened")


cand["foras_stage"] = cand.apply(cand_stage, axis=1)
cand["era"] = pd.cut(
    cand["year"].fillna(-1), bins=[-2, 1919, 1944, 1979, 1999, 2025],
    labels=["<1920", "1920-44", "1945-79", "1980-99", "2000+"]
).astype(str)

print("\nIn-FORAS overlap stages:")
print(cand[cand["in_foras"]]["foras_stage"].value_counts())

n_any_to_ft = ((cand["edges_to_foras_ft"] > 0) | (cand["edges_from_foras_ft"] > 0)).sum()
n_any_to_tiab = ((cand["edges_to_foras_tiab"] > 0) | (cand["edges_from_foras_tiab"] > 0)).sum()
n_any_to_foras = (cand["edges_total"] > 0).sum()
print(f"\nEdge summary (per candidate, any direction):")
print(f"  >=1 edge to/from FT-included   : {n_any_to_ft}")
print(f"  >=1 edge to/from TIAB-broad    : {n_any_to_tiab}")
print(f"  >=1 edge to/from any FORAS     : {n_any_to_foras}")
print(f"  total cross-edges (sum)        : {int(cand['edges_total'].sum())}")

keep_cols = [
    "openalex_id", "doi", "title", "year", "era", "language", "type",
    "pools", "found_via_terms", "n_terms",
    "in_foras", "foras_stage", "foras_label_FT", "foras_label_TIAB",
    "edges_to_foras_ft", "edges_to_foras_tiab", "edges_to_foras_all",
    "edges_from_foras_ft", "edges_from_foras_tiab", "edges_from_foras_all",
    "edges_total", "abstract",
]
cand_out = cand[keep_cols].copy()
cand_out.to_parquet(f"{OUT_DIR}/candidates.parquet", index=False)
print(f"\nWritten: {OUT_DIR}/candidates.parquet ({len(cand_out)} rows)")

# Cross-edges list
cross = []
for _, r in cand.iterrows():
    cid = r["openalex_id"]
    rw = str(r.get("referenced_works") or "").strip()
    refs = [short_id(p) for p in re.split(r'[;,\s]+', rw) if short_id(p)] if rw else []
    for x in refs:
        if x in foras_pids:
            cross.append((cid, x, "candidate_to_foras", pid_to_stage.get(x, "screened")))
for fpid, refs in foras_refs.items():
    fst = pid_to_stage.get(fpid)
    for cid in (refs & cand_set):
        cross.append((fpid, cid, "foras_to_candidate", fst))
ce_df = pd.DataFrame(cross, columns=["src", "dst", "direction", "foras_stage"])
ce_df.to_parquet(f"{OUT_DIR}/candidate_foras_edges.parquet", index=False)
print(f"Cross-edges file: {len(ce_df)} rows")
