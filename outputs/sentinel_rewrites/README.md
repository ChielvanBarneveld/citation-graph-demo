# Sentinel rewrites — Optie 1 data-prep

Branch: `feat/sentinel-rewrites` (logical only — repo not under git on this
machine on 5 May 2026).
Generated: 2026-05-05 in dispatch-werkstand modus 3 (5 mei festival, Wageningen).
Source dispatch prompt: `outputs/dispatch_prompts_5mei.md` § "Dispatch-prompt 3".

This folder contains, for each of the three sentinel papers (Solomon 1993 /
Kardiner 1941 / Southard 1920), a pair of JSON files:

- `<id>_original.json` — title, year, authors, journal, OpenAlex ID, DOI,
  original abstract (or a manually-sourced description where no abstract
  exists), `cited_by_foras_*` counts, and a per-paper criterion-met analysis.
- `<id>_rewrite.json` — the rewritten abstract, `original_criteria_met`,
  `criteria_check` with verbatim quotes, anachronisms flagged, banned-token
  check, and iteration count.

`prompt_templates.md` documents the exact criterion-check and rewrite prompts.

## Aggregaat-tabel

| Sentinel | Year | Difficulty | Orig. crit. (1/2/3/4) | Rewrite crit. (1/2/3/4) | Iter. | Words | Anachronisms | Banned-token check |
|---|---:|:-:|:-:|:-:|:-:|---:|---:|:-:|
| **Solomon** *Combat Stress Reaction: The Enduring Toll of War* | 1993 | EASY | ✗ ✓ ✗ ✗ | ✓ ✓ ✓ ✓ | 1 | 222 | 2 | pass |
| **Kardiner** *The Traumatic Neuroses of War* | 1941 | MEDIUM | ✗ ✓ ✗ ✗ | ✓ ✓ ✓ ✓ | 1 | 214 | 4 | pass |
| **Southard** *Shell-Shock and Other Neuropsychiatric Problems* | 1920 | HARD | ✗ ✓ ✗ ✗ | ✓ ✓ ✓ ✓ | 1 | 205 | 4 | pass |

All three rewrites pass the four FORAS inclusion criteria after one iteration.
None contain any banned modern-PTSD token (`PTSD`, `posttraumatic stress
disorder`, `post-traumatic stress`, `posttraumatic`, `DSM-III/IV/V/5`,
`trauma-related disorder`).

## Per-sentinel paragrafen

### Solomon 1993 — *Combat Stress Reaction: The Enduring Toll of War* (EASY)

Original (chapter-list "abstract" from OpenAlex; 22/7/1 cited-by-FORAS
any/TIAB/FT). The 1993 monograph itself summarises Solomon's Israeli-veteran
longitudinal research programme around the 1982 Lebanon war. Crit. 2 (post-
trauma) is satisfied by the subject; crit. 1, 3 and 4 are not derivable from
the chapter list. The rewrite reconstructs a plausible empirical
formulation: 382 combat-stress-reaction casualties + 334 controls, three
measurement waves at 1/2/3 years post-exposure, SCL-90 War-Stress Subscale and
IES as validated instruments, LGMM identifying four trajectory classes
(resilient / recovery / chronic / delayed-onset). The historical term
"combat stress reaction" is preserved throughout. **Anachronisms**: LGMM
(Muthén & Shedden 1999) postdates the 1993 publication; SCL-90 (1973) and IES
(1979) are era-plausible.

### Kardiner 1941 — *The Traumatic Neuroses of War* (MEDIUM)

Original is a 258-page Psychosomatic Medicine Monograph (II–III, NRC
Washington); 17/1/0 cited-by-FORAS counts. The original has **no
abstract_inverted_index in OpenAlex**, and the sandbox cannot reach
api.openalex.org (cowork-egress-blocked). The original-abstract record is
therefore manually sourced (`abstract_source: "manual"`) — a synthesised
descriptive summary of Kardiner's well-known argument (physioneurosis +
five cardinal features + 15-year VA follow-up). Crit. 2 is satisfied; the
other three are not. The rewrite reconstructs a hypothetical empirical
study around Kardiner's casuïstiek: 50 WWI veterans at the U.S. Veterans'
Bureau, four equally-spaced visits, a fictional but plausibly-named
"Veterans' Bureau War-Neurosis Inventory" (36 items, κ = .79 inter-rater
reliability), LGMM identifying three classes (stable-symptomatic /
oscillating / partial-recovery). The historical terms "war neurosis",
"traumatic neurosis" and "physioneurosis" are preserved. **Anachronisms**: the
Inventory is fictional, Cohen's κ (1960) and LGMM (1999) postdate 1941, and
the underlying multi-wave validated-scale design is itself anachronistic to
1941 clinical practice.

### Southard 1920 — *Shell-Shock and Other Neuropsychiatric Problems* (HARD)

Original is a posthumous compilation of 589 case histories of war neuroses
and psychoses; 0/0/0 cited-by-FORAS counts (P2 negative-control case in
`context/hypothesis.md`: zero edge density → expected unfindable in any
citation-based method). Crit. 2 is satisfied; the other three are not. The
rewrite reconstructs a 120-soldier longitudinal field study around the 1916
Battle of the Somme, with four observation points and a fictional "Maghull
Shell-Shock Schedule" (24 items; r = .82 split-half reliability), LGMM
identifying three classes (rapid-remission / protracted-symptom / relapsing).
The historical terms "shell-shock" and "war neuroses" are preserved.
**Anachronisms** are heavy: LGMM (1999) and the Maghull Schedule are
ahistorical; split-half reliability (Spearman-Brown 1910) was barely
available; reporting a 4-wave validated-instrument cohort study from a
clearing-station in 1916 is methodologically beyond what the era supported.
The HARD label is therefore a methodological-anachronism marker as much as a
citation-edge marker.

## Rol in het experiment

These six JSON files are inputs for two downstream consumers:

1. **Dispatch-prompt 4 (`feat/candidates-tab`)** — sentinel cards in tab 4
   render the original abstract next to the rewrite plus the criterium-check
   verdicts (✓/✗ per criterion).
2. **Dispatch-prompts 1 and 2 (`feat/asreview-tab`, `feat/gnn-tab`)** — the
   rewrites form the FORAS-criterion-passing version of each sentinel that
   gets injected into the FORAS hold-out for the 70/30 ranking experiment in
   `outputs/comparison/sentinel_ranks.parquet`. Whether ASReview and the GCN
   rank a sentinel high or low when given only the rewrite measures the
   citation-structure / text-similarity contribution to recall under
   historical terminology drift.

The intended ablation runs both the rewrite and the original through the
ranker. A large rewrite-vs-original delta indicates that vocabulary alone
drives the recall difference; a small delta indicates the
citation-structure carries the signal.

## Reproduceerbaarheid

The scripts that fetched and shaped the originals (and assembled the rewrite
JSONs) live in this conversation rather than in `code/` (a deliberate
pure-data-prep deviation per the dispatch prompt's "no v5 changes" constraint
extended here to "no `code/` additions"). To re-run, copy the inline Python
from this session, or re-run the prompts in `prompt_templates.md` against any
LLM. The originals can be regenerated from
`data/historical-terminology/candidates.csv` (Solomon, Southard) and from the
manual-source provenance recorded in `kardiner_1941_original.json`.

## Open punten

- `ANTHROPIC_API_KEY` is not present in `.env`; the `anthropic` Python-SDK
  call route was therefore not used. Inline-LLM was used instead. When the
  key is set, re-running `prompt_templates.md` against
  `claude-haiku-4-5-20251001` at `temperature=0` is the canonical path.
- The repository is not under git on this machine; the requested branch
  `feat/sentinel-rewrites` is a label only. When git is initialised,
  the contents of `outputs/sentinel_rewrites/` constitute the branch's diff
  versus `main`.
- Kardiner 1941 has no OpenAlex abstract; the manual abstract is
  conservative but is not a substitute for the original 258-page text.
  Anyone re-checking against the actual Kardiner monograph should expect
  more nuance and wider scope than the manual summary captures.
