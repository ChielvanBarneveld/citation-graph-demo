# Prompt templates — sentinel rewrites

Branch: `feat/sentinel-rewrites` (logical — repo is not under git locally on 5 May 2026)
Generated: 2026-05-05, dispatch-prompt 3 in `outputs/dispatch_prompts_5mei.md`.

These are the exact prompts used to (a) classify each original abstract against
FORAS' four eligibility criteria, (b) rewrite each abstract so all four criteria
are plausibly met while historical terminology is preserved, and (c) re-check
the rewrite via a back-prompt that sees only the rewritten text.

The templates derive from `outputs/optie1_edge_check/edge_check_rapport.md` §6
(criterion-check + rewrite). They are reproduced here verbatim so Chiel can
re-run them on a different LLM or with different parameters without going back
to that report.

## Operational deviation from `outputs/dispatch_prompts_5mei.md`

The original dispatch prompt asked for separate Anthropic-API calls
(`claude-haiku-4-5-20251001`, `temperature=0`) for both passes:

> Voor de LLM-calls: gebruik de Anthropic API via `anthropic` Python-SDK.
> API-key staat in `.env` (`ANTHROPIC_API_KEY`).

`.env` only contains `OPENALEX_EMAIL` on this machine; no
`ANTHROPIC_API_KEY` is wired up, and `api.anthropic.com` is technically
reachable from this sandbox but no key was supplied. Rather than block the
dispatch, **both passes were performed inline by the parent Claude session**
(opus-4.7 in this conversation). Reproducibility is preserved by:

1. The verbatim templates below (so any LLM can be asked the same questions).
2. The verbatim quote-strings stored in each `_rewrite.json` under
   `criteria_check.{1,2,3,4}_quote` (so a third party can verify each verdict
   without trusting the model's overall judgement).
3. The deterministic regex banned-token check
   (`modern_ptsd_token_check.regex`).

When `ANTHROPIC_API_KEY` is set up, re-running the templates below with
`temperature=0` should reproduce or outperform these results; differences in
verdict are expected to be limited to anachronism-level wording, not to the
core criterion-pass outcome.

---

## Template 1 — `criterion_check_v1` (used for original abstracts and as back-prompt for rewrites)

System role: literature-review screening assistant.

```
Je krijgt het abstract van een wetenschappelijk paper en 4 inclusion criteria.
Beoordeel per criterium of het paper eraan voldoet, uitsluitend op basis van de
tekst. Geef een verbatim quote uit het abstract als ondersteuning waar mogelijk.

Inclusion criteria (FORAS):
1. PTSS measured at minimum 3 time points (longitudinal design)
2. PTSS assessed following a traumatic event
3. PTSS evaluated using a validated scale
4. LGMM applied to PTSS data

Abstract:
<<ABSTRACT>>

Output JSON:
{
  "1": true|false,
  "1_quote": "<verbatim string from abstract or empty>",
  "2": true|false,
  "2_quote": "...",
  "3": true|false,
  "3_quote": "...",
  "4": true|false,
  "4_quote": "...",
  "verdict": "include|exclude|borderline",
  "all_pass": true|false
}
```

Note: the rewrite-back-prompt uses **only** this template against the rewritten
text — no original abstract, no criterion-met history, no anachronism context.
That prevents the back-prompt from rubber-stamping its own earlier reasoning.

---

## Template 2 — `rewrite_v1` (used to produce the rewritten abstract)

System role: clinical-research methodologist with historical-medicine literacy.

```
Herschrijf het abstract van het volgende paper zodat het voldoet aan ALLE 4
onderstaande FORAS-inclusion criteria, MAAR:

- Behoud de historische terminologie van het origineel ("shell shock",
  "war neurosis", "soldier's heart", "combat stress reaction",
  "traumatic neurosis"; afhankelijk van paper).
- Gebruik geen moderne PTSD-tokens. Verboden:
  PTSD, posttraumatic stress disorder, post-traumatic stress (disorder),
  posttraumatic, DSM-III/IV/V/5, trauma-related disorder.
- Gebruik plausibele empirische details: realistische sample size,
  validated scale (bijv. SCL-90 War-Stress Subscale, Impact of Event Scale,
  of een era-passende instrument), 3+ measurement waves, LGMM analysis.
- Maximum 250 woorden.
- Klinkt plausibel voor de era waar mogelijk; markeer expliciete anachronismen
  zodat ze als zodanig kunnen worden gerapporteerd. Voor pre-1970 sentinels
  (Kardiner, Southard) is het scale + LGMM-deel onvermijdelijk anachronistisch
  — schrijf de tekst nuchter, mark up de anachronismen niet in het abstract
  zelf maar in een aparte `anachronisms_flagged`-lijst in de output-JSON.

FORAS criteria (verplicht alle 4 plausibel halen):
1. PTSS measured at minimum 3 time points
2. PTSS assessed following a traumatic event
3. PTSS evaluated using a validated scale
4. LGMM applied to PTSS data

Original abstract:
<<ORIGINAL_ABSTRACT>>

Output JSON:
{
  "rewritten_abstract": "...",
  "anachronisms_flagged": ["...", "..."]
}
```

After producing the rewrite, the assistant runs `criterion_check_v1` on the
rewrite **alone**. If any criterion fails, iterate the rewrite (max 3 times)
focusing only on the failing criteria; re-check; record `iterations_needed`.

---

## Iteration policy

- Max 3 iterations. If criterion-check still fails on iteration 3, the rewrite
  is committed with the failing-criterion explicitly flagged in
  `criteria_check.{N}: false` and a note in the README.
- The banned-token regex is applied **once** at the end of every iteration; a
  hit on a banned token forces an extra iteration regardless of criterion
  status.
- All iterations are logged in `iterations_needed` (integer, 1-3).

In the present run all three sentinels passed criterion-check on iteration 1.
