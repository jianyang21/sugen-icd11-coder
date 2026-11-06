# ICD-11 coder

Reads an outpatient note (typed consult, one-liner or ambient transcript, any language) and proposes up to three ICD-11 MMS codes, best first.

```
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

**Pipeline.** One generic pipeline for every note style. (1) An LLM *extract* call returns the diagnosis in ICD-style English, with specifiers (organism, acuity, site, pregnancy...) and guessed codes. (2) Local BM25 search over the 18,375 MMS titles, their ancestors' titles and generated synonyms builds a ~40-code shortlist, with parent, children and `.Z` siblings added. (3) An LLM *select* call picks 3 codes from that shortlist only, with candidates grouped by parent category. (4) Code-side rules: invalid or non-shortlist codes are dropped, a category with children becomes its leaf, `.Z` goes before `.Y`, and the answer is padded to 3. With no key, or on any API failure, the search alone answers. Each line is written and flushed as it finishes. A hard deadline (50 min), a spend guard ($5), a circuit breaker, model fallback chains and parameter repair keep every note answered inside the caps. The note is treated as untrusted data: per-call nonce delimiters, an embedded JSON schema, and validation of every reply.

## Scores

| | Dev, 60 notes (/75) | Synthetic, 235 notes (/25) |
|---|---|---|
| No LLM (search only) | **35.8** (top1 .417, top3 .600) | 6.3 (top1 .196, top3 .362) |
| LLM path, `sarvam-105b` | **67.9** (top1 .883, top3 .950) | **18.3** (top1 .702, top3 .791) |

**Estimate of the hidden score: ~70-80 / 100** if the graded model codes as well as Sarvam did here, and ~40 if every LLM call fails. We had no OpenAI or Anthropic key during development. Every LLM number above comes from Sarvam through the same code path, so the graded model (Anthropic `claude-haiku-5-5` or OpenAI `gpt-6-luna` by default) is unmeasured. Dev scores were also used for tuning, and differ by ~3 points between identical runs. I discount the measured ~86 accordingly.

## Data added

- `data/synonyms.jsonl`: up to 5 English alternative names for 9,017 codes (eponyms, abbreviations, lay terms), generated with `sarvam-105b` (`tools/make_synonyms.py`, ~$0.44). Indexed at 0.3 of the title weight, and only words not already in the title. WHO index terms would be the authoritative source; they need API registration we did not do.
- `data/synth/synth_notes.jsonl`: 235 synthetic gold-labelled notes (`tools/make_synth.py`) used for evaluation only, never for search. They are not in the image.

## Evaluation beyond the 60 dev notes

- **Synthetic set** (a stand-in for the generalisation slice): leaf codes stratified over all 26 chapters, ~30% `.Z`, 40% non-English (Hindi, Tamil, Bengali, Marathi, French, Spanish, Portuguese), all three styles. The generator sees the target's siblings and must not fit them. Notes that copy the official title are retried, then flagged (31 notes; the score without them is 70.6%). Limits: chapter-uniform sampling, some mislabelled notes, and the same model family writes and codes the notes. Swahili was dropped after a trial because the output was incoherent.
- **Breakdowns** by language, style and chapter (`tools/breakdown.py`). The weakest areas are Spanish, French and Portuguese notes (49-67% of points) and one-line notes (66%).
- **Replays without new LLM calls:** `tools/replay_shortlist.py` re-runs retrieval on recorded extractions; `tools/replay_select.py` re-runs only the select step on identical shortlists. A/B results are compared only on notes where every arm's API calls succeeded, because rate-limit failures had been mixing into model quality.
- **Robustness tests** (28, `pytest tests`): messy input, LLM errors, junk and hung LLM calls, prompt injection, budget guard, breaker, model chain, parameter repair. A clean `git clone` → `build --no-cache` → graded command gives 60/60 predictions and 0 invalid codes.

## Where the AI tooling was wrong and I caught it

>Adding synonyms to the index: the AI's implementation looked like a small win in its own retrieval numbers. A unit test then failed: an adult note saying "uncomplicated falciparum malaria" now ranked *congenital* falciparum malaria (KA64.20) first. The cause was that generated synonyms ("Neonatal falciparum malaria") repeated words already in other codes' titles, so those codes got double credit. The fix: each code's synonym field keeps only words absent from its own title. Dev shortlist recall rose from .967 to .983, and the no-LLM top-1 from .367 to .417. In the same step, the AI's replay tool had overstated the gain by truncating one long shortlist instead of rebuilding each size the way the live coder does.

See `DECISIONS.md` for every choice, its evidence and its commit.

**Hours spent:** 8 hrs
