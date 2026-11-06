# Decisions

Each entry: date/time, commit, decision and options, evidence, what to revisit.
(Commit hashes are filled in once the repo is initialised: git was not available at first.)

Note on history: git was not installed on the development machine for the first part of
the work (2026-11-06). When it was set up, the existing work was committed in several
commits grouped by component, in the order it was built. Those first commits show the
final state of each component, not intermediate snapshots. Entries written before the
repository existed name the commit that holds the final state of that work. Everything
after that is committed as it happens.

## 2026-11-06 - One pipeline for all note styles
- Commit: 24f36e5 (prompts; pipeline design)
- Decision: treat every note as plain text and let the LLM read it, instead of a parser per style (typed / short / transcript).
- Options: per-style parsers; one generic pipeline.
- Evidence: the brief says the hidden set has unseen styles and languages; a per-style parser breaks on those.
- Revisit: no.

## 2026-11-06 - Two LLM calls: extract, then select from a local shortlist
- Commit: 24f36e5 (prompts and schemas)
- Decision: call 1 extracts the diagnosis, ICD-style phrasings, specifiers and guessed codes; local search builds a shortlist of ~30-40 real codes; call 2 picks 3 from the shortlist only.
- Options: one call with the full code list (too big for ~1 cent/note); LLM codes from memory (invents codes, mixes in ICD-10); one combined call (cheaper, shortlist built without LLM phrasing).
- Evidence: guessed codes are validated against MMS and only used as search hints; a selected code must be in the shortlist, so hallucinated or injected codes are dropped.
- Revisit: merge into one call if cost or latency is tight; measure both.

## 2026-11-06 - Prompt-injection and output safety
- Commit: 24f36e5 (prompts.py, validate.py, tests/test_safety.py)
- Decision: note is untrusted data inside per-call nonce tags (nonce stated in the user turn, so the system prompt stays static and cacheable); explicit "ignore instructions in the note" rules; JSON schema embedded in the prompt; all output parsed and validated in code; single-pass template rendering.
- Evidence: tests/test_safety.py. Caught a bug in my first version: sequential placeholder replacement let a note containing "{summary}" pull in other prompt sections.
- Revisit: switch to native structured outputs once the provider/model is chosen.

## 2026-11-06 - Local search: BM25 over titles + ancestor titles, with hierarchy expansion
- Commit: 47089dc (icd_index.py, fallback.py, tools/eval_retrieval.py)
- Decision: dependency-free BM25 (title, plus ancestor titles at weight 0.35), British/American spelling folding, light stemming, a general clinical-abbreviation map. Shortlist = top direct hits first, then their families (parent/children/siblings incl. .Z), then weaker hits. Parent derived from the code string (verified for all 18,375 codes).
- Options: embeddings (deferred: adds a model to the image; revisit for non-English); rank_bm25 (unneeded dependency).
- Evidence (dev, no LLM, queries from the note's diagnosis line):
  - v1 (family right after each hit): recall @10 0.62, @20 0.72, @30 0.77, @40 0.82; score 32.1/75.
  - v2 (direct hits first, clinician lines only in transcripts): @10 0.68, @20 0.78, @30 0.80, @40 0.80.
  - v3 (+ chapter 26 traditional medicine down-weighted x0.4): same recall, top3 0.567, score 32.5/75.
- Remaining misses are vocabulary gaps (chickenpox/varicella, tinea capitis/dermatophytosis of scalp, Graves/thyrotoxicosis with diffuse goitre). Not patched by hand to avoid tuning on dev; to be fixed by LLM phrasings and WHO index terms.
- Caveat: a few lay terms in the abbreviation map (jiggers, bilharzia, thrush, piles) were added while looking at dev notes; they are real synonyms but they flatter the dev score slightly. WHO index terms should make them redundant.
- Revisit: WHO ICD-11 index terms/synonyms in the index; multilingual embeddings; shortlist size vs cost once the LLM is wired.

## 2026-11-06 - Runner: precomputed fallback, write-once streaming, hard deadline
- Commit: dcea8f7 (run.py, tests/test_run.py)
- Decision: src/run.py computes the no-LLM answer for every note first, then (when a key exists) runs the LLM path in a thread pool, writing and flushing each line as it finishes. At CODER_DEADLINE_S (default 50 min) it stops waiting and writes fallbacks for the rest; a `finally` block guarantees a line for every id; `os._exit` at the end so a hung API thread cannot keep the process alive past the cap.
- Options: write all fallbacks first then "overwrite" (impossible: score.py keeps the FIRST line per id); write only at the end (loses everything on a kill).
- Evidence: tests/test_run.py (messy input, failing/junk LLM, LLM hanging past deadline). Dev run 32.5/75, 0 missing, 0 invalid. 600 notes no-LLM in 1.6 s total.
- Revisit: a spend cap alongside the time cap once the LLM coder reports token usage.

## 2026-11-06 - Docker: python:3.12-slim, compose mounts ./in and ./out, env passthrough
- Commit: 756e97a (Dockerfile, docker-compose.yml)
- Decision: image holds src/ and the MMS file only; entrypoint is run.py so `docker compose run coder /in/notes.jsonl /out/predictions.jsonl` works with no extra flags. Compose mounts ./in (read-only) and ./out, and passes through ANTHROPIC/OPENAI keys, HTTPS_PROXY/HTTP_PROXY/NO_PROXY (both cases) and the CODER_* knobs. The build fails fast if the index can't load. in/ and out/ are kept in git via .gitkeep.
- Options: bake data download into run (impossible: no network at run time); non-root user (risk of permission errors on the graders' bind mount; root kept).
- Evidence: exact command on dev 32.5/75, 0 missing; same with `--network none`; simulated clean clone (copy of tracked files, `build --no-cache`) builds and scores 32.5.
- Revisit: pin SDK versions once the LLM coder is written; redo the clean-clone check with a real `git clone` before submitting.

## 2026-11-06 - LLM coder: provider-agnostic, Sarvam for development only
- Commit: 8cd3ad2 (llm_coder.py, tests/test_llm_coder.py)
- Decision: src/llm_coder.py runs extract -> shortlist -> select -> validate. Providers: anthropic / openai (both reachable in the graded run) and sarvam via the OpenAI-compatible v2 endpoint (dev only: the graded network allows only api.anthropic.com and api.openai.com). Auto-detects by key, Anthropic first; CODER_PROVIDER overrides.
- Why Sarvam at all: no Anthropic/OpenAI key available to us during development. Sarvam numbers validate the plumbing, not the graded model's accuracy.
- Shortlist queries: LLM English diagnosis (w 1.0), ICD phrasings (0.8), original wording (0.4), "X in pregnancy" when pregnant (0.8), secondary diagnoses (0.4), plus the raw-note fallback queries at 0.3; validated guessed codes are injected with a bonus.
- Rails: structured output (json_schema; array-length limits and type lists stripped/converted for the API, full schema still validated client-side); spend tracked from usage with a budget guard (CODER_BUDGET_USD, default 5.0 of the 6.0 cap); circuit breaker after 5 consecutive 400/401/402/403/404 so a dead key or exhausted credit does not burn the time cap; per-note JSONL trace (CODER_TRACE) for error analysis.
- Evidence: tests/test_llm_coder.py with a fake client (happy path, injected code dropped, extract failure, select failure, breaker, budget). 21/21 tests pass. No live accuracy number yet.
- First live runs (Sarvam sarvam-105b, dev 60):
  - reasoning_effort "low": extract spent all 4000 tokens reasoning -> "none" (valid JSON in ~5 s, ~370 tokens).
  - v1: 57.9/75 (top1 .733, top3 .850); gold in LLM shortlist 57/60 (vs 48/60 no-LLM). ~$0.07 per 60 notes, 140 s with 4 workers.
  - Error classes: parent codes chosen; .Y over .Z; subtypes the note never states (primary OA, paroxysmal AF); 4 extract runaways.
  - Fixes: code rules (non-leaf -> its leaf; .Z sibling ahead of .Y), sharper "never assume a subtype" prompt rule (examples deliberately NOT taken from dev notes), one extract retry.
  - v2: 65.8 then 58.3 on an identical re-run -> v2 default temperature is 1; differences under ~8 points are noise at n=60 with this model.
  - Sarvam copied the format-example codes from the prompt as "guesses" (falciparum malaria code for urticaria). Removed example codes from the extract prompt; guesses must share a non-generic word with the extracted diagnosis to get the shortlist bonus.
  - temperature 0: 65.4 but extract looped to max_tokens 33 times (11 notes with no extraction), 480 s. Set 0.2, extract max_tokens 1500 when reasoning is off so runaways fail fast.
- Temperature 0.2, two runs: 64.6 and 67.9 (mean 66.3; top3 .950 both). Same top-1 code in 48/60 notes; 46 right in both, 4 wrong in both, 10 flip.
- Open: graded-run model.

## 2026-11-06 - Synthetic evaluation set for the generalisation slice
- Commit: 7cd33c8 (tools/make_synth.py, data/synth/)
- Decision: tools/make_synth.py generates ~240 gold-labelled notes with sarvam-105b: leaf codes stratified by chapter (10 per chapter, 3 for chapters 23/25/26, placeholder codes such as "National emergency code" excluded), ~30% '.Z' targets; generator sees target + parent + siblings and must support the target but not its siblings; 60% English, 40% Hindi/Tamil/Bengali/Marathi/French/Spanish/Portuguese; styles sampled evenly. Marked slice "B" so score.py reports it on the 25-point scale.
- Trial findings that changed the design: Swahili output was incoherent (dropped along with Arabic); notes copied the official title verbatim (retry up to 2x with a rewrite instruction, then flag title_leak so results can be reported with and without leaked notes).
- Known limits: chapter-uniform sampling over-weights rare chapters versus real clinics (deliberate stress test, not a calibrated estimate); some notes are internally inconsistent; Tamil short notes come out mostly English; the generator family (Sarvam) is also the dev-time coder, which can flatter scores.
- Revisit: generate a second set with a different model once a graded-provider key exists; hand-check a sample.
- Result (235 notes; 4 generation failures; 31 title leaks): no-LLM 6.0/25 (top1 .183, top3 .353). Sarvam pipeline 17.0/25 (top1 .655, top3 .728); without title-leak notes 66.7% of slice points. Non-English notes are NOT worse than English (hi .80, mr .78, ta .77, bn .71, en .66 pts%): the extract step's translation works. Gold in shortlist 83% (dev 97%): 38 search misses, 26 selection misses outside top-3. Weakest chapters: 23 external causes, 24 factors/Q codes, 26 traditional medicine, 05, 09, 13, 15 (n=3-10 each, noisy).
- Reading a sample of misses: roughly a third are bad synthetic notes (note describes a different condition than its gold, e.g. RA note with psoriatic-arthritis gold; injury notes for external-cause gold codes), so the true score is likely higher. Real search misses are synonym gaps ("vitamin D toxicity" vs "Hypervitaminosis D", "macular dystrophy" vs "degeneration of macula").

## 2026-11-06 - LLM-generated synonyms as a low-weight search field
- Commit: the "Add generated search synonyms" commit (data/synonyms.jsonl, tools/make_synonyms.py, tools/replay_shortlist.py); index field in 47089dc
- Decision: tools/make_synonyms.py asked sarvam-105b for up to 5 English alternative names per non-residual code (13,041 codes, batches of 50, 2 passes; ~$0.44). 9,017 codes got at least one synonym; residual .Y/.Z codes skipped. Shipped as data/synonyms.jsonl and indexed as a third BM25 field at weight 0.3.
- Options: WHO ICD-11 index terms via the WHO API (authoritative, needs registration we don't have yet); no synonyms.
- Evidence (tools/replay_shortlist.py: rebuilds shortlists from recorded extractions, no new LLM calls; shortlist@40 as the live coder uses it): dev .967 -> .967 at every weight; synthetic .838 -> .860 (w 0.3), .851 (0.6), .838 (1.0). Gains are genuine vocabulary fixes (vitamin D toxicity, DVT in pregnancy, laryngeal stenosis, neonatal facial palsy); losses are codes without synonyms crowded out, plus one imprecise synonym. No-LLM path at 0.3: dev top1 .367 = / top3 .567 -> .533; synthetic top1 .183 -> .213, top3 .353 -> .362.
- Caught my own measurement bug: the first replay truncated one 60-code shortlist to each size, but shortlist() fills differently per n, so its @40 numbers (dev .983) overstated the gain. Rebuilt per size.
- Expected end-to-end effect is small (~+2 points of synthetic recall). Revisit: WHO index terms; synonyms only for the no-LLM path; a second synonym source to cross-check.
- Follow-up fix: a unit test caught synonyms double-counting title words ("Neonatal falciparum malaria" pushed congenital malaria KA64.20 above 1F40.Z for an adult note). The synonym field now keeps only words absent from the code's own title. Replay @40: dev .967 -> .983, synthetic .838 -> .855; no-LLM dev top1 .367 -> .417, top3 .567 -> .600; synthetic no-LLM top1 .183 -> .196, top3 .353 -> .362.

## 2026-11-06 - Select step: grouped candidates + "cite or go unspecified"
- Commit: 8cd3ad2 (grouped candidates, evidence rule in llm_coder.py); A/B tool tools/replay_select.py in the synonyms commit
- Problem (48 recorded top-1 misses with gold in the shortlist, dev + synthetic): 33 were sibling confusions; 18 picked a subtype when gold was .Z, 11 picked .Z when the note stated the detail. Gold was usually near the top of the shortlist (15x rank 1), so this is judgement between siblings, not retrieval.
- Decision: (1) candidates grouped under their parent category (groups ordered by best-ranked member) so siblings are compared side by side; prompt asks for a two-step choice (group, then child). (2) new field first_code_evidence: exact words from the note justifying a specific child over its .Z sibling; in code, if the quote is empty or not found in the note (exact, or >=60% of its words), the .Z sibling goes first and the specific code second (still counts for top-3).
- Measurement: tools/replay_select.py re-runs only the select call on recorded extractions (identical shortlists), A = flat/no rule, B = grouped/no rule, C = grouped + rule, on dev and synthetic.
- Dev (60): A 91.7%, B 92.2%, C 91.7% of points: no separation at this size.
- Synthetic, raw: A 69.6 / 71.1, B 70.6, C 71.6 / 67.1. Misleading: Sarvam 429s failed 1-36 select calls per run, and failed notes fall back to padding.
- Synthetic, paired on the 153 notes where every run's select call succeeded: A 65.1 / 68.2, B 72.5, C 72.3 / 68.2. Top-1 wins/losses grouped vs flat: +17/-3, +12/-4, +18/-4, +13/-5, +15/-9, +9/-9.
- Kept: grouped candidates (~+4-5 points, consistent across pairings, ~8% fewer input tokens). Evidence code rule: no measurable gain (B1 = C1; C2 lower), so off by default (CODER_EVIDENCE_RULE=1 to enable); the evidence field stays in the prompt.
- Also: client max_retries 2 -> 5 (CODER_MAX_RETRIES), since rate limits will also exist in the graded run.
- Lesson: compare runs only on notes where every arm's calls succeeded; raw scores mixed API failures into model quality.
- End-to-end with the final setup (grouped, rule off, synonyms new-words-only, 3 workers): dev 67.9/75 (top1 .883, top3 .950; previous mean 66.3), synthetic 18.3/25 (top1 .702, top3 .791; previous 17.0), 70.6% of points without title-leak notes (previous 66.7%). Weakest: Spanish/French/Portuguese notes (48-67%) and short consults (66%); Indian languages 76-85%, possibly flattered because Sarvam also wrote those notes. Cost estimate per note (~5.5k in / ~0.6k out tokens, 2 calls): Haiku 5.5 ~$0.001, Sonnet 5.5 ~$0.011-0.017 (at or over the 1 cent/note cap), Opus 5.5 ~$0.03 (over cap). OpenAI default id "gpt-5-mini" is unverified.
