"""Generate a synthetic, gold-labelled evaluation set with an LLM (Sarvam by default).

  python tools/make_synth.py --out data/synth/synth_notes.jsonl [--per-chapter 10] [--seed 11]

Why: the 60 dev notes are narrow (few chapters, English only), and the hidden set has a
150-note slice from a different distribution: every chapter, not always English. This set
estimates that slice and gives enough notes to tell real gains from noise.

How each note is made:
  - a leaf code is sampled (stratified by chapter, ~30% unspecified '.Z' codes, fixed seed);
  - the generator sees the target title plus its parent and siblings, and must write a note
    that supports the target but NOT its siblings (for '.Z': no subtype detail at all);
  - it must use natural clinician wording, not the official title, so search isn't flattered;
  - language and style are sampled; the gold is the sampled code.
Lines are appended as they finish; rerunning resumes and skips ids already written.

Caveat: the same model family may later code these notes, which can flatter scores.
Spot-check a sample by hand (tools/make_synth.py --show 10).
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from icd_index import IcdIndex, parent_of  # noqa: E402

# Swahili and Arabic were dropped after a trial: the generator (sarvam-105b) produced
# incoherent Swahili notes. Indian languages are its strength; Romance languages read fine.
LANGS = [("en", "English")] * 6 + [
    ("hi", "Hindi"), ("ta", "Tamil"), ("bn", "Bengali"), ("mr", "Marathi"),
    ("fr", "French"), ("es", "Spanish"), ("pt", "Portuguese"),
]
STYLES = {
    "typed_consult": "a structured outpatient consult note with S:, O:, A:, P: sections, using common clinical abbreviations",
    "short_consult": "a terse one- or two-line outpatient note full of abbreviations, like a busy clinician writes",
    "ambient_transcript": "a raw transcript of the doctor-patient conversation (lines starting 'Doctor:' and 'Patient:'), no headings; the doctor states the conclusion in plain words somewhere in the conversation",
}
SMALL_CHAPTERS = {"23": 3, "25": 3, "26": 3}

GEN_SYSTEM = """You write realistic synthetic outpatient clinical notes for testing a medical coding system. The notes are fictional and contain no real patient data.

Rules:
- Write the note in the requested language and style. Medical terms may stay in English if clinicians in that language commonly write them that way.
- The note must clearly support the TARGET diagnosis, and must NOT support any of the listed sibling diagnoses better than the target.
- If the target is an "unspecified" category, the note must give NO detail that would allow a more specific sibling (no subtype, no organism, no stage, no site detail beyond what the target says).
- If the target is specific, the note must state the details that distinguish it from its siblings.
- Do NOT copy the official title. Write the diagnosis the way a clinician naturally would: abbreviations, common names, local or lay terms, or a description, as fits the style and language.
- Make it realistic: some history, findings and a plan; optionally a stable unrelated background condition or a ruled-out differential. The visit's main diagnosis must remain the target.
- Use invented details only. No real names, phone numbers or ID numbers.
- Return JSON: {"text": "<the note>"}"""

GEN_USER = """Language: {lang}
Style: {style}
Patient: {patient}

TARGET diagnosis: {title}
Its parent category: {parent}
Sibling diagnoses the note must NOT fit better: {siblings}
{z_rule}"""

SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def sample_codes(idx: IcdIndex, per_chapter: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_ch: dict[str, list[str]] = {}
    placeholder = ("emergency code", "reserved", "code for", "provisional assignment")
    for c in idx.codes:
        if idx.is_leaf(c) and not any(p in idx.title(c).lower() for p in placeholder):
            by_ch.setdefault(idx.rows[c]["chapter"], []).append(c)
    picked = []
    for ch in sorted(by_ch):
        leaves = by_ch[ch]
        n = SMALL_CHAPTERS.get(ch, per_chapter)
        z = [c for c in leaves if c.endswith("Z") and parent_of(c)]
        nz = [c for c in leaves if not c.endswith(("Z", "Y"))]  # skip 'other specified'
        n_z = min(len(z), round(n * 0.3))
        picked += rng.sample(z, n_z) + rng.sample(nz, min(len(nz), n - n_z))
    return picked


def build_job(idx: IcdIndex, code: str, rng: random.Random) -> dict:
    p = parent_of(code)
    sibs = [s for s in (idx.children.get(p, []) if p else []) if s != code][:12]
    lang_code, lang = rng.choice(LANGS)
    style = rng.choice(list(STYLES))
    age = rng.choice(["child", "adolescent", "adult", "adult", "older adult"])
    sex = rng.choice(["female", "male"])
    z_rule = ("The target is UNSPECIFIED: give no subtype/organism/stage/site detail beyond the target."
              if code.endswith("Z") else "")
    return {
        "code": code, "lang": lang_code, "style": style,
        "user": GEN_USER.format(
            lang=lang, style=STYLES[style], patient=f"{age}, {sex} (adjust if the diagnosis requires otherwise)",
            title=idx.title(code), parent=idx.title(p) if p else "(top-level category)",
            siblings="; ".join(idx.title(s) for s in sibs) or "(none)", z_rule=z_rule,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synth", "synth_notes.jsonl"))
    ap.add_argument("--per-chapter", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="sarvam-105b")
    ap.add_argument("--show", type=int, default=0, help="print N existing notes and exit")
    ap.add_argument("--limit", type=int, default=0, help="generate at most N new notes (trial runs)")
    a = ap.parse_args()

    if a.show:
        rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        idx = IcdIndex.load(os.path.join(ROOT, "public", "data", "mms-2026-01.json"))
        for r in random.Random(0).sample(rows, min(a.show, len(rows))):
            print(f"--- {r['id']} [{r['lang']}/{r['style']}] gold {r['gold'][0]} {idx.title(r['gold'][0])}\n{r['text']}\n")
        return

    import openai
    client = openai.OpenAI(api_key=os.environ["SARVAM_API_KEY"],
                           base_url=os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai/v2"),
                           timeout=120, max_retries=3)
    idx = IcdIndex.load(os.path.join(ROOT, "public", "data", "mms-2026-01.json"))
    codes = sample_codes(idx, a.per_chapter, a.seed)
    rng = random.Random(a.seed + 1)
    jobs = []
    for i, c in enumerate(codes):
        j = build_job(idx, c, rng)
        j["id"] = f"syn-{i:03d}"
        jobs.append(j)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [j for j in jobs if j["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(jobs)} notes planned, {len(done)} already written, {len(todo)} to generate", flush=True)

    lock = threading.Lock()
    f = open(a.out, "a", encoding="utf-8")
    usage = {"in": 0, "out": 0, "fail": 0}

    def leaks_title(code, text):
        t = idx.title(code).lower().replace(", unspecified", "").strip()
        return len(t) > 6 and t in text.lower()

    def gen(j):
        user = j["user"]
        tin = tout = 0
        for attempt in range(3):
            r = client.chat.completions.create(
                model=a.model, max_tokens=1500, temperature=0.8, reasoning_effort="none",
                messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "note", "schema": SCHEMA, "strict": True}},
            )
            if r.usage:
                tin += r.usage.prompt_tokens or 0
                tout += r.usage.completion_tokens or 0
            text = json.loads(r.choices[0].message.content)["text"].strip()
            if len(text) < 40:
                continue
            leak = leaks_title(j["code"], text)
            if not leak:
                break
            user = j["user"] + ("\n\nYour previous note copied the official title word for word. "
                                "Rewrite it using the clinician's own wording, an abbreviation or a description.")
        else:
            if len(text) < 40:
                raise ValueError("note too short")
        return j, text, leak, (tin, tout)

    with ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(gen, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                j, text, leak, (tin, tout) = fut.result()
            except Exception as e:
                usage["fail"] += 1
                print(f"  failed: {e!r}"[:200], flush=True)
                continue
            row = {"id": j["id"], "text": text, "gold": [j["code"]], "style": j["style"],
                   "lang": j["lang"], "chapter": idx.rows[j["code"]]["chapter"], "slice": "B",
                   "title_leak": leak}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                usage["in"] += tin
                usage["out"] += tout
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    f.close()
    usd = (usage["in"] * 0.18 + usage["out"] * 0.71) / 1e6
    print(f"done. failures {usage['fail']}; tokens {usage['in']} in / {usage['out']} out (~${usd:.3f})")


if __name__ == "__main__":
    main()
