"""Generate search synonyms for every MMS code with an LLM (Sarvam by default).

  python tools/make_synonyms.py [--out data/synonyms.jsonl] [--batch 50] [--workers 4]

Why: ICD-11 titles alone miss common clinical vocabulary ("Graves disease" for
"Thyrotoxicosis with diffuse goitre", "vitamin D toxicity" for "Hypervitaminosis D").
Those misses mean the gold code never reaches the shortlist, so the select step cannot
pick it. The synonyms are an extra, lower-weight search field in icd_index.py.

Output: one line per code, {"code": ..., "synonyms": [...]} (English, up to 5).
Runs are resumable: codes already in the file are skipped. Built ahead of time and
shipped in the repo, since the graded run has no network except the model APIs.

Caveat: generated synonyms can be wrong. They are weighted below the official title,
and recall is measured before and after on dev and synthetic notes (tools/replay_shortlist.py).
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from icd_index import IcdIndex, parent_of  # noqa: E402

SYSTEM = """You are an expert clinical terminologist. For each ICD-11 category you are given, list other names clinicians and patients use for exactly that condition, to help a search engine find the category from a clinical note.

For each code give up to 5 short English terms:
- common clinical names and eponyms (e.g. a named syndrome or disease),
- standard abbreviations and acronyms,
- lay or colloquial terms,
- older or alternative terminology.

Rules:
- Each term must mean the SAME condition as the category, at the same level of detail. Do not give broader terms, narrower subtypes, symptoms, causes, treatments or related conditions.
- Do not repeat the title or trivially reword it (word order, singular/plural).
- If the category has no other common names (e.g. "Other specified ...", "..., unspecified", administrative or highly technical categories), return an empty list for it.
- Return every code you were given, exactly as given.

Return JSON: {"items": [{"code": "...", "synonyms": ["...", ...]}, ...]}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "synonyms": {"type": "array", "items": {"type": "string"}}},
                "required": ["code", "synonyms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synonyms.jsonl"))
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="sarvam-105b")
    ap.add_argument("--limit", type=int, default=0, help="at most N batches (trial)")
    a = ap.parse_args()

    import openai
    client = openai.OpenAI(api_key=os.environ["SARVAM_API_KEY"],
                           base_url=os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai/v2"),
                           timeout=180, max_retries=3)
    idx = IcdIndex.load(os.path.join(ROOT, "public", "data", "mms-2026-01.json"))

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["code"] for l in open(a.out, encoding="utf-8") if l.strip()}
    # Residual categories ("other specified", "unspecified") have no real synonyms: skip
    # the call and let them inherit context from their parent's title in the index.
    todo = [c for c in idx.codes if c not in done and not c.endswith(("Y", "Z"))]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    if a.limit:
        batches = batches[:a.limit]
    print(f"{len(done)} codes done, {len(todo)} to do in {len(batches)} batches", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    f = open(a.out, "a", encoding="utf-8")
    lock = threading.Lock()
    stats = {"in": 0, "out": 0, "fail": 0, "missing": 0}

    def run(batch):
        lines = []
        for c in batch:
            p = parent_of(c)
            lines.append(f"{c} | {idx.title(c)}" + (f" | parent: {idx.title(p)}" if p else ""))
        r = client.chat.completions.create(
            model=a.model, max_tokens=6000, temperature=0.2, reasoning_effort="none",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
            response_format={"type": "json_schema", "json_schema": {"name": "synonyms", "schema": SCHEMA, "strict": True}},
        )
        if r.choices[0].finish_reason == "length":
            raise RuntimeError("hit max_tokens")
        items = json.loads(r.choices[0].message.content)["items"]
        return batch, items, r.usage

    with ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(run, b) for b in batches]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                batch, items, u = fut.result()
            except Exception as e:
                stats["fail"] += 1
                print(f"  batch failed: {e!r}"[:200], flush=True)
                continue
            want = set(batch)
            got = {}
            for it in items:
                c = it.get("code", "").strip()
                if c in want:
                    title = idx.title(c).lower()
                    syns = []
                    for s in it.get("synonyms", [])[:5]:
                        s = " ".join(str(s).split())[:80]
                        if s and s.lower() != title and s.lower() not in (x.lower() for x in syns):
                            syns.append(s)
                    got[c] = syns
            stats["missing"] += len(want - set(got))
            with lock:
                for c in batch:
                    if c in got:  # codes the model skipped are retried on the next run
                        f.write(json.dumps({"code": c, "synonyms": got[c]}, ensure_ascii=False) + "\n")
                f.flush()
                if u:
                    stats["in"] += u.prompt_tokens or 0
                    stats["out"] += u.completion_tokens or 0
            if n % 25 == 0:
                print(f"  {n}/{len(batches)} batches", flush=True)
    f.close()
    usd = (stats["in"] * 0.18 + stats["out"] * 0.71) / 1e6
    print(f"done. failed batches {stats['fail']}, codes skipped by model {stats['missing']}; "
          f"tokens {stats['in']} in / {stats['out']} out (~${usd:.3f})")


if __name__ == "__main__":
    main()
