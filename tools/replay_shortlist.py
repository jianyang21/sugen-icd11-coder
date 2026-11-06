"""Measure retrieval changes on recorded LLM extractions, with no new LLM calls.

  python tools/replay_shortlist.py --notes NOTES.jsonl --trace TRACE.jsonl [--syn data/synonyms.jsonl]

For each traced note, rebuilds the shortlist exactly as the live coder does
(llm_coder.build_shortlist) from the recorded extraction, with and without synonyms,
and reports the share of notes whose gold code is in the shortlist at several sizes.
Also reports the no-LLM path (top-3 from the raw note) with and without synonyms.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fallback import queries_from_note  # noqa: E402
from icd_index import IcdIndex  # noqa: E402
from llm_coder import build_shortlist  # noqa: E402

MMS = os.path.join(ROOT, "public", "data", "mms-2026-01.json")
SIZES = [10, 20, 30, 40, 60]


def evaluate(idx, notes, trace):
    hit = {n: 0 for n in SIZES}
    nk1 = nk3 = 0
    for r in notes:
        gold = set(r["gold"])
        t = trace.get(r["id"])
        ext = t.get("extraction") if t else None
        # Build each size separately: shortlist() fills differently for different n
        # (direct hits = n//3), so truncating one long list would not match the live coder.
        for n in SIZES:
            sl, _ = build_shortlist(idx, ext, r["text"], n)
            hit[n] += bool(gold & set(sl))
        p = idx.top_leaves(queries_from_note(r["text"]), 3)
        nk1 += bool(p) and p[0] in gold
        nk3 += bool(gold & set(p))
    N = len(notes)
    return {n: hit[n] / N for n in SIZES}, nk1 / N, nk3 / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--syn", default=os.path.join(ROOT, "data", "synonyms.jsonl"))
    a = ap.parse_args()

    notes = [json.loads(l) for l in open(a.notes, encoding="utf-8") if l.strip()]
    trace = {t["id"]: t for t in map(json.loads, open(a.trace, encoding="utf-8"))}
    print(f"{len(notes)} notes, {sum(1 for r in notes if (trace.get(r['id']) or {}).get('extraction'))} with a recorded extraction")
    for label, syn in [("titles only", None), ("+ synonyms", a.syn)]:
        idx = IcdIndex.load(MMS, syn)
        rec, t1, t3 = evaluate(idx, notes, trace)
        print(f"{label:<12} ({len(idx.synonyms):>5} codes w/ synonyms)  LLM-path recall: "
              + "  ".join(f"@{n} {v:.3f}" for n, v in rec.items())
              + f"   | no-LLM top1 {t1:.3f} top3 {t3:.3f}")


if __name__ == "__main__":
    main()
