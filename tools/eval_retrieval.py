"""Measure the local search on a gold-labelled JSONL file.

  python tools/eval_retrieval.py [--notes public/data/dev_notes.jsonl] [--show-misses]

Reports:
  - shortlist recall@n: is any gold code in the shortlist we would send the LLM?
    This caps the whole system's accuracy.
  - no-key top-1 / top-3: what the fallback path scores on its own.
Also writes the no-key predictions so they can be checked with the official score.py.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fallback import dx_spans, queries_from_note  # noqa: E402
from icd_index import IcdIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=os.path.join(ROOT, "public", "data", "dev_notes.jsonl"))
    ap.add_argument("--mms", default=os.path.join(ROOT, "public", "data", "mms-2026-01.json"))
    ap.add_argument("--pred-out", default=os.path.join(ROOT, "out", "dev_nokey_predictions.jsonl"))
    ap.add_argument("--show-misses", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    idx = IcdIndex.load(a.mms)
    print(f"index built in {time.time() - t0:.2f}s")

    notes = [json.loads(l) for l in open(a.notes, encoding="utf-8") if l.strip()]
    sizes = [5, 10, 20, 30, 40, 60]
    hit = {n: 0 for n in sizes}
    top1 = top3 = 0
    misses = []
    os.makedirs(os.path.dirname(a.pred_out), exist_ok=True)
    t0 = time.time()
    with open(a.pred_out, "w", encoding="utf-8") as out:
        for r in notes:
            gold = set(r["gold"])
            qs = queries_from_note(r["text"])
            sl = idx.shortlist(qs, n=max(sizes))
            for n in sizes:
                hit[n] += bool(gold & set(sl[:n]))
            pred = idx.top_leaves(qs, 3)
            out.write(json.dumps({"id": r["id"], "codes": pred}) + "\n")
            top1 += bool(pred) and pred[0] in gold
            top3 += bool(gold & set(pred))
            if not gold & set(sl[:40]):
                misses.append((r, sl[:5], pred))
    dt = time.time() - t0

    N = len(notes)
    print(f"notes {N}   {1000 * dt / N:.0f} ms/note")
    print("shortlist recall: " + "  ".join(f"@{n} {hit[n] / N:.2f}" for n in sizes))
    print(f"no-key: top1 {top1 / N:.3f}  top3 {top3 / N:.3f}  "
          f"(~{75 * (2 / 3 * top1 / N + 1 / 3 * top3 / N):.1f}/75 on slice A)")
    print(f"predictions -> {a.pred_out}")
    if a.show_misses:
        print(f"\n--- {len(misses)} notes with gold not in shortlist@40 ---")
        for r, sl, pred in misses:
            g = ", ".join(f"{c} {idx.title(c)}" for c in r["gold"])
            print(f"{r['id']} [{r['style']}] gold: {g}")
            print(f"   dx spans: {dx_spans(r['text'])[:2]}")
            print(f"   shortlist top5: {[(c, idx.title(c)[:40]) for c in sl]}")


if __name__ == "__main__":
    main()
