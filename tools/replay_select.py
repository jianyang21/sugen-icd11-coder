"""A/B the select step on recorded extractions: same shortlists, only selection re-run.

  python tools/replay_select.py --notes NOTES.jsonl --trace TRACE.jsonl --out OUT_PER_NOTE.jsonl
         [--flat] [--no-evidence-rule] [--workers 4]

Rebuilds each note's shortlist from its recorded extraction (deterministic), runs only
the select call with the current prompt, applies the code-side rules, pads to 3 codes
like the live coder, and scores top-1/top-3 against gold. Halves the cost of an
end-to-end run and removes extraction noise from the comparison.
Uses whatever provider/model env the live coder would (CODER_PROVIDER, keys, ...).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import llm_coder  # noqa: E402
from icd_index import IcdIndex  # noqa: E402
from validate import fill_to_three, norm_code  # noqa: E402

MMS = os.path.join(ROOT, "public", "data", "mms-2026-01.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat", action="store_true", help="flat candidate list (old format)")
    ap.add_argument("--no-evidence-rule", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    idx = IcdIndex.load(MMS)
    valid = {norm_code(c) for c in idx.codes}
    coder = llm_coder.make_coder(idx, valid)
    coder.grouped = not a.flat
    coder.evidence_rule = not a.no_evidence_rule
    notes = [json.loads(l) for l in open(a.notes, encoding="utf-8") if l.strip()]
    trace = {t["id"]: t for t in map(json.loads, open(a.trace, encoding="utf-8"))}

    def one(r):
        ext = (trace.get(r["id"]) or {}).get("extraction")
        shortlist, guesses = llm_coder.build_shortlist(idx, ext, r["text"], coder.n_short)
        tr = {"id": r["id"]}
        try:
            codes = coder.select(r["text"], ext, shortlist, tr)
        except Exception as e:
            codes, tr["select_errors"] = [], [repr(e)]
        out = list(codes)
        for c in [*guesses, *shortlist]:
            if c in valid:
                leaf = idx.to_leaf(c)
                if leaf not in out:
                    out.append(leaf)
            if len(out) >= 3:
                break
        out = fill_to_three(out, [], valid)
        gold = set(r["gold"])
        return {"id": r["id"], "gold": sorted(gold), "pred": out, "top1": bool(out) and out[0] in gold,
                "top3": bool(gold & set(out)), "evidence": tr.get("evidence", ""),
                "evidence_rule": tr.get("evidence_rule"), "errors": tr.get("select_errors")}

    with ThreadPoolExecutor(a.workers) as pool:
        res = list(pool.map(one, notes))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    N = len(res)
    t1 = sum(r["top1"] for r in res) / N
    t3 = sum(r["top3"] for r in res) / N
    fired = sum(1 for r in res if r["evidence_rule"])
    print(f"{'flat' if a.flat else 'grouped'} / evidence rule {'off' if a.no_evidence_rule else 'on'}: "
          f"n={N} top1 {t1:.3f} top3 {t3:.3f} -> {100 * (2 / 3 * t1 + 1 / 3 * t3):.1f}% of points; "
          f"evidence rule fired {fired}x; {coder.report()}")


if __name__ == "__main__":
    main()
