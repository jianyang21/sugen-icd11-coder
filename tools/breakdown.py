"""Break score.py --per-note results down by language, style, chapter and title leak.

  python tools/breakdown.py --notes data/synth/synth_notes.jsonl --per-note out/x_per_note.jsonl [--trace out/x_trace.jsonl]

With --trace (from CODER_TRACE), also reports shortlist recall: whether the gold code
was in the shortlist sent to the select step (an upper bound on what selection can do).
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--per-note", required=True)
    ap.add_argument("--trace")
    a = ap.parse_args()

    meta = {r["id"]: r for r in map(json.loads, open(a.notes, encoding="utf-8")) if r}
    res = [json.loads(l) for l in open(a.per_note, encoding="utf-8") if l.strip()]
    trace = {}
    if a.trace:
        trace = {t["id"]: t for t in map(json.loads, open(a.trace, encoding="utf-8"))}

    def pts(rows):
        n = len(rows)
        t1 = sum(r["top1"] for r in rows) / n
        t3 = sum(r["top3"] for r in rows) / n
        return n, t1, t3, 100 * (2 / 3 * t1 + 1 / 3 * t3)

    def show(title, key):
        groups = defaultdict(list)
        for r in res:
            groups[key(r)].append(r)
        print(f"\n{title:<14} {'n':>4} {'top1':>6} {'top3':>6} {'pts%':>6}" + ("  shortlist" if trace else ""))
        for g in sorted(groups, key=lambda x: str(x)):
            n, t1, t3, p = pts(groups[g])
            line = f"{str(g):<14} {n:>4} {t1:>6.2f} {t3:>6.2f} {p:>6.1f}"
            if trace:
                rows = [r for r in groups[g] if r["id"] in trace]
                if rows:
                    rec = sum(bool(set(r["gold"]) & set(trace[r["id"]].get("shortlist", []))) for r in rows) / len(rows)
                    line += f"  {rec:>9.2f}"
            print(line)

    n, t1, t3, p = pts(res)
    print(f"ALL n={n} top1 {t1:.3f} top3 {t3:.3f}  -> {p:.1f}% of slice points")
    show("language", lambda r: meta[r["id"]].get("lang", "en"))
    show("style", lambda r: meta[r["id"]].get("style", "?"))
    show("title_leak", lambda r: meta[r["id"]].get("title_leak", False))
    show("chapter", lambda r: meta[r["id"]].get("chapter", "?"))


if __name__ == "__main__":
    main()
