#!/usr/bin/env python3
"""Score ICD-11 predictions against gold labels.

Usage:
  python3 score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms mms-2026-01.json

The same script grades the hidden set. Only the gold file differs.

gold lines:  {"id": "...", "gold": ["1F40.0", ...], "slice": "A"|"B" (optional, default A), ...}
pred lines:  {"id": "...", "codes": ["1F40.0", "1F45", "CA40.0"]}   up to 3, best first

Rules:
- Only the first 3 codes count. A postcoordinated cluster ("NC72.2&XK8G") is scored by its stem, the part before the first & or /.
- A code that is not a category in the MMS release is invalid and counts as a miss.
- A gold note with no prediction line counts as a miss.
- Score = 75 * (2/3 top1_A + 1/3 top3_A) + 25 * (2/3 top1_B + 1/3 top3_B).
  When a slice is empty (the dev set has no B), its share is reported but not used,
  and the dev score is shown out of the slices present.
"""
import argparse
import json
import re
import sys

STEM = re.compile(r"[&/]")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: not JSON ({e})")
    return rows


def norm(code):
    if not isinstance(code, str):
        return None
    return STEM.split(code.strip())[0].strip().upper() or None


def load_valid(mms_path):
    with open(mms_path, encoding="utf-8") as f:
        data = json.load(f)
    return {norm(r["code"]) for r in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mms", required=True, help="mms-2026-01.json: the list of valid diagnostic codes")
    ap.add_argument("--per-note", help="write per-note results (JSONL) here")
    a = ap.parse_args()

    valid = load_valid(a.mms)
    gold = load_jsonl(a.gold)
    preds = {}
    dupes = 0
    for p in load_jsonl(a.pred):
        pid = p.get("id")
        if pid in preds:
            dupes += 1  # the first line for an id wins
            continue
        preds[pid] = p

    stats = {}
    invalid = missing = 0
    out = []
    for g in gold:
        s = g.get("slice", "A")
        st = stats.setdefault(s, {"n": 0, "top1": 0, "top3": 0})
        st["n"] += 1
        gset = {norm(c) for c in g["gold"]}
        p = preds.get(g["id"])
        codes = []
        if p is None:
            missing += 1
        else:
            for c in (p.get("codes") or [])[:3]:
                c = norm(c)
                if c is None or c not in valid:
                    invalid += 1
                    c = None
                codes.append(c)
        top1 = bool(codes) and codes[0] in gset
        top3 = any(c in gset for c in codes)
        st["top1"] += top1
        st["top3"] += top3
        out.append({"id": g["id"], "slice": s, "gold": sorted(gset), "pred": codes, "top1": top1, "top3": top3})

    weights = {"A": 75.0, "B": 25.0}
    total = 0.0
    present = 0.0
    print(f"notes {len(gold)}  missing predictions {missing}  invalid codes {invalid}  duplicate ids {dupes}")
    for s in sorted(stats):
        st = stats[s]
        t1 = st["top1"] / st["n"]
        t3 = st["top3"] / st["n"]
        w = weights.get(s, 0.0)
        pts = w * (2 / 3 * t1 + 1 / 3 * t3)
        total += pts
        present += w
        print(f"slice {s}: n={st['n']}  top1 {t1:.3f}  top3 {t3:.3f}  points {pts:.1f} / {w:.0f}")
    print(f"SCORE {total:.1f} / {present:.0f}")

    if a.per_note:
        with open(a.per_note, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
