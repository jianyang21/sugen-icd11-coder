"""Entry point:  python src/run.py /in/notes.jsonl /out/predictions.jsonl

Guarantees, in priority order:
  1. Every input id gets exactly one output line (score.py keeps the first line per id,
     so we never write an id twice).
  2. Each line is written and flushed as soon as that note is done, so a time or money
     cap only costs the notes that were still in flight.
  3. Every code written is a valid MMS code, and there are 3 per note whenever possible.

Flow:
  - Read input tolerantly (BOM, blank or broken lines, missing text, duplicate ids).
  - Compute the no-LLM answer for every note up front (~3 ms each). This is the answer
    for any note the LLM path fails on or does not finish before the deadline.
  - If an LLM coder is available, run notes through it in parallel; otherwise write
    the no-LLM answers straight away.
  - At the deadline, stop waiting and write the no-LLM answer for every unfinished note.

Environment:
  CODER_DEADLINE_S  seconds from start after which no new LLM results are awaited (default 3000 = 50 min)
  CODER_WORKERS     parallel LLM workers (default 12)
  CODER_NO_LLM=1    force the no-LLM path even if a key is set
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fallback import queries_from_note  # noqa: E402
from icd_index import IcdIndex  # noqa: E402
from validate import fill_to_three, norm_code  # noqa: E402

START = time.monotonic()
DEFAULT_MMS = os.path.join(os.path.dirname(HERE), "public", "data", "mms-2026-01.json")

# Used only when a note gives the search nothing at all (e.g. empty text). Common
# outpatient codes, verified against MMS at startup; still better than a missing line.
LAST_RESORT = ["CA07.0", "1A40.Z", "GC08.Z"]


def log(msg: str) -> None:
    print(f"[{time.monotonic() - START:7.1f}s] {msg}", file=sys.stderr, flush=True)


def read_notes(path: str) -> list[dict]:
    """Return [{"id", "text"}] in input order, one per unique id. Never raises on bad lines."""
    notes, seen, bad = [], set(), 0
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                log(f"input line {n}: not JSON, skipped")
                continue
            if not isinstance(obj, dict) or obj.get("id") is None:
                bad += 1
                log(f"input line {n}: no id, skipped")
                continue
            nid = obj["id"]
            if nid in seen:
                log(f"input line {n}: duplicate id {nid!r}, skipped")
                continue
            seen.add(nid)
            text = obj.get("text")
            notes.append({"id": nid, "text": text if isinstance(text, str) else ""})
    log(f"read {len(notes)} notes ({bad} bad lines)")
    return notes


class Writer:
    """Thread-safe, write-once-per-id, flush-every-line output."""

    def __init__(self, path: str):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()
        self.done: set = set()

    def write(self, nid, codes: list[str]) -> bool:
        with self.lock:
            if nid in self.done:
                return False
            self.f.write(json.dumps({"id": nid, "codes": codes}, ensure_ascii=False) + "\n")
            self.f.flush()
            self.done.add(nid)
            return True

    def close(self):
        with self.lock:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


def no_llm_codes(idx: IcdIndex, text: str, valid: set[str], last_resort: list[str]) -> list[str]:
    try:
        ranked = idx.top_leaves(queries_from_note(text), 6)
    except Exception as e:  # never let one odd note kill the run
        log(f"fallback search error: {e!r}")
        ranked = []
    return fill_to_three(ranked, last_resort, valid)


def load_llm_coder(idx: IcdIndex, valid: set[str]):
    """Return a callable note_text -> list[str], or None for the no-LLM path."""
    if os.environ.get("CODER_NO_LLM") == "1":
        log("CODER_NO_LLM=1: no-LLM path")
        return None
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SARVAM_API_KEY")
    if not (any(os.environ.get(k) for k in keys) or os.environ.get("CODER_PROVIDER")):
        log("no API key: no-LLM path")
        return None
    try:
        from llm_coder import make_coder  # step 4; absent until then
    except ImportError:
        log("llm_coder not available yet: no-LLM path")
        return None
    try:
        return make_coder(idx, valid)
    except Exception as e:
        log(f"LLM coder failed to start ({e!r}): no-LLM path")
        return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: run.py IN_NOTES.jsonl OUT_PREDICTIONS.jsonl", file=sys.stderr)
        return 2
    in_path, out_path = argv[0], argv[1]
    mms_path = os.environ.get("CODER_MMS", DEFAULT_MMS)
    deadline = START + float(os.environ.get("CODER_DEADLINE_S", "3000"))
    workers = int(os.environ.get("CODER_WORKERS", "12"))

    writer = Writer(out_path)  # open output first: even a crash below leaves a valid file
    notes = read_notes(in_path)

    idx = IcdIndex.load(mms_path)
    valid = {norm_code(c) for c in idx.codes}
    last_resort = [c for c in LAST_RESORT if c in valid]
    log(f"index ready: {len(idx.codes)} codes")

    fallback = {}
    for n in notes:
        fallback[n["id"]] = no_llm_codes(idx, n["text"], valid, last_resort)
    log(f"no-LLM answers ready for {len(fallback)} notes")

    coder = load_llm_coder(idx, valid)
    stats = {"llm": 0, "fallback": 0}

    try:
        if coder is None:
            for n in notes:
                writer.write(n["id"], fallback[n["id"]])
                stats["fallback"] += 1
        else:
            def job(note):
                codes = coder(note["text"], note["id"])
                return fill_to_three(codes, fallback[note["id"]], valid)

            pool = ThreadPoolExecutor(max_workers=workers)
            futs = {pool.submit(job, n): n for n in notes}
            pending = set(futs)
            while pending:
                left = deadline - time.monotonic()
                if left <= 0:
                    log(f"deadline reached with {len(pending)} notes unfinished")
                    break
                finished, pending = wait(pending, timeout=min(left, 30), return_when=FIRST_COMPLETED)
                for fut in finished:
                    nid = futs[fut]["id"]
                    try:
                        codes = fut.result()
                        src = "llm"
                    except Exception as e:
                        log(f"{nid}: LLM path failed ({e!r}), using fallback")
                        codes, src = fallback[nid], "fallback"
                    if writer.write(nid, codes or fallback[nid]):
                        stats[src] += 1
                done = len(writer.done)
                if done % 50 == 0 or not pending:
                    log(f"{done}/{len(notes)} written")
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
    finally:
        # Whatever happened above, every note gets a line.
        for n in notes:
            if writer.write(n["id"], fallback[n["id"]]):
                stats["fallback"] += 1
        writer.close()

    log(f"done: {len(writer.done)} predictions ({stats['llm']} LLM, {stats['fallback']} no-LLM) -> {out_path}")
    if coder is not None and hasattr(coder, "report"):
        log(f"LLM usage: {coder.report()}")
    return 0


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard exit: normal interpreter shutdown joins worker threads, so one hung API call
    # after the deadline would keep the process alive past the cap. Output is closed already.
    os._exit(rc)
