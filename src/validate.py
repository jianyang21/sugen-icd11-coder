"""Validate LLM output. The model is never trusted: everything it returns is parsed,
checked against the schema and filtered before it can reach predictions.jsonl.

- parse_json_object: tolerant extraction of one JSON object from the reply
  (handles code fences or stray prose around it).
- check_schema: small JSON-Schema subset validator (type, properties, required,
  additionalProperties, items, minItems, maxItems). No third-party dependency.
- clean_extraction / clean_selection: schema check plus domain rules. Codes must
  exist in MMS, and selected codes must also be in the shortlist we sent.
- fill_to_three: always return 3 valid codes, padding from the local search ranking.
"""

import json
import re

from prompts import EXTRACT_SCHEMA, SELECT_SCHEMA

_STEM = re.compile(r"[&/]")
_CODE_SHAPE = re.compile(r"^[0-9A-Z]{4}(\.[0-9A-Z]{1,3})?$")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def norm_code(code) -> str | None:
    """Same normalisation as score.py: stem before & or /, strip, upper-case."""
    if not isinstance(code, str):
        return None
    c = _STEM.split(code.strip())[0].strip().upper()
    return c if _CODE_SHAPE.match(c) else None


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #

def parse_json_object(text: str) -> dict | None:
    """Return the first JSON object in text, or None."""
    if not isinstance(text, str):
        return None
    s = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Schema check
# --------------------------------------------------------------------------- #

_TYPES = {
    "string": str, "boolean": bool, "object": dict, "array": list, "null": type(None),
    "number": (int, float), "integer": int,
}


def _type_ok(value, t) -> bool:
    types = t if isinstance(t, list) else [t]
    for name in types:
        py = _TYPES[name]
        if name in ("number", "integer") and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def check_schema(value, schema: dict, path: str = "$") -> list[str]:
    """Return a list of error strings; empty means valid."""
    errs = []
    if "type" in schema and not _type_ok(value, schema["type"]):
        return [f"{path}: expected {schema['type']}, got {type(value).__name__}"]
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k}: missing")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}.{k}: not allowed")
        for k, sub in props.items():
            if k in value:
                errs += check_schema(value[k], sub, f"{path}.{k}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                errs += check_schema(item, schema["items"], f"{path}[{i}]")
    return errs


def _strip_extra(value, schema: dict):
    """Drop fields the schema does not allow (lenient repair, not rejection)."""
    if isinstance(value, dict) and schema.get("additionalProperties") is False:
        props = schema.get("properties", {})
        return {k: _strip_extra(v, props[k]) for k, v in value.items() if k in props}
    if isinstance(value, list) and "items" in schema:
        return [_strip_extra(v, schema["items"]) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Domain checks
# --------------------------------------------------------------------------- #

def clean_extraction(raw_text: str, valid_codes: set[str]) -> tuple[dict | None, list[str]]:
    """Parse + validate an EXTRACT reply.

    Returns (extraction or None, errors). Extra fields are dropped. Guessed codes
    that are not real MMS codes are removed, so ICD-10 or invented codes never
    reach the shortlist.
    """
    obj = parse_json_object(raw_text)
    if obj is None:
        return None, ["no JSON object in reply"]
    obj = _strip_extra(obj, EXTRACT_SCHEMA)
    errs = check_schema(obj, EXTRACT_SCHEMA)
    if errs:
        return None, errs
    for dx in [obj["main_diagnosis"], *obj["secondary_diagnoses"]]:
        kept = []
        for g in dx["guessed_codes"][:5]:
            c = norm_code(g.get("code"))
            if c and c in valid_codes:
                kept.append({"code": c, "title": g.get("title", "")})
        dx["guessed_codes"] = kept
    return obj, []


def clean_selection(
    raw_text: str, shortlist: list[str], valid_codes: set[str]
) -> tuple[list[str], bool, list[str]]:
    """Parse + validate a SELECT reply.

    Returns (codes, injection_detected, errors). Codes are normalised, deduplicated,
    capped at 3, and kept only if they are both in the shortlist we sent and in MMS.
    That stops the model from inventing codes or taking a code planted in the note.
    """
    obj = parse_json_object(raw_text)
    if obj is None:
        return [], False, ["no JSON object in reply"]
    obj = _strip_extra(obj, SELECT_SCHEMA)
    errs = check_schema(obj, SELECT_SCHEMA)
    raw_codes = obj.get("codes") if isinstance(obj.get("codes"), list) else []
    allowed = {norm_code(c) for c in shortlist} & valid_codes
    codes, seen = [], set()
    for c in raw_codes:
        c = norm_code(c)
        if c and c in allowed and c not in seen:
            codes.append(c)
            seen.add(c)
        elif c and c not in allowed:
            errs.append(f"code {c} not in shortlist/MMS, dropped")
    injection = obj.get("injection_detected") is True
    return codes[:3], injection, errs


def selection_evidence(raw_text: str) -> str:
    """The select step's quoted evidence for its first code ('' if absent or malformed)."""
    obj = parse_json_object(raw_text) or {}
    ev = obj.get("first_code_evidence")
    return ev.strip() if isinstance(ev, str) else ""


def evidence_in_note(evidence: str, note: str) -> bool:
    """Is the quoted evidence really in the note? Exact match after normalising case and
    whitespace, or at least 60% of its words present (models trim or re-punctuate quotes).
    Works for any language: it compares the note's own words."""
    ev = " ".join(evidence.lower().split())
    if not ev:
        return False
    nt = " ".join(note.lower().split())
    if ev in nt:
        return True
    words = [w for w in re.findall(r"\w+", ev) if len(w) > 2]
    if not words:
        return False
    note_words = set(re.findall(r"\w+", nt))
    return sum(w in note_words for w in words) / len(words) >= 0.6


def fill_to_three(codes: list[str], fallback_ranked: list[str], valid_codes: set[str]) -> list[str]:
    """Pad to 3 codes from the local search ranking. Every returned code is valid MMS."""
    out, seen = [], set()
    for c in [*codes, *fallback_ranked]:
        c = norm_code(c)
        if c and c in valid_codes and c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) == 3:
            break
    return out
