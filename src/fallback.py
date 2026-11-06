"""No-LLM query building: turn a raw note into weighted search queries.

Used when no API key is set, when the LLM fails, and as an extra shortlist signal.
Deliberately generic: it looks for diagnosis cues common to most note styles and
falls back to the whole note when none are found.
"""

import re

# Section / cue words that introduce the clinician's conclusion. A few non-English
# cues are included because the hidden set has non-English notes.
_DX_CUE = re.compile(
    r"(?:^|\n|[.;]\s*)\s*(?:"
    r"(?:A|Ax)\s*:"  # single-letter SOAP header needs a colon, or it matches "A 32-year-old..."
    r"|(?:Dx|DDx|Imp|Impr|Impression|Assessment|Diagnosis|Diagnoses|Diag|Working\s+diagnosis|"
    r"Final\s+diagnosis|Diagnostic|Diagn[oó]stico|Diagnose|Diagnosi|Utambuzi)\s*[:\-–—]?"
    r")\s*(.+?)(?=\n|\s(?:P|Plan|Rx|Tx|Mx|Management|Treatment|Plan de)\s*[:\-]|$)",
    re.IGNORECASE,
)
# Transcript-style conclusions: "This is an asthma attack", "you have ...", "looks like ..."
_TALK_CUE = re.compile(
    r"\b(?:this is|that is|it is|it's|you have|you've got|he has|she has|your child has|looks like|"
    r"sounds like|consistent with|in keeping with|suggestive of|diagnosis is|I think (?:it's|this is|you have))"
    r"\s+(?:an?\s+|the\s+)?([^.?!\n]{3,120})",
    re.IGNORECASE,
)
# Clauses that negate or rule out a condition. Note: "without" is kept on purpose,
# because ICD titles use it ("Varicella without complication").
_NEG = re.compile(
    r"\b(?:no|not|nil|neg|negative|excluded|ruled out|r/o|unlikely|denies|absent|normal)\b",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"[,;.\n]|\s-\s|\s–\s")


def drop_negated(text: str) -> str:
    """Remove clauses containing a negation cue ("malaria excluded", "mRDT NEG")."""
    parts = [p for p in _SPLIT.split(text) if p.strip()]
    return " , ".join(p for p in parts if not _NEG.search(p))


# Speaker labels in transcripts ("Doctor:", "Dr:", "Clinician:", "Médecin:" ...)
_CLINICIAN = re.compile(
    r"^\s*(?:doctor|dr\.?|clinician|physician|provider|nurse|co|médecin|medecin|docteur|médico|medico|daktari)\s*:",
    re.IGNORECASE,
)
_SPEAKER = re.compile(r"^\s*[A-Za-zÀ-ÿ .]{2,20}:\s")


def clinician_text(note: str) -> str:
    """For labelled transcripts, keep only the clinician's lines (patients say
    'my thyroid' too). Unlabelled text is returned unchanged."""
    lines = note.splitlines()
    labelled = [l for l in lines if _SPEAKER.match(l)]
    if len(labelled) < 3:
        return note
    doc = [l for l in lines if _CLINICIAN.match(l)]
    return "\n".join(doc) if doc else note


def dx_spans(note: str) -> list[str]:
    spans = [m.group(1).strip() for m in _DX_CUE.finditer(note)]
    if not spans:
        spans = [m.group(1).strip() for m in _TALK_CUE.finditer(clinician_text(note))]
    return [s for s in spans if s]


def queries_from_note(note: str) -> list[tuple[str, float]]:
    """Weighted queries for IcdIndex.scored/shortlist/top_leaves."""
    note = note or ""
    qs: list[tuple[str, float]] = []
    spans = dx_spans(note)
    for s in spans:
        clean = drop_negated(s) or s
        qs.append((clean, 1.0))
    # whole note (clinician lines for transcripts), negated clauses removed, as a weak backup
    body = clinician_text(note)
    qs.append((drop_negated(body) or body, 0.35 if spans else 1.0))
    return qs
