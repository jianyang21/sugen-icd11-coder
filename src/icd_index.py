"""Local ICD-11 MMS search: BM25 over code titles plus their ancestors' titles,
and hierarchy helpers for building an LLM shortlist.

No third-party dependencies, no network. Builds in about a second at startup.

Main entry points:
  idx = IcdIndex.load(path)
  idx.search(text, k)                -> [(code, score)]  raw BM25 hits, any depth
  idx.shortlist(queries, guesses, n) -> [code]           hits + family, for the SELECT prompt
  idx.to_leaf(code)                  -> code             non-leaf -> its unspecified/best leaf
"""

import json
import math
import os
import re
from collections import Counter, defaultdict

DEFAULT_SYNONYMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synonyms.jsonl")

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

# British -> American spelling stems, so "haemorrhoids" matches "hemorrhoids", etc.
_SPELLING = [
    (r"haem", "hem"), (r"aemi", "emi"), (r"oesoph", "esoph"), (r"oedem", "edem"),
    (r"rrhoe", "rrhe"), (r"paed", "ped"), (r"amoeb", "ameb"), (r"foet", "fet"),
    (r"tumour", "tumor"), (r"ischaem", "ischem"), (r"caes", "ces"), (r"oestr", "estr"),
    (r"orthopaed", "orthoped"), (r"glycaem", "glycem"), (r"leukaem", "leukem"),
    (r"coeli", "celi"), (r"dyspnoe", "dyspne"), (r"apnoe", "apne"),
]

_STOP = set("""
a an and are as at be by for from has have in is it its of on or the to was were with
which this that these those other others than due unspecified specified certain
""".split())

# Common clinical abbreviations -> expansion. General medical knowledge, not tuned to
# the dev notes. Matched case-insensitively as whole tokens.
ABBREV = {
    "urti": "upper respiratory tract infection", "lrti": "lower respiratory tract infection",
    "uti": "urinary tract infection", "htn": "hypertension", "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus", "t1dm": "type 1 diabetes mellitus",
    "af": "atrial fibrillation", "oa": "osteoarthritis", "ra": "rheumatoid arthritis",
    "tb": "tuberculosis", "ptb": "pulmonary tuberculosis", "copd": "chronic obstructive pulmonary disease",
    "dvt": "deep vein thrombosis", "gerd":"gastro-oesophageal reflux disease",
    "gord": "gastro-oesophageal reflux disease", "pud": "peptic ulcer disease",
    "pid": "pelvic inflammatory disease", "ccf": "congestive heart failure",
    "chf": "congestive heart failure", "hf": "heart failure", "mi": "myocardial infarction",
    "stemi": "st elevation myocardial infarction", "nstemi": "non st elevation myocardial infarction",
    "acs": "acute coronary syndrome", "cva": "stroke cerebrovascular", "tia": "transient ischaemic attack",
    "ckd": "chronic kidney disease", "aki": "acute kidney injury", "bph": "benign prostatic hyperplasia",
    "ibs": "irritable bowel syndrome", "ibd": "inflammatory bowel disease", "gi": "gastrointestinal",
    "aom": "acute otitis media",
    "sle": "systemic lupus erythematosus", "hiv":"human immunodeficiency virus hiv",
    "aids": "acquired immunodeficiency syndrome", "sti": "sexually transmitted infection",
    "std": "sexually transmitted disease", "bv": "bacterial vaginosis", "vvc": "vulvovaginal candidosis",
    "ida": "iron deficiency anaemia", "mam": "moderate acute malnutrition",
    "gdm": "gestational diabetes mellitus pregnancy", "pih": "pregnancy induced hypertension",
    "aph": "antepartum haemorrhage", "pph": "postpartum haemorrhage",
    "pprom": "preterm premature rupture of membranes", "anc":"antenatal care pregnancy supervision",
    "lbp": "low back pain", "ocd": "obsessive compulsive disorder", "adhd": "attention deficit hyperactivity disorder",
    "ptsd": "post traumatic stress disorder", "mdd":"major depressive disorder",
    "ards": "acute respiratory distress syndrome", "osa": "obstructive sleep apnoea",
    "bppv": "benign paroxysmal positional vertigo",
    "dka":"diabetic ketoacidosis", "hhs": "hyperosmolar hyperglycaemic state",
    "nafld": "non alcoholic fatty liver disease", "ugib": "upper gastrointestinal bleeding",
    "scc": "squamous cell carcinoma", "bcc": "basal cell carcinoma",
    "fx": "fracture", "mps": "malaria parasites",
    "uncx": "uncomplicated", "exac": "exacerbation", "aecopd": "chronic obstructive pulmonary disease acute exacerbation",
    "thrush":"candidosis candidiasis", "jiggers": "tungiasis", "bilharzia": "schistosomiasis",
    "flu": "influenza", "piles": "haemorrhoids", "dx": "", "imp": "", "ddx": "",
}


def _spell(s: str) -> str:
    for a, b in _SPELLING:
        s = re.sub(a, b, s)
    return s


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is", "sis")):
        return t[:-1]
    return t


def tokenize(text: str, expand: bool = False) -> list[str]:
    text = text.lower()
    if expand:
        text = " ".join(ABBREV.get(w, w) for w in re.split(r"(\W+)", text))
    text = _spell(text)
    toks = re.findall(r"[a-z0-9]+", text)
    return [_stem(t) for t in toks if t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(self.n, 1)
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()
        }

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t, qf in Counter(query).items():
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, f in self.post[t]:
                norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                out[i] += idf * norm * (1 + 0.2 * (qf - 1))  # mild credit for repeated query terms
        return out


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

def parent_of(code: str) -> str | None:
    """CA23.30 -> CA23.3 -> CA23 -> None. Verified against all 18,375 MMS codes."""
    if "." not in code:
        return None
    stem, ext = code.split(".", 1)
    return code[:-1] if len(ext) > 1 else stem


class IcdIndex:
    ANCESTOR_WEIGHT = 0.35  # ancestor titles add context ("Amoebiasis" for "Acute amoebiasis")
    # Chapter 26 (traditional medicine) is supplementary and rarely the main outpatient
    # code; down-weight rather than exclude so a real TM note can still reach it.
    CHAPTER_PRIOR = {"26": 0.4}

    # Generated synonyms (data/synonyms.jsonl) are less trustworthy than official titles.
    # Replayed on recorded extractions, shortlist@40 (dev / synthetic): titles only .967/.838;
    # weight 0.3 .967/.860; 0.6 .967/.851; 1.0 .967/.838. Higher weights let noisy synonyms crowd out titles.
    SYN_WEIGHT = float(os.environ.get("CODER_SYN_WEIGHT", "0.3"))

    def __init__(self, rows: list[dict], synonyms: dict[str, list[str]] | None = None):
        self.synonyms = synonyms or {}
        self.rows = {r["code"]: r for r in rows}
        self.codes = [r["code"] for r in rows]
        self.children: dict[str, list[str]] = defaultdict(list)
        for c in self.codes:
            p = parent_of(c)
            if p:
                self.children[p].append(c)
        self._title = _BM25([tokenize(self.rows[c]["title"]) for c in self.codes])
        self._anc = _BM25([tokenize(" ".join(self.title(a) for a in self.ancestors(c))) for c in self.codes])
        self._prior = [self.CHAPTER_PRIOR.get(self.rows[c]["chapter"], 1.0) for c in self.codes]
        # Synonym field = only words NOT already in the code's own title. Otherwise synonyms
        # double-count title words: "Neonatal falciparum malaria" pushed congenital malaria
        # (KA64.20) above plain falciparum malaria for an adult note.
        if self.synonyms:
            docs = []
            for c in self.codes:
                own = set(tokenize(self.rows[c]["title"]))
                docs.append([t for t in tokenize(" ; ".join(self.synonyms.get(c, []))) if t not in own])
            self._syn = _BM25(docs)
        else:
            self._syn = None

    @classmethod
    def load(cls, path: str, synonyms_path: str | None = "default") -> "IcdIndex":
        """synonyms_path: 'default' -> data/synonyms.jsonl next to src/ if present; None -> no synonyms."""
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        if synonyms_path == "default":
            synonyms_path = os.environ.get("CODER_SYNONYMS", DEFAULT_SYNONYMS)
        syn = {}
        if synonyms_path and os.path.exists(synonyms_path):
            with open(synonyms_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("synonyms"):
                            syn[r["code"]] = r["synonyms"]
        return cls(rows, syn)

    # ---- hierarchy ----------------------------------------------------------
    def title(self, code: str) -> str:
        return self.rows[code]["title"]

    def is_leaf(self, code: str) -> bool:
        return bool(self.rows[code]["leaf"])

    def parent(self, code: str) -> str | None:
        return parent_of(code)

    def ancestors(self, code: str) -> list[str]:
        out, p = [], parent_of(code)
        while p:
            out.append(p)
            p = parent_of(p)
        return out

    def siblings(self, code: str) -> list[str]:
        p = parent_of(code)
        return [c for c in self.children.get(p, []) if c != code] if p else []

    def unspecified_child(self, code: str) -> str | None:
        for c in self.children.get(code, []):
            if c.endswith("Z"):
                return c
        return None

    def to_leaf(self, code: str, prefer: dict[str, float] | None = None) -> str:
        """Walk down from a non-leaf code: take the unspecified child if there is one
        (right when the detail is missing), else the best-scoring or first child."""
        while not self.is_leaf(code):
            kids = self.children.get(code, [])
            if not kids:
                break
            z = self.unspecified_child(code)
            if z:
                code = z
            elif prefer:
                code = max(kids, key=lambda k: prefer.get(k, 0.0))
            else:
                code = kids[0]
        return code

    def family(self, code: str) -> list[str]:
        """Parent, the code's children, its unspecified sibling and its other siblings."""
        out = []
        p = parent_of(code)
        if p:
            out.append(p)
        out += self.children.get(code, [])
        out += self.siblings(code)
        return out

    # ---- search -------------------------------------------------------------
    def search(self, text: str, k: int = 50) -> list[tuple[str, float]]:
        q = tokenize(text, expand=True)
        if not q:
            return []
        s = self._title.scores(q)
        for i, v in self._anc.scores(q).items():
            s[i] += self.ANCESTOR_WEIGHT * v
        if self._syn is not None:
            for i, v in self._syn.scores(q).items():
                s[i] += self.SYN_WEIGHT * v
        for i in s:
            s[i] *= self._prior[i]
        top = sorted(s.items(), key=lambda x: -x[1])[:k]
        return [(self.codes[i], v) for i, v in top]

    def scored(self, queries: list[tuple[str, float]], k: int = 80) -> dict[str, float]:
        """Weighted sum of search scores over several queries, each normalised to its top hit."""
        agg: dict[str, float] = defaultdict(float)
        for text, w in queries:
            hits = self.search(text, k)
            if not hits:
                continue
            top = hits[0][1] or 1.0
            for c, v in hits:
                agg[c] += w * v / top
        return agg

    def shortlist(
        self, queries: list[tuple[str, float]], guesses: list[str] = (), n: int = 40
    ) -> list[str]:
        """Candidate codes for the SELECT prompt, best first.

        queries: [(text, weight)], e.g. the English diagnosis and its ICD phrasings.
        guesses: codes the LLM proposed from memory (already validated as real MMS codes).
        Each strong hit pulls in its family, so the most specific child and the
        unspecified variant are always available to choose from.
        """
        agg = self.scored(queries)
        for g in guesses:
            if g in self.rows:
                agg[g] = max(agg.get(g, 0.0), 1.0) + 0.5
        ranked = sorted(agg.items(), key=lambda x: -x[1])
        out: list[str] = []
        seen: set[str] = set()

        def add(c: str):
            if c in self.rows and c not in seen:
                seen.add(c)
                out.append(c)

        # 1) direct hits (each with its leaf), so one big family can't crowd out other hits
        direct = max(n // 3, 5)
        for c, _ in ranked[:direct]:
            add(c)
            if not self.is_leaf(c):
                add(self.to_leaf(c, agg))
        # 2) families of the strongest hits: parent, children, siblings (incl. the .Z variant)
        for c, _ in ranked[:direct]:
            if len(out) >= n:
                break
            fam = sorted(self.family(c), key=lambda f: -agg.get(f, 0.0))
            for f in fam[:8]:
                add(f)
        # 3) weaker hits fill the rest
        for c, _ in ranked[direct:]:
            if len(out) >= n:
                break
            add(c)
        return out[:n]

    def top_leaves(self, queries: list[tuple[str, float]], n: int = 3) -> list[str]:
        """No-LLM ranking: best-scoring codes mapped to leaves, deduplicated."""
        agg = self.scored(queries)
        out: list[str] = []
        for c, _ in sorted(agg.items(), key=lambda x: -x[1]):
            leaf = self.to_leaf(c, agg)
            if leaf not in out:
                out.append(leaf)
            if len(out) == n:
                break
        return out
