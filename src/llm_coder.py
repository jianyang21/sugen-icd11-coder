"""LLM path: extract -> shortlist -> select -> validate.

make_coder(idx, valid) returns a callable coder(text, note_id=None) -> list[str],
or raises if no provider is usable (run.py then takes the no-LLM path).

Providers (CODER_PROVIDER, else auto-detected from keys in this order):
  anthropic  ANTHROPIC_API_KEY  (graded run: allowed)
  openai     OPENAI_API_KEY     (graded run: allowed)
  sarvam     SARVAM_API_KEY     (development only: the graded run cannot reach it)

Both official SDKs honour HTTPS_PROXY, which the graded run requires.

Safety rails:
  - Every reply is parsed and validated in validate.py; selected codes must be in the
    shortlist we sent and in MMS.
  - Spend is tracked from response usage; past CODER_BUDGET_USD the coder raises and
    run.py writes the no-LLM answer.
  - Circuit breaker: after CODER_MAX_FATAL consecutive auth/quota/model errors the LLM
    path switches itself off, so a bad key or exhausted credit costs seconds, not minutes.

Environment knobs: CODER_MODEL, CODER_EFFORT, CODER_SHORTLIST_N, CODER_BUDGET_USD,
CODER_TIMEOUT_S, CODER_TRACE (path to write a per-note JSONL trace for error analysis),
CODER_PRICE_IN / CODER_PRICE_OUT (USD per 1M tokens, overrides the table below).
"""
import copy
import json
import os
import re
import sys
import threading
import time

from fallback import queries_from_note
from prompts import (
    EXTRACT_SCHEMA, SELECT_SCHEMA, build_extract_messages, build_select_messages, format_candidate,
)
from validate import clean_extraction, clean_selection, evidence_in_note, norm_code, selection_evidence

# Model chains: if a model id is rejected as unknown, the client moves to the next one,
# so one wrong or retired id costs a single failed call instead of the whole LLM path.
# Chosen for the ~1 cent/note budget; CODER_MODEL (comma-separated) overrides.
DEFAULT_MODELS = {
    "anthropic": ["claude-haiku-5-5", "claude-haiku-4-5"],
    # OpenAI pricing/models pages (read 2026-11-06): gpt-6-luna is the "most efficient
    # model for focused, high-volume tasks"; older cheap ids still priced as backups.
    "openai": ["gpt-6-luna", "gpt-5.6-luna", "gpt-5-mini", "gpt-4o-mini"],
    "sarvam": ["sarvam-105b"],
}

# USD per 1M tokens: (input, output, cache read). Unknown models use a pessimistic default
# so the budget guard errs towards stopping early.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "gpt-6-luna": (0.10, 0.50, 0.01),
    "gpt-5.6-luna": (0.20, 1.20, 0.02),
    "gpt-5-mini": (0.25, 2.00, 0.025),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "sarvam-105b": (0.18, 0.71, 0.06),  # INR 15/60/5 per 1M at ~85 INR/USD
}
UNKNOWN_PRICE = (3.00, 15.00, 0.30)

FATAL_STATUS = {400, 401, 402, 403, 404}


def _log(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def api_schema(schema: dict) -> dict:
    """Copy of a schema the providers' strict modes accept: drop array length limits
    (validated client-side) and turn type lists into anyOf."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            node.pop("minItems", None)
            node.pop("maxItems", None)
            t = node.get("type")
            if isinstance(t, list):
                node.pop("type")
                desc = node.pop("description", None)
                node["anyOf"] = [{"type": x} for x in t]
                if desc:
                    node["description"] = desc
            for v in list(node.values()):
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s


class BudgetExceeded(RuntimeError):
    pass


class LLMDisabled(RuntimeError):
    pass


def price_of(model: str) -> tuple[float, float, float]:
    p = PRICES.get(model, UNKNOWN_PRICE)
    if os.environ.get("CODER_PRICE_IN"):
        p = (float(os.environ["CODER_PRICE_IN"]), float(os.environ.get("CODER_PRICE_OUT", p[1])), p[2])
    return p


class Spend:
    def __init__(self, budget: float):
        self.budget = budget
        self.usd = 0.0
        self.calls = 0
        self.tok_in = self.tok_out = 0
        self.lock = threading.Lock()

    def add(self, model: str, tin: int, tout: int, cache_read: int = 0, cache_write: int = 0):
        p_in, p_out, p_cache = price_of(model)  # per model: the chain may switch models mid-run
        with self.lock:
            self.calls += 1
            self.tok_in += tin + cache_read + cache_write
            self.tok_out += tout
            self.usd += (tin * p_in + cache_write * p_in * 1.25 + cache_read * p_cache + tout * p_out) / 1e6

    def check(self):
        if self.usd >= self.budget:
            raise BudgetExceeded(f"spent ${self.usd:.3f} >= budget ${self.budget:.2f}")


# --------------------------------------------------------------------------- #
# Provider clients: call(system, user, schema, name, max_tokens) -> text
# --------------------------------------------------------------------------- #

_QUOTED = re.compile(r"['\"`]([a-z_][a-z0-9_.]*)['\"`]")
_MODEL_MISSING = ("not found", "does not exist", "unknown model", "invalid model", "no such model",
                  "model_not_found", "not available", "do not have access")


class _ChainClient:
    """Shared retry logic for provider clients.

    - Model chain: a request rejected because the model id is unknown or unavailable
      moves the whole client to the next model in the list (thread-safe, logged).
    - Parameter repair: a 400 naming one of our optional parameters (e.g. a model that
      rejects 'temperature' or wants 'max_completion_tokens' instead of 'max_tokens')
      drops or swaps that parameter for all later calls and retries once.
    Both are bounded per call, then the error is raised to the coder (breaker/fallback).
    """
    OPTIONAL: tuple = ()

    def __init__(self, models: list[str], spend: "Spend"):
        self.models = list(models)
        self.i = 0
        self.spend = spend
        self.dropped: set[str] = set()
        self.lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.models[self.i]

    def status(self, e: Exception) -> int | None:
        return getattr(e, "status_code", None)

    def _bad_param(self, msg: str) -> str | None:
        for tok in _QUOTED.findall(msg):  # first quoted name is usually the offending one
            if tok in self.OPTIONAL and tok not in self.dropped:
                return tok
        for p in self.OPTIONAL:
            if p in msg and p not in self.dropped:
                return p
        return None

    def repair_param(self, p: str) -> None:
        self.dropped.add(p)

    def call(self, system, user, schema, name, max_tokens) -> str:
        for _ in range(len(self.models) + 4):
            model = self.model
            try:
                return self._once(model, system, user, schema, name, max_tokens)
            except Exception as e:
                st = self.status(e)
                msg = str(e).lower()
                if st == 404 or (st in (400, 403) and "model" in msg and any(k in msg for k in _MODEL_MISSING)):
                    with self.lock:
                        if self.model == model and self.i + 1 < len(self.models):
                            self.i += 1
                            _log(f"model {model!r} rejected (HTTP {st}); switching to {self.model!r}")
                            continue
                    if self.model != model:
                        continue  # another thread already switched
                    raise
                if st == 400:
                    p = self._bad_param(msg)
                    if p:
                        with self.lock:
                            if p not in self.dropped:
                                self.repair_param(p)
                                _log(f"{model}: parameter {p!r} rejected; dropping it for later calls")
                        continue
                raise
        raise RuntimeError("model chain / parameter repair exhausted")


class AnthropicClient(_ChainClient):
    OPTIONAL = ("effort",)

    def __init__(self, models: list[str], spend: Spend, timeout: float, effort: str | None):
        import anthropic
        super().__init__(models, spend)
        self.anthropic = anthropic
        # 429s were the main failure in Sarvam replays (up to 36/235 select calls);
        # the SDK backs off exponentially and honours retry-after.
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=int(os.environ.get("CODER_MAX_RETRIES", "5")))
        self.effort = effort

    def status(self, e: Exception) -> int | None:
        return getattr(e, "status_code", None) if isinstance(e, self.anthropic.APIStatusError) else None

    def _once(self, model, system, user, schema, name, max_tokens) -> str:
        output_config = {"format": {"type": "json_schema", "schema": api_schema(schema)}}
        if self.effort and "effort" not in self.dropped:
            output_config["effort"] = self.effort
        r = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            # static system prompt -> cacheable prefix across all notes
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        u = r.usage
        self.spend.add(model, u.input_tokens or 0, u.output_tokens or 0,
                       getattr(u, "cache_read_input_tokens", 0) or 0,
                       getattr(u, "cache_creation_input_tokens", 0) or 0)
        if r.stop_reason == "refusal":
            raise RuntimeError("model refused")
        if r.stop_reason == "max_tokens":
            raise RuntimeError("hit max_tokens")
        return next((b.text for b in r.content if b.type == "text"), "")


class OpenAICompatClient(_ChainClient):
    """OpenAI, or any OpenAI-compatible endpoint (Sarvam) via base_url.

    max_param: "max_completion_tokens" for OpenAI (reasoning models reject max_tokens),
    "max_tokens" for Sarvam (its documented name). A 400 naming either swaps to the other.
    temperature: None means "don't send" (OpenAI reasoning models may reject it).
    """
    OPTIONAL = ("temperature", "reasoning_effort", "max_tokens", "max_completion_tokens")

    def __init__(self, models: list[str], spend: Spend, timeout: float, effort: str | None,
                 api_key: str | None = None, base_url: str | None = None,
                 max_param: str = "max_completion_tokens", temperature: float | None = None):
        import openai
        super().__init__(models, spend)
        self.openai = openai
        kw = {"timeout": timeout, "max_retries": int(os.environ.get("CODER_MAX_RETRIES", "5"))}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        self.client = openai.OpenAI(**kw)
        self.effort, self.max_param, self.temperature = effort, max_param, temperature

    def status(self, e: Exception) -> int | None:
        return getattr(e, "status_code", None) if isinstance(e, self.openai.APIStatusError) else None

    def repair_param(self, p: str) -> None:
        if p in ("max_tokens", "max_completion_tokens"):
            if p == self.max_param:  # the token-limit name we use was rejected: use the other
                self.max_param = "max_tokens" if p == "max_completion_tokens" else "max_completion_tokens"
            return
        self.dropped.add(p)

    def _once(self, model, system, user, schema, name, max_tokens) -> str:
        kw = {self.max_param: max_tokens}
        if self.effort and "reasoning_effort" not in self.dropped:
            kw["reasoning_effort"] = self.effort
        if self.temperature is not None and "temperature" not in self.dropped:
            kw["temperature"] = self.temperature
        r = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "schema": api_schema(schema), "strict": True}},
            **kw,
        )
        u = r.usage
        if u:
            details = getattr(u, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            self.spend.add(model, (u.prompt_tokens or 0) - cached, u.completion_tokens or 0, cached)
        ch = r.choices[0]
        if ch.finish_reason == "length":
            raise RuntimeError("hit max_tokens")
        return ch.message.content or ""


_GENERIC = {"disease", "disorder", "acute", "chronic", "infection", "syndrome", "condition",
            "type", "primary", "secondary", "severe", "mild", "moderate", "complication",
            "without", "mention", "site", "organism", "agent", "unknown", "uncomplicated"}


def plausible_guesses(idx, dx: dict) -> list[str]:
    """Keep a guessed code only if its title, an ancestor's title or one of its synonyms
    shares a non-generic word with the extracted diagnosis. Weak models emit real codes
    for the wrong disease (seen: a malaria code guessed for urticaria), and guesses get a
    ranking bonus in the shortlist."""
    from icd_index import tokenize
    words = set(tokenize(" ".join([dx.get("english", ""), *dx.get("icd_phrasings", [])]), expand=True)) - _GENERIC
    out = []
    for g in dx.get("guessed_codes", []):
        c = g["code"]
        if c not in idx.rows:
            continue
        text = " ".join([idx.title(c), *(idx.title(a) for a in idx.ancestors(c)), *idx.synonyms.get(c, [])])
        if words & set(tokenize(text)):
            out.append(c)
    return out


def build_shortlist(idx, ext: dict | None, text: str, n: int) -> tuple[list[str], list[str]]:
    """Shortlist for the select step from an extraction (may be None) and the raw note.
    Shared by the live coder and tools/replay_shortlist.py, so retrieval changes can be
    measured on recorded extractions without new LLM calls."""
    queries, guesses = [], []
    if ext:
        md = ext["main_diagnosis"]
        queries.append((md["english"], 1.0))
        queries += [(p, 0.8) for p in md["icd_phrasings"][:5]]
        if md["as_written"] and md["as_written"] != md["english"]:
            queries.append((md["as_written"], 0.4))
        ctx = ext["patient_context"].get("pregnancy_status") or ""
        if "pregnan" in ctx.lower() or "postpartum" in ctx.lower():
            queries.append((f"{md['english']} in pregnancy", 0.8))
        guesses = plausible_guesses(idx, md)
        for sd in ext["secondary_diagnoses"][:2]:
            queries.append((sd["english"], 0.4))
            guesses += plausible_guesses(idx, sd)[:2]
    queries += [(q, w * (0.3 if ext else 1.0)) for q, w in queries_from_note(text)]
    return idx.shortlist(queries, guesses, n=n), guesses


def format_candidates(idx, shortlist: list[str], grouped: bool = True) -> list[str]:
    """Candidate lines for the select prompt.

    grouped=True: codes are grouped under their parent category, groups in order of their
    best-ranked member, so siblings sit side by side (33 of 48 recorded top-1 misses with
    the gold in the shortlist were sibling confusions). grouped=False: the old flat list.
    """
    if not grouped:
        return ["## (all candidates)"] + [
            format_candidate(i + 1, c, idx.title(c), idx.parent(c), idx.is_leaf(c)) for i, c in enumerate(shortlist)]
    groups: dict[str, list[str]] = {}
    for c in shortlist:
        key = idx.parent(c) or "(top level)"
        groups.setdefault(key, []).append(c)
    lines = []
    for key, members in groups.items():  # dict keeps first-seen order = best rank first
        if key == "(top level)":
            lines.append("## Top-level categories")
        else:
            grand = idx.parent(key)
            ctx = f" (within: {idx.title(grand)})" if grand else ""
            lines.append(f"## {key} {idx.title(key)}{ctx}")
        for c in members:
            lines.append(f"- {c} | {idx.title(c)} | {'leaf' if idx.is_leaf(c) else 'has children'}")
    return lines


def pick_provider() -> str | None:
    p = os.environ.get("CODER_PROVIDER", "").strip().lower()
    if p:
        return p
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("SARVAM_API_KEY"):
        return "sarvam"
    return None


# --------------------------------------------------------------------------- #
# Coder
# --------------------------------------------------------------------------- #

class Coder:
    def __init__(self, idx, valid: set[str]):
        provider = pick_provider()
        if provider is None:
            raise RuntimeError("no provider key")
        if provider not in DEFAULT_MODELS:
            raise RuntimeError(f"unknown provider {provider!r}")
        env_models = [m.strip() for m in os.environ.get("CODER_MODEL", "").split(",") if m.strip()]
        models = env_models or DEFAULT_MODELS[provider]
        timeout = float(os.environ.get("CODER_TIMEOUT_S", "60"))
        # sarvam-105b reasons by default and reasoning tokens count against max_tokens:
        # at "low" it spent all 4000 tokens reasoning; "none" gave valid JSON in ~5 s.
        default_effort = {"sarvam": "none"}.get(provider, "low")
        effort = os.environ.get("CODER_EFFORT", default_effort) or None
        temp_env = os.environ.get("CODER_TEMPERATURE")
        self.spend = Spend(float(os.environ.get("CODER_BUDGET_USD", "5.0")))
        if provider == "anthropic":
            self.client = AnthropicClient(models, self.spend, timeout, effort)
        elif provider == "openai":
            # No temperature by default: OpenAI reasoning models may reject non-default values.
            self.client = OpenAICompatClient(
                models, self.spend, timeout, effort, max_param="max_completion_tokens",
                temperature=float(temp_env) if temp_env else None)
        else:  # sarvam
            # Sarvam v2 defaults to temperature 1 (two identical dev runs: 65.8 vs 58.3);
            # temperature 0 made extraction loop until max_tokens. 0.2 is its v1 default.
            self.client = OpenAICompatClient(
                models, self.spend, timeout, effort,
                api_key=os.environ["SARVAM_API_KEY"],
                base_url=os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai/v2"),
                max_param="max_tokens", temperature=float(temp_env) if temp_env else 0.2)
        self.idx, self.valid = idx, valid
        self.n_short = int(os.environ.get("CODER_SHORTLIST_N", "40"))
        self.max_fatal = int(os.environ.get("CODER_MAX_FATAL", "5"))
        self.grouped = os.environ.get("CODER_GROUPED", "1") == "1"
        # Off by default: in paired replays it changed ~15 of 235 answers with no net gain.
        self.evidence_rule = os.environ.get("CODER_EVIDENCE_RULE", "0") == "1"
        # Reasoning models spend output tokens thinking; give them more room.
        self.extract_max_tokens = int(os.environ.get(
            "CODER_EXTRACT_MAX_TOKENS", "1500" if effort in (None, "none") else "6000"))
        self.select_max_tokens = int(os.environ.get(
            "CODER_SELECT_MAX_TOKENS", "800" if effort in (None, "none") else "4000"))
        self.fatal_streak = 0
        self.disabled = False
        self.lock = threading.Lock()
        trace = os.environ.get("CODER_TRACE")
        self.trace = open(trace, "w", encoding="utf-8") if trace else None
        self.provider = provider
        _log(f"provider={provider} models={models} effort={effort} shortlist={self.n_short} "
             f"budget=${self.spend.budget:.2f}")

    @property
    def model(self) -> str:
        return self.client.model

    # -- one API call with breaker + budget ----------------------------------
    def _call(self, system, user, schema, name, max_tokens) -> str:
        if self.disabled:
            raise LLMDisabled("LLM path disabled")
        self.spend.check()
        try:
            text = self.client.call(system, user, schema, name, max_tokens)
        except Exception as e:
            st = self.client.status(e)
            with self.lock:
                if st in FATAL_STATUS:
                    self.fatal_streak += 1
                    if self.fatal_streak >= self.max_fatal and not self.disabled:
                        self.disabled = True
                        _log(f"{self.fatal_streak} consecutive HTTP {st} errors: disabling LLM path ({e!r})")
            raise
        with self.lock:
            self.fatal_streak = 0
        return text

    # -- pipeline -------------------------------------------------------------
    def __call__(self, text: str, note_id=None) -> list[str]:
        t0 = time.monotonic()
        tr = {"id": note_id}

        # 1. extract (one retry: a fresh nonce and sample often fixes a malformed or runaway reply)
        ext, errs = None, []
        for attempt in range(2):
            system, user = build_extract_messages(text)
            try:
                # normal replies are ~400 tokens; a low cap makes runaway repetition fail fast
                ext, e2 = clean_extraction(self._call(system, user, EXTRACT_SCHEMA, "extraction",
                                                      self.extract_max_tokens), self.valid)
                errs += e2
            except (BudgetExceeded, LLMDisabled):
                raise
            except Exception as e:
                errs.append(f"extract call failed: {e!r}")
            if ext:
                break
        tr["extract_errors"] = errs[:5]
        tr["extraction"] = ext

        # 2. shortlist: LLM phrasings (if any) + the raw-note signal
        if ext and ext.get("injection_detected"):
            _log(f"{note_id}: injection attempt flagged by extract step")
            tr["injection_extract"] = True
        shortlist, guesses = build_shortlist(self.idx, ext, text, self.n_short)
        tr["guesses"] = guesses
        tr["shortlist"] = shortlist

        # 3. select
        try:
            codes = self.select(text, ext, shortlist, tr)
        except (BudgetExceeded, LLMDisabled):
            if not ext:
                raise
            codes = []
            tr["select_errors"] = ["skipped: budget/disabled"]

        # 4. pad: valid guesses mapped to leaves, then shortlist leaves (run.py pads further)
        out = list(codes)
        for c in [*guesses, *shortlist]:
            c = norm_code(c)
            if c and c in self.valid:
                leaf = self.idx.to_leaf(c)
                if leaf not in out:
                    out.append(leaf)
            if len(out) >= 3:
                break
        tr["codes"] = out[:3]
        tr["ms"] = int(1000 * (time.monotonic() - t0))
        tr["usd_total"] = round(self.spend.usd, 4)
        if self.trace:
            with self.lock:
                self.trace.write(json.dumps(tr, ensure_ascii=False) + "\n")
                self.trace.flush()
        return out[:3]

    def select(self, text: str, ext: dict | None, shortlist: list[str], tr: dict) -> list[str]:
        """The select call plus code-side rules. Separate so tools/replay_select.py can
        re-run selection alone on recorded extractions and shortlists."""
        lines = format_candidates(self.idx, shortlist, grouped=self.grouped)
        summary = ext if ext else {"note": "diagnosis extraction failed; rely on the note"}
        system, user = build_select_messages(text, summary, lines)
        codes, evidence = [], ""
        try:
            raw = self._call(system, user, SELECT_SCHEMA, "selection", self.select_max_tokens)
            codes, inj, errs = clean_selection(raw, shortlist, self.valid)
            evidence = selection_evidence(raw)
            tr["select_errors"] = errs[:5]
            if inj:
                _log(f"{tr.get('id')}: injection attempt flagged by select step")
                tr["injection_select"] = True
        except (BudgetExceeded, LLMDisabled):
            raise
        except Exception as e:
            tr["select_errors"] = [f"select call failed: {e!r}"]
        tr["evidence"] = evidence
        codes = self.apply_coding_rules(codes)
        if self.evidence_rule:
            codes = self.require_evidence(codes, evidence, text, tr)
        tr["selected"] = codes
        return codes

    def require_evidence(self, codes: list[str], evidence: str, text: str, tr: dict) -> list[str]:
        """'Cite or go unspecified': if the first code is a specific child that has an
        unspecified (.Z) sibling, and the model quoted no words from the note supporting
        that detail (or the quote is not actually in the note), put the .Z sibling first
        and keep the specific code second, so a correct specific pick still counts in top-3."""
        if not codes:
            return codes
        first = codes[0]
        p = self.idx.parent(first)
        z = self.idx.unspecified_child(p) if p else None
        if not z or z == first or first.endswith("Z"):
            return codes
        if evidence_in_note(evidence, text):
            return codes
        tr["evidence_rule"] = f"{first} -> {z}"
        return [z] + [c for c in codes if c != z]

    def apply_coding_rules(self, codes: list[str]) -> list[str]:
        """ICD-11 conventions enforced in code, since models break them even when told:
        - a category with children is not a final code -> walk down to its leaf (the .Z
          child when there is one, i.e. 'unspecified');
        - 'other specified' (.Y) is not for missing detail -> put the .Z sibling first.
        """
        out: list[str] = []

        def add(c):
            if c and c in self.valid and c not in out:
                out.append(c)

        for c in codes:
            if not self.idx.is_leaf(c):
                c = self.idx.to_leaf(c)
            if c.endswith("Y"):
                p = self.idx.parent(c)
                z = self.idx.unspecified_child(p) if p else None
                if z:
                    add(z)
            add(c)
        return out

    def report(self) -> str:
        s = self.spend
        return (f"{s.calls} calls, {s.tok_in} in / {s.tok_out} out tokens, ~${s.usd:.4f} "
                f"({self.provider}/{self.model})")


def make_coder(idx, valid: set[str]) -> Coder:
    return Coder(idx, valid)
