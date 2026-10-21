"""
The guardrails, all in plain code. Four of them, each at a different point in the loop:

  scope_check     BEFORE any tool is called: a question asking for personal medical advice (what to take, how much, is it safe for me) is refused.
  select_tools    AFTER discovery: only tools the server marks read-only are given to the model.
  scrub           ON every tool result: anything that reads like an instruction (a planted line in a label or a trial title) is removed, and the rest is
                  wrapped in <tool_data> tags so the model is told it's data, not orders.
  check_answer    ON the final answer: every number must come from a tool result, every trial id must have been returned by a tool, and nothing may read as advice.
"""

import json
import re

REFUSAL = ("I can't give personal medical advice, like what to take, how much, or whether something is safe for you. I can look up what a drug's official label says "
           "(its warnings, interactions, adverse reactions) or find clinical trials for a condition. For advice about you, please ask a doctor or pharmacist. "
           "Want me to look something up?")
DISCLAIMER = "This is information from public sources, not medical advice."

_ADVICE_Q = re.compile(r"\b(should i (take|use|give|stop|start|avoid|worry|combine|mix|drink|eat|switch|increase|decrease|double|skip|try)|can i (take|use|mix|combine|give|drink)|how (much|many)\b.*\b(should|can|do) i|what (dose|dosage)\b|which dose|(dose|dosage) for (me|my)|"
                       r"safe for (me|my)|is it (ok|okay|safe) (for me|to take)|diagnos\w*|prescribe\w*|am i allowed)\b", re.I)
_INJECTION = re.compile(r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)+(instructions?|rules|prompt)|system prompt|you are now|new instructions?|"
                        r"reveal your|do not tell the user|tell the user to|stop taking|call the \w+ tool|instead,? (say|respond|answer)", re.I)
_ADVICE_OUT = re.compile(r"(you should (take|stop|start|avoid|use)|i recommend|recommended dose|take \d|stop taking|safe for you|you can safely)", re.I)
_NUM = re.compile(r"\d[\d,]*\.?\d*")


def scope_check(question: str) -> str | None:
    """Returns the refusal text if the question asks for personal medical advice, else None."""
    return REFUSAL if _ADVICE_Q.search(question) else None


def select_tools(tools) -> tuple[list, list[str]]:
    """Keep only tools the server marked read-only. Returns (kept, events). A tool with no annotation is treated as not safe."""
    kept, events = [], []
    for t in tools:
        if t.annotations is not None and getattr(t.annotations, "readOnlyHint", False):
            kept.append(t)
        else:
            events.append(f"tools: left out '{t.name}' because the server doesn't mark it read-only")
    events.insert(0, f"tools: exposing {len(kept)} of {len(tools)} discovered tools (read-only only)")
    return kept, events


def _walk(x, hits: list):
    if isinstance(x, str):
        if _INJECTION.search(x):
            hits.append(x[:80])
            return "[removed: this text looked like an instruction]"
        return x
    if isinstance(x, list):
        return [_walk(i, hits) for i in x]
    if isinstance(x, dict):
        return {k: _walk(v, hits) for k, v in x.items()}
    return x


def scrub(text: str) -> tuple[str, int]:
    """Removes instruction-like strings from a tool result. Returns (clean JSON or text, how many were removed)."""
    hits: list = []
    try:
        clean = json.dumps(_walk(json.loads(text), hits), indent=1)
    except ValueError:
        clean = _walk(text, hits)
    return clean, len(hits)


def wrap(clean: str) -> str:
    return f"<tool_data>\n{clean}\n</tool_data>"


def _nums(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def check_answer(answer: str, evidence_text: str, question: str) -> list[str]:
    """Reasons to reject a final answer. Numbers must come from the tool results (or the question), trial ids must have been returned, no advice."""
    problems = []
    allowed = _nums(evidence_text) | _nums(question)
    for x in _nums(answer):
        if x > 10 and not any(abs(x - a) < 0.5 for a in allowed):
            problems.append(f"the number {x:g} is not in the tool results")
    for nct in set(re.findall(r"NCT\d{8}", answer)):
        if nct not in evidence_text:
            problems.append(f"{nct} was not returned by any tool")
    if _ADVICE_OUT.search(answer):
        problems.append("it reads as medical advice")
    return problems
