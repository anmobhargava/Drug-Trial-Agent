"""
The agent loop. It connects to the MCP server, asks it what tools it has, and then lets the brain decide what to call, step by step, until it has an answer.

    ask -> scope check -> connect -> discover tools -> keep the read-only ones
        -> loop (at most MAX_STEPS):  brain picks tool calls -> call them over MCP (several at once if asked) -> scrub the results -> give them back
        -> final answer -> check it -> (one retry, then a plain summary of the tool results if it still fails)
        -> add sources and the disclaimer in code

Nothing about the tools is hard-coded here: names, descriptions and input schemas all come from the server at run time. Swap in a different MCP server and
the same loop works.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .brains import ERROR_MARK, ToolCall, plain_summary
from .guards import DISCLAIMER, check_answer, scope_check, scrub, select_tools, wrap

MAX_STEPS = 6
TOOL_TIMEOUT = 20.0
SRC = str(Path(__file__).resolve().parents[1])


@dataclass
class Event:
    kind: str          # "guard" | "tools" | "call" | "result" | "step"
    text: str


@dataclass
class Answer:
    text: str
    refused: bool = False
    events: list = field(default_factory=list)
    calls: list = field(default_factory=list)          # [{"name","args","ok"}]


def server_params(live: bool = False, poison: bool = False) -> StdioServerParameters:
    """How to start the MCP server. The demo data is fictional; live needs internet. `poison` plants an instruction in one trial title (demo only)."""
    env = {**os.environ, "PYTHONPATH": SRC}
    if poison:
        env["HEALTHMCP_POISON"] = "1"
    return StdioServerParameters(command=sys.executable, args=["-m", "healthmcp"] + ([] if live else ["--demo-data"]), env=env)


class DrugAgent:
    def __init__(self, brain, server: StdioServerParameters | None = None):
        self.brain, self.server = brain, server or server_params()

    async def ask(self, question: str) -> Answer:
        refusal = scope_check(question)
        if refusal:                                              # refused before any tool is even listed
            return Answer(refusal, refused=True, events=[Event("guard", "scope: refused, this asks for personal medical advice. No tools were called.")])
        async with stdio_client(self.server) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return await self.run(session, question)

    async def run(self, session, question: str) -> Answer:
        events: list[Event] = []
        calls: list[dict] = []
        evidence: list[dict] = []

        tools, notes = select_tools((await session.list_tools()).tools)
        events += [Event("tools", n) for n in notes]
        tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in tools]
        allowed = {t.name for t in tools}

        messages: list = [{"role": "user", "content": question}]
        final, retried = None, False
        for step in range(MAX_STEPS):
            turn = await self.brain.next(messages, tool_defs)
            if turn.calls:
                messages.append({"role": "assistant", "content": turn.text or None,
                                 "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.input)}} for c in turn.calls]})
                for c in turn.calls:
                    events.append(Event("call", f"{c.name}({json.dumps(c.input)})"))
                results = await asyncio.gather(*[self._call(session, c, allowed) for c in turn.calls])       # several calls in one turn run together
                for c, (ok, text, removed) in zip(turn.calls, results):
                    calls.append({"name": c.name, "args": c.input, "ok": ok})
                    events.append(Event("result", f"{c.name}: {'ok' if ok else 'error'}"))
                    if removed:
                        events.append(Event("guard", f"tool data: removed {removed} piece(s) of text from {c.name} that looked like an instruction"))
                    try:
                        data = json.loads(text)
                    except ValueError:
                        data = text
                    evidence.append({"name": c.name, "args": c.input, "ok": ok, "data": data})
                    # OpenAI has no "is_error" flag on tool messages, so a failure is marked in the text itself
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": wrap(text if ok else ERROR_MARK + text)})
                continue
            problems = check_answer(turn.text, json.dumps([e["data"] for e in evidence]), question)
            if not problems:
                final = turn.text
                break
            events.append(Event("guard", "answer check failed: " + "; ".join(problems)))
            if retried:
                break
            retried = True
            messages += [{"role": "assistant", "content": turn.text}, {"role": "user", "content": "That answer was rejected because: " + "; ".join(problems) + ". Rewrite it using only what the tool results say."}]
        else:
            events.append(Event("guard", f"stopped after {MAX_STEPS} steps without a final answer"))

        if final is None:
            events.append(Event("guard", "fell back to a plain summary of the tool results"))
            final = plain_summary(evidence)
        elif not calls:
            final += "\n\n(I answered without looking anything up.)"
        return Answer(self._finish(final, evidence), events=events, calls=calls)

    async def _call(self, session, call: ToolCall, allowed: set) -> tuple[bool, str, int]:
        if call.name not in allowed:
            return False, f"Unknown or unavailable tool '{call.name}'.", 0
        try:
            res = await asyncio.wait_for(session.call_tool(call.name, call.input), TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            return False, "The tool timed out. Try again or ask something simpler.", 0
        except Exception as e:                                   # a broken tool call should not crash the whole run
            return False, f"The tool call failed ({type(e).__name__}).", 0
        clean, removed = scrub("\n".join(getattr(b, "text", "") for b in res.content))
        return (not res.isError), clean, removed

    @staticmethod
    def _finish(text: str, evidence: list[dict]) -> str:
        """Sources and the disclaimer are added by code from what the tools returned, not written by the model."""
        urls, seen = [], set()
        for e in evidence:
            d = e["data"] if e["ok"] and isinstance(e["data"], dict) else {}
            for u in [d.get("source_url")] + [f"https://clinicaltrials.gov/study/{t['nct_id']}" for t in d.get("trials", [])]:
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
            if d.get("source") and d["source"] not in seen:
                seen.add(d["source"])
                urls.append(f"data: {d['source']}")
        return text.rstrip() + ("\n\nSources:\n" + "\n".join(f"- {u}" for u in urls) if urls else "") + f"\n\n{DISCLAIMER}"
