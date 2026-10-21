import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(coro):
    return asyncio.run(coro)


def tool(name, read_only=True):
    return NS(name=name, description=f"{name} tool", inputSchema={"type": "object", "properties": {}}, annotations=NS(readOnlyHint=True) if read_only else None)


def result(obj, error=False):
    return NS(isError=error, content=[NS(text=obj if isinstance(obj, str) else json.dumps(obj))])


class FakeSession:
    """Stands in for an MCP session: lists the tools it is given and answers calls from a function. Records the calls and how many ran at once."""
    def __init__(self, tools, handler):
        self._tools, self.handler, self.calls, self.running, self.peak = tools, handler, [], 0, 0

    async def list_tools(self):
        return NS(tools=self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.02)
        self.running -= 1
        return self.handler(name, args)


class FakeClient:
    """Plays back scripted OpenAI-style replies (each a message: text and/or tool calls) and records every request, so the loop is tested with no API key."""
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": copy.deepcopy(kw["messages"])})
        r = self.replies.pop(0)
        return NS(choices=[NS(message=r[0] if isinstance(r, list) else r)])


def text(t):
    return NS(content=t, tool_calls=None)


def use(id, tool_name, **inp):
    return NS(content=None, tool_calls=[NS(id=id, function=NS(name=tool_name, arguments=json.dumps(inp)))])


class ScriptBrain:
    """A brain that follows a script of Turns, for testing the loop itself."""
    name = "script"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    async def next(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
