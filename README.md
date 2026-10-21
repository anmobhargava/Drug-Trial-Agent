# Agent 2: Drug and trial lookup agent

An agent that answers questions about drugs and clinical trials by calling tools on an MCP server. This repo is about the agent side: how it discovers tools, decides what to call, reads the results, and stays inside its lane. The server it talks to is my Agent 3 repo.

## What it does

You ask something like "What are the warnings for Cardiodex?" or "Find recruiting trials for hypertension". The agent:

1. Checks the question is in scope. Personal medical advice ("how much should I take") is refused before any tool runs.
2. Connects to the MCP server over stdio and asks it what tools exist (nothing is hard-coded). Only read-only tools are kept.
3. Runs a loop: the model picks tool calls (OpenAI function calling), I run them (in parallel when there are several), the results go back, repeat. Max 6 steps, 20 second timeout per call.
4. Checks the draft answer against the tool data. Numbers and trial ids must appear in the evidence, and no advice phrasing. One retry, then a plain summary of the raw data.
5. Appends the sources and a disclaimer in code, so the model can't leave them out.

Tool results are treated as untrusted. Text that looks like an instruction is scrubbed and the rest is wrapped in `<tool_data>` tags. The demo page has a checkbox that plants "IGNORE PREVIOUS INSTRUCTIONS..." inside a trial title so you can watch this happen.

## Concepts covered
MCP client (stdio, tool discovery), tool-calling loop with the OpenAI SDK (function calling: tool_calls and tool messages), parallel tool calls, error recovery, prompt injection defence, output grounding checks, scope guardrails.

## Run it
```
pip install -r requirements.txt
PYTHONPATH=src python -m drugagent.app      # http://localhost:7860
PYTHONPATH=src pytest -q
```
Set `OPENAI_API_KEY` to use an OpenAI model as the brain (model name is in `brains.py`, check it is the one you want). Without a key it uses a small keyword-rule stand-in so everything still runs. Set `DRUGAGENT_LIVE=1` for real openFDA / ClinicalTrials.gov data.

## Layout
- `src/drugagent/agent.py`: the loop
- `src/drugagent/guards.py`: scope check, tool filter, scrubbing, grounding check
- `src/drugagent/brains.py`: OpenAI brain and the offline stand-in
- `src/drugagent/app.py`: Gradio page showing answer, tool calls and guardrail events
- `src/healthmcp/`: a vendored copy of the Agent 3 server, with one addition: `HEALTHMCP_POISON=1` plants the hostile text in a trial title. Agent 3 is the standalone server repo.

## What I did and didn't test
- 32 tests pass: guards, the loop against the real MCP server over stdio, and the OpenAI path against a scripted fake client.
- Not tested: the live openFDA and ClinicalTrials.gov calls (those sites were unreachable where I built this), and the real OpenAI API.
- The offline stand-in only understands its own phrasings.
- Demo data is fictional (invented drug names, trial ids starting NCT9). Nothing here is medical information or advice.
# Drug-Trial-Agent
