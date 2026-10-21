"""
The demo page. Ask a question about a drug or a clinical trial. Next to the answer you can see what the agent actually did: which MCP tools it called and
with what arguments, and every time a guardrail stepped in.

    PYTHONPATH=src python -m drugagent.app           -> http://localhost:7860    (fictional demo data)
    DRUGAGENT_LIVE=1 PYTHONPATH=src python -m drugagent.app                       (real openFDA / ClinicalTrials.gov; needs internet)

The demo data is made up (invented drug names, trial ids starting NCT9), so nothing on the page is real medical information.
"""

import os

import gradio as gr

from healthmcp.demo_data import KNOWN_DRUGS

from .agent import DrugAgent, server_params
from .brains import OpenAIBrain, RuleBrain

LIVE = os.environ.get("DRUGAGENT_LIVE") == "1"


def make_brain():
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            return OpenAIBrain(openai.OpenAI())
        except ImportError:
            pass
    return RuleBrain()


BRAIN = make_brain()
D1, D2 = ("ibuprofen", "metformin") if LIVE else (KNOWN_DRUGS[0], KNOWN_DRUGS[1])
EXAMPLES = [f"What are the warnings for {D1}?", f"Does {D1} interact with other drugs?", f"What are the most reported side effects of {D2}?",
            "Find recruiting trials for " + ("type 2 diabetes" if LIVE else "hypertension"), f"How much {D1} should I take for a headache?", "Tell me about the trials for hypertension (try with the hostile-data box ticked)"]
ICON = {"tools": "🧰", "call": "🔧", "result": "📥", "guard": "🛡️", "step": "•"}


async def answer(question: str, poison: bool):
    if not question.strip():
        return "Ask something first.", "", ""
    a = await DrugAgent(BRAIN, server_params(live=LIVE, poison=poison)).ask(question)
    steps = "\n".join(f"- {ICON.get(e.kind, '•')} {e.text}" for e in a.events) or "- (no steps)"
    return a.text, steps, f"**{len(a.calls)} tool call(s)**" + (" · refused before any tool was used" if a.refused else "")


def build():
    with gr.Blocks(title="Drug and trial lookup agent") as demo:
        gr.Markdown(f"# Drug and trial lookup agent\nAn agent that answers by calling tools on an MCP server. Data: **{'live openFDA + ClinicalTrials.gov' if LIVE else 'fictional demo data'}**. "
                    f"Reader: *{BRAIN.name}*. It looks things up; it doesn't give medical advice.")
        q = gr.Textbox(label="Ask about a drug or a clinical trial", lines=2, value=EXAMPLES[0])
        gr.Examples(EXAMPLES, q)
        poison = gr.Checkbox(False, label="Simulate hostile data: plant an instruction inside one trial title (demo data only) and watch the agent refuse to follow it", visible=not LIVE)
        go = gr.Button("Ask", variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                out = gr.Markdown(label="Answer")
            with gr.Column(scale=2):
                n = gr.Markdown()
                steps = gr.Markdown()
        go.click(answer, [q, poison], [out, steps, n])
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860)
