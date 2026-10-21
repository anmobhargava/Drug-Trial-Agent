# Drug and trial lookup agent: plain-language explainer

**The idea.** Instead of the model guessing about drugs from memory, it gets a toolbox (an MCP server) and has to look things up. The agent is the part that decides which tool to use, reads what comes back, and writes an answer.

**How a question flows.** First a quick check: is this a lookup, or a request for personal medical advice? Advice gets a polite refusal and nothing else happens. Otherwise the agent asks the server what tools it offers, keeps only the read-only ones, and lets the model choose. The model can ask for several lookups at once; they run together and the results go back to it. It can then look again or answer, up to 6 rounds.

**Why MCP.** The agent has no drug-specific code. It learns the tools at runtime, so swapping in a different server needs no changes to the agent.

**Not trusting the data.** Trial titles are free text written by strangers, so they could contain instructions aimed at an AI. Anything that looks like one is removed and everything else is labelled as data. The demo has a switch to try this.

**Checking the answer.** Before showing it, code checks every number and trial id against what the tools returned and rejects advice-style wording. If the answer fails twice, the user gets a plain summary of the raw data instead. Sources and the disclaimer are added by code.

**Honest limits.** Built and tested on fictional data; live APIs and the real OpenAI call weren't run in my build environment.
