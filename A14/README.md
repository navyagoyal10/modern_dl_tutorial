# A14 — Agentic AI: Building Tool-Using Agents with Gemini

## Overview

An "agent" is a language model augmented with the ability to take actions — calling APIs, executing code, reading files, searching the web — and to loop until a goal is satisfied. This is not a single forward pass: the model observes state, decides an action, observes the result, and continues until it finishes.

This assignment builds two agentic systems using the **Google Gemini API** (`google.genai` SDK):

1. **Single ReAct agent** — one model with a full toolset, automatic function calling, and persistent chat history
2. **Multi-agent pipeline** — an Orchestrator that decomposes tasks and delegates to specialist sub-agents (Research, Code, Math), each with a focused toolset

---

## Theory

### What makes something an "agent"?

A language model alone is a function $f: \text{context} \to \text{text}$. An agent wraps this in a loop:

```
while not done:
    thought  = model(context)
    action   = parse_action(thought)
    result   = execute(action)
    context += (thought, action, result)
```

The key addition is **tool use** — the model can call external functions, and those results flow back into the context. This unlocks access to real-time information, computation, file systems, and APIs that the model cannot handle purely from weights.

### ReAct (Reason + Act)

The ReAct framework (Yao et al., 2022) interleaves **reasoning** (verbal thought steps) with **acting** (tool calls). Each step is:

1. **Thought** — the model reasons about what to do next
2. **Action** — the model calls a tool with specific arguments
3. **Observation** — the tool result is appended to context

The model iterates until it determines it has enough information to give a final answer.

Modern LLM APIs (including Gemini) implement this loop natively. You declare tools as Python functions, and the SDK handles:
- Converting function signatures to JSON schema
- Detecting when the model wants to call a tool
- Executing the function
- Returning the result to the model

### Function Calling in `google.genai`

The `google-genai` Python SDK (v1.x, 2025+) uses a simple pattern:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def my_tool(query: str) -> str:
    """Search for information about query."""
    return some_api_call(query)

config = types.GenerateContentConfig(
    tools=[my_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        maximum_remote_calls=10,
    ),
)

chat = client.chats.create(model="gemini-3.1-flash-lite", config=config)
response = chat.send_message("What is the current prime minister of India?")
print(response.text)
```

Key points:
- Pass **Python functions directly** as tools — the SDK extracts the docstring and type annotations to build the JSON schema automatically
- `automatic_function_calling` makes the SDK execute tool calls and return final text, hiding the round-trip from you
- Set `maximum_remote_calls` to bound the agent's tool-calling depth per turn
- Use `client.chats.create(...)` for stateful multi-turn conversations

### Multi-Agent Systems

A single agent with many tools can get confused. Multi-agent systems address this by:

1. **Specialisation** — each agent has a focused toolset and system prompt; it is better at one thing
2. **Parallelism** — subtasks can be dispatched concurrently (not shown here but easy to add with `asyncio`)
3. **Isolation** — context windows stay small; specialist agents don't accumulate irrelevant history

The pattern here:

```
User → Orchestrator → ResearchAgent → Wikipedia
                    → CodeAgent     → Python executor
                    → MathAgent     → Calculator
              ↑ collects results
              → final synthesis → User
```

The Orchestrator's tools are itself Python functions that call the specialist agents. This is a clean abstraction: Gemini sees the specialists as tools just like any other function.

### Agentic Failure Modes

Understanding where agents break is as important as building them:

| Failure | Cause | Fix |
|---------|-------|-----|
| Tool call loop | Model keeps calling tools without making progress | `maximum_remote_calls` cap |
| Wrong tool | Model misreads task scope | Better system prompt, clearer tool docstrings |
| Hallucinated results | Model ignores tool output, answers from weights | Explicit "always use tools" in system prompt |
| Context overflow | Long tool results fill the window | Summarise results before returning to model |
| Stale data | Model answers from training knowledge | Ground with a search tool |



---

## Reading Material

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022): https://arxiv.org/abs/2210.03629
- Google Gen AI Python SDK docs: https://googleapis.github.io/python-genai/
- Gemini function calling guide: https://ai.google.dev/gemini-api/docs/function-calling
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022): https://arxiv.org/abs/2201.11903
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023): https://arxiv.org/abs/2302.04761

---

## Assignment

### Setup

```bash
pip install google-genai
export GEMINI_API_KEY=your_key_here
```

### Task

**Part 1 — Single agent with tools:** Run `agent.py` interactively. Ask it multi-step questions that require tool use (e.g., "Find out who invented the Transformer and compute the number of years since then"). Observe how the SDK interleaves tool calls and reasoning.

**Part 2 — Implement a custom tool:** Add a new tool to `tools.py` — for example:
- `fetch_url(url)` — fetch raw text from a URL
- `summarise_text(text, max_words)` — use Gemini to recursively summarise
- `solve_equation(equation)` — solve symbolic equations

Register it in `ALL_TOOLS` and verify the agent uses it.

**Part 3 — Multi-agent pipeline:** Run `multi_agent.py`. Give it a task that spans all three specialists, e.g. "Research what GRPO is, write Python code to simulate 100 GRPO reward comparisons, and compute the expected advantage for a group of 6 completions with rewards [0, 0, 0.2, 0.4, 1.0, 1.0]."

**Part 4 — Build your own agent task:** Design an original agentic task that:
- Requires at least 3 sequential tool calls (each depends on the previous)
- Has a verifiable answer (so you can check if the agent succeeded)

Examples: a fact-checking agent, a code-review agent, a study-plan generator.

**Part 5 — Failure analysis:** Intentionally break your agent:
- Give it a task with an ambiguous tool choice
- Give it incorrect information and ask it to verify
- Ask it to call a tool more than `maximum_remote_calls` times

Report what happens and how the system degrades.

### What to implement

1. `tools.py` — provided + your custom tool ✓ (extend the file)
2. `agent.py` — single agent with automatic function calling ✓ (provided)
3. `multi_agent.py` — orchestrator + specialists ✓ (provided)
4. `my_agent_task.py` — your original agentic task (you implement)

### Deliverables

- [ ] `transcript_single_agent.txt` — 3 interesting multi-step conversations with the single agent
- [ ] `custom_tool.py` — your added tool + a demo showing it works
- [ ] `transcript_multi_agent.txt` — 2 multi-specialist tasks with agent outputs
- [ ] `my_agent_task.py` — your original agent design with documentation
- [ ] `failure_analysis.md` — 3 failure modes observed and analysis

---

## Starter Workflow

```bash
# Install
pip install google-genai

# Set API key
export GEMINI_API_KEY=your_key_from_aistudio

# Part 1: single agent
python agent.py
python agent.py --task research    # pre-set task
python agent.py --task code
python agent.py --task math

# Part 3: multi-agent
python multi_agent.py
```

---

