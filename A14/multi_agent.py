"""
Multi-agent pipeline with Gemini.

Architecture:
  Orchestrator agent — receives user task, decomposes into subtasks, delegates
  Specialist agents  — each has a focused toolset and system prompt
    • ResearchAgent  — Wikipedia search + summarise
    • CodeAgent      — Python execution + file I/O
    • MathAgent      — calculation + numerical reasoning

The Orchestrator collects specialist outputs and synthesises a final answer.
All agents use the google.genai client with function calling.
"""

import os
from typing import Optional

from google import genai
from google.genai import types

from tools import calculate, search_wikipedia, run_python, get_current_time, read_file, write_file


# --------------------------------------------------------------------------- #
# Specialist agents                                                            #
# --------------------------------------------------------------------------- #

class SpecialistAgent:
    def __init__(self, name: str, system_prompt: str, tools: list, model: str = "gemini-3.1-flash-lite"):
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.name = name
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=5,
            ),
            temperature=0.3,
        )
        self.model = model

    def run(self, task: str) -> str:
        """Execute a single-turn task (no persistent history)."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=task,
            config=self.config,
        )
        return response.text


def make_research_agent() -> SpecialistAgent:
    return SpecialistAgent(
        name="ResearchAgent",
        system_prompt=(
            "You are a research specialist. When given a topic, search Wikipedia, "
            "synthesise the key facts, and return a concise structured summary. "
            "Always cite what you found."
        ),
        tools=[search_wikipedia, get_current_time],
    )


def make_code_agent() -> SpecialistAgent:
    return SpecialistAgent(
        name="CodeAgent",
        system_prompt=(
            "You are a Python coding specialist. Write correct, minimal Python code "
            "to solve the given task, run it, and report the result. "
            "Show the code and its output."
        ),
        tools=[run_python, read_file, write_file],
    )


def make_math_agent() -> SpecialistAgent:
    return SpecialistAgent(
        name="MathAgent",
        system_prompt=(
            "You are a mathematical reasoning specialist. "
            "Use the calculate tool for all numerical work. "
            "Show formulas, intermediate steps, and final answers."
        ),
        tools=[calculate],
    )


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #

ORCHESTRATOR_SYSTEM = """You are an orchestrator AI that coordinates specialist agents.

Given a complex user task, you will:
1. Break it into sub-tasks suited for each specialist
2. Call each specialist with the relevant sub-task
3. Collect the results
4. Synthesise a final coherent answer

You have three specialists you can call:
- research_specialist(task): searches Wikipedia and summarises information
- code_specialist(task): writes and runs Python code
- math_specialist(task): solves mathematical problems

Always use specialists when their domain is relevant. Synthesise at the end."""


class OrchestratorAgent:
    def __init__(self, model: str = "gemini-3.1-flash-lite"):
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Specialist agents
        self._research = make_research_agent()
        self._code = make_code_agent()
        self._math = make_math_agent()

        # Tool wrappers that delegate to specialists
        def research_specialist(task: str) -> str:
            """
            Delegate a research/information-lookup task to the research specialist agent.

            Args:
                task: The research question or topic to investigate.
            """
            print(f"  [ResearchAgent] ← {task[:60]}...")
            result = self._research.run(task)
            print(f"  [ResearchAgent] → done")
            return result

        def code_specialist(task: str) -> str:
            """
            Delegate a coding or computation task to the code specialist agent.

            Args:
                task: Description of the code to write and run.
            """
            print(f"  [CodeAgent] ← {task[:60]}...")
            result = self._code.run(task)
            print(f"  [CodeAgent] → done")
            return result

        def math_specialist(task: str) -> str:
            """
            Delegate a mathematical problem to the math specialist agent.

            Args:
                task: The mathematical problem or calculation to solve.
            """
            print(f"  [MathAgent] ← {task[:60]}...")
            result = self._math.run(task)
            print(f"  [MathAgent] → done")
            return result

        self.config = types.GenerateContentConfig(
            system_instruction=ORCHESTRATOR_SYSTEM,
            tools=[research_specialist, code_specialist, math_specialist],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=10,
            ),
            temperature=0.5,
        )

        self.chat = self.client.chats.create(model=self.model, config=self.config)

    def send(self, message: str) -> str:
        response = self.chat.send_message(message)
        return response.text


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #

DEMO_TASKS = [
    (
        "Research task",
        "Explain what Transformers are in machine learning, then write Python code to count "
        "how many parameters a Transformer with 6 layers, 512 hidden dim, 8 heads has."
    ),
    (
        "Math + code task",
        "Calculate the first 10 prime numbers mathematically, then verify with a Python script."
    ),
]


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY first: export GEMINI_API_KEY=your_key")
        return

    print("Initialising multi-agent system...")
    orchestrator = OrchestratorAgent()
    print("Multi-agent system ready.\n")

    for title, task in DEMO_TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {title}")
        print(f"{'='*60}")
        print(f"User: {task}\n")
        response = orchestrator.send(task)
        print(f"\nOrchestrator: {response}")
        print()

    # Interactive mode
    print("\n=== Interactive mode ===")
    print("Type your question (multi-domain questions work best). 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user.lower() == "quit":
            break
        if not user:
            continue
        response = orchestrator.send(user)
        print(f"\nOrchestrator: {response}\n")


if __name__ == "__main__":
    main()
