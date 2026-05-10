"""
Agentic AI with Gemini API (google.genai SDK).

Architecture:
  - Single ReAct-style agent with tool use
  - Gemini handles the agentic loop automatically via automatic_function_calling
  - Configurable system prompt, tools, and model

Usage:
    export GEMINI_API_KEY=your_key_here
    python agent.py                          # interactive chat
    python agent.py --task "research"        # pre-set research task
    python agent.py --task "code"            # coding assistant task
"""

import argparse
import os
import sys
from typing import Optional

from google import genai
from google.genai import types

from tools import ALL_TOOLS


# --------------------------------------------------------------------------- #
# Model config                                                                 #
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """You are a capable AI assistant with access to tools.
When answering questions that require computation, file access, or web search,
use your tools rather than guessing. Think step by step.

Available tools:
- calculate(expression): evaluate math expressions
- read_file(path): read local files
- write_file(path, content): write local files
- list_directory(path): list directory contents
- search_wikipedia(query): look up Wikipedia articles
- run_python(code): run Python snippets
- get_current_time(): current date/time

Always explain your reasoning and show tool results before giving a final answer."""


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #

class GeminiAgent:
    """
    Stateful conversational agent backed by Gemini with automatic tool calling.

    The google.genai SDK handles the full agentic loop:
      1. Send user message to model
      2. Model may request tool calls
      3. SDK executes tools automatically
      4. Tool results sent back to model
      5. Model generates final text response
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        tools: list = None,
        temperature: float = 0.7,
    ):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/app/apikey"
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tools = tools or ALL_TOOLS

        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=self.tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=10,  # cap tool call depth per turn
            ),
            temperature=temperature,
        )

        # Persistent chat session maintains conversation history
        self.chat = self.client.chats.create(
            model=self.model,
            config=self.config,
        )

        print(f"Agent ready. Model: {self.model}, Tools: {[t.__name__ for t in self.tools]}")

    def send(self, message: str) -> str:
        """Send a user message; return the assistant's response text."""
        response = self.chat.send_message(message)
        return response.text

    def reset(self):
        """Start a fresh conversation."""
        self.chat = self.client.chats.create(
            model=self.model,
            config=self.config,
        )
        print("Conversation reset.")


# --------------------------------------------------------------------------- #
# Pre-set tasks                                                                #
# --------------------------------------------------------------------------- #

TASKS = {
    "research": [
        "Search Wikipedia for 'Transformer neural network' and give me a 3-bullet summary.",
        "Now search for 'Attention mechanism' and compare it to what you just found about Transformers.",
        "Calculate: how many parameters does a Transformer with 12 layers, 768 hidden dim, and 12 attention heads have? Assume standard FFN (4x hidden) and weight tying.",
    ],
    "code": [
        "Write a Python function to compute the nth Fibonacci number using memoisation. Then run it to compute F(40).",
        "Now write the iterative version and run both to compare their outputs for n in [10, 20, 30, 40].",
        "Save both implementations to a file called 'fibonacci.py' in the current directory.",
    ],
    "filesystem": [
        "List the files in the current directory.",
        "Read the README.md file if it exists, otherwise tell me what you see.",
        "Create a file called 'agent_output.txt' with a summary of what you've done so far.",
    ],
    "math": [
        "What is the area of a circle with radius 7.5?",
        "Calculate the compound interest on $10,000 at 5% annual rate for 10 years.",
        "What is log base 2 of 1024?",
    ],
}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def interactive_loop(agent: GeminiAgent):
    print("\nInteractive mode. Type 'quit' to exit, 'reset' to clear history.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        response = agent.send(user_input)
        print(f"\nAgent: {response}\n")


def run_task(agent: GeminiAgent, task_name: str):
    messages = TASKS.get(task_name)
    if not messages:
        print(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
        return

    print(f"\n=== Running task: {task_name} ===\n")
    for msg in messages:
        print(f"User: {msg}")
        response = agent.send(msg)
        print(f"Agent: {response}\n")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Gemini agentic AI demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Run a pre-set task sequence")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    agent = GeminiAgent(model=args.model, temperature=args.temperature)

    if args.task:
        run_task(agent, args.task)
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()
