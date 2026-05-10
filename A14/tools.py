"""
Tool implementations for the Gemini agentic AI assignment.

Tools expose Python functions that Gemini can call via function calling.
The SDK automatically converts Python functions to JSON schema tool declarations.
"""

import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Math / reasoning tools                                                       #
# --------------------------------------------------------------------------- #

def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports arithmetic, sqrt, log, sin, cos, pow, etc.
    Returns the numeric result as a string.

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144) + 3*7"
    """
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --------------------------------------------------------------------------- #
# File system tools                                                            #
# --------------------------------------------------------------------------- #

def read_file(path: str) -> str:
    """
    Read the contents of a local file and return it as a string.

    Args:
        path: Absolute or relative path to the file.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found at {path}"
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    """
    Write content to a local file, creating parent directories as needed.

    Args:
        path: File path to write to.
        content: Text content to write.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error: {e}"


def list_directory(path: str = ".") -> str:
    """
    List files and directories at the given path.

    Args:
        path: Directory to list (default: current directory).
    """
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{'[DIR]' if e.is_dir() else '[FILE]'} {e.name}" for e in entries]
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


# --------------------------------------------------------------------------- #
# Web / search simulation tools                                               #
# --------------------------------------------------------------------------- #

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for a query and return the introductory summary.
    Uses the Wikipedia REST API (no API key required).

    Args:
        query: Search term or article title.
    """
    import urllib.request
    import urllib.parse

    term = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ModernDLTutorial/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("extract", "No summary found.")
    except Exception as e:
        return f"Error fetching Wikipedia: {e}"


# --------------------------------------------------------------------------- #
# Code execution tool                                                          #
# --------------------------------------------------------------------------- #

def run_python(code: str) -> str:
    """
    Execute a Python code snippet in a subprocess and return stdout + stderr.
    Execution is time-limited to 10 seconds.

    Args:
        code: Python source code to execute.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\nSTDERR: {result.stderr.strip()}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (>10s)"
    except Exception as e:
        return f"Error: {e}"


# --------------------------------------------------------------------------- #
# Time / datetime tool                                                         #
# --------------------------------------------------------------------------- #

def get_current_time(timezone: Optional[str] = None) -> str:
    """
    Return the current date and time.

    Args:
        timezone: Optional timezone name (e.g. 'UTC', 'US/Eastern'). If None, uses system local time.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


# --------------------------------------------------------------------------- #
# All tools exported for the agent                                             #
# --------------------------------------------------------------------------- #

ALL_TOOLS = [calculate, read_file, write_file, list_directory,
             search_wikipedia, run_python, get_current_time]
