"""Build prompts for the language model."""

import json
from typing import Any


def build_prompt(user_prompt: str, functions_definition: Any) -> str:
    """Build the input prompt for function calling."""
    definitions = json.dumps(
        functions_definition,
        ensure_ascii=False,
        indent=2,
    )

    system_message = (
        "You are a function calling assistant.\n"
        "Select exactly one function from the provided definitions.\n"
        "Use the user's exact argument values. Never invent values.\n"
        "Use only the declared function names and parameter names.\n"
        "Do not execute functions or answer the user's request.\n"
        "Return exactly one JSON object with these keys: "
        '"name" and "parameters".\n'
        "Do not include explanations or Markdown fences.\n"
        "Do not include any text before or after the JSON object.\n"
        "Available function definitions:\n"
        f"{definitions}"
    )

    return (
        "<|im_start|>system\n"
        f"{system_message}\n"
        "/no_think"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
