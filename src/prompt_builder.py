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

    return (
        "You are a function calling assistant.\n"
        "Choose the function that best matches the user's request.\n"
        "Do not execute the function or answer the user directly.\n"
        "Return only a JSON object with the keys "
        '"name" and "parameters".\n\n'
        f"Available function definitions:\n{definitions}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Function call:"
    )