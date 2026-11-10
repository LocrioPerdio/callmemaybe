"""Build compact prompts for the language model."""

import json
from typing import Any


def build_prompt(user_prompt: str, functions_definition: Any) -> str:
    """Build a compact prompt using the available function definitions."""
    compact_definitions = [
        {
            "name": definition["name"],
            "description": definition.get("description", ""),
            "parameters": definition.get("parameters", {}),
        }
        for definition in functions_definition
    ]

    definitions = json.dumps(
        compact_definitions,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    system_message = (
        "Select the single best function for the user's request. "
        "Extract the arguments using the declared names and types. "
        "Do not invent values or execute functions. "
        'Return only JSON: {"name":"function_name",'
        '"parameters":{"argument":"value"}}. '
        "No explanation or Markdown. "
        "Numeric parameters must be JSON numbers without quotes. "
        "String parameters must be enclosed in double quotes. "
        f"Functions: {definitions}"
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