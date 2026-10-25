"""Generate function calls using the language model."""

import json

from pydantic import ValidationError
from llm_sdk import Small_LLM_Model

from src.decoding import select_next_token
from src.models import GeneratedFunctionCall
from src.prompt_builder import build_prompt
from src.validator import  validate_function_call

def find_function_call(
    generated_text: str,
    functions_definition: object,
) -> GeneratedFunctionCall | None:
    """Find the first valid function call in generated text."""
    decoder = json.JSONDecoder()

    for index, character in enumerate(generated_text):
        if character != "{":
            continue

        try:
            data, _ = decoder.raw_decode(generated_text[index:])
        except json.JSONDecodeError:
            continue

        try:
            result = GeneratedFunctionCall.model_validate(data)
        except ValidationError:
            continue

        if validate_function_call(result, functions_definition):
            return result

    return None

def generate_function_call(
    user_prompt: str,
    functions_definition: object,
    model: Small_LLM_Model,
) -> str:
    """Generate a function call from a user's request."""
    prompt = build_prompt(user_prompt, functions_definition)
    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []

    for _ in range(80):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_next_token(logits)

        input_ids.append(next_token)
        generated_ids.append(next_token)

        generated_text = model.decode(generated_ids)
        result = find_function_call(
            generated_text,
            functions_definition,
        )

        if result is not None:
            return result.model_dump_json()

    raise ValueError(
        "The model did not generate a valid function call. "
        f"Generated text: {generated_text!r}"
    )