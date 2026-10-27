"""Generate function calls using constrained decoding."""

import json

from pydantic import ValidationError
from llm_sdk import Small_LLM_Model

from src.decoding import select_constrained_token
from src.models import GeneratedFunctionCall
from src.prompt_builder import build_prompt
from src.validator import validate_function_call


def _has_balanced_json_structure(text: str) -> bool:
    """Check balanced braces while respecting JSON strings."""
    depth = 0
    in_string = False
    escaped = False

    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False

    return in_string or depth > 0


def is_valid_json_prefix(text: str) -> bool:
    """Check whether text can be extended into a valid JSON object."""
    if not text.startswith("{"):
        return False

    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        # Incomplete JSON objects are allowed during generation.
        return _has_balanced_json_structure(text)


def find_function_call(
    generated_text: str,
    functions_definition: object,
) -> GeneratedFunctionCall | None:
    """Find a valid function call anywhere in generated text."""
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
    """Generate a function call using JSON prefix constraints."""
    prompt = build_prompt(user_prompt, functions_definition)
    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []

    for _ in range(200):
        logits = model.get_logits_from_input_ids(input_ids)

        ranked_token_ids = sorted(
            range(len(logits)),
            key=logits.__getitem__,
            reverse=True,
        )

        allowed_token_ids: set[int] = set()

        # Keep several high-scoring tokens that preserve valid JSON.
        for token_id in ranked_token_ids:
            candidate_ids = generated_ids + [token_id]
            candidate_text = model.decode(candidate_ids)

            if is_valid_json_prefix(candidate_text):
                allowed_token_ids.add(token_id)

                if len(allowed_token_ids) >= 50:
                    break

        if not allowed_token_ids:
            raise ValueError(
                "No token can continue a valid JSON prefix. "
                f"Generated text: {model.decode(generated_ids)!r}"
            )

        next_token = select_constrained_token(
            logits,
            allowed_token_ids,
        )

        generated_ids.append(next_token)
        input_ids.append(next_token)

        generated_text = model.decode(generated_ids)

        result = find_function_call(
            generated_text,
            functions_definition,
        )

        if result is not None:
            return result.model_dump_json()

    raise ValueError(
        "The model did not generate a valid function call. "
        f"Generated text: {model.decode(generated_ids)!r}"
    )
