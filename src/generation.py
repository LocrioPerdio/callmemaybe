"""Generate function calls using constrained decoding."""

import json

from pydantic import ValidationError
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.decoding import select_next_token
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
    """Find and validate a function call in generated text."""
    decoder = json.JSONDecoder()

    if not isinstance(functions_definition, list):
        return None

    for index, character in enumerate(generated_text):
        if character != "{":
            continue

        try:
            data, _ = decoder.raw_decode(generated_text[index:])
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        function_name = data.get("name")
        parameters = data.get("parameters")

        if not isinstance(function_name, str):
            continue

        if not isinstance(parameters, dict):
            continue

        definition = next(
            (
                item
                for item in functions_definition
                if isinstance(item, dict)
                and item.get("name") == function_name
            ),
            None,
        )

        if definition is None:
            continue

        expected_parameters = definition.get("parameters")

        if not isinstance(expected_parameters, dict):
            continue

        normalized_parameters = parameters.copy()
        conversion_failed = False

        for parameter_name, specification in expected_parameters.items():
            if not isinstance(specification, dict):
                conversion_failed = True
                break

            if specification.get("type") != "number":
                continue

            value = normalized_parameters.get(parameter_name)

            if isinstance(value, str):
                try:
                    converted_value = json.loads(value)
                except json.JSONDecodeError:
                    conversion_failed = True
                    break

                if (
                    isinstance(converted_value, bool)
                    or not isinstance(converted_value, (int, float))
                ):
                    conversion_failed = True
                    break

                normalized_parameters[parameter_name] = converted_value

        if conversion_failed:
            continue

        try:
            result = GeneratedFunctionCall.model_validate({
                "name": function_name,
                "parameters": normalized_parameters,
            })
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
    """Generate and validate a function call for the user's request."""
    print("Building prompt...", flush=True)
    prompt = build_prompt(user_prompt, functions_definition)
    print("Tokenizing prompt...", flush=True)
    input_ids = model.encode(prompt)[0].tolist()

    generated_ids: list[int] = []

    for step in range(120):
        print(f"Generating token {step + 1}/120...", flush=True)
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_next_token(logits)

        input_ids.append(next_token)
        generated_ids.append(next_token)

        # Decode and validate only every 4 tokens.
        if (step + 1) % 4 == 0:
            generated_text = model.decode(generated_ids)

            result = find_function_call(
                generated_text,
                functions_definition,
            )

            if result is not None:
                return result.model_dump_json()

    # Validate the remaining generated text.
    generated_text = model.decode(generated_ids)
    result = find_function_call(
        generated_text,
        functions_definition,
    )

    if result is not None:
        print("Valid function call generated.", flush=True)
        return result.model_dump_json()

    print(f"\nFAILED PROMPT: {user_prompt!r}", flush=True)
    print(f"GENERATED TEXT: {generated_text!r}", flush=True)

    raise ValueError(
        "The model did not generate a valid function call. "
        f"Generated text: {generated_text!r}"
    )