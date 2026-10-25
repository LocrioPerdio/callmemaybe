"""Entry point for the function calling project."""

import json

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.argument_parser import parse_arguments
from src.generation import generate_function_call
from src.io_utils import load_json, save_json
from src.models import FunctionCallResult, GeneratedFunctionCall


def main() -> None:
    """Run the application."""
    args = parse_arguments()

    try:
        functions_definition = load_json(args.functions_definition)
        test_cases = load_json(args.input)
    except (OSError, ValueError) as error:
        print(f"Error loading input files: {error}")
        return

    if not isinstance(test_cases, list):
        print("Error: the input file must contain a JSON list.")
        return

    try:
        model = Small_LLM_Model()
    except Exception as error:
        print(f"Error initializing the language model: {error}")
        return

    results: list[dict[str, object]] = []

    for index, test_case in enumerate(test_cases, start=1):
        if not isinstance(test_case, dict):
            print(f"Skipping test case {index}: expected a JSON object.")
            continue

        user_prompt = test_case.get("prompt")

        if not isinstance(user_prompt, str):
            print(f"Skipping test case {index}: missing a valid prompt.")
            continue

        try:
            generated_json = generate_function_call(
                user_prompt,
                functions_definition,
                model,
            )

            function_call = GeneratedFunctionCall.model_validate_json(
                generated_json
            )

            result = FunctionCallResult(
                prompt=user_prompt,
                name=function_call.name,
                parameters=function_call.parameters,
            )

            results.append(result.model_dump())

        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            print(f"Error processing test case {index}: {error}")
        except Exception as error:
            print(f"Unexpected error in test case {index}: {error}")

    try:
        save_json(args.output, results)
    except OSError as error:
        print(f"Error saving results: {error}")
        return

    print(f"Processed {len(results)} of {len(test_cases)} test cases.")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()