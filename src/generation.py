"""Generate function calls using the language model."""

from pydantic import ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model

from src.decoding import select_next_token
from src.models import GeneratedFunctionCall
from src.prompt_builder import build_prompt


def generate_function_call(
    user_prompt: str,
    functions_definition: object,
    model: Small_LLM_Model,
) -> str:
    """Generate a function call from a user's request."""
    prompt = build_prompt(user_prompt, functions_definition)
    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []

    for _ in range(200):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_next_token(logits)

        input_ids.append(next_token)
        generated_ids.append(next_token)

        generated_text = model.decode(generated_ids).strip()

        try:
            result = GeneratedFunctionCall.model_validate_json(
                generated_text
            )
        except ValidationError:
            continue

        return result.model_dump_json()

    raise ValueError("The model did not generate a valid function call.")