
from src.models import GeneratedFunctionCall


def validate_function_call(
    function_call: GeneratedFunctionCall,
    definitions: object,
) -> bool:
    """Check a function call against its declared parameter schema."""
    if not isinstance(definitions, list):
        return False

    definition = next(
        (
            item
            for item in definitions
            if isinstance(item, dict)
            and item.get("name") == function_call.name
        ),
        None,
    )

    if definition is None:
        return False

    expected_parameters = definition.get("parameters")
    actual_parameters = function_call.parameters

    if not isinstance(expected_parameters, dict):
        return False

    if set(actual_parameters) != set(expected_parameters):
        return False

    for parameter_name, specification in expected_parameters.items():
        if not isinstance(specification, dict):
            return False

        parameter_type = specification.get("type")
        value = actual_parameters[parameter_name]

        if parameter_type == "string":
            if not isinstance(value, str):
                return False

        elif parameter_type == "number":
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                return False

        else:
            return False

    return True
