import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Converts natural language promts into "
        "funcion calls."
        )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path where the results JSON file will be saved."
    )
    return parser.parse_args()
