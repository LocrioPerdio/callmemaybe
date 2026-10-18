from src.argument_parser import parse_arguments
from src.io_utils import load_json


def main() -> None:
    
    args = parse_arguments()
    try:
        functions_definition = load_json(args.functions_definition)
        test_cases = load_json(args.input)
    except (OSError, ValueError) as error:
        print(f"Error loading input files: {error}")
        return
    
    print(f"Loaded functions: {len(functions_definition)}")
    print(f"Loaded test cases: {len(test_cases)}")


if __name__ == "__main__":
    main()