from src.argument_parser import parse_arguments

def main() -> None:
    args = parse_arguments()
    print(f"Functions definition: {args.functions_definition}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")

if __name__ == "__main__":
    main()