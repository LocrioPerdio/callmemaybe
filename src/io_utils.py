import json
from typing import Any
from pathlib import Path


def load_json(file_path: Path) -> list[Any]:
    """Description: Load data from a json file
        Parameters: file_path
        Returns: list
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Not JSON file found to be loaded")
        return []
    except json.JSONDecodeError:
        print("JSON file has invalid format")
        return []


def save_json(file_path: Path, results: list[Any]) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    except Exception:
        print("Error saving results")
