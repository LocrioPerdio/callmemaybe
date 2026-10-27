"""Data models for function calling results."""

from pydantic import BaseModel


class GeneratedFunctionCall(BaseModel):
    """Represent a generated function call."""

    name: str
    parameters: dict[str, object]


class FunctionCallResult(BaseModel):
    """Represent one structured function calling result."""

    prompt: str
    name: str
    parameters: dict[str, object]
