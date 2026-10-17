from pydantic import BaseModel

class FunctionCallResult(BaseModel):
    """Represent one structured function calling result"""

    prompt: str
    name: str
    parameters: dict[str, object]