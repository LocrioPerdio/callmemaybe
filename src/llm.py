from llm_sdk.llm_sdk import Small_LLM_Model

class LLM:

    def __init__(self) -> None:
        self.model = Small_LLM_Model()

    def encode (self, text: str) -> list[int]:
        return self.model.encode(text)[0].tolist()

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(input_ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)

