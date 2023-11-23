from pydantic import BaseModel
from datetime import datetime

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float


class LLMResponse(BaseModel):
    user_prompt: str
    system_response: str | None = None
    usage: TokenUsage
    created_on: int = datetime.now().isoformat()