from pydantic import BaseModel
from typing import Dict


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    compound: float
