from fastapi import APIRouter, Depends

from api.dto.SentimentRequest import SentimentRequest
from api.dto.SentimentResponse import SentimentResponse
from api.helper.sentiment_model import Model, get_model

router = APIRouter()


@router.post("/feeling_analysis/", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    """
    The injection of the parameter model is done by the function get_model
    """
    sentiment, compound, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, probabilities=probabilities, compound=compound
    )
