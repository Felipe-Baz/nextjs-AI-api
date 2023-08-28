from typing import Dict, Tuple

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

from deep_translator import GoogleTranslator


class Model:
    def __init__(self):
        self.classifier = get_classifier()

    def _translate(self, text: str, target_lang: str = "en") -> str:
        """Translate a given text into the specified target language.

        Args:
            text (str): The text to be translated.
            target_lang (str, optional): The target language code for translation. Defaults to 'en'.

        Returns:
            str: The translated text in the specified target language.
        """
        return GoogleTranslator(source="auto", target=target_lang).translate(text)

    def _analyze_feeling(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of the given text and provide sentiment scores.

        This function uses a sentiment analysis classifier to evaluate the sentiment
        of the provided text and returns a dictionary of sentiment scores.

        Args:
            text (str): The text to be analyzed for sentiment.

        Returns:
            Dict[str, float]: A dictionary containing sentiment scores. The keys are:
            - 'neg': Negative sentiment score
            - 'neu': Neutral sentiment score
            - 'pos': Positive sentiment score
            - 'compound': Compound sentiment score (overall sentiment)
        """
        return self.classifier.polarity_scores(text)

    def _transform_feeling(self, feeling: Dict[str, float]) -> str:
        """
        Transforms a sentiment analysis score represented as a dictionary into a feeling label.

        This private method takes a sentiment analysis score represented as a dictionary and
        categorizes it into a feeling label. The sentiment score is expected to have at least
        a 'compound' key with a float value representing the sentiment polarity.

        Args:
            feeling (Dict[str, float]): A dictionary containing sentiment analysis scores.
                It should have at least a 'compound' key representing the sentiment polarity.

        Returns:
            str: A string representing the feeling label derived from the sentiment score.
                Possible values are 'Positivo' for positive sentiment, 'Negativo' for negative sentiment,
                and 'Neutro' for neutral sentiment.

        Note:
            The cutoff values for categorizing sentiment into 'Positivo', 'Negativo', and 'Neutro'
            are based on comparison with the 'compound' score:
            - If feeling['compound'] > 0.05, the sentiment is categorized as 'Positivo'.
            - If feeling['compound'] < -0.05, the sentiment is categorized as 'Negativo'.
            - Otherwise, the sentiment is categorized as 'Neutro'.

        Example:
            - sentiment_score = {'compound': 0.2, 'positive': 0.6, 'negative': 0.1, 'neutral': 0.3}
            - feeling_label = _transform_feeling(sentiment_score)
            - Result: 'Positivo'
        """
        if feeling["compound"] > 0.05:
            return "Positivo"
        elif feeling["compound"] < -0.05:
            return "Negativo"
        else:
            return "Neutro"

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts sentiment and provides sentiment analysis results for a given text.

        This method takes a text input, translates it, performs sentiment analysis, and returns
        a sentiment label, sentiment compound score, and a dictionary of sentiment analysis scores.

        Args:
            text (str): The input text for sentiment analysis.

        Returns:
            Tuple[str, float, Dict[str, float]]: A tuple containing:
                - Sentiment label as a string ('Positivo', 'Negativo', 'Neutro').
                - Sentiment compound score as a float.
                - A dictionary containing sentiment analysis scores with keys 'positive', 'negative',
                and 'neutral'.

        Note:
            The `_translate` method is used to translate the input text, `_analyze_feeling` is used
            for sentiment analysis, and `_transform_feeling` categorizes the sentiment into labels.

        Example:
            sentiment_predictor = SentimentPredictor()  # Create an instance of the SentimentPredictor class

            input_text = "I'm so excited about this!"

            sentiment_label, sentiment_score, sentiment_scores = sentiment_predictor.predict(input_text)

            Result: sentiment_label='Positivo', sentiment_score=0.875, sentiment_scores={'positive': 0.875, 'negative': 0.0, 'neutral': 0.125}
        """
        _translated_text: str = self._translate(text)
        _result = self._analyze_feeling(_translated_text)
        _compound = _result.get("compound")
        _sentiment = self._transform_feeling(_result)
        del _result["compound"]
        return _sentiment, _compound, _result


def get_classifier() -> SentimentIntensityAnalyzer:
    """
    Retrieves a SentimentIntensityAnalyzer classifier instance.

    This function returns an instance of the SentimentIntensityAnalyzer class, which is used for
    sentiment analysis. The SentimentIntensityAnalyzer provides a pre-trained model for
    sentiment analysis based on the intensity of sentiment words in text.

    Returns:
        SentimentIntensityAnalyzer: An instance of the SentimentIntensityAnalyzer classifier.

    Example:
        classifier = get_classifier()

        text = "This movie is fantastic!"

        sentiment_scores = classifier.polarity_scores(text)

        Result: sentiment_scores={'neg': 0.0, 'neu': 0.218, 'pos': 0.782, 'compound': 0.6696}
    """
    return SentimentIntensityAnalyzer()


model = Model()


def get_model():
    return model
