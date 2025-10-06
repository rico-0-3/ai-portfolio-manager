"""
Advanced sentiment analysis using FinBERT and VADER.
Provides sophisticated NLP-based sentiment scoring for financial texts.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Optional imports - will be installed via requirements
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers torch")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available. Install with: pip install vaderSentiment")


class FinBERTAnalyzer:
    """FinBERT-based sentiment analysis for financial texts."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT analyzer.

        Args:
            model_name: HuggingFace model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required. Install: pip install transformers torch")

        logger.info(f"Loading FinBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"FinBERT loaded on {self.device}")

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to probabilities
        probs = predictions.cpu().numpy()[0]

        # FinBERT output: [positive, negative, neutral]
        sentiment = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'compound': float(probs[0] - probs[1])  # Compound score
        }

        return sentiment

    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of sentiment dictionaries
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = predictions.cpu().numpy()

            for prob in probs:
                results.append({
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2]),
                    'compound': float(prob[0] - prob[1])
                })

        return results


class VADERAnalyzer:
    """VADER-based sentiment analysis (faster, rule-based)."""

    def __init__(self):
        """Initialize VADER analyzer."""
        if not VADER_AVAILABLE:
            raise ImportError("vaderSentiment required. Install: pip install vaderSentiment")

        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER analyzer initialized")

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of sentiment dictionaries
        """
        return [self.analyze(text) for text in texts]


class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer combining FinBERT and VADER.
    Uses ensemble approach for more robust predictions.
    """

    def __init__(self, use_finbert: bool = True, finbert_weight: float = 0.7):
        """
        Initialize hybrid analyzer.

        Args:
            use_finbert: Whether to use FinBERT (requires GPU for best performance)
            finbert_weight: Weight for FinBERT scores (0-1), remaining weight for VADER
        """
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE
        self.finbert_weight = finbert_weight
        self.vader_weight = 1.0 - finbert_weight

        # Initialize analyzers
        if self.use_finbert:
            try:
                self.finbert = FinBERTAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}. Using VADER only.")
                self.use_finbert = False

        if VADER_AVAILABLE:
            self.vader = VADERAnalyzer()
        else:
            logger.warning("VADER not available. Using FinBERT only if available.")

        logger.info(f"Hybrid analyzer initialized (FinBERT: {self.use_finbert}, VADER: {VADER_AVAILABLE})")

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using ensemble method.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with ensemble sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}

        # FinBERT analysis
        if self.use_finbert:
            finbert_scores = self.finbert.analyze(text)
            for key in scores:
                scores[key] += finbert_scores[key] * self.finbert_weight

        # VADER analysis
        if VADER_AVAILABLE:
            vader_scores = self.vader.analyze(text)
            for key in scores:
                scores[key] += vader_scores[key] * self.vader_weight

        return scores

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of texts.

        Args:
            df: DataFrame with text data
            text_column: Name of column containing text

        Returns:
            DataFrame with sentiment scores added
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts")

        texts = df[text_column].fillna('').tolist()

        # Use batch processing if using FinBERT
        if self.use_finbert:
            finbert_results = self.finbert.analyze_batch(texts)
        else:
            finbert_results = [None] * len(texts)

        if VADER_AVAILABLE:
            vader_results = self.vader.analyze_batch(texts)
        else:
            vader_results = [None] * len(texts)

        # Combine results
        sentiment_data = []
        for i in range(len(texts)):
            combined = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}

            if finbert_results[i]:
                for key in combined:
                    combined[key] += finbert_results[i][key] * self.finbert_weight

            if vader_results[i]:
                for key in combined:
                    combined[key] += vader_results[i][key] * self.vader_weight

            sentiment_data.append(combined)

        # Add to DataFrame
        df_result = df.copy()
        df_result['sentiment_positive'] = [s['positive'] for s in sentiment_data]
        df_result['sentiment_negative'] = [s['negative'] for s in sentiment_data]
        df_result['sentiment_neutral'] = [s['neutral'] for s in sentiment_data]
        df_result['sentiment_compound'] = [s['compound'] for s in sentiment_data]

        # Add category
        df_result['sentiment_category'] = df_result['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )

        logger.info("Sentiment analysis completed")
        return df_result
