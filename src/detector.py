"""
Main AI text detector interface.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
try:
    from feature_extractor_lite import FeatureExtractor
except ImportError:
    from feature_extractor import FeatureExtractor
from classifiers import XGBoostClassifier, NeuralNetTrainer


class AITextDetector:
    """Main interface for AI text detection."""

    def __init__(self, model_type='xgboost', model_path=None, language='en'):
        """
        Initialize AI text detector.

        Args:
            model_type: Type of model ('xgboost' or 'neural')
            model_path: Path to saved model (optional)
            language: Language code ('en' or 'ru')
        """
        self.model_type = model_type
        self.language = language
        self.feature_extractor = FeatureExtractor(language=language)

        if model_type == 'xgboost':
            self.model = XGBoostClassifier()
        elif model_type == 'neural':
            self.model = None  # Will be initialized with input size
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def detect(self, text: str) -> Dict[str, any]:
        """
        Detect if text is AI-generated.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train a model first.")

        # Extract features
        features = self.feature_extractor.extract_all_features(text)
        feature_vector = np.array([list(features.values())])

        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]

        result = {
            'is_ai_generated': bool(prediction),
            'confidence': float(probabilities[1] if prediction else probabilities[0]),
            'ai_probability': float(probabilities[1]),
            'human_probability': float(probabilities[0]),
            'prediction_label': 'AI-generated' if prediction else 'Human-written',
            'features': features
        }

        return result

    def detect_batch(self, texts: list) -> list:
        """
        Detect AI-generated texts in batch.

        Args:
            texts: List of texts

        Returns:
            List of detection results
        """
        return [self.detect(text) for text in texts]

    def explain_prediction(self, text: str, top_n=5) -> Dict:
        """
        Explain prediction with feature contributions.

        Args:
            text: Input text
            top_n: Number of top features to show

        Returns:
            Dictionary with explanation
        """
        result = self.detect(text)

        if self.model_type == 'xgboost' and hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

            result['top_contributing_features'] = [
                {
                    'feature': feat,
                    'importance': score,
                    'value': result['features'].get(feat, 0)
                }
                for feat, score in top_features
            ]

        return result

    def load_model(self, model_path: str):
        """Load trained model."""
        if self.model_type == 'xgboost':
            self.model.load(model_path)
        elif self.model_type == 'neural':
            # For neural network, we need to know input size
            # This will be handled during training
            pass

    def save_model(self, model_path: str):
        """Save trained model."""
        self.model.save(model_path)
