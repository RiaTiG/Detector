"""
Dataset preparation and loading utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import os

# Check if we should use full version
if os.environ.get('USE_FULL_FEATURE_EXTRACTOR'):
    from feature_extractor import FeatureExtractor
else:
    try:
        from feature_extractor_lite import FeatureExtractor
    except ImportError:
        from feature_extractor import FeatureExtractor


class DatasetLoader:
    """Load and prepare datasets for training."""

    def __init__(self, language='en'):
        """
        Initialize dataset loader.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language
        self.feature_extractor = FeatureExtractor(language=language)

    def load_from_csv(self, csv_path: str, text_column='text', label_column='label') -> pd.DataFrame:
        """
        Load dataset from CSV file.

        Args:
            csv_path: Path to CSV file
            text_column: Name of column containing text
            label_column: Name of column containing labels (0=human, 1=AI)

        Returns:
            DataFrame with text and labels
        """
        df = pd.read_csv(csv_path)
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"CSV must contain '{text_column}' and '{label_column}' columns")
        return df[[text_column, label_column]]

    def load_from_text_files(self, human_dir: str, ai_dir: str) -> pd.DataFrame:
        """
        Load dataset from text files in directories.

        Args:
            human_dir: Directory containing human-written texts
            ai_dir: Directory containing AI-generated texts

        Returns:
            DataFrame with text and labels
        """
        data = []

        # Load human texts
        human_path = Path(human_dir)
        if human_path.exists():
            for file_path in human_path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        data.append({'text': text, 'label': 0, 'source': 'human'})

        # Load AI texts
        ai_path = Path(ai_dir)
        if ai_path.exists():
            for file_path in ai_path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        data.append({'text': text, 'label': 1, 'source': 'ai'})

        return pd.DataFrame(data)

    def extract_features_from_dataset(self, df: pd.DataFrame, text_column='text') -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from all texts in dataset.

        Args:
            df: DataFrame containing texts
            text_column: Name of column containing text

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        all_features = []
        feature_names = None

        print("Extracting features...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row[text_column]
            features = self.feature_extractor.extract_all_features(text)

            if feature_names is None:
                feature_names = sorted(features.keys())

            # Ensure consistent feature order
            feature_vector = [features[name] for name in feature_names]
            all_features.append(feature_vector)

        return np.array(all_features), feature_names

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        text_column='text',
        label_column='label',
        test_size=0.2,
        random_state=42
    ) -> Dict:
        """
        Prepare complete training dataset with features.

        Args:
            df: DataFrame with texts and labels
            text_column: Name of text column
            label_column: Name of label column
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Dictionary with train/test splits and metadata
        """
        from sklearn.model_selection import train_test_split

        # Extract features
        X, feature_names = self.extract_features_from_dataset(df, text_column)
        y = df[label_column].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }

    def create_sample_dataset(self, output_dir: str, n_samples_per_class=50):
        """
        Create sample dataset for testing.

        Args:
            output_dir: Directory to save sample texts
            n_samples_per_class: Number of samples per class
        """
        output_path = Path(output_dir)
        human_dir = output_path / 'human'
        ai_dir = output_path / 'ai'

        human_dir.mkdir(parents=True, exist_ok=True)
        ai_dir.mkdir(parents=True, exist_ok=True)

        # Sample human-like texts
        human_samples = [
            "I can't believe how crazy yesterday was! Met up with Sarah and we grabbed coffee at that new place downtown. The barista was super nice, gave us extra whipped cream lol. Anyway, we talked for hours about everything - work, relationships, life stuff. It's amazing how time flies when you're with good people.",
            "Been thinking a lot about my career lately. Not sure if I'm on the right path, you know? Sometimes I feel like I should've studied something else, but then other days I love what I do. Life's complicated like that I guess.",
            "Just finished reading this book my friend recommended. Honestly? Wasn't that great. The plot was kinda predictable and the characters felt flat. But hey, at least I can say I read it!",
            "My cat did the funniest thing this morning - jumped into a box that was way too small for him and just sat there looking confused. Classic cat behavior 😂",
            "Anyone else feel like Mondays are getting harder? I swear I can't get out of bed anymore. Need more vacation time...",
        ]

        # Sample AI-like texts
        ai_samples = [
            "The implementation of artificial intelligence in modern business practices represents a significant paradigm shift in operational efficiency. Organizations that leverage machine learning algorithms demonstrate improved decision-making capabilities and enhanced productivity metrics across multiple operational domains.",
            "Climate change mitigation strategies require comprehensive policy frameworks that integrate technological innovation with sustainable development practices. The transition to renewable energy sources, coupled with carbon capture technologies, presents viable pathways toward achieving long-term environmental sustainability goals.",
            "Effective communication in professional settings necessitates clear articulation of objectives, active listening skills, and the ability to adapt messaging to diverse audiences. These fundamental competencies contribute to enhanced collaboration and organizational success.",
            "The evolution of digital marketing strategies reflects changing consumer behaviors and technological advancements. Data-driven approaches enable targeted campaigns that optimize engagement rates and conversion metrics while maintaining brand consistency across multiple platforms.",
            "Educational systems worldwide face challenges in adapting to rapid technological changes. Integration of digital learning tools and personalized instruction methods can enhance student outcomes while preparing learners for future workforce demands.",
        ]

        # Save human samples
        for i, text in enumerate(human_samples[:n_samples_per_class]):
            with open(human_dir / f'human_{i:03d}.txt', 'w', encoding='utf-8') as f:
                f.write(text)

        # Save AI samples
        for i, text in enumerate(ai_samples[:n_samples_per_class]):
            with open(ai_dir / f'ai_{i:03d}.txt', 'w', encoding='utf-8') as f:
                f.write(text)

        print(f"Sample dataset created in {output_dir}")
        print(f"Human texts: {len(human_samples)}")
        print(f"AI texts: {len(ai_samples)}")
