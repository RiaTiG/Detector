"""
Lightweight feature extraction module without spaCy (Python 3.14 compatible).
Extracts stylometric and basic syntactic features that are hard for AI to imitate.
"""

import re
import numpy as np
from collections import Counter
from typing import Dict, List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class FeatureExtractor:
    """Extract features from text that distinguish human from AI writing."""

    def __init__(self, language='en'):
        """
        Initialize feature extractor.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language

        # Load stopwords
        try:
            self.stopwords = set(stopwords.words('english' if language == 'en' else 'russian'))
        except:
            self.stopwords = set()

    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of feature names and values
        """
        features = {}

        # Stylometric features
        features.update(self.extract_stylometric_features(text))

        # Syntactic features (simplified without spaCy)
        features.update(self.extract_syntactic_features(text))

        # Semantic features (simplified without spaCy)
        features.update(self.extract_semantic_features(text))

        return features

    def extract_stylometric_features(self, text: str) -> Dict[str, float]:
        """
        Extract stylometric features.

        Features:
        - Average sentence length (words)
        - Sentence length variance
        - Lexical diversity (TTR - Type-Token Ratio)
        - Hapax legomena ratio (words appearing once)
        - Function word frequency
        - Punctuation density
        - Average word length
        - Word length variance
        """
        features = {}

        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_alpha = [w for w in words if w.isalpha()]

        if not sentences or not words_alpha:
            return self._get_zero_stylometric_features()

        # Sentence length statistics
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        features['avg_sentence_length'] = np.mean(sentence_lengths)
        features['sentence_length_variance'] = np.var(sentence_lengths)
        features['sentence_length_std'] = np.std(sentence_lengths)
        features['max_sentence_length'] = max(sentence_lengths)
        features['min_sentence_length'] = min(sentence_lengths)

        # Lexical diversity
        features['type_token_ratio'] = len(set(words_alpha)) / len(words_alpha)

        # Hapax legomena (words appearing once)
        word_counts = Counter(words_alpha)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        features['hapax_ratio'] = hapax_count / len(word_counts)

        # Dis legomena (words appearing twice)
        dis_count = sum(1 for count in word_counts.values() if count == 2)
        features['dis_legomena_ratio'] = dis_count / len(word_counts)

        # Function words (stopwords) frequency
        stopword_count = sum(1 for w in words_alpha if w in self.stopwords)
        features['stopword_ratio'] = stopword_count / len(words_alpha)

        # Punctuation density
        punctuation_count = sum(1 for c in text if c in '.,;:!?-—()[]{}"\'"')
        features['punctuation_density'] = punctuation_count / len(words)

        # Specific punctuation
        features['comma_ratio'] = text.count(',') / len(sentences)
        features['semicolon_ratio'] = text.count(';') / len(sentences)
        features['exclamation_ratio'] = text.count('!') / len(sentences)
        features['question_ratio'] = text.count('?') / len(sentences)

        # Word length statistics
        word_lengths = [len(w) for w in words_alpha]
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_variance'] = np.var(word_lengths)

        # Long words ratio (>6 characters)
        long_words = sum(1 for w in words_alpha if len(w) > 6)
        features['long_word_ratio'] = long_words / len(words_alpha)

        # Short words ratio (<=3 characters)
        short_words = sum(1 for w in words_alpha if len(w) <= 3)
        features['short_word_ratio'] = short_words / len(words_alpha)

        return features

    def extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """
        Extract syntactic features using basic NLP.

        Features:
        - POS tag patterns
        - Comma patterns
        - Parentheses usage
        """
        features = {}

        sentences = sent_tokenize(text)

        if not sentences:
            return self._get_zero_syntactic_features()

        # Comma patterns
        commas_per_sentence = [s.count(',') for s in sentences]
        features['avg_commas_per_sentence'] = np.mean(commas_per_sentence) if commas_per_sentence else 0
        features['comma_variance'] = np.var(commas_per_sentence) if commas_per_sentence else 0

        # Parentheses and brackets
        features['parentheses_ratio'] = (text.count('(') + text.count(')')) / len(sentences)
        features['bracket_ratio'] = (text.count('[') + text.count(']')) / len(sentences)

        # Quote usage
        features['quote_ratio'] = (text.count('"') + text.count("'")) / len(sentences)

        # POS tagging (simplified)
        try:
            all_words = word_tokenize(text)
            pos_tags = nltk.pos_tag(all_words)

            # Count POS categories
            noun_count = sum(1 for _, pos in pos_tags if pos.startswith('NN'))
            verb_count = sum(1 for _, pos in pos_tags if pos.startswith('VB'))
            adj_count = sum(1 for _, pos in pos_tags if pos.startswith('JJ'))
            adv_count = sum(1 for _, pos in pos_tags if pos.startswith('RB'))

            total_words = len([w for w in all_words if w.isalpha()])
            if total_words > 0:
                features['noun_ratio'] = noun_count / total_words
                features['verb_ratio'] = verb_count / total_words
                features['adj_ratio'] = adj_count / total_words
                features['adv_ratio'] = adv_count / total_words
            else:
                features['noun_ratio'] = 0
                features['verb_ratio'] = 0
                features['adj_ratio'] = 0
                features['adv_ratio'] = 0
        except:
            features['noun_ratio'] = 0
            features['verb_ratio'] = 0
            features['adj_ratio'] = 0
            features['adv_ratio'] = 0

        return features

    def extract_semantic_features(self, text: str) -> Dict[str, float]:
        """
        Extract semantic features.

        Features:
        - Pronoun usage patterns
        - Negation frequency
        - Modal verb usage
        """
        features = {}

        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_alpha = [w for w in words if w.isalpha()]

        if not words_alpha:
            return self._get_zero_semantic_features()

        # Pronoun patterns
        pronouns_en = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your',
                       'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their'}
        pronouns_ru = {'я', 'мы', 'ты', 'вы', 'он', 'она', 'оно', 'они', 'меня', 'мне', 'нас'}
        pronouns = pronouns_en if self.language == 'en' else pronouns_ru

        pronoun_count = sum(1 for w in words_alpha if w in pronouns)
        features['pronoun_ratio'] = pronoun_count / len(words_alpha)

        # First person pronouns (more human-like)
        first_person_en = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
        first_person_ru = {'я', 'мы', 'меня', 'мне', 'мой', 'наш'}
        first_person = first_person_en if self.language == 'en' else first_person_ru

        first_person_count = sum(1 for w in words_alpha if w in first_person)
        features['first_person_ratio'] = first_person_count / len(words_alpha)

        # Negation frequency
        negations_en = {'not', 'no', 'never', 'nothing', 'nobody', 'none', "n't"}
        negations_ru = {'не', 'нет', 'ни', 'никогда', 'ничего', 'никто'}
        negations = negations_en if self.language == 'en' else negations_ru

        negation_count = sum(1 for w in words_alpha if w in negations)
        features['negation_ratio'] = negation_count / len(sentences)

        # Modal verbs
        modals_en = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
        modals_ru = {'можно', 'нужно', 'должен', 'следует', 'может'}
        modals = modals_en if self.language == 'en' else modals_ru

        modal_count = sum(1 for w in words_alpha if w in modals)
        features['modal_ratio'] = modal_count / len(words_alpha)

        # Contractions (informal language, more human)
        contractions = sum(1 for w in words if "'" in w or "'" in w)
        features['contraction_ratio'] = contractions / len(words_alpha)

        # Intensifiers
        intensifiers_en = {'very', 'really', 'so', 'extremely', 'incredibly', 'absolutely',
                          'totally', 'completely', 'quite', 'rather'}
        intensifiers_ru = {'очень', 'весьма', 'крайне', 'совершенно', 'абсолютно'}
        intensifiers = intensifiers_en if self.language == 'en' else intensifiers_ru

        intensifier_count = sum(1 for w in words_alpha if w in intensifiers)
        features['intensifier_ratio'] = intensifier_count / len(words_alpha)

        # Discourse markers (human conversation patterns)
        discourse_en = {'well', 'so', 'like', 'you know', 'i mean', 'actually', 'basically',
                       'honestly', 'literally', 'anyway'}
        discourse_ru = {'ну', 'вот', 'так', 'типа', 'короче', 'значит'}
        discourse = discourse_en if self.language == 'en' else discourse_ru

        discourse_count = sum(1 for w in words_alpha if w in discourse)
        features['discourse_marker_ratio'] = discourse_count / len(words_alpha)

        return features

    def _get_zero_stylometric_features(self) -> Dict[str, float]:
        """Return zero-valued stylometric features."""
        return {
            'avg_sentence_length': 0.0,
            'sentence_length_variance': 0.0,
            'sentence_length_std': 0.0,
            'max_sentence_length': 0.0,
            'min_sentence_length': 0.0,
            'type_token_ratio': 0.0,
            'hapax_ratio': 0.0,
            'dis_legomena_ratio': 0.0,
            'stopword_ratio': 0.0,
            'punctuation_density': 0.0,
            'comma_ratio': 0.0,
            'semicolon_ratio': 0.0,
            'exclamation_ratio': 0.0,
            'question_ratio': 0.0,
            'avg_word_length': 0.0,
            'word_length_variance': 0.0,
            'long_word_ratio': 0.0,
            'short_word_ratio': 0.0,
        }

    def _get_zero_syntactic_features(self) -> Dict[str, float]:
        """Return zero-valued syntactic features."""
        return {
            'avg_commas_per_sentence': 0.0,
            'comma_variance': 0.0,
            'parentheses_ratio': 0.0,
            'bracket_ratio': 0.0,
            'quote_ratio': 0.0,
            'noun_ratio': 0.0,
            'verb_ratio': 0.0,
            'adj_ratio': 0.0,
            'adv_ratio': 0.0,
        }

    def _get_zero_semantic_features(self) -> Dict[str, float]:
        """Return zero-valued semantic features."""
        return {
            'pronoun_ratio': 0.0,
            'first_person_ratio': 0.0,
            'negation_ratio': 0.0,
            'modal_ratio': 0.0,
            'contraction_ratio': 0.0,
            'intensifier_ratio': 0.0,
            'discourse_marker_ratio': 0.0,
        }
