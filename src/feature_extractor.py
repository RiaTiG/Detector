"""
Feature extraction module for AI text detection.
Extracts stylometric, syntactic, and semantic features that are hard for AI to imitate.
"""

import re
import numpy as np
import spacy
from collections import Counter
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class FeatureExtractor:
    """Extract features from text that distinguish human from AI writing."""

    def __init__(self, language='en'):
        """
        Initialize feature extractor.

        Args:
            language: Language code ('en' or 'ru')
        """
        self.language = language

        # Load spaCy model
        if language == 'en':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("Please install: python -m spacy download en_core_web_sm")
                self.nlp = None
        elif language == 'ru':
            try:
                self.nlp = spacy.load('ru_core_news_sm')
            except:
                print("Please install: python -m spacy download ru_core_news_sm")
                self.nlp = None

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

        # Syntactic features
        features.update(self.extract_syntactic_features(text))

        # Semantic features
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

        # Lexical diversity
        features['type_token_ratio'] = len(set(words_alpha)) / len(words_alpha)

        # Hapax legomena (words appearing once)
        word_counts = Counter(words_alpha)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        features['hapax_ratio'] = hapax_count / len(word_counts)

        # Function words (stopwords) frequency
        stopword_count = sum(1 for w in words_alpha if w in self.stopwords)
        features['stopword_ratio'] = stopword_count / len(words_alpha)

        # Punctuation density
        punctuation_count = sum(1 for c in text if c in '.,;:!?-—()[]{}"\'"')
        features['punctuation_density'] = punctuation_count / len(words)

        # Word length statistics
        word_lengths = [len(w) for w in words_alpha]
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_variance'] = np.var(word_lengths)

        # Long words ratio (>6 characters)
        long_words = sum(1 for w in words_alpha if len(w) > 6)
        features['long_word_ratio'] = long_words / len(words_alpha)

        return features

    def extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """
        Extract syntactic features using dependency parsing.

        Features:
        - Average parse tree depth
        - Parse tree depth variance
        - Comma patterns (frequency, placement)
        - Dependency arc lengths
        - Clause complexity
        """
        features = {}

        if self.nlp is None:
            return self._get_zero_syntactic_features()

        doc = self.nlp(text)

        if not list(doc.sents):
            return self._get_zero_syntactic_features()

        # Parse tree depth
        depths = []
        for sent in doc.sents:
            depths.append(self._get_tree_depth(sent.root))

        features['avg_tree_depth'] = np.mean(depths)
        features['tree_depth_variance'] = np.var(depths)

        # Comma patterns
        commas = [token for token in doc if token.text == ',']
        features['comma_frequency'] = len(commas) / len(list(doc.sents))

        # Comma position variance (position in sentence)
        if commas:
            comma_positions = []
            for sent in doc.sents:
                sent_commas = [token.i - sent.start for token in sent if token.text == ',']
                if sent_commas:
                    sent_len = len(sent)
                    comma_positions.extend([pos / sent_len for pos in sent_commas])
            features['comma_position_variance'] = np.var(comma_positions) if comma_positions else 0
        else:
            features['comma_position_variance'] = 0

        # Dependency arc lengths
        arc_lengths = []
        for token in doc:
            if token.head != token:
                arc_lengths.append(abs(token.i - token.head.i))

        features['avg_dependency_distance'] = np.mean(arc_lengths) if arc_lengths else 0
        features['dependency_distance_variance'] = np.var(arc_lengths) if arc_lengths else 0

        # Clause complexity (subordinate clauses)
        subordinate_clauses = sum(1 for token in doc if token.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp'])
        features['subordinate_clause_ratio'] = subordinate_clauses / len(list(doc.sents))

        # POS tag diversity
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        features['pos_diversity'] = len(set(pos_tags)) / len(pos_tags) if pos_tags else 0

        return features

    def extract_semantic_features(self, text: str) -> Dict[str, float]:
        """
        Extract semantic features.

        Features:
        - Abstractness level (ratio of abstract to concrete words)
        - Named entity density
        - Pronoun usage patterns
        - Negation frequency
        - Modal verb usage
        """
        features = {}

        if self.nlp is None:
            return self._get_zero_semantic_features()

        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_space and not token.is_punct]

        if not tokens:
            return self._get_zero_semantic_features()

        # Named entity density
        entities = list(doc.ents)
        features['entity_density'] = len(entities) / len(tokens)

        # Pronoun patterns
        pronouns = [token for token in tokens if token.pos_ == 'PRON']
        features['pronoun_ratio'] = len(pronouns) / len(tokens)

        # First person pronouns (more human-like)
        first_person = sum(1 for token in pronouns if token.text.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'я', 'мы', 'меня', 'мне'])
        features['first_person_ratio'] = first_person / len(tokens)

        # Negation frequency
        negations = sum(1 for token in doc if token.dep_ == 'neg')
        features['negation_ratio'] = negations / len(list(doc.sents))

        # Modal verbs
        modals = sum(1 for token in tokens if token.tag_ in ['MD'] or token.text.lower() in ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'])
        features['modal_ratio'] = modals / len(tokens)

        # Adjective to noun ratio (abstraction indicator)
        adjectives = sum(1 for token in tokens if token.pos_ == 'ADJ')
        nouns = sum(1 for token in tokens if token.pos_ == 'NOUN')
        features['adj_noun_ratio'] = adjectives / nouns if nouns > 0 else 0

        # Adverb usage
        adverbs = sum(1 for token in tokens if token.pos_ == 'ADV')
        features['adverb_ratio'] = adverbs / len(tokens)

        return features

    def _get_tree_depth(self, token) -> int:
        """Recursively compute tree depth."""
        if not list(token.children):
            return 1
        return 1 + max(self._get_tree_depth(child) for child in token.children)

    def _get_zero_stylometric_features(self) -> Dict[str, float]:
        """Return zero-valued stylometric features."""
        return {
            'avg_sentence_length': 0.0,
            'sentence_length_variance': 0.0,
            'sentence_length_std': 0.0,
            'type_token_ratio': 0.0,
            'hapax_ratio': 0.0,
            'stopword_ratio': 0.0,
            'punctuation_density': 0.0,
            'avg_word_length': 0.0,
            'word_length_variance': 0.0,
            'long_word_ratio': 0.0,
        }

    def _get_zero_syntactic_features(self) -> Dict[str, float]:
        """Return zero-valued syntactic features."""
        return {
            'avg_tree_depth': 0.0,
            'tree_depth_variance': 0.0,
            'comma_frequency': 0.0,
            'comma_position_variance': 0.0,
            'avg_dependency_distance': 0.0,
            'dependency_distance_variance': 0.0,
            'subordinate_clause_ratio': 0.0,
            'pos_diversity': 0.0,
        }

    def _get_zero_semantic_features(self) -> Dict[str, float]:
        """Return zero-valued semantic features."""
        return {
            'entity_density': 0.0,
            'pronoun_ratio': 0.0,
            'first_person_ratio': 0.0,
            'negation_ratio': 0.0,
            'modal_ratio': 0.0,
            'adj_noun_ratio': 0.0,
            'adverb_ratio': 0.0,
        }
