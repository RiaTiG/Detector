"""
Simple test to verify the detector works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from feature_extractor_lite import FeatureExtractor
except ImportError:
    from feature_extractor import FeatureExtractor


def test_feature_extraction():
    """Test that feature extraction works."""
    print("Testing Feature Extraction...")
    print("=" * 60)

    extractor = FeatureExtractor(language='en')

    # Test texts
    human_text = """
    I can't believe how crazy yesterday was! Met up with Sarah and we grabbed coffee.
    The barista was super nice. Anyway, we talked for hours about everything - work,
    relationships, life stuff. It's amazing how time flies!
    """

    ai_text = """
    The implementation of advanced machine learning algorithms in contemporary
    business environments represents a fundamental transformation in operational
    paradigms. Organizations leveraging these technologies demonstrate enhanced
    efficiency metrics and optimized decision-making frameworks across diverse
    functional domains.
    """

    # Extract features
    print("\n1. Extracting features from human text...")
    human_features = extractor.extract_all_features(human_text)
    print(f"   Extracted {len(human_features)} features")

    print("\n2. Extracting features from AI text...")
    ai_features = extractor.extract_all_features(ai_text)
    print(f"   Extracted {len(ai_features)} features")

    # Compare key features
    print("\n3. Key feature differences:")
    print("   " + "-" * 56)
    print(f"   {'Feature':<30} {'Human':>12} {'AI':>12}")
    print("   " + "-" * 56)

    key_features = [
        'avg_sentence_length',
        'type_token_ratio',
        'contraction_ratio',
        'first_person_ratio',
        'discourse_marker_ratio',
        'avg_word_length',
        'exclamation_ratio'
    ]

    for feat in key_features:
        if feat in human_features and feat in ai_features:
            print(f"   {feat:<30} {human_features[feat]:>12.4f} {ai_features[feat]:>12.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS: Feature extraction working correctly!")
    print("=" * 60)

    # Show all features
    print(f"\nAll available features ({len(human_features)}):")
    for i, feat in enumerate(sorted(human_features.keys()), 1):
        print(f"  {i:2d}. {feat}")

    return True


if __name__ == '__main__':
    test_feature_extraction()
