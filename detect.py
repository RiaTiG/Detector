"""
Detection script for analyzing texts.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detector import AITextDetector


def detect_text(detector, text, show_features=False):
    """
    Detect and print results for a single text.

    Args:
        detector: AITextDetector instance
        text: Text to analyze
        show_features: Whether to show extracted features
    """
    result = detector.explain_prediction(text, top_n=5)

    print("\n" + "=" * 60)
    print("Detection Result")
    print("=" * 60)
    print(f"\nText: {text[:200]}{'...' if len(text) > 200 else ''}\n")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Human-written: {result['human_probability']:.2%}")
    print(f"  AI-generated:  {result['ai_probability']:.2%}")

    if 'top_contributing_features' in result:
        print(f"\nTop Contributing Features:")
        print("-" * 60)
        for feat_info in result['top_contributing_features']:
            print(f"  {feat_info['feature']:30s}: {feat_info['value']:.4f} (importance: {feat_info['importance']:.4f})")

    if show_features:
        print(f"\nAll Extracted Features:")
        print("-" * 60)
        for feature, value in sorted(result['features'].items()):
            print(f"  {feature:30s}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Detect AI-generated text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--model', type=str, default='models/xgboost_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--model-type', type=str, default='xgboost', choices=['xgboost', 'neural'],
                        help='Type of model')
    parser.add_argument('--show-features', action='store_true',
                        help='Show all extracted features')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'ru'],
                        help='Text language')

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first using train.py")
        return

    # Load detector
    print(f"Loading detector with {args.model_type} model...")
    detector = AITextDetector(
        model_type=args.model_type,
        model_path=args.model,
        language=args.language
    )

    # Get text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Interactive mode
        print("\nInteractive mode - enter text to analyze (Ctrl+D or Ctrl+Z to finish):")
        print("-" * 60)
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        text = '\n'.join(lines)

    if not text.strip():
        print("Error: No text provided")
        return

    # Detect
    detect_text(detector, text, args.show_features)


if __name__ == '__main__':
    main()
