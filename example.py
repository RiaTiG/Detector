"""
Example usage of the AI text detector.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset_loader import DatasetLoader
from classifiers import XGBoostClassifier
from visualizer import Visualizer


def main():
    print("AI Text Detector - Example Usage")
    print("=" * 60)

    # Step 1: Create sample dataset
    print("\n1. Creating sample dataset...")
    loader = DatasetLoader(language='en')
    loader.create_sample_dataset('data/sample', n_samples_per_class=5)

    # Step 2: Load and prepare data
    print("\n2. Loading and preparing data...")
    df = loader.load_from_text_files('data/sample/human', 'data/sample/ai')
    print(f"   Loaded {len(df)} samples")

    data_dict = loader.prepare_training_data(df, test_size=0.2, random_state=42)
    print(f"   Training samples: {len(data_dict['X_train'])}")
    print(f"   Test samples: {len(data_dict['X_test'])}")
    print(f"   Features: {data_dict['n_features']}")

    # Step 3: Train XGBoost classifier
    print("\n3. Training XGBoost classifier...")
    classifier = XGBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    classifier.train(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_test'],
        data_dict['y_test']
    )

    # Step 4: Evaluate
    print("\n4. Evaluating model...")
    metrics = classifier.evaluate(data_dict['X_test'], data_dict['y_test'])

    print("\n   Performance Metrics:")
    print("   " + "-" * 40)
    for metric, value in metrics.items():
        print(f"   {metric.capitalize():15s}: {value:.4f}")

    # Step 5: Feature importance
    print("\n5. Analyzing feature importance...")
    feature_importance = classifier.get_feature_importance(data_dict['feature_names'])
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n   Top 10 Most Important Features:")
    print("   " + "-" * 40)
    for idx, (feature, score) in enumerate(sorted_features, 1):
        print(f"   {idx:2d}. {feature:30s}: {score:.4f}")

    # Step 6: Test on new text
    print("\n6. Testing on new samples...")

    test_texts = [
        ("I honestly can't believe it's already Friday! This week flew by so fast. "
         "Got a lot done though, which feels good. Anyone else feel like time moves "
         "differently depending on how busy you are? Weird how that works."),

        ("The implementation of advanced machine learning algorithms in contemporary "
         "business environments represents a fundamental transformation in operational "
         "paradigms. Organizations leveraging these technologies demonstrate enhanced "
         "efficiency metrics and optimized decision-making frameworks across diverse "
         "functional domains.")
    ]

    try:
        from feature_extractor_lite import FeatureExtractor
    except ImportError:
        from feature_extractor import FeatureExtractor
    import numpy as np

    extractor = FeatureExtractor(language='en')

    for idx, text in enumerate(test_texts, 1):
        print(f"\n   Test {idx}:")
        print(f"   Text: {text[:80]}...")

        # Extract features
        features = extractor.extract_all_features(text)
        feature_vector = np.array([[features[name] for name in data_dict['feature_names']]])

        # Predict
        prediction = classifier.predict(feature_vector)[0]
        proba = classifier.predict_proba(feature_vector)[0]

        print(f"   Prediction: {'AI-generated' if prediction else 'Human-written'}")
        print(f"   Confidence: {max(proba):.2%}")
        print(f"   Probabilities: Human={proba[0]:.2%}, AI={proba[1]:.2%}")

    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    viz = Visualizer(output_dir='results')

    # Feature importance plot
    viz.plot_feature_importance(
        feature_importance,
        top_n=15,
        filename='example_feature_importance.png'
    )

    # Confusion matrix
    y_pred = classifier.predict(data_dict['X_test'])
    viz.plot_confusion_matrix(
        data_dict['y_test'],
        y_pred,
        filename='example_confusion_matrix.png'
    )

    # Feature distributions
    viz.plot_feature_distributions(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['feature_names'],
        filename='example_feature_distributions.png'
    )

    print("\n" + "=" * 60)
    print("Example complete! Check the 'results' directory for visualizations.")
    print("=" * 60)


if __name__ == '__main__':
    main()
