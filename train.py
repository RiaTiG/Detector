"""
Training script for AI text detection models.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset_loader import DatasetLoader
from classifiers import XGBoostClassifier, NeuralNetTrainer
from visualizer import Visualizer


def train_xgboost(data_dict, output_dir='models', visualize=True):
    """
    Train XGBoost classifier.

    Args:
        data_dict: Dictionary with training data
        output_dir: Directory to save model
        visualize: Whether to create visualizations
    """
    print("\n" + "=" * 60)
    print("Training XGBoost Classifier")
    print("=" * 60)

    # Initialize classifier
    classifier = XGBoostClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    # Train
    print("\nTraining model...")
    classifier.train(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_test'],
        data_dict['y_test']
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = classifier.evaluate(data_dict['X_test'], data_dict['y_test'])

    print("\nTest Set Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.capitalize():15s}: {value:.4f}")

    # Feature importance
    feature_importance = classifier.get_feature_importance(data_dict['feature_names'])

    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for idx, (feature, score) in enumerate(sorted_features, 1):
        print(f"{idx:2d}. {feature:30s}: {score:.4f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    model_path = output_path / 'xgboost_model.pkl'
    classifier.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Visualizations
    if visualize:
        print("\nCreating visualizations...")
        viz = Visualizer(output_dir='results')

        # Feature importance
        viz.plot_feature_importance(
            feature_importance,
            top_n=20,
            title='XGBoost Feature Importance',
            filename='xgboost_feature_importance.png'
        )

        # Confusion matrix
        y_pred = classifier.predict(data_dict['X_test'])
        viz.plot_confusion_matrix(
            data_dict['y_test'],
            y_pred,
            filename='xgboost_confusion_matrix.png'
        )

        # ROC curve
        y_proba = classifier.predict_proba(data_dict['X_test'])[:, 1]
        viz.plot_roc_curve(
            data_dict['y_test'],
            y_proba,
            filename='xgboost_roc_curve.png'
        )

        # Report
        viz.create_report(
            'XGBoost',
            metrics,
            feature_importance,
            filename='xgboost_report.txt'
        )

    return classifier, metrics


def train_neural_network(data_dict, output_dir='models', visualize=True):
    """
    Train neural network classifier.

    Args:
        data_dict: Dictionary with training data
        output_dir: Directory to save model
        visualize: Whether to create visualizations
    """
    print("\n" + "=" * 60)
    print("Training Neural Network Classifier")
    print("=" * 60)

    # Initialize trainer
    trainer = NeuralNetTrainer(
        input_size=data_dict['n_features'],
        hidden_sizes=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        device='cpu'
    )

    # Train
    print("\nTraining model...")
    trainer.train(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_test'],
        data_dict['y_test'],
        epochs=100,
        batch_size=32,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(data_dict['X_test'], data_dict['y_test'])

    print("\nTest Set Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.capitalize():15s}: {value:.4f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    model_path = output_path / 'neural_net_model.pth'
    trainer.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Visualizations
    if visualize:
        print("\nCreating visualizations...")
        viz = Visualizer(output_dir='results')

        # Confusion matrix
        y_pred = trainer.predict(data_dict['X_test'])
        viz.plot_confusion_matrix(
            data_dict['y_test'],
            y_pred,
            filename='neural_net_confusion_matrix.png'
        )

        # ROC curve
        y_proba = trainer.predict_proba(data_dict['X_test'])[:, 1]
        viz.plot_roc_curve(
            data_dict['y_test'],
            y_proba,
            filename='neural_net_roc_curve.png'
        )

        # Report
        viz.create_report(
            'Neural Network',
            metrics,
            filename='neural_net_report.txt'
        )

    return trainer, metrics


def main():
    parser = argparse.ArgumentParser(description='Train AI text detection models')
    parser.add_argument('--data-dir', type=str, help='Directory containing human/ and ai/ subdirectories')
    parser.add_argument('--csv-path', type=str, help='Path to CSV file with text and label columns')
    parser.add_argument('--model', type=str, default='both', choices=['xgboost', 'neural', 'both'],
                        help='Model type to train')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualizations')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')

    args = parser.parse_args()

    # Initialize loader
    loader = DatasetLoader(language='en')

    # Create sample dataset if requested
    if args.create_sample:
        print("Creating sample dataset...")
        loader.create_sample_dataset('data/sample', n_samples_per_class=5)
        args.data_dir = 'data/sample'

    # Load dataset
    if args.csv_path:
        print(f"Loading dataset from CSV: {args.csv_path}")
        df = loader.load_from_csv(args.csv_path)
    elif args.data_dir:
        print(f"Loading dataset from directory: {args.data_dir}")
        human_dir = Path(args.data_dir) / 'human'
        ai_dir = Path(args.data_dir) / 'ai'
        df = loader.load_from_text_files(str(human_dir), str(ai_dir))
    else:
        print("Error: Either --data-dir or --csv-path must be specified")
        print("Use --create-sample to generate sample data")
        return

    print(f"\nDataset loaded: {len(df)} samples")
    print(f"  Human texts: {sum(df['label'] == 0)}")
    print(f"  AI texts: {sum(df['label'] == 1)}")

    # Prepare data
    print("\nPreparing training data...")
    data_dict = loader.prepare_training_data(df, test_size=0.2, random_state=42)

    print(f"\nTraining set: {len(data_dict['X_train'])} samples")
    print(f"Test set: {len(data_dict['X_test'])} samples")
    print(f"Number of features: {data_dict['n_features']}")

    # Train models
    results = {}

    if args.model in ['xgboost', 'both']:
        classifier, metrics = train_xgboost(data_dict, args.output_dir, not args.no_viz)
        results['XGBoost'] = metrics

    if args.model in ['neural', 'both']:
        trainer, metrics = train_neural_network(data_dict, args.output_dir, not args.no_viz)
        results['Neural Network'] = metrics

    # Compare models if both were trained
    if args.model == 'both' and not args.no_viz:
        print("\nCreating model comparison...")
        viz = Visualizer(output_dir='results')
        viz.plot_model_comparison(results, filename='model_comparison.png')

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
