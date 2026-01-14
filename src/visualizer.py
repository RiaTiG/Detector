"""
Visualization and evaluation tools.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from typing import Dict, List


class Visualizer:
    """Visualization tools for model evaluation."""

    def __init__(self, output_dir='results'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        sns.set_style('whitegrid')

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n=20,
        title='Feature Importance',
        filename='feature_importance.png'
    ):
        """
        Plot feature importance.

        Args:
            importance_dict: Dictionary of feature names and importance scores
            top_n: Number of top features to show
            title: Plot title
            filename: Output filename
        """
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature importance plot saved to {self.output_dir / filename}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels=['Human', 'AI'],
        filename='confusion_matrix.png'
    ):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            filename: Output filename
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {self.output_dir / filename}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        filename='roc_curve.png'
    ):
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            filename: Output filename
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC curve saved to {self.output_dir / filename}")

    def plot_feature_distributions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        filename='feature_distributions.png'
    ):
        """
        Plot feature distributions for human vs AI texts.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            filename: Output filename
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['class'] = df['label'].map({0: 'Human', 1: 'AI'})

        # Select top features by variance
        variances = df[feature_names].var()
        top_features = variances.nlargest(9).index.tolist()

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            for label, class_name in [(0, 'Human'), (1, 'AI')]:
                data = df[df['label'] == label][feature]
                ax.hist(data, alpha=0.6, label=class_name, bins=20)

            ax.set_xlabel(feature, fontsize=9)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Feature Distributions: Human vs AI', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature distributions saved to {self.output_dir / filename}")

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        filename='model_comparison.png'
    ):
        """
        Compare multiple models.

        Args:
            results: Dictionary of model names and their metrics
            filename: Output filename
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        models = list(results.keys())

        data = {metric: [results[model].get(metric, 0) for model in models] for metric in metrics}

        x = np.arange(len(models))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, metric in enumerate(metrics):
            offset = width * (idx - 2)
            ax.bar(x + offset, data[metric], width, label=metric.upper())

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Model comparison saved to {self.output_dir / filename}")

    def create_report(
        self,
        model_name: str,
        metrics: Dict[str, float],
        feature_importance: Dict[str, float] = None,
        filename='report.txt'
    ):
        """
        Create text report.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            feature_importance: Optional feature importance scores
            filename: Output filename
        """
        report_path = self.output_dir / filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"AI Text Detection - Model Report\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Model: {model_name}\n\n")

            f.write(f"Performance Metrics:\n")
            f.write(f"{'-' * 50}\n")
            for metric, value in metrics.items():
                f.write(f"{metric.capitalize():15s}: {value:.4f}\n")

            if feature_importance:
                f.write(f"\n\nTop 10 Most Important Features:\n")
                f.write(f"{'-' * 50}\n")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for idx, (feature, score) in enumerate(sorted_features, 1):
                    f.write(f"{idx:2d}. {feature:30s}: {score:.4f}\n")

        print(f"Report saved to {report_path}")
