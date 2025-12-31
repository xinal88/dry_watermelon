"""
Evaluation Metrics for Emotion Recognition

Implements:
- UAR (Unweighted Average Recall) - Average recall across all classes
- WAR (Weighted Average Recall) - Weighted recall by class frequency
- WA-F1 (Weighted Average F1) - Weighted F1 score
- Confusion Matrix
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class EmotionMetrics:
    """
    Compute emotion recognition metrics.
    
    Metrics:
    - UAR: Unweighted Average Recall (macro recall)
    - WAR: Weighted Average Recall (weighted recall)
    - WA-F1: Weighted Average F1 Score
    - Accuracy
    - Per-class metrics
    """
    
    def __init__(self, num_classes: int = 8, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        
        if class_names is None:
            self.class_names = [
                "neutral", "calm", "happy", "sad",
                "angry", "fearful", "disgust", "surprised"
            ]
        else:
            self.class_names = class_names
    
    def compute(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            predictions: [N] - Predicted class indices
            labels: [N] - Ground truth class indices
            
        Returns:
            Dict with all metrics
        """
        # Accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # UAR (Unweighted Average Recall) = Macro Recall
        uar = recall_score(labels, predictions, average='macro', zero_division=0)
        
        # WAR (Weighted Average Recall)
        war = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        # WA-F1 (Weighted Average F1)
        wa_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Macro F1
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Per-class recall
        per_class_recall = recall_score(
            labels, predictions,
            average=None,
            zero_division=0,
            labels=range(self.num_classes)
        )
        
        # Per-class F1
        per_class_f1 = f1_score(
            labels, predictions,
            average=None,
            zero_division=0,
            labels=range(self.num_classes)
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=range(self.num_classes))
        
        metrics = {
            "accuracy": accuracy,
            "uar": uar,  # Unweighted Average Recall
            "war": war,  # Weighted Average Recall
            "wa_f1": wa_f1,  # Weighted Average F1
            "macro_f1": macro_f1,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way."""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  UAR:          {metrics['uar']:.4f} ({metrics['uar']*100:.2f}%)")
        print(f"  WAR:          {metrics['war']:.4f} ({metrics['war']*100:.2f}%)")
        print(f"  WA-F1:        {metrics['wa_f1']:.4f} ({metrics['wa_f1']*100:.2f}%)")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-"*32)
        
        for i, name in enumerate(self.class_names):
            recall = metrics['per_class_recall'][i]
            f1 = metrics['per_class_f1'][i]
            print(f"{name:<12} {recall:.4f}     {f1:.4f}")
        
        print("="*60)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save figure
            normalize: Normalize by row (true labels)
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def get_classification_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> str:
        """Get detailed classification report."""
        return classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )


def compute_metrics_from_outputs(
    all_predictions: List[int],
    all_labels: List[int],
    num_classes: int = 8,
) -> Dict[str, float]:
    """
    Convenience function to compute metrics from lists.
    
    Args:
        all_predictions: List of predicted class indices
        all_labels: List of ground truth class indices
        num_classes: Number of classes
        
    Returns:
        Dict with metrics
    """
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    metrics_calculator = EmotionMetrics(num_classes=num_classes)
    metrics = metrics_calculator.compute(predictions, labels)
    
    return metrics
