"""
Evaluation Script with UAR, WAR, WA-F1 Metrics

Load checkpoint and evaluate on test samples.
Computes:
- UAR (Unweighted Average Recall)
- WAR (Weighted Average Recall)
- WA-F1 (Weighted Average F1)
- Confusion Matrix
- Per-class metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalFER
from training.metrics import EmotionMetrics
from data.test_dataset import create_test_dataloader


class Evaluator:
    """Evaluator for emotion recognition model."""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        num_classes: int = 8,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        
        # Metrics calculator
        self.metrics_calculator = EmotionMetrics(num_classes=num_classes)
    
    @torch.no_grad()
    def evaluate(self, return_predictions: bool = False):
        """
        Evaluate model on dataset.
        
        Args:
            return_predictions: Return predictions and labels
            
        Returns:
            metrics: Dict with all metrics
            predictions: (optional) List of predictions
            labels: (optional) List of labels
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_metadata = []
        
        print("\nEvaluating...")
        for audio, video, labels, metadata in tqdm(self.dataloader):
            audio = audio.to(self.device)
            video = video.to(self.device)
            
            # Forward
            outputs = self.model(audio, video)
            
            # Get predictions
            probabilities = outputs["probabilities"]
            predictions = probabilities.argmax(dim=1)
            
            # Accumulate
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_metadata.extend(metadata)
        
        # Convert to numpy
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute(predictions, labels)
        
        # Print metrics
        self.metrics_calculator.print_metrics(metrics)
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(self.metrics_calculator.get_classification_report(predictions, labels))
        
        if return_predictions:
            return metrics, predictions, labels, probabilities, all_metadata
        else:
            return metrics
    
    def evaluate_per_sample(self):
        """Evaluate and show per-sample results."""
        self.model.eval()
        
        print("\n" + "="*70)
        print("PER-SAMPLE EVALUATION")
        print("="*70)
        
        emotion_names = self.metrics_calculator.class_names
        
        for idx, (audio, video, label, metadata) in enumerate(self.dataloader):
            if audio.shape[0] > 1:
                # Skip batches, only process single samples
                continue
            
            audio = audio.to(self.device)
            video = video.to(self.device)
            
            # Forward
            with torch.no_grad():
                outputs = self.model(audio, video)
            
            # Get prediction
            probabilities = outputs["probabilities"][0]
            prediction = probabilities.argmax().item()
            confidence = probabilities[prediction].item()
            
            true_label = label.item()
            
            # Print result
            print(f"\nSample {idx + 1}:")
            print(f"  File: {metadata['filename'][0]}")
            print(f"  True: {emotion_names[true_label]}")
            print(f"  Pred: {emotion_names[prediction]} (confidence: {confidence:.2%})")
            print(f"  Correct: {'✓' if prediction == true_label else '✗'}")
            
            # Top-3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            print(f"  Top-3:")
            for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                print(f"    {i+1}. {emotion_names[idx]}: {prob:.2%}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: str = "cuda"):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load model on
        
    Returns:
        model: Model with loaded weights
        checkpoint: Full checkpoint dict
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"  Metrics:")
        print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"    UAR: {metrics.get('uar', 0):.4f}")
        print(f"    WAR: {metrics.get('war', 0):.4f}")
        print(f"    WA-F1: {metrics.get('wa_f1', 0):.4f}")
    
    return model, checkpoint


def save_results(
    metrics: dict,
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: list,
    save_dir: str,
):
    """Save evaluation results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_path = save_dir / "evaluation_metrics.json"
    
    # Convert numpy arrays to lists for JSON
    metrics_json = {
        "accuracy": float(metrics["accuracy"]),
        "uar": float(metrics["uar"]),
        "war": float(metrics["war"]),
        "wa_f1": float(metrics["wa_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_recall": metrics["per_class_recall"].tolist(),
        "per_class_f1": metrics["per_class_f1"].tolist(),
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save confusion matrix plot
    cm_path = save_dir / "confusion_matrix.png"
    metrics_calculator = EmotionMetrics(num_classes=8)
    metrics_calculator.plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=cm_path,
        normalize=True,
    )
    
    # Save predictions
    predictions_path = save_dir / "predictions.npz"
    np.savez(
        predictions_path,
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
    )
    
    print(f"Predictions saved to: {predictions_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Multimodal FER Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test_samples",
        help="Directory with test samples"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        help="Show per-sample results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MULTIMODAL FER - EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Create model
    print("\n[1/4] Creating model...")
    model = MultimodalFER(num_classes=8, num_segments=8)
    
    # Load checkpoint
    print("\n[2/4] Loading checkpoint...")
    model, checkpoint = load_checkpoint(args.checkpoint, model, args.device)
    
    # Create dataloader
    print("\n[3/4] Creating dataloader...")
    dataloader = create_test_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    print(f"  Test samples: {len(dataloader.dataset)}")
    
    # Create evaluator
    print("\n[4/4] Evaluating...")
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=args.device,
        num_classes=8,
    )
    
    # Evaluate
    metrics, predictions, labels, probabilities, metadata = evaluator.evaluate(
        return_predictions=True
    )
    
    # Save results
    save_results(
        metrics=metrics,
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        metadata=metadata,
        save_dir=args.save_dir,
    )
    
    # Per-sample evaluation
    if args.per_sample:
        evaluator.evaluate_per_sample()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
