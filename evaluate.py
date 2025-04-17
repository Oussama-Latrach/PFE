"""
Evaluation script for DGCNN model on test data.
Computes metrics and generates classification reports.
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import get_loaders
from models.dgcnn import DGCNN
import os


def plot_confusion_matrix(cm, class_names):
    """Visualize confusion matrix with Seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('experiments/confusion_matrix.png')
    plt.close()


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data and generate reports."""
    model.eval()
    all_preds = []
    all_targets = []

    print("\n Starting evaluation on test set...")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Generate classification report
    class_names = [f"Class {i + 1}" for i in range(6)]
    print("\n Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=class_names, digits=4))

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names)

    # Calculate overall accuracy
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    print(f"\n Test Accuracy: {accuracy:.2%}")


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")

    # Create directories for outputs
    os.makedirs('experiments', exist_ok=True)

    # Load data
    _, _, test_loader = get_loaders(batch_size=32)

    # Initialize model
    model = DGCNN(num_classes=6).to(device)

    # Load best model weights
    model_path = 'experiments/best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f" Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Evaluate
    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()