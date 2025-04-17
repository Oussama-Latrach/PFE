"""
Evaluation metrics for classification tasks.
"""

from sklearn.metrics import accuracy_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        accuracy: Overall accuracy
        cm: Confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, cm