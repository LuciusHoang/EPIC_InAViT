import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Computes overall accuracy, per-class F1-scores, and confusion matrix.

    Args:
        y_true (list or np.array): Ground-truth labels
        y_pred (list or np.array): Predicted labels
        class_names (list): Optional, class names for display.

    Returns:
        dict: metrics dictionary
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }

    if class_names:
        print("\nPer-class F1-scores:")
        for i, score in enumerate(f1_per_class):
            print(f"  {class_names[i]}: {score:.3f}")

    return metrics


def plot_confusion_matrix(cm, class_names, normalize=False, figsize=(10, 8), cmap='Blues', save_path=None):
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        cm (np.array): Confusion matrix.
        class_names (list): List of class names.
        normalize (bool): If True, normalize the confusion matrix.
        figsize (tuple): Figure size.
        cmap (str): Color map for the heatmap.
        save_path (str): If provided, saves the plot to this path.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 2)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
