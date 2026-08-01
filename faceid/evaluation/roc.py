import matplotlib.pyplot as plt
import numpy as np
import torch

class ROCEvaluator:
    """
    Computes ROC curve for face verification.
    """

    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.thresholds = None

    def compute(self, distances: torch.Tensor, labels: torch.Tensor, num_thresholds: int = 100):
        """
        Compute ROC curve.

        Args:
            distances (torch.Tensor):
                Euclidean distances between pairs of embeddings.
            labels (torch.Tensor):
                Ground truth labels (1 for same person, 0 for different).
            num_thresholds (int):
                Number of thresholds to evaluate.

        Returns:
            fpr (np.ndarray):
                False positive rates.
            tpr (np.ndarray):
                True positive rates.
            thresholds (np.ndarray):
                Thresholds used for evaluation.
        """

        thresholds = np.linspace(0, distances.max().item(), num_thresholds)
        tpr = []
        fpr = []

        for threshold in thresholds:
            predictions = distances <= threshold
            tp = ((predictions == 1) & (labels == 1)).sum().item()
            fp = ((predictions == 1) & (labels == 0)).sum().item()
            fn = ((predictions == 0) & (labels == 1)).sum().item()
            tn = ((predictions == 0) & (labels == 0)).sum().item()

            tpr.append(tp / (tp + fn + 1e-8))
            fpr.append(fp / (fp + tn + 1e-8))

        self.fpr = np.array(fpr)
        self.tpr = np.array(tpr)
        self.thresholds = thresholds

        return self.fpr, self.tpr, self.thresholds


    def plot(self):
        """
        Plot ROC curve.
        """

        if self.fpr is None or self.tpr is None:
            raise ValueError("ROC curve not computed. Call compute() first.")

        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.tpr, label='ROC Curve', color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def auc(self) -> float:
        """
        Compute Area Under the Curve (AUC) for the ROC curve.

        Returns:
            float: AUC value.
        """

        if self.fpr is None or self.tpr is None:
            raise ValueError("ROC curve not computed. Call compute() first.")

        return np.trapz(self.tpr, self.fpr)
