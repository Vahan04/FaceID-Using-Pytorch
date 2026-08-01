import torch

def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """

    correct = (predictions == targets).sum().item()

    return correct / len(targets)
def verification_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """

    correct = (predictions == targets).sum().item()

    return correct / len(targets)
def precision(predictions: torch.Tensor, targets: torch.Tensor) -> float:

    tp = ((predictions == 1) & (targets == 1)).sum().item()

    fp = ((predictions == 1) & (targets == 0)).sum().item()

    return tp / (tp + fp + 1e-8)

def recall(predictions: torch.Tensor, targets: torch.Tensor) -> float:

    tp = ((predictions == 1) & (targets == 1)).sum().item()

    fn = ((predictions == 0) & (targets == 1)).sum().item()

    return tp / (tp + fn + 1e-8)
def f1_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:

    p = precision(predictions, targets)
    r = recall(predictions, targets)

    return 2 * p * r / (p + r + 1e-8)
def euclidean_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance between two embeddings.

    Args:
        embedding1 (torch.Tensor): First embedding.
        embedding2 (torch.Tensor): Second embedding.

    Returns:
        torch.Tensor: Euclidean distance between the two embeddings.
    """

    return torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=1))
def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity between two embeddings.

    Args:
        embedding1 (torch.Tensor): First embedding.
        embedding2 (torch.Tensor): Second embedding.

    Returns:
        torch.Tensor: Cosine similarity between the two embeddings.
    """

    dot_product = torch.sum(embedding1 * embedding2, dim=1)
    
    return dot_product   # Adding a small epsilon to avoid division by zero

import torch


class Verifier:
    """
    Verifies whether two face embeddings belong to the same person
    based on a distance threshold.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def verify(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Verify whether two embeddings belong to the same identity.

        Args:
            distance (torch.Tensor):
                Euclidean distance between two embeddings.

        Returns:
            torch.Tensor:
                Boolean tensor where True means the embeddings
                belong to the same person.
        """

        return distance <= self.threshold