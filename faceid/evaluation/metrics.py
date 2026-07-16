import torch


def euclidean_distance(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Euclidean (L2) distance between two batches of embeddings.

    Args:
        embedding1: Tensor of shape (batch_size, embedding_dim)
        embedding2: Tensor of shape (batch_size, embedding_dim)

    Returns:
        Tensor of shape (batch_size,) containing the distance for each pair.
    """

    return torch.norm(embedding1 - embedding2, p=2, dim=1)


def verification_accuracy(
    distances: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> float:
    """
    Compute face verification accuracy.

    Args:
        distances: Euclidean distances between embedding pairs.
        labels:
            1 -> same identity
            0 -> different identity
        threshold: Distance threshold.

    Returns:
        Verification accuracy.
    """

    predictions = (distances < threshold).long()

    correct = (predictions == labels).sum().item()

    return correct / len(labels)