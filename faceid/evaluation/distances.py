import torch

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