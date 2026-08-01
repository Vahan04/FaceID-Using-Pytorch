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