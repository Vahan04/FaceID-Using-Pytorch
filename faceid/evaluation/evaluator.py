import torch

from .metrics import (
    euclidean_distance,
    verification_accuracy,
)


class Evaluator:
    """
    Evaluates a face recognition model.
    """

    def __init__(
        self,
        model,
        dataloader,
        device,
        threshold: float = 1.0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

    def evaluate(self):
        """
        Evaluate the model.

        Returns:
            Dictionary containing evaluation metrics.
        """

        self.model.eval()

        with torch.no_grad():
            # We'll implement this together later.
            pass