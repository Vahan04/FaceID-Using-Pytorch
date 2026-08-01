from pathlib import Path

import torch
from torch import nn
from PIL import Image

from .preprocessing import preprocess_image
from faceid.evaluation.distances import euclidean_distance


class FaceRecognizer:
    """
    Face recognition using a trained embedding model.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint: str | Path,
        device: torch.device,
        transform=None,
    ):
        self.device = device
        self.model = model.to(device)
        self.transform = transform

        checkpoint = torch.load(
            checkpoint,
            map_location=device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()
    def get_embedding(self, image_path: str | Path) -> torch.Tensor:
        """
        Compute embedding for one image.
        """

        image = preprocess_image(
            image_path,
            self.transform,
        )

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding
    
    def compare(self, image1: str | Path, image2: str | Path) -> float:
        """
        Compare two faces.
        """

        embedding1 = self.get_embedding(image1)
        embedding2 = self.get_embedding(image2)

        distance = euclidean_distance(
            embedding1,
            embedding2,
        )

        return distance.item()
    
    def verify(self, image1: str | Path, image2: str | Path, threshold: float = 0.8) -> bool:
        """
        Verify if two images belong to the same person.
        """

        distance = self.compare(
            image1,
            image2,
        )

        return distance <= threshold