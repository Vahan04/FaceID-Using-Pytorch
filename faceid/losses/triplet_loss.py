import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Args:
            anchor: Anchor embeddings of shape (batch_size, embedding_dim)
            positive: Positive embeddings of shape (batch_size, embedding_dim)
            negative: Negative embeddings of shape (batch_size, embedding_dim)

        Returns:
            loss: Scalar tensor representing the triplet loss
        """
        # Compute pairwise distances
        distance_ap = torch.norm(anchor - positive, p=2, dim=1)
        distance_an = torch.norm(anchor - negative, p=2, dim=1)

        loss = torch.clamp(distance_ap - distance_an + self.margin, min=0.0)

        return loss.mean()
