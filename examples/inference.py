import torch
from torchvision import transforms

from faceid.models.embedding_net import FaceEmbeddingNet
from faceid.inference import FaceRecognizer

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

recognizer = FaceRecognizer(
    model=FaceEmbeddingNet(),
    checkpoint="checkpoints/best_model.pth",
    device=device,
    transform=transform,
)

distance = recognizer.compare(
    "image1.jpg",
    "image2.jpg",
)

print(distance)

print(
    recognizer.verify(
        "image1.jpg",
        "image2.jpg",
    )
)