import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from faceid.datasets.face_dataset import FaceDataset
from faceid.models.embedding_net import FaceEmbeddingNet
from faceid.losses.triplet_loss import TripletLoss
from faceid.trainers.trainer import Trainer


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])

    dataset = FaceDataset(
        root_dir="data/lfw_funneled",
        transform=transform,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )
    model = FaceEmbeddingNet().to(device)
    loss_fn = TripletLoss(margin=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.parameters()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, train_loader=train_loader, device=device)
    trainer.fit(num_epochs=1)


if __name__ == "__main__":
    main()