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
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = FaceDataset(
        root_dir="dataset",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None

    model = FaceEmbeddingNet().to(device)

    criterion = TripletLoss(margin=0.2)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )

    trainer = Trainer(
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
    )

    trainer.fit(num_epochs=20)
if __name__ == "__main__":
    main()