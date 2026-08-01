import torch

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from faceid.datasets.face_dataset import FaceDataset
from faceid.models.embedding_net import FaceEmbeddingNet
from faceid.losses.triplet_loss import TripletLoss
from faceid.trainers.trainer import Trainer
from faceid.datasets import (FaceDataset, split_dataset_by_identity)

DATASET_DIR = "data/lfw_funneled"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 32
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = FaceDataset(root_dir=DATASET_DIR, transform=transform)

    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = split_dataset_by_identity(dataset, train_ratio=0.8, seed=42)

    print("=" * 50)
    print("Dataset Split")
    print("=" * 50)
    print(f"Train identities : {len(train_dataset.person_to_images)}")
    print(f"Validation IDs   : {len(val_dataset.person_to_images)}")
    print(f"Train images     : {len(train_dataset)}")
    print(f"Validation imgs  : {len(val_dataset)}")
    print("=" * 50)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

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
        checkpoint_dir=CHECKPOINT_DIR,
    )

    trainer.fit(num_epochs=20)


if __name__ == "__main__":
    main()