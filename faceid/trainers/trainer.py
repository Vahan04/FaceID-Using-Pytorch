from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


class Trainer:
    """
    Handles model training, validation and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | Path | None = None,
    ):

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.scheduler = scheduler

        self.best_val_loss = float("inf")

        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def train_one_epoch(self) -> float:
        """
        Train for one epoch.
        """

        self.model.train()

        running_loss = 0.0

        for anchor, positive, negative in self.train_loader:

            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            self.optimizer.zero_grad()

            anchor_embedding = self.model(anchor)
            positive_embedding = self.model(positive)
            negative_embedding = self.model(negative)

            loss = self.loss_fn(
                anchor_embedding,
                positive_embedding,
                negative_embedding,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Evaluate one validation epoch.
        """

        if self.val_loader is None:
            return 0.0

        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for anchor, positive, negative in self.val_loader:

                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                anchor_embedding = self.model(anchor)
                positive_embedding = self.model(positive)
                negative_embedding = self.model(negative)

                loss = self.loss_fn(
                    anchor_embedding,
                    positive_embedding,
                    negative_embedding,
                )

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def fit(self, num_epochs: int):

        for epoch in range(num_epochs):

            train_loss = self.train_one_epoch()

            val_loss = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if (
                self.checkpoint_dir is not None
                and val_loss < self.best_val_loss
            ):

                self.best_val_loss = val_loss

                self.save_checkpoint(
                    epoch,
                    train_loss,
                    val_loss,
                )

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:

        checkpoint_path = (
            self.checkpoint_dir /
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
            },
            checkpoint_path,
        )

        print(f"Best model saved -> {checkpoint_path}")