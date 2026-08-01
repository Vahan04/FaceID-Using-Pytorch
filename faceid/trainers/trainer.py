import torch
import time

from pathlib import Path
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        train_history: list[float] | None = None,
        val_history: list[float] | None = None,
       
    ):

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        print("INIT train_loader:", self.train_loader)
        print("INIT val_loader:", self.val_loader)

        self.device = device
        self.scheduler = scheduler

        self.best_val_loss = float("inf")
        self.train_history = train_history or []
        self.val_history = val_history or []
        self.writer = SummaryWriter(log_dir="runs/faceid")

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
        
        progress_bar = tqdm(self.train_loader,desc="Training")

        for anchor, positive, negative, label in progress_bar:
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
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Evaluate one validation epoch.
        """
        print("val_loader =", self.val_loader)
        print("Is None =", self.val_loader is None)

        if self.val_loader is None:
            return running_loss

        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for anchor, positive, negative, label in self.val_loader:

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
            start_time = time.time()

            train_loss = self.train_one_epoch()

            val_loss = self.validate()
            print("DEBUG validation loss:", val_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            print("=" * 60)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")
            print(f"Best Val   : {self.best_val_loss:.4f}")
            print(f"LR         : {current_lr:.6f}")
            print(f"Time       : {elapsed_time:.2f} s")
            print("=" * 60)

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
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            
            self.writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

            self.writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch,
        )

            self.writer.add_scalar(
            "Learning Rate",
            self.optimizer.param_groups[0]["lr"],
            epoch,
        )
        self.writer.close()

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

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
                ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        },
        checkpoint_path,
        )

        print(f"Best model saved -> {checkpoint_path}")