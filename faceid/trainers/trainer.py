import torch

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, device, checkpoint_path=None):
        """
        Initializes the Trainer.

        Args:
            model: The neural network model.
            loss_fn: The loss function.
            optimizer: The optimizer.
            train_loader: The data loader for training data.
            device: The device to run the training on.
            checkpoint_path: The path to save checkpoints.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.checkpoint_path = checkpoint_path

    def train_one_epoch(self):

        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:


            anchor, positive, negative, label = batch
            # unpack batch

            # move to device
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            label = label.to(self.device)
            # forward

            # compute loss
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
            # zero gradients
            self.optimizer.zero_grad()
            # backward
            loss.backward()
            self.optimizer.step()
            # optimizer step
            running_loss += loss.item()
        average_loss = running_loss / len(self.train_loader)
        #print(f"Average loss: {average_loss:.4f}")
        return average_loss
    
    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            average_loss = self.train_one_epoch()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}")
    
    def save_checkpoint(self, epoch, loss):

        checkpoint_path = (
            self.checkpoint_dir /
            f"epoch_{epoch + 1:03d}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved: {checkpoint_path}")