import os
import time
import torch

class Logger:
    """
    Simple logger for training progress and model checkpoints.
    """

    def __init__(self, log_dir):
        """
        Initializes the logger.

        Args:
            log_dir (str): Directory to save logs and checkpoints.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')

    def log(self, message):
        """
        Logs a message to the console and the log file.

        Args:
            message (str): The message to log.
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")

    def save_checkpoint(self, model, optimizer, epoch, metrics, checkpoint_path):
        """
        Saves a model checkpoint, including model state, optimizer state, and metrics.

        Args:
            model (torch.nn.Module): Trained model to save.
            optimizer (torch.optim.Optimizer): Optimizer to save.
            epoch (int): Current training epoch.
            metrics (dict): Dictionary of training metrics.
            checkpoint_path (str): Full path for the checkpoint file.
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.log(f"✅ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """
        Loads a model checkpoint into the provided model and optimizer.

        Args:
            model (torch.nn.Module): Model to load state into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            tuple: (epoch (int), metrics (dict))
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        self.log(f"📦 Checkpoint loaded: {checkpoint_path} (epoch {epoch})")
        return epoch, metrics
