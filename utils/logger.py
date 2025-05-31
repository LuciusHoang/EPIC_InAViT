import os
import time
import torch


class Logger:
    """
    Simple logger for training progress and model checkpoints.
    """

    def __init__(self, log_dir):
        """
        Args:
            log_dir (str): Directory to save logs and checkpoints.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'training_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')

    def log(self, message):
        """
        Prints a message and appends it to the log file.
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")

    def save_checkpoint(self, model, optimizer, epoch, metrics, checkpoint_name='checkpoint.pth'):
        """
        Saves model, optimizer, and metrics as a checkpoint.

        Args:
            model (torch.nn.Module): Trained model.
            optimizer (torch.optim.Optimizer): Optimizer.
            epoch (int): Current epoch.
            metrics (dict): Dictionary of metrics.
            checkpoint_name (str): Filename for the checkpoint.
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        checkpoint_path = os.path.join(self.log_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        self.log(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """
        Loads model, optimizer, and metrics from a checkpoint.

        Args:
            model (torch.nn.Module): Model to load state into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            checkpoint_path (str): Path to checkpoint file.

        Returns:
            tuple: epoch, metrics
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        self.log(f"Checkpoint loaded: {checkpoint_path} (epoch {epoch})")
        return epoch, metrics
