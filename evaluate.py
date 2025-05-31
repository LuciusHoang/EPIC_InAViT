import os
import torch
from torch.utils.data import DataLoader

from config import *
from utils.data_loader import get_dataloader
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.logger import Logger
from utils.utils import set_seed, get_device, ensure_dir, labels_to_names

from models.mlp_bc import MLPBC
from models.cnn_lstm import CNNLSTM
from models.transformer import PoseTransformer

def select_model(model_name):
    """
    Factory function to select model architecture.
    """
    if model_name == 'mlp_bc':
        return MLPBC(input_size=48, seq_length=SEQ_LENGTH, num_classes=NUM_CLASSES)
    elif model_name == 'cnn_lstm':
        return CNNLSTM(num_classes=NUM_CLASSES)
    elif model_name == 'transformer':
        return PoseTransformer(input_size=48, seq_length=SEQ_LENGTH, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pose'].to(device) if 'pose' in batch else None
            video_inputs = batch['video'].to(device) if 'video' in batch else None
            labels = batch['label'].to(device)

            if isinstance(model, MLPBC) or isinstance(model, PoseTransformer):
                outputs = model(inputs)
            elif isinstance(model, CNNLSTM):
                outputs = model(video_inputs)
            else:
                raise ValueError("Model type not supported in evaluate.")

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, class_names=SELECTED_TASKS)
    return metrics

def main(model_name='mlp_bc', checkpoint_path='logs/checkpoints/best_model.pth'):
    set_seed(SEED)
    device = get_device()

    ensure_dir(LOG_DIR)
    logger = Logger(LOG_DIR)
    logger.log(f"Evaluating model: {model_name}")

    # Load model
    model = select_model(model_name).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        logger.log(f"Loading checkpoint from {checkpoint_path}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Dummy optimizer for compatibility
        _, _ = logger.load_checkpoint(model, optimizer, checkpoint_path)
    else:
        logger.log(f"Checkpoint not found: {checkpoint_path}")
        return

    # Data loader
    test_loader = get_dataloader(TEST_DATA_DIR, SELECTED_TASKS, batch_size=BATCH_SIZE, mode='test', shuffle=False, num_workers=NUM_WORKERS)

    # Evaluate
    metrics = evaluate(model, test_loader, device)

    # Print results
    logger.log(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.log(f"Test Macro F1-score: {metrics['f1_macro']:.4f}")

    for idx, f1 in enumerate(metrics['f1_per_class']):
        logger.log(f"{SELECTED_TASKS[idx]}: F1-score: {f1:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], SELECTED_TASKS, normalize=True, save_path=os.path.join(LOG_DIR, "confusion_matrix.png"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate EgoDex classification models.")
    parser.add_argument('--model', type=str, default='mlp_bc', choices=['mlp_bc', 'cnn_lstm', 'transformer'],
                        help='Model architecture to evaluate.')
    parser.add_argument('--checkpoint', type=str, default='logs/checkpoints/best_model.pth',
                        help='Path to model checkpoint.')
    args = parser.parse_args()

    main(model_name=args.model, checkpoint_path=args.checkpoint)
