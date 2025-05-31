import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from utils.data_loader import get_dataloader
from utils.metrics import compute_metrics
from utils.logger import Logger
from utils.utils import set_seed, get_device, ensure_dir

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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch['pose'].to(device) if 'pose' in batch else None
        video_inputs = batch['video'].to(device) if 'video' in batch else None
        labels = batch['label'].to(device)

        if isinstance(model, MLPBC) or isinstance(model, PoseTransformer):
            outputs = model(inputs)
        elif isinstance(model, CNNLSTM):
            outputs = model(video_inputs)
        else:
            raise ValueError("Model type not supported in train_one_epoch.")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
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
                raise ValueError("Model type not supported in validate.")

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics

def main(model_name='mlp_bc'):
    set_seed(SEED)
    device = get_device()

    ensure_dir(LOG_DIR)
    ensure_dir(CHECKPOINT_DIR)
    logger = Logger(LOG_DIR)
    logger.log(f"Starting training with model: {model_name}")

    # Initialize model
    model = select_model(model_name).to(device)
    logger.log(str(model))

    # Data loaders
    train_loader = get_dataloader(TRAIN_DATA_DIR, SELECTED_TASKS, batch_size=BATCH_SIZE, mode='train', shuffle=True, num_workers=NUM_WORKERS)
    val_loader = get_dataloader(TEST_DATA_DIR, SELECTED_TASKS, batch_size=BATCH_SIZE, mode='test', shuffle=False, num_workers=NUM_WORKERS)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        logger.log(f"Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        logger.log(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['f1_macro']:.4f}")
        logger.log(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            logger.save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_name='best_model.pth')

    logger.log("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train EgoDex classification models.")
    parser.add_argument('--model', type=str, default='mlp_bc', choices=['mlp_bc', 'cnn_lstm', 'transformer'],
                        help='Model architecture to train.')
    args = parser.parse_args()

    main(model_name=args.model)
