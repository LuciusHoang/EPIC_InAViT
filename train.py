import os
import torch
from utils.data_loader import get_dataloader
from models.mlp_bc import MLPBC
from models.cnn_lstm import CNNLSTM
from models.transformer import PoseTransformer
from config import Config
from utils.logger import Logger

def train():
    config = Config()
    logger = Logger(config.log_dir)

    train_loader = get_dataloader(
        root_dir=config.train_data_path,
        task_list=config.task_list,
        batch_size=config.batch_size,
        shuffle=True,
        seq_length=config.seq_length,
        num_workers=config.num_workers
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_type in config.model_types:
        logger.log(f"\n🚀 Starting training for model: {model_type}")

        if model_type == 'mlp':
            model = MLPBC(
                input_dim=config.input_dim,
                seq_length=config.seq_length,
                num_classes=config.num_classes
            )
            checkpoint_name = 'mlp_bc_checkpoint.pth'
        elif model_type == 'cnn_lstm':
            model = CNNLSTM(
                input_dim=config.input_dim,
                num_classes=config.num_classes
            )
            checkpoint_name = 'cnn_lstm_checkpoint.pth'
        elif model_type == 'transformer':
            model = PoseTransformer(
                input_size=config.input_dim,
                seq_length=config.seq_length,
                num_classes=config.num_classes
            )
            checkpoint_name = 'transformer_checkpoint.pth'
        else:
            logger.log(f"⚠️ Unsupported model type: {model_type}. Skipping.")
            continue

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(config.epochs):
            running_loss = 0.0

            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)

                valid_mask = labels != -1
                if valid_mask.sum() == 0:
                    continue

                data = data[valid_mask]
                labels = labels[valid_mask]

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    logger.log(f"[{model_type}] Epoch [{epoch+1}/{config.epochs}], "
                               f"Step [{batch_idx+1}/{len(train_loader)}], "
                               f"Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(train_loader)
            logger.log(f"[{model_type}] Epoch [{epoch+1}/{config.epochs}] completed. "
                       f"Average Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
        logger.save_checkpoint(model, optimizer, config.epochs, {'loss': avg_loss}, checkpoint_path)
        logger.log(f"✅ Training completed for model: {model_type}")

    logger.log('🎉 All models have been trained!')

if __name__ == '__main__':
    train()