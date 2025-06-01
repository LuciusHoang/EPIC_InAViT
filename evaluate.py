import os
import torch
from tqdm import tqdm
from utils.data_loader import get_dataloader
from utils.metrics import compute_metrics, plot_confusion_matrix
from config import (
    TEST_DATA_DIR,
    SELECTED_TASKS,
    NUM_CLASSES,
    BATCH_SIZE,
    SEQ_LENGTH,
    NUM_WORKERS,
    USE_MPS,
    CHECKPOINT_DIR,
    INPUT_DIM,
    MODEL_TYPES
)
from models import mlp_bc, cnn_lstm, transformer

def evaluate():
    device = torch.device("mps" if USE_MPS and torch.backends.mps.is_available() else "cpu")

    test_loader = get_dataloader(
        root_dir=TEST_DATA_DIR,
        task_list=SELECTED_TASKS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seq_length=SEQ_LENGTH,
        num_workers=NUM_WORKERS
    )

    for model_name in MODEL_TYPES:
        print(f"\n🚀 Evaluating model: {model_name}")

        # Model selection
        if model_name == 'mlp':
            model = mlp_bc.MLPBC(
                input_dim=INPUT_DIM,
                seq_length=SEQ_LENGTH,
                num_classes=NUM_CLASSES
            )
            checkpoint_name = 'mlp_bc_checkpoint.pth'
        elif model_name == 'cnn_lstm':
            model = cnn_lstm.CNNLSTM(
                input_dim=INPUT_DIM,
                num_classes=NUM_CLASSES
            )
            checkpoint_name = 'cnn_lstm_checkpoint.pth'
        elif model_name == 'transformer':
            model = transformer.PoseTransformer(
                input_size=INPUT_DIM,
                seq_length=SEQ_LENGTH,
                num_classes=NUM_CLASSES
            )
            checkpoint_name = 'transformer_checkpoint.pth'
        else:
            print(f"⚠️ Unsupported model type: {model_name}. Skipping.")
            continue

        model.to(device)

        # Load checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found at {checkpoint_path}. Skipping {model_name}.")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Filter invalid labels
                valid_mask = labels != -1
                if valid_mask.sum() == 0:
                    continue

                inputs = inputs[valid_mask]
                labels = labels[valid_mask]

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if not all_preds or not all_labels:
            print(f"⚠️ No valid samples found for {model_name}. Skipping metrics.")
            continue

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = compute_metrics(all_labels, all_preds, class_names=SELECTED_TASKS)
        print(f"✅ {model_name.upper()} Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"✅ {model_name.upper()} Test Macro F1: {metrics['f1_macro']:.4f}")

        plot_confusion_matrix(metrics['confusion_matrix'], SELECTED_TASKS, normalize=True)

if __name__ == "__main__":
    evaluate()