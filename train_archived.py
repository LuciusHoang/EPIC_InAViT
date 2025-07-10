# train_archived.py (with logging and checkpoint saving)
import os
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, Dataset
from utils.utils import set_seed
from tqdm import tqdm

# Configuration
torch.set_num_threads(6)
torch.set_num_interop_threads(4)
EMBEDDING_TYPE = 'egom2p'  # or 'egopack'
EMBEDDING_DIR = f'embeddings/{EMBEDDING_TYPE}'
LABELS_PATH = 'labels.npy'
LOG_DIR = f'logs/{EMBEDDING_TYPE}'
os.makedirs(LOG_DIR, exist_ok=True)
CHECKPOINT_PATH = f'{LOG_DIR}/{EMBEDDING_TYPE}_model.pth'
LOG_PATH = f'{LOG_DIR}/train.log'

# Setup logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

BATCH_SIZE = 64
EPOCHS = 20

class EmbeddingDataset(Dataset):
    def __init__(self, emb_dir, label_path):
        self.emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith('.npy')])
        self.emb_dir = emb_dir
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.emb_files)

    def __getitem__(self, idx):
        emb = np.load(os.path.join(self.emb_dir, self.emb_files[idx]))
        label = self.labels[idx]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, output_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Load data
dataset = EmbeddingDataset(EMBEDDING_DIR, LABELS_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
model = MLPClassifier()
model = model.to('cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    logging.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Save model checkpoint
torch.save(model.state_dict(), CHECKPOINT_PATH)
logging.info(f"Model saved to {CHECKPOINT_PATH}")
