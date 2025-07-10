# evaluate.py

import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from torch.nn import functional as F

from utils.data_loader import EgoDataset
from models.egom2p_encoder import EgoM2PEncoder
from models.egopack_gnn import EgoPackGNN

def evaluate(embedding_type='egom2p'):
    DEVICE = torch.device('cpu')
    LABELS_PATH = 'test_labels.npy'
    CLIP_IDS_PATH = 'clip_ids.npy'
    LOGIT_SAVE_PATH = f'logits/{embedding_type}_logits.npy'
    LOG_PATH = f'logs/{embedding_type}/evaluate.log'

    os.makedirs('logits', exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    dataset = EgoDataset(mode='precompute')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if embedding_type == 'egom2p':
        model = EgoM2PEncoder(add_classifier=True).to(DEVICE).eval()
    elif embedding_type == 'egopack':
        model = EgoPackGNN(add_classifier=True).to(DEVICE).eval()
    else:
        raise ValueError("Unsupported model type")

    probs_all = []
    clip_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if embedding_type == 'egom2p':
                rgb = batch['rgb'].to(DEVICE)
                pose = batch['pose'].to(DEVICE)
                logits = model(rgb, pose)
            else:
                graph = batch['graph'].to(DEVICE)
                logits = model(graph)

            probs = F.softmax(logits, dim=1)
            probs_all.append(probs.cpu().numpy())
            clip_ids.append(batch['clip_ids'])

    probs_all = np.concatenate(probs_all, axis=0)
    clip_ids = np.array(clip_ids).squeeze()
    np.save(LOGIT_SAVE_PATH, probs_all)
    np.save("clip_ids.npy", clip_ids)
    logging.info(f"Saved probabilities to {LOGIT_SAVE_PATH}")
    logging.info(f"Saved clip IDs to clip_ids.npy")

    true_labels = np.load(LABELS_PATH)
    preds = np.argmax(probs_all, axis=1)

    acc = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds)

    print(f"Accuracy: {acc:.4f}")
    print(report)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{report}")

    for cid, pred in zip(clip_ids, preds):
        logging.info(f"{cid}: predicted class {pred}")

if __name__ == "__main__":
    evaluate('egom2p')