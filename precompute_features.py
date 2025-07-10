# precompute_features.py

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.data_loader import EgoDataset
from models.egom2p_encoder import EgoM2PEncoder
from models.egopack_gnn import EgoPackGNN

def run_feature_extraction(embedding_type='egom2p'):
    SAVE_DIR = 'embeddings'
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    DEVICE = torch.device('cpu')

    os.makedirs(f'{SAVE_DIR}/{embedding_type}', exist_ok=True)

    dataset = EgoDataset(mode='precompute')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if embedding_type == 'egom2p':
        model = EgoM2PEncoder(add_classifier=True).to(DEVICE).eval()
    elif embedding_type == 'egopack':
        model = EgoPackGNN(add_classifier=True).to(DEVICE).eval()
    else:
        raise ValueError("Unsupported embedding type")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            clip_ids = batch['clip_ids']

            if embedding_type == 'egom2p':
                rgb = batch['rgb'].to(DEVICE)
                pose = batch['pose'].to(DEVICE)
                features = model(rgb, pose, return_embedding=True)
            else:
                graph = batch['graph'].to(DEVICE)
                features = model(graph, return_embedding=True)

            for i, cid in enumerate(clip_ids):
                np.save(f'{SAVE_DIR}/{embedding_type}/{cid}.npy', features[i].cpu().numpy())

if __name__ == "__main__":
    run_feature_extraction('egom2p')
