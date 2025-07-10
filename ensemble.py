# ensemble.py

import os
import numpy as np
import logging
from sklearn.metrics import accuracy_score, classification_report

def run_ensemble():
    LOG_DIR = 'logs/ensemble'
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, 'ensemble.log')

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    egom2p_logits = np.load("logits/egom2p_logits.npy")
    egopack_logits = np.load("logits/egopack_logits.npy")
    true_labels = np.load("test_labels.npy")

    # Fusion: simple average (can adjust weights if needed)
    ensemble_logits = (egom2p_logits + egopack_logits) / 2
    preds = np.argmax(ensemble_logits, axis=1)

    # Evaluation
    acc = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds)

    print(f"Ensemble accuracy: {acc:.4f}")
    print(report)

    logging.info(f"Ensemble accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{report}")

if __name__ == "__main__":
    run_ensemble()