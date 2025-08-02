import os
import json
import torch
import time
import sys
import pandas as pd

# Add custom module path for InAViT
sys.path.append(os.path.abspath("InAViT"))

from slowfast.config.defaults import get_cfg
from inavit_inference import predict_segment, build_and_load_model

# ✅ Path to EPIC test segment CSV file (must contain 'segment_id' column)
EPIC_TEST_LIST = "EPIC_100_test_segments.csv"

# ✅ Path to the raw video clips (test set)
VIDEO_BASE = "/Volumes/T7_Shield/videos/test"

def main():
    # Load and configure model
    cfg = get_cfg()
    cfg.merge_from_file("EK_INAVIT_MF_ant.yaml")
    cfg.MODEL.NUM_CLASSES = 3806  # EPIC-KITCHENS action classes
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    model = build_and_load_model(cfg)
    model.eval()

    df = pd.read_csv(EPIC_TEST_LIST)
    segment_ids = df["segment_id"].tolist()

    results = []
    total = len(segment_ids)
    start_time = time.time()

    for i, segment_id in enumerate(segment_ids, 1):
        participant = segment_id.split("_")[0]
        video_path = os.path.join(VIDEO_BASE, participant, f"{segment_id}.MP4")

        if not os.path.exists(video_path):
            print(f"[WARN] Video path missing: {video_path}")
            continue

        print(f"\n▶️ [{i}/{total}] Predicting {segment_id}")
        pred_idx, conf = predict_segment(video_path, None, model, cfg)

        results.append({
            "segment_id": segment_id,
            "pred_idx": pred_idx,
            "confidence": float(conf)
        })

        elapsed = time.time() - start_time
        avg_time = elapsed / i
        eta = avg_time * (total - i)

        print(f"⏱️  Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        print(f"✅ Action ID: {pred_idx} | Confidence: {conf:.4f}")

    # Save predictions to results
    os.makedirs("results", exist_ok=True)
    with open("results/inavit_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Inference complete. Predictions saved to results/inavit_predictions.json.")

if __name__ == "__main__":
    main()
