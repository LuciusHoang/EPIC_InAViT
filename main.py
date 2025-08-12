import os
import json
import time
import sys
import pandas as pd

# Add custom module path for InAViT
sys.path.append(os.path.abspath("InAViT"))

from slowfast.config.defaults import get_cfg
from inavit_inference import predict_segment, build_and_load_model

# ✅ Paths
EPIC_TEST_LIST = "EPIC_100_test_segments.csv"  # must contain 'narration_id' or 'segment_id'
FRAMES_BASE = "/Volumes/T7_Shield/inference/frames_rgb_flow/rgb/test"  # extracted RGB frames

def _get_segment_ids(csv_path):
    df = pd.read_csv(csv_path)

    # Prefer 'narration_id', fallback to 'segment_id'
    id_col = "narration_id" if "narration_id" in df.columns else (
        "segment_id" if "segment_id" in df.columns else None
    )
    if id_col is None:
        raise KeyError("CSV must contain 'narration_id' or 'segment_id'.")

    # If present, preserve any prior sorting (e.g., by start_frame) in the file
    seg_ids = df[id_col].astype(str).tolist()
    return seg_ids

def _frames_dir_for(segment_id):
    """
    Convert 'P01_11_XXXX' or 'P01_11' → {FRAMES_BASE}/P01/P01_11/
    Only the first two tokens (PXX_YY) map to the folder name.
    """
    base_id = "_".join(segment_id.split("_")[:2])  # PXX_YY
    participant = base_id.split("_")[0]            # PXX
    return os.path.join(FRAMES_BASE, participant, base_id)

def main():
    # Load and configure model
    cfg = get_cfg()
    cfg.merge_from_file("EK_INAVIT_MF_ant.yaml")
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    model = build_and_load_model(cfg)
    model.eval()

    segment_ids = _get_segment_ids(EPIC_TEST_LIST)

    results = []
    total = len(segment_ids)
    start_time = time.time()

    for i, segment_id in enumerate(segment_ids, 1):
        frames_dir = _frames_dir_for(segment_id)

        if not os.path.isdir(frames_dir):
            print(f"[WARN] Missing frames dir → {frames_dir}")
            continue

        print(f"\n▶️ [{i}/{total}] {segment_id}")
        pred_idx, conf = predict_segment(frames_dir, obj_dir=None, model=model, cfg=cfg)

        results.append({
            "segment_id": segment_id,       # <-- matches evaluate_predictions.py
            "pred_idx": int(pred_idx),      # <-- matches evaluate_predictions.py
            "confidence": float(conf),      # <-- matches evaluate_predictions.py
        })

        elapsed = time.time() - start_time
        avg = elapsed / i
        eta = avg * (total - i)
        print(f"⏱  elapsed {elapsed:.1f}s | ETA {eta:.1f}s | pred {pred_idx} (conf {conf:.4f})")

    # Save predictions
    os.makedirs("results", exist_ok=True)
    with open("results/inavit_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Inference complete → results/inavit_predictions.json")

if __name__ == "__main__":
    main()
