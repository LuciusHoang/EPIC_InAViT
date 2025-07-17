import os
import json
import torch
import time
import sys

# Add custom module path for InAViT
sys.path.append(os.path.abspath("InAViT"))

from slowfast.config.defaults import get_cfg
from inavit_inference import predict_video, build_and_load_model
from ek100_to_egodex_map import ek100_to_egodex_map

def main():
    # Load and configure model
    cfg = get_cfg()
    cfg.merge_from_file("EK_INAVIT_MF_ant.yaml")
    cfg.MODEL.NUM_CLASSES = 24
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    model = build_and_load_model(cfg)
    model.eval()

    test_root = "test"
    results = []

    # Count all videos to track progress
    total_videos = sum(
        len(files) for _, _, files in os.walk(test_root) if files
    )
    processed = 0
    start_time = time.time()

    for class_folder in sorted(os.listdir(test_root)):
        class_path = os.path.join(test_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        for video_file in sorted(os.listdir(class_path)):
            video_path = os.path.join(class_path, video_file)

            print(f"\n▶️ [{processed + 1}/{total_videos}] Predicting: {video_path}")
            pred_idx, conf = predict_video(video_path, model, cfg)
            pred_label = ek100_to_egodex_map.get(pred_idx, "Unknown")

            results.append({
                "file": video_path,
                "pred_idx": pred_idx,
                "confidence": float(conf),
                "pred_label": pred_label
            })

            processed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            eta = avg_time * (total_videos - processed)

            print(f"⏱️  Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            print(f"✅ Confidence: {conf:.4f} | Label: {pred_label}")

    # Save final predictions
    os.makedirs("results", exist_ok=True)
    with open("results/inavit_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Inference complete. Predictions saved to results/inavit_predictions.json.")

if __name__ == "__main__":
    main()
