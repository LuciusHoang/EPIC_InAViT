import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from inavit_inference import load_model, predict_video
from ek100_to_egodex_map import ek100_to_egodex_map  # <- make sure the dictionary is named like this

def read_video(video_path, num_frames=16, size=224):
    """
    Load and resize video frames uniformly sampled from the video file.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        raise ValueError(f"Video too short: {video_path} ({total_frames} frames)")

    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Normalize and convert to tensor [B, C, T, H, W]
    frames = np.stack(frames).astype(np.float32) / 255.0
    frames = (frames - 0.5) / 0.5
    frames = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
    return frames


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint)
    model.eval().to(device)
    print("✅ Model loaded and ready.\n")

    predictions = {}

    class_dirs = sorted(os.listdir(args.test_dir))
    for class_name in class_dirs:
        class_path = os.path.join(args.test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for video_name in sorted(os.listdir(class_path)):
            if not video_name.endswith(".mp4"):
                continue

            video_path = os.path.join(class_path, video_name)

            try:
                frames = read_video(video_path, num_frames=args.num_frames, size=args.input_size).to(device)
            except Exception as e:
                print(f"[!] Skipped {video_name}: {e}")
                continue

            with torch.no_grad():
                logits = predict_video(model, frames)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                egodex_class = ek100_to_egodex_map.get(pred_idx, "unknown")

            predictions[video_name] = {
                "epic_pred_idx": pred_idx,
                "egodex_class": egodex_class,
                "probability": float(probs[pred_idx])
            }

            print(f"[✓] {video_name} → {egodex_class} ({probs[pred_idx]:.4f})")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"\n✅ Inference complete. Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="test", help="Path to EgoDex test set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_00081.pyth", help="Path to pretrained checkpoint")
    parser.add_argument("--input_size", type=int, default=224, help="Frame resize dimension (square)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames sampled per video")
    parser.add_argument("--output_file", type=str, default="results/inavit_predictions.json", help="Where to save JSON output")
    args = parser.parse_args()

    main(args)
