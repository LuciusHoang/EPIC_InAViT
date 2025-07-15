import os
import cv2
import shutil
import tempfile
import subprocess
import numpy as np

# Optional: import your label mapping
try:
    from scripts.ek100_to_egodex_map import ek100_to_egodex_map
except ImportError:
    ek100_to_egodex_map = {}

def extract_frames(video_path, output_dir, num_frames=16, resolution=(224, 224)):
    """Extracts 16 uniformly sampled 224x224 frames from a .mp4 file."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print(f"⚠️  Skipping {video_path}: only {total_frames} frames.")
        return False

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    saved = 0
    idx = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i == frame_indices[idx]:
            frame = cv2.resize(frame, resolution)
            out_path = os.path.join(output_dir, f"{idx + 1:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            idx += 1
            if saved == num_frames:
                break

    cap.release()
    return saved == num_frames

def run_inavit_on_frames(frame_root, checkpoint_path):
    """Runs InAViT using subprocess and parses predicted class index."""
    cmd = [
        "python", "tools/run_net.py",
        "--cfg", "",  # using CLI overrides only
        "TRAIN.ENABLE", "False",
        "TEST.ENABLE", "True",
        "TEST.CHECKPOINT_FILE_PATH", checkpoint_path,
        "DATA.PATH_TO_DATA_DIR", frame_root,
        "TEST.BATCH_SIZE", "1"
    ]

    print("🔁 Running InAViT...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout

    # Try to find predicted class index in output
    for line in stdout.splitlines():
        if "Predicted class" in line or "preds" in line:
            try:
                pred_id = int("".join(filter(str.isdigit, line)))
                return pred_id
            except ValueError:
                continue

    print("⚠️ Could not extract prediction from output.")
    return None

def infer(video_path, checkpoint_path="checkpoints/checkpoint_epoch_00081.pyth"):
    """Main function: extracts frames, prepares input, runs InAViT, maps prediction."""
    print(f"\n▶️ Processing video: {video_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        if not extract_frames(video_path, frame_dir):
            return None, None

        # Prepare expected folder layout: ek100/videos/[clip_name]/
        clip_folder = os.path.join(tmpdir, "ek100", "videos", "clip_0001")
        os.makedirs(clip_folder, exist_ok=True)

        for f in sorted(os.listdir(frame_dir)):
            shutil.copy(os.path.join(frame_dir, f), os.path.join(clip_folder, f))

        pred_class = run_inavit_on_frames(os.path.join(tmpdir, "ek100"), checkpoint_path)

        if pred_class is not None:
            label = ek100_to_egodex_map.get(pred_class, f"unmapped_class_{pred_class}")
            print(f"✅ Predicted EK100 class index: {pred_class}")
            print(f"🔁 Mapped to EgoDex action: {label}")
            return pred_class, label

        return None, None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inavit_inference.py path/to/video.mp4")
        sys.exit(1)

    infer(sys.argv[1])
