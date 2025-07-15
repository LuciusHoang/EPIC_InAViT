# main.py

import os
import json
from inavit_inference import infer

# Path settings
TEST_DIR = "test"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_00081.pyth"
OUTPUT_JSON = "results/inavit_predictions.json"

def main():
    predictions = {}

    for action_folder in sorted(os.listdir(TEST_DIR)):
        action_path = os.path.join(TEST_DIR, action_folder)
        if not os.path.isdir(action_path):
            continue

        for file_name in sorted(os.listdir(action_path)):
            if not file_name.endswith(".mp4"):
                continue

            video_path = os.path.join(action_path, file_name)
            print(f"🔍 Running inference on: {video_path}")

            pred_class, mapped_label = infer(video_path, CHECKPOINT_PATH)

            predictions[video_path] = {
                "class_index": pred_class,
                "mapped_label": mapped_label or "unknown",
                "true_label": action_folder
            }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\n✅ Saved predictions to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
