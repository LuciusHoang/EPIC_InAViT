# generate_test_labels.py

import os
import numpy as np
from config import SELECTED_TASKS, TEST_DATA_DIR

def generate_test_labels_and_ids():
    labels = []
    clip_ids = []

    for task_name in sorted(os.listdir(TEST_DATA_DIR)):
        task_path = os.path.join(TEST_DATA_DIR, task_name)
        if not os.path.isdir(task_path):
            continue

        if task_name not in SELECTED_TASKS:
            print(f"⚠️ Skipping unknown task: {task_name}")
            continue

        label_index = SELECTED_TASKS.index(task_name)

        for fname in sorted(os.listdir(task_path)):
            if fname.endswith('.hdf5'):
                clip_num = fname.replace('.hdf5', '')
                clip_id = f"{task_name}_{clip_num}"
                clip_ids.append(clip_id)
                labels.append(label_index)

    labels = np.array(labels, dtype=np.int64)
    clip_ids = np.array(clip_ids)

    np.save("test_labels.npy", labels)
    np.save("clip_ids.npy", clip_ids)

    print(f"✅ Saved test_labels.npy with shape {labels.shape}")
    print(f"✅ Saved clip_ids.npy with shape {clip_ids.shape}")

if __name__ == "__main__":
    generate_test_labels_and_ids()
