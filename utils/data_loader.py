import os
import h5py
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.graph_utils import build_temporal_graph  # Define separately

class EgoDataset(Dataset):
    """
    Dataset for extracting embeddings from egocentric video and pose data.
    Returns RGB clips, pose tensors, temporal graphs, and clip IDs.
    """
    def __init__(self, root_dir='test', mode='precompute', seq_length=16, image_size=(224, 224)):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.image_size = image_size
        self.mode = mode
        self.samples = self._gather_samples()

    def _gather_samples(self):
        clips = []
        for task in sorted(os.listdir(self.root_dir)):
            task_dir = os.path.join(self.root_dir, task)
            if not os.path.isdir(task_dir): continue

            for fname in sorted(os.listdir(task_dir)):
                if fname.endswith('.mp4'):
                    base = fname[:-4]
                    mp4_path = os.path.join(task_dir, base + '.mp4')
                    h5_path = os.path.join(task_dir, base + '.hdf5')
                    if os.path.exists(h5_path):
                        clips.append({
                            'clip_id': f'{task}_{base}',
                            'mp4': mp4_path,
                            'h5': h5_path
                        })
        return clips

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_id = sample['clip_id']
        rgb_tensor = self._load_rgb_frames(sample['mp4'])
        pose_tensor = self._load_pose(sample['h5'])

        graph_tensor = build_temporal_graph(pose_tensor)  # shape: [T, J, D] → graph

        return {
            'clip_ids': clip_id,
            'rgb': rgb_tensor,       # [T, C, H, W]
            'pose': pose_tensor,     # [T, J, D]
            'graph': graph_tensor    # for EgoPack GNN
        }

    def _load_rgb_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(total_frames)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.resize(frame, self.image_size)
                frame = frame[:, :, ::-1]  # BGR to RGB
                frame = frame.transpose(2, 0, 1)  # [H, W, C] → [C, H, W]
                frames.append(frame)
        cap.release()

        frames = np.stack(frames, axis=0)  # [T, C, H, W]
        return torch.from_numpy(frames).float() / 255.0

    def _load_pose(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            joints = list(f['transforms'].keys())
            joint_data = [f['transforms'][j][:, :3, 3] for j in joints]  # (T, 3)
            poses = np.stack(joint_data, axis=1)  # (T, J, 3)
        return torch.from_numpy(poses).float()

    def _sample_indices(self, total_frames):
        if total_frames <= self.seq_length:
            return list(range(total_frames)) + [total_frames - 1] * (self.seq_length - total_frames)
        start = np.random.randint(0, total_frames - self.seq_length + 1)
        return list(range(start, start + self.seq_length))
