import os
import torch
import torch.utils.data as data
import numpy as np
import h5py
import cv2

class EgoDexDataset(data.Dataset):
    """
    EgoDex Dataset Loader
    Loads .hdf5 (skeletal data) and .mp4 (video) files.
    """

    def __init__(self, root_dir, tasks, mode='train', seq_length=60, frame_size=(224, 224), fps=10):
        """
        Args:
            root_dir (str): Base directory path.
            tasks (list): List of selected tasks (folders).
            mode (str): 'train' or 'test'.
            seq_length (int): Number of frames/steps per sample.
            frame_size (tuple): (height, width) for video frames.
            fps (int): Frames per second for sampling videos.
        """
        self.root_dir = root_dir
        self.tasks = tasks
        self.mode = mode
        self.seq_length = seq_length
        self.frame_size = frame_size
        self.fps = fps
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for task in self.tasks:
            task_dir = os.path.join(self.root_dir, task)
            if not os.path.exists(task_dir):
                continue
            files = os.listdir(task_dir)
            hdf5_files = sorted([f for f in files if f.endswith('.hdf5')])
            for f in hdf5_files:
                index = f.split('.')[0]
                hdf5_path = os.path.join(task_dir, f"{index}.hdf5")
                mp4_path = os.path.join(task_dir, f"{index}.mp4")
                if os.path.exists(mp4_path):
                    samples.append((task, hdf5_path, mp4_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_hdf5(self, path):
        with h5py.File(path, 'r') as f:
            # Assuming dataset is stored under 'pose'
            if 'pose' in f:
                pose = np.array(f['pose'])
                if pose.shape[0] > self.seq_length:
                    start_idx = np.random.randint(0, pose.shape[0] - self.seq_length)
                    pose = pose[start_idx:start_idx + self.seq_length]
                else:
                    pad = self.seq_length - pose.shape[0]
                    pose = np.pad(pose, ((0, pad), (0, 0)), mode='constant')
                pose = torch.tensor(pose, dtype=torch.float32)
                return pose
            else:
                raise ValueError(f"Pose data not found in {path}")

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(cap.get(cv2.CAP_PROP_FPS) / self.fps))

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frame = frame / 255.0  # normalize to [0,1]
            frames.append(frame)
            if len(frames) >= self.seq_length:
                break

        cap.release()

        if len(frames) < self.seq_length:
            pad = self.seq_length - len(frames)
            frames.extend([np.zeros_like(frames[0]) for _ in range(pad)])

        video = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)
        return video

    def __getitem__(self, idx):
        task, hdf5_path, mp4_path = self.samples[idx]
        label = self.tasks.index(task)  # integer label
        pose_seq = self._load_hdf5(hdf5_path)
        video_seq = self._load_video(mp4_path)
        return {
            'pose': pose_seq,       # shape: (seq_length, features)
            'video': video_seq,     # shape: (seq_length, 3, H, W)
            'label': torch.tensor(label, dtype=torch.long),
            'task_name': task
        }

def get_dataloader(root_dir, tasks, batch_size=8, mode='train', shuffle=True, num_workers=0):
    dataset = EgoDexDataset(root_dir=root_dir, tasks=tasks, mode=mode)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
