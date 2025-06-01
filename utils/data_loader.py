import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EgoDexDataset(Dataset):
    """
    Dataset class for EgoDex pose sequences using PyTorch DataLoader.
    Loads selected joint transforms from HDF5 files and organizes them
    into sequences for training models.
    """
    def __init__(self, root_dir, task_list, seq_length=60, selected_joints=None, transform=None):
        """
        Args:
            root_dir (str): Path to the train/test directory.
            task_list (list): List of tasks to include.
            seq_length (int): Number of frames per sequence.
            selected_joints (list, optional): List of joints to extract.
            transform (callable, optional): Optional transform to apply.
        """
        self.root_dir = root_dir
        self.task_list = task_list
        self.seq_length = seq_length
        self.transform = transform

        # Default selected joints
        self.selected_joints = selected_joints or [
            'leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip',
            'leftRingFingerTip', 'leftLittleFingerTip',
            'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip',
            'rightRingFingerTip', 'rightLittleFingerTip'
        ]

        self.data_files = []
        for task in task_list:
            task_dir = os.path.join(root_dir, task)
            if not os.path.exists(task_dir):
                continue
            for file in os.listdir(task_dir):
                if file.endswith('.hdf5'):
                    self.data_files.append((task, os.path.join(task_dir, file)))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        task, file_path = self.data_files[idx]
        label = self.task_list.index(task)

        try:
            with h5py.File(file_path, 'r') as f:
                transforms_group = f['transforms']

                positions_list = []
                for joint_name in self.selected_joints:
                    if joint_name not in transforms_group:
                        raise KeyError(f"Joint '{joint_name}' not found in transforms.")
                    joint_data = transforms_group[joint_name][:]
                    joint_positions = joint_data[:, :3, 3]  # Extract positions
                    positions_list.append(joint_positions)

                positions = np.concatenate(positions_list, axis=1)  # shape: (T, 3*num_joints)

                total_frames = positions.shape[0]
                if total_frames >= self.seq_length:
                    start = np.random.randint(0, total_frames - self.seq_length + 1)
                    positions_seq = positions[start:start + self.seq_length]
                else:
                    pad_len = self.seq_length - total_frames
                    positions_seq = np.pad(positions, ((0, pad_len), (0, 0)), mode='edge')

                positions_seq = torch.from_numpy(positions_seq).float()
                label = torch.tensor(label).long()

                if self.transform:
                    positions_seq = self.transform(positions_seq)

                return positions_seq, label

        except (OSError, KeyError, ValueError) as e:
            print(f"Skipping file {file_path}: {e}")
            # Return dummy tensor and dummy label
            dummy_shape = (self.seq_length, 3 * len(self.selected_joints))
            positions_seq = torch.zeros(dummy_shape)
            label = torch.tensor(-1).long()
            return positions_seq, label

def get_dataloader(root_dir, task_list, batch_size=16, shuffle=True, seq_length=60, num_workers=4, selected_joints=None):
    """
    Utility function to create a PyTorch DataLoader from the EgoDexDataset.
    """
    dataset = EgoDexDataset(
        root_dir=root_dir,
        task_list=task_list,
        seq_length=seq_length,
        selected_joints=selected_joints
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class HDF5Dataset:
    """
    A simpler, NumPy-based dataset class to load stacked datasets from an HDF5 file.
    Does not rely on torch.utils.data.Dataset, useful for quick data inspection or non-PyTorch tasks.
    """
    def __init__(self, file_path, data_group='transforms', label_dataset='confidences/hip'):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            data_group (str): Group name containing feature datasets.
            label_dataset (str): Path to the label dataset.
        """
        self.file = h5py.File(file_path, 'r')

        # Load all datasets from the data group
        data_items = []
        for key in self.file[data_group]:
            data_items.append(self.file[f"{data_group}/{key}"][:])
        self.data = np.stack(data_items, axis=-1)  # shape: (N, ..., num_features)

        # Load labels
        self.labels = self.file[label_dataset][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def close(self):
        self.file.close()
