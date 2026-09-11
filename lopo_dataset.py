from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class LOPOGazeDataset(Dataset):
    def __init__(self, mat_paths, include_metadata=False):
        self.mat_paths = list(mat_paths)
        self.include_metadata = include_metadata

        self.images = []
        self.gazes = []
        self.poses = []
        self.file_lengths = []
        self.cumulative_ends = []

        total = 0

        for path in self.mat_paths:
            data = loadmat(str(path), struct_as_record=False)

            subject = data["data"][0, 0]
            right = subject.right[0, 0]

            images = right.image
            gazes = right.gaze
            poses = right.pose

            if len(images) != len(gazes) or len(images) != len(poses):
                raise ValueError(f"Length mismatch in {path}")

            self.images.append(images)
            self.gazes.append(gazes)
            self.poses.append(poses)

            file_length = len(images)
            self.file_lengths.append(file_length)

            total += file_length
            self.cumulative_ends.append(total)

        self.total = total

    def __len__(self):
        return self.total

    def _locate(self, idx):
        if idx < 0:
            idx += self.total

        if idx < 0 or idx >= self.total:
            raise IndexError(f"dataset index out of range: {idx}")

        file_idx = bisect_right(self.cumulative_ends, idx)
        previous_end = 0

        if file_idx > 0:
            previous_end = self.cumulative_ends[file_idx - 1]

        local_idx = idx - previous_end
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)

        image = self.images[file_idx][local_idx]
        gaze = self.gazes[file_idx][local_idx]

        x, y, z = gaze

        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0)

        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        label = torch.tensor([pitch, yaw], dtype=torch.float32)

        if not self.include_metadata:
            return image, label

        pose = torch.tensor(
            self.poses[file_idx][local_idx],
            dtype=torch.float32,
        )

        metadata = {
            "file_path": str(self.mat_paths[file_idx]),
            "sample_index": local_idx,
            "pose": pose,
            "gaze": label,
        }

        return image, label, metadata


if __name__ == "__main__":
    base = Path("./dataset/Data/Normalized")
    people = [f"p{i:02d}" for i in range(15)]
    test_person = "p00"

    test_files = sorted((base / test_person).rglob("*.mat"))

    train_files = []
    for person in people:
        if person != test_person:
            train_files.extend(sorted((base / person).rglob("*.mat")))

    dataset = LOPOGazeDataset(train_files)
    metadata_dataset = LOPOGazeDataset(
        test_files,
        include_metadata=True,
    )

    first_image, first_label = dataset[0]
    negative_image, negative_label = dataset[-1]
    last_image, last_label = dataset[len(dataset) - 1]

    print("Total training samples:", len(dataset))
    print("First sample shapes:", first_image.shape, first_label.shape)
    print(
        "Negative index matches last index:",
        torch.equal(negative_image, last_image)
        and torch.equal(negative_label, last_label),
    )

    image, label, metadata = metadata_dataset[0]
    print("Test image shape:", image.shape)
    print("Test label shape:", label.shape)
    print("Metadata keys:", list(metadata.keys()))
    print("Pose:", metadata["pose"])
    print("Source file:", metadata["file_path"])