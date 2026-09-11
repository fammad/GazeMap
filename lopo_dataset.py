from pathlib import Path
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset


class LOPOGazeDataset(Dataset):
    def __init__(self, mat_paths):
        self.mat_paths = list(mat_paths)
        self.file_lengths = []
        self.cum_lengths = [0]

        self.images = []
        self.gazes = []

        total = 0
        for p in self.mat_paths:
            data = loadmat(str(p), struct_as_record=False)
            d = data["data"][0, 0]
            right = d.right[0, 0]

            imgs = right.image  # (N, H, W)
            gz = right.gaze     # (N, 3)

            self.images.append(imgs)
            self.gazes.append(gz)

            L = len(imgs)
            self.file_lengths.append(L)
            total += L
            self.cum_lengths.append(total)

        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Find file index
        file_idx = 0
        while file_idx < len(self.file_lengths) and idx >= self.cum_lengths[file_idx + 1]:
            file_idx += 1

        local_idx = idx - self.cum_lengths[file_idx]
        image = self.images[file_idx][local_idx]
        x, y, z = self.gazes[file_idx][local_idx]

        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0)

        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        label = torch.tensor([pitch, yaw], dtype=torch.float32)

        return image, label


if __name__ == "__main__":
    base = Path("./dataset/Data/Normalized")
    test_person = "p00"
    people = [f"p{i:02d}" for i in range(15)]

    test_dir = base / test_person
    test_files = sorted(test_dir.rglob("*.mat"))

    train_people = [p for p in people if p != test_person]
    train_files = []
    for person in train_people:
        person_dir = base / person
        train_files.extend(sorted(person_dir.rglob("*.mat")))

    dataset = LOPOGazeDataset(train_files)

    print("Total samples in train dataset:", len(dataset))
    img, label = dataset[0]
    print("Sample image shape:", img.shape)
    print("Sample label shape:", label.shape)

    img_last, label_last = dataset[-1]
    print("Last sample image shape:", img_last.shape)
    print("Last sample label shape:", label_last.shape)