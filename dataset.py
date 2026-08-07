from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GazeDataset(Dataset):
    def __init__(self, mat_path):
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        
        self.images = data["data"].right.image
        self.gaze = data["data"].right.gaze

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        image = torch.from_numpy(image).float() / 255.0 
        image = image.unsqueeze(0)

        x, y, z = self.gaze[i]
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)

        label = torch.tensor([pitch, yaw], dtype=torch.float32)
        return image, label
    
    

if __name__ == "__main__":
    mat_path = "./dataset/Data/Normalized/p00/day01.mat"

    dataset = GazeDataset(mat_path)

    print(len(dataset))

    img, label = dataset[0]
    print(img.shape, img.dtype)
    print(label.shape)

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    images, labels = next(iter(loader))
    print(images.shape, labels.shape)