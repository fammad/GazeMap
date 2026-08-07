import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import GazeDataset

class GazeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*6*12, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    mat_path = "./dataset/Data/Normalized/p00/day01.mat"
    dataset = GazeDataset(mat_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    images, labels = next(iter(loader))

    net = GazeCNN()
    out = net(images)

    print(out.shape)
 


