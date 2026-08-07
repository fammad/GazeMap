import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import GazeDataset
from pipeline import GazeCNN


mat_path = "./dataset/Data/Normalized/p00/day01.mat"
full_dataset = GazeDataset(mat_path)

train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size

split_generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size],
    generator=split_generator,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

model = GazeCNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    train_total = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        train_total += loss.item()

    train_average = train_total / len(train_loader)

    model.eval()
    validation_total = 0.0

    with torch.no_grad():
        for images, labels in validation_loader:
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            validation_total += loss.item()

    validation_average = validation_total / len(validation_loader)

    print(
        f"Epoch {epoch + 1}/5 | "
        f"train loss: {train_average:.4f} | "
        f"validation loss: {validation_average:.4f}"
    )
