from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lopo_dataset import LOPOGazeDataset
from pipeline import GazeCNN
from metrics import angular_error_degrees


torch.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

base = Path("./dataset/Data/Normalized")
people = [f"p{i:02d}" for i in range(15)]

batch_size = 128
num_epochs = 5
fold_maes = []

for test_person in people:
    print(f"\nStarting fold {test_person}")

    test_dir = base / test_person
    test_files = sorted(test_dir.rglob("*.mat"))

    train_people = [person for person in people if person != test_person]
    train_files = []

    for person in train_people:
        person_dir = base / person
        train_files.extend(sorted(person_dir.rglob("*.mat")))

    train_dataset = LOPOGazeDataset(train_files)
    test_dataset = LOPOGazeDataset(test_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = GazeCNN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            loss.backward()
            optimizer.step()

            train_total += loss.item()

        train_average = train_total / len(train_loader)

        model.eval()
        all_errors = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                errors = angular_error_degrees(predictions, labels)
                all_errors.append(errors.cpu())

        test_mae = torch.cat(all_errors).mean().item()

        print(
            f"Fold {test_person} | "
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_average:.4f} | "
            f"test MAE: {test_mae:.2f}°"
        )

    fold_maes.append(test_mae)
    print(f"Fold {test_person} final MAE: {test_mae:.2f}°")

fold_maes = np.array(fold_maes)

print("\nLOPO results over 15 folds:")
print(f"Mean MAE: {fold_maes.mean():.2f}°")
print(f"Std MAE:  {fold_maes.std():.2f}°")
print(f"Min MAE:  {fold_maes.min():.2f}°")
print(f"Max MAE:  {fold_maes.max():.2f}°")