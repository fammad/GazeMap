import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lopo_dataset import LOPOGazeDataset
from metrics import angular_error_degrees
from pipeline import GazeCNN


SEED = 42
BASE = Path("./dataset/Data/Normalized")
PEOPLE = [f"p{i:02d}" for i in range(15)]
BATCH_SIZE = 128
NUM_EPOCHS = 5


def build_fold_files(test_person):
    test_files = sorted((BASE / test_person).rglob("*.mat"))

    train_files = []
    for person in PEOPLE:
        if person != test_person:
            train_files.extend(
                sorted((BASE / person).rglob("*.mat"))
            )

    train_paths = set(train_files)
    test_paths = set(test_files)

    assert train_paths.isdisjoint(test_paths), (
        f"Leakage detected in fold {test_person}"
    )

    return train_files, test_files


def evaluate(model, loader, device):
    model.eval()
    all_errors = []
    all_metadata = []

    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            errors = angular_error_degrees(
                predictions,
                labels,
            ).cpu()

            all_errors.append(errors)

            batch_size = len(errors)

            for i in range(batch_size):
                all_metadata.append(
                    {
                        "file_path": metadata["file_path"][i],
                        "sample_index": int(
                            metadata["sample_index"][i]
                        ),
                        "pose": metadata["pose"][i].numpy(),
                        "gaze_pitch": float(
                            metadata["gaze"][i][0]
                        ),
                        "gaze_yaw": float(
                            metadata["gaze"][i][1]
                        ),
                        "error_degrees": float(
                            errors[i].item()
                        ),
                    }
                )

    return torch.cat(all_errors), all_metadata


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    fold_rows = []
    all_errors = []
    all_metadata = []

    for test_person in PEOPLE:
        torch.manual_seed(SEED)

        print(f"\nStarting fold {test_person}")

        train_files, test_files = build_fold_files(
            test_person
        )

        train_dataset = LOPOGazeDataset(
            train_files,
            include_metadata=False,
        )

        test_dataset = LOPOGazeDataset(
            test_files,
            include_metadata=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = GazeCNN().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        epoch_maes = []

        for epoch in range(NUM_EPOCHS):
            model.train()

            train_loss_total = 0.0
            train_sample_total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                predictions = model(images)
                batch_mean_loss = loss_fn(
                    predictions,
                    labels,
                )

                batch_size = images.shape[0]
                batch_mean_loss.backward()
                optimizer.step()

                train_loss_total += (
                    batch_mean_loss.item() * batch_size
                )
                train_sample_total += batch_size

            train_average = (
                train_loss_total / train_sample_total
            )

            errors, epoch_metadata = evaluate(
                model,
                test_loader,
                device,
            )

            test_mae = errors.mean().item()
            epoch_maes.append(test_mae)

            print(
                f"Fold {test_person} | "
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"train loss: {train_average:.6f} | "
                f"test MAE: {test_mae:.2f}°"
            )

        final_errors, final_metadata = evaluate(
            model,
            test_loader,
            device,
        )

        for record in final_metadata:
            record["person_id"] = test_person

        all_errors.append(final_errors.numpy())
        all_metadata.extend(final_metadata)

        row = {
            "person_id": test_person,
            "n_test_samples": len(test_dataset),
            "final_mae_degrees": epoch_maes[-1],
        }

        for epoch_number, mae in enumerate(
            epoch_maes,
            start=1,
        ):
            row[
                f"epoch_{epoch_number}_mae_degrees"
            ] = mae

        fold_rows.append(row)

        print(
            f"Fold {test_person} final MAE: "
            f"{epoch_maes[-1]:.2f}°"
        )

    fold_maes = np.array(
        [row["final_mae_degrees"] for row in fold_rows]
    )

    counts = np.array(
        [row["n_test_samples"] for row in fold_rows]
    )

    pooled_errors = np.concatenate(all_errors)

    person_equal_mean = fold_maes.mean()
    population_std = fold_maes.std()
    sample_std = fold_maes.std(ddof=1)
    standard_error = population_std / np.sqrt(
        len(fold_maes)
    )

    sample_weighted_mean = np.average(
        fold_maes,
        weights=counts,
    )

    per_sample_mean = pooled_errors.mean()

    assert np.isclose(
        sample_weighted_mean,
        per_sample_mean,
        atol=1e-5,
    )

    fold_fields = [
        "person_id",
        "n_test_samples",
        "epoch_1_mae_degrees",
        "epoch_2_mae_degrees",
        "epoch_3_mae_degrees",
        "epoch_4_mae_degrees",
        "epoch_5_mae_degrees",
        "final_mae_degrees",
    ]

    with open(
        "lopo_results.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fold_fields,
        )
        writer.writeheader()
        writer.writerows(fold_rows)

    metadata_fields = [
        "person_id",
        "file_path",
        "sample_index",
        "pose_0",
        "pose_1",
        "pose_2",
        "gaze_pitch",
        "gaze_yaw",
        "error_degrees",
    ]

    with open(
        "lopo_sample_results.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata_fields,
        )
        writer.writeheader()

        for record in all_metadata:
            pose = record["pose"]

            writer.writerow(
                {
                    "person_id": record["person_id"],
                    "file_path": record["file_path"],
                    "sample_index": record[
                        "sample_index"
                    ],
                    "pose_0": pose[0],
                    "pose_1": pose[1],
                    "pose_2": pose[2],
                    "gaze_pitch": record[
                        "gaze_pitch"
                    ],
                    "gaze_yaw": record[
                        "gaze_yaw"
                    ],
                    "error_degrees": record[
                        "error_degrees"
                    ],
                }
            )

    print("\nLOPO results over 15 folds:")
    print(
        f"Person-equal mean MAE: "
        f"{person_equal_mean:.2f}°"
    )
    print(
        f"Population std: "
        f"{population_std:.2f}°"
    )
    print(
        f"Sample std: "
        f"{sample_std:.2f}°"
    )
    print(
        f"Standard error of mean: "
        f"{standard_error:.2f}°"
    )
    print(
        f"Sample-weighted mean MAE: "
        f"{sample_weighted_mean:.2f}°"
    )
    print(
        f"Pooled per-sample MAE: "
        f"{per_sample_mean:.2f}°"
    )
    print(
        f"Min fold MAE: "
        f"{fold_maes.min():.2f}°"
    )
    print(
        f"Max fold MAE: "
        f"{fold_maes.max():.2f}°"
    )
    print("Saved: lopo_results.csv")
    print("Saved: lopo_sample_results.csv")


if __name__ == "__main__":
    main()