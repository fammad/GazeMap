import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lopo_dataset import LOPOGazeDataset
from metrics import angular_error_degrees
from pipeline import GazeCNN


BASE = Path("./dataset/Data/Normalized")
MODELS_DIR = Path("models")
PEOPLE = [f"p{i:02d}" for i in range(15)]
BATCH_SIZE = 512
SEED = 42

# Severity 0 must reproduce the LOPO result within this many degrees.
CLEAN_TOLERANCE = 0.05

RESULTS_CSV = "robustness_results.csv"
SUMMARY_CSV = "robustness_summary.csv"
OUTPUT_PLOT = "robustness_error.png"
EXAMPLES_PLOT = "robustness_examples.png"

# Each condition has 5 levels. The first level is always the clean image.
CONDITIONS = {
    "brightness": [1.0, 0.8, 0.6, 0.4, 0.2],  # multiply pixel values
    "blur": [0.0, 0.5, 1.0, 1.5, 2.0],  # Gaussian sigma in pixels
    "noise": [0.0, 0.02, 0.05, 0.1, 0.2],  # noise std on the 0-1 pixel scale
    "occlusion": [0.0, 0.1, 0.2, 0.3, 0.4],  # fraction of rows blacked out from the top
}

AXIS_LABELS = {
    "brightness": "Brightness factor (1.0 = original)",
    "blur": "Blur sigma (pixels)",
    "noise": "Noise std (0-1 pixel scale)",
    "occlusion": "Fraction of image covered from top",
}


# ---------- corruptions: each takes a batch (N, 1, 36, 60) in 0-1 ----------

def darken(images, factor):
    return images * factor


def gaussian_kernel(sigma):
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def blur(images, sigma):
    if sigma == 0:
        return images

    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2

    # Blur left-right, then up-down. Same result as one 2D Gaussian, cheaper.
    padded = F.pad(images, (radius, radius, 0, 0), mode="reflect")
    images = F.conv2d(padded, kernel.view(1, 1, 1, -1))

    padded = F.pad(images, (0, 0, radius, radius), mode="reflect")
    images = F.conv2d(padded, kernel.view(1, 1, -1, 1))

    return images


def add_noise(images, std, generator):
    if std == 0:
        return images

    noise = torch.randn(images.shape, generator=generator) * std
    return images + noise


def occlude(images, fraction):
    rows = round(images.shape[2] * fraction)
    covered = images.clone()
    covered[:, :, :rows, :] = 0.0
    return covered


def corrupt(images, condition, value, generator):
    if condition == "brightness":
        images = darken(images, value)
    elif condition == "blur":
        images = blur(images, value)
    elif condition == "noise":
        images = add_noise(images, value, generator)
    elif condition == "occlusion":
        images = occlude(images, value)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return images.clamp(0.0, 1.0)


# ---------- data and models ----------

def load_person(person):
    files = sorted((BASE / person).rglob("*.mat"))
    dataset = LOPOGazeDataset(files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_images = []
    all_labels = []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)

    return torch.cat(all_images), torch.cat(all_labels)


def load_model(person, device):
    model = GazeCNN()
    state = torch.load(MODELS_DIR / f"lopo_{person}.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_clean_reference(path="lopo_results.csv"):
    reference = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            reference[row["person_id"]] = float(row["final_mae_degrees"])
    return reference


def evaluate(model, images, labels, condition, value, device):
    # Fresh generator per evaluation, so noise is identical on every run.
    generator = torch.Generator().manual_seed(SEED)
    errors = []

    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]
            batch_labels = labels[start:start + BATCH_SIZE]

            batch = corrupt(batch, condition, value, generator)

            predictions = model(batch.to(device))
            batch_errors = angular_error_degrees(
                predictions,
                batch_labels.to(device),
            )
            errors.append(batch_errors.cpu())

    return torch.cat(errors).mean().item()


# ---------- outputs ----------

def summarize(rows):
    summary = []
    for condition, values in CONDITIONS.items():
        for level, value in enumerate(values):
            maes = [
                row["mae_degrees"]
                for row in rows
                if row["condition"] == condition and row["level"] == level
            ]
            summary.append(
                {
                    "condition": condition,
                    "level": level,
                    "value": value,
                    "n_people": len(maes),
                    "person_equal_mae": float(np.mean(maes)),
                }
            )
    return summary


def save_csv(rows, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    clean_mae = summary[0]["person_equal_mae"]

    for ax, condition in zip(axes.flat, CONDITIONS):
        points = [row for row in summary if row["condition"] == condition]
        x = [row["value"] for row in points]
        y = [row["person_equal_mae"] for row in points]

        ax.axhline(clean_mae, color="gray", linestyle=":", linewidth=1, label="clean")
        ax.plot(x, y, marker="o")
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}°", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)

        ax.set_title(condition)
        ax.set_xlabel(AXIS_LABELS[condition])
        ax.set_xticks(x)
        if condition == "brightness":
            ax.invert_xaxis()  # left = clean, right = darkest

    axes[0, 0].set_ylabel("Person-equal MAE (degrees)")
    axes[1, 0].set_ylabel("Person-equal MAE (degrees)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("LOPO error under cabin-like image conditions (15 held-out people)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_examples(image, path):
    # image: one eye, shape (1, 36, 60)
    fig, axes = plt.subplots(len(CONDITIONS), 5, figsize=(10, 5))
    generator = torch.Generator().manual_seed(SEED)

    for r, (condition, values) in enumerate(CONDITIONS.items()):
        for c, value in enumerate(values):
            corrupted = corrupt(image.unsqueeze(0), condition, value, generator)
            ax = axes[r, c]
            ax.imshow(corrupted[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{condition} {value}", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    reference = load_clean_reference()
    rows = []

    for person in PEOPLE:
        images, labels = load_person(person)
        model = load_model(person, device)
        print(f"\n{person}: {len(images)} test samples")

        if person == PEOPLE[0]:
            plot_examples(images[len(images) // 2], EXAMPLES_PLOT)

        for condition, values in CONDITIONS.items():
            maes = []
            for level, value in enumerate(values):
                mae = evaluate(model, images, labels, condition, value, device)
                maes.append(mae)
                rows.append(
                    {
                        "person_id": person,
                        "condition": condition,
                        "level": level,
                        "value": value,
                        "mae_degrees": mae,
                    }
                )

            # Sanity check: the clean level must match the LOPO training run.
            difference = abs(maes[0] - reference[person])
            if difference > CLEAN_TOLERANCE:
                raise RuntimeError(
                    f"{person} clean MAE {maes[0]:.3f}° does not match "
                    f"lopo_results.csv {reference[person]:.3f}°"
                )

            print(f"  {condition:<10} " + "  ".join(f"{m:.2f}°" for m in maes))

    summary = summarize(rows)

    print("\nPerson-equal MAE by condition and level:")
    for condition in CONDITIONS:
        points = [row for row in summary if row["condition"] == condition]
        print(f"  {condition:<10} " + "  ".join(
            f"{row['value']}: {row['person_equal_mae']:.2f}°" for row in points
        ))

    save_csv(rows, RESULTS_CSV)
    save_csv(summary, SUMMARY_CSV)
    plot_summary(summary, OUTPUT_PLOT)
    print(f"\nSaved: {RESULTS_CSV}, {SUMMARY_CSV}, {OUTPUT_PLOT}, {EXAMPLES_PLOT}")


if __name__ == "__main__":
    main()
