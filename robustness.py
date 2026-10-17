"""Cabin-condition robustness of the trained LOPO models.

Each fold model is tested on its own held-out person after a synthetic image
corruption at 5 severity levels. Nothing is retrained.

By default the corrupted eye image is histogram-equalized again before it
reaches the model, because that is what the real pipeline does (the dataset
images and gaze_demo.py both equalize after cutting out the eye). Without this
step a darker image would be a situation the pipeline never produces.
Use --raw to switch the re-equalization off and test the model on its own.

Usage:
  python robustness.py                               # final models, realistic pipeline
  python robustness.py --raw                         # without re-equalization
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from gaze_model import GazeNet, list_people, load_all
from metrics import angular_error_degrees


BATCH_SIZE = 512
OUT_DIR = Path("results/robustness")
SEED = 42
CLEAN_TOLERANCE = 0.05  # degrees; only checked when re-equalization is off

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
    padded = F.pad(images, (radius, radius, 0, 0), mode="reflect")
    images = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(images, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel.view(1, 1, -1, 1))


def corrupt(images, condition, value, generator):
    if condition == "brightness":
        images = images * value
    elif condition == "blur":
        images = blur(images, value)
    elif condition == "noise" and value > 0:
        images = images + torch.randn(images.shape, generator=generator) * value
    elif condition == "occlusion":
        images = images.clone()
        images[:, :, :round(images.shape[2] * value), :] = 0.0
    return images.clamp(0.0, 1.0)


def equalize(images):
    """Histogram-equalize every image again, as the real pipeline does."""
    as_uint8 = (images[:, 0] * 255).round().to(torch.uint8).numpy()
    out = np.stack([cv2.equalizeHist(img) for img in as_uint8])
    return torch.from_numpy(out).float().div(255).unsqueeze(1)


# ---------- models ----------

def load_config(model_dir):
    with open(Path(model_dir).parent / "config.json") as file:
        return json.load(file)


def load_model(model_dir, person, config, device):
    model = GazeNet(config["conv1"], config["conv2"], config["hidden"], config["head_pose"])
    model.load_state_dict(torch.load(Path(model_dir) / f"lopo_{person}.pt", map_location=device))
    return model.to(device).eval()


def clean_reference(model_dir):
    """Per-person test error recorded at training time."""
    with open(Path(model_dir).parent / "fold_results.csv", newline="") as file:
        return {row["person_id"]: float(row["mae_degrees"]) for row in csv.DictReader(file)}


def evaluate(model, images, labels, heads, condition, value, device, reequalize):
    generator = torch.Generator().manual_seed(SEED)
    errors = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE].float().div(255).unsqueeze(1)
            batch = corrupt(batch, condition, value, generator)
            if reequalize:
                batch = equalize(batch)
            predictions = model(batch.to(device), heads[start:start + BATCH_SIZE].to(device))
            errors.append(angular_error_degrees(predictions, labels[start:start + BATCH_SIZE].to(device)).cpu())
    return torch.cat(errors).mean().item()


# ---------- outputs ----------

def plot_summary(summary, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    clean_mae = summary[0]["person_equal_mae"]
    for ax, condition in zip(axes.flat, CONDITIONS):
        points = [row for row in summary if row["condition"] == condition]
        x = [row["value"] for row in points]
        y = [row["person_equal_mae"] for row in points]
        ax.axhline(clean_mae, color="gray", linestyle=":", linewidth=1, label="clean")
        ax.plot(x, y, marker="o")
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}°", (xi, yi), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8)
        ax.set_title(condition)
        ax.set_xlabel(AXIS_LABELS[condition])
        ax.set_xticks(x)
        if condition == "brightness":
            ax.invert_xaxis()
    axes[0, 0].set_ylabel("Person-equal MAE (degrees)")
    axes[1, 0].set_ylabel("Person-equal MAE (degrees)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_examples(image_uint8, path, reequalize):
    fig, axes = plt.subplots(len(CONDITIONS), 5, figsize=(10, 5))
    generator = torch.Generator().manual_seed(SEED)
    image = torch.from_numpy(image_uint8).float().div(255)[None, None]
    for r, (condition, values) in enumerate(CONDITIONS.items()):
        for c, value in enumerate(values):
            shown = corrupt(image, condition, value, generator)
            if reequalize:
                shown = equalize(shown)
            ax = axes[r, c]
            ax.imshow(shown[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{condition} {value}", fontsize=8)
    fig.suptitle("What the model receives" + (" (after re-equalization)" if reequalize else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/final/models")
    parser.add_argument("--raw", action="store_true", help="do not re-equalize after corruption")
    args = parser.parse_args()

    reequalize = not args.raw
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_raw" if args.raw else ""
    config = load_config(args.model_dir)
    eye_ids = [0, 1] if "left" in config.get("eyes", ["right"]) else [0]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    reference = clean_reference(args.model_dir)
    people = [p for p in list_people() if (Path(args.model_dir) / f"lopo_{p}.pt").exists()]
    print(f"Device: {device} | models: {args.model_dir} | people: {len(people)} | "
          f"re-equalize: {reequalize}")

    rows = []
    for k, person in enumerate(people):
        data = load_all([person])
        keep = np.isin(data["eye"], eye_ids)
        images = torch.from_numpy(data["images"][keep])
        labels = torch.from_numpy(data["gaze"][keep])
        heads = torch.from_numpy(data["head"][keep])
        model = load_model(args.model_dir, person, config, device)
        print(f"\n{person}: {len(images)} test eye images")

        if k == 0:
            plot_examples(data["images"][keep][len(images) // 2], OUT_DIR / f"robustness_examples{suffix}.png", reequalize)

        for condition, values in CONDITIONS.items():
            maes = [evaluate(model, images, labels, heads, condition, v, device, reequalize) for v in values]
            for level, (value, mae) in enumerate(zip(values, maes)):
                rows.append({"person_id": person, "condition": condition, "level": level,
                             "value": value, "mae_degrees": mae})
            print(f"  {condition:<10} " + "  ".join(f"{m:.2f}°" for m in maes))

        difference = abs(maes[0] - reference[person])
        if not reequalize and difference > CLEAN_TOLERANCE:
            raise RuntimeError(f"{person}: clean MAE {maes[0]:.3f}° does not match training-time "
                               f"{reference[person]:.3f}°. Wrong model or data. Stop.")
        print(f"  clean vs training-time result: {maes[0]:.2f}° vs {reference[person]:.2f}°")

    summary = []
    for condition, values in CONDITIONS.items():
        for level, value in enumerate(values):
            maes = [r["mae_degrees"] for r in rows if r["condition"] == condition and r["level"] == level]
            summary.append({"condition": condition, "level": level, "value": value,
                            "n_people": len(maes), "person_equal_mae": float(np.mean(maes))})

    print("\nPerson-equal MAE by condition and level:")
    for condition in CONDITIONS:
        pts = [r for r in summary if r["condition"] == condition]
        print(f"  {condition:<10} " + "  ".join(f"{r['value']}: {r['person_equal_mae']:.2f}°" for r in pts))

    for data_rows, name in ((rows, OUT_DIR / f"robustness_results{suffix}.csv"), (summary, OUT_DIR / f"robustness_summary{suffix}.csv")):
        with open(name, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data_rows[0].keys()))
            writer.writeheader()
            writer.writerows(data_rows)

    title = "Error under cabin-like image conditions (held-out people)"
    title += "" if reequalize else ", no re-equalization"
    plot_summary(summary, OUT_DIR / f"robustness_error{suffix}.png", title)
    print(f"\nSaved to {OUT_DIR}: robustness_results{suffix}.csv, robustness_summary{suffix}.csv, "
          f"robustness_error{suffix}.png, robustness_examples{suffix}.png")


if __name__ == "__main__":
    main()
