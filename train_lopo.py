"""GazeMap: leave-one-person-out training and the ablation steps.

Usage:
  python train_lopo.py --quick            # 2-minute smoke test + time estimate
  python train_lopo.py --config final     # one configuration
  python train_lopo.py --all              # final first, then the ablation steps

Results go to results/<config>/. An interrupted run can be restarted with the
same command: finished folds are skipped.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gaze_model import build_model, list_people, load_all, to_batch
from metrics import angular_error_degrees


SEED = 42
OUTPUT_ROOT = Path("results")

# Cumulative ablation ladder. Each step adds one change to the one before it.
# All settings are fixed before running; no epoch is chosen by looking at test error.
BASE_CONFIG = {
    "eyes": ["right", "left"],
    "conv1": 8, "conv2": 16, "hidden": 16,
    "head_pose": False,
    "epochs": 5, "lr": 1e-3, "batch_size": 128, "lr_drop_epochs": [],
}

CONFIGS = {
    # baseline: small network, right eye only (the original model)
    "baseline": {**BASE_CONFIG, "eyes": ["right"]},
    # same small network, trained and tested on both eyes
    "a_both_eyes": {**BASE_CONFIG},
    # + paper-size network (MnistNet: 20 / 50 / 500)
    "b_wide": {**BASE_CONFIG, "conv1": 20, "conv2": 50, "hidden": 500},
    # + head pose fed into the last layer
    "c_head_pose": {**BASE_CONFIG, "conv1": 20, "conv2": 50, "hidden": 500, "head_pose": True},
    # + longer training: 15 epochs, learning rate x0.1 after epoch 10
    "final": {**BASE_CONFIG, "conv1": 20, "conv2": 50, "hidden": 500, "head_pose": True,
              "epochs": 15, "lr_drop_epochs": [10]},
}

RUN_ORDER_ALL = ["final", "a_both_eyes", "b_wide", "c_head_pose"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, data, indices, device, batch_size=1024):
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            images = to_batch(data["images_t"][idx], device)
            labels = data["gaze_t"][idx].to(device)
            head = data["head_t"][idx].to(device)
            predictions = model(images, head)
            errors.append(angular_error_degrees(predictions, labels).cpu())
    return torch.cat(errors).numpy()


def train_fold(config, data, train_idx, test_idx, device, max_steps_per_epoch=None, log_prefix=""):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_drop_epochs"], gamma=0.1
    )
    loss_fn = nn.MSELoss()
    train_idx_t = torch.from_numpy(train_idx)

    epoch_maes = []
    for epoch in range(config["epochs"]):
        model.train()
        order = train_idx_t[torch.randperm(len(train_idx_t), generator=generator)]
        loss_sum, seen = 0.0, 0

        for step, start in enumerate(range(0, len(order), config["batch_size"])):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            idx = order[start:start + config["batch_size"]]
            images = to_batch(data["images_t"][idx], device)
            labels = data["gaze_t"][idx].to(device)
            head = data["head_t"][idx].to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(images, head), labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(idx)
            seen += len(idx)

        scheduler.step()

        # Logged for the learning curve only; never used to pick an epoch.
        test_errors = evaluate(model, data, test_idx, device)
        epoch_maes.append(float(test_errors.mean()))
        print(f"{log_prefix} epoch {epoch + 1}/{config['epochs']} | "
              f"train loss {loss_sum / seen:.6f} | test MAE {epoch_maes[-1]:.2f}°", flush=True)

    return model, test_errors, epoch_maes


def write_fold_samples(path, data, people, test_idx, errors):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["person_id", "eye", "file_path", "sample_index",
                         "pose_0", "pose_1", "pose_2", "gaze_pitch", "gaze_yaw",
                         "error_degrees"])
        for i, err in zip(test_idx, errors):
            pose = data["pose_raw"][i]
            writer.writerow([
                people[data["person"][i]],
                "right" if data["eye"][i] == 0 else "left",
                data["file_path"][i], int(data["sample_index"][i]),
                float(pose[0]), float(pose[1]), float(pose[2]),
                float(data["gaze"][i][0]), float(data["gaze"][i][1]),
                float(err),
            ])


def run_config(name, config, data, people, device, quick=False):
    out_dir = OUTPUT_ROOT / ("quick_" + name if quick else name)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)

    eye_ids = [0, 1] if "left" in config["eyes"] else [0]
    use = np.isin(data["eye"], eye_ids)

    fold_people = people[:2] if quick else people
    epochs = 1 if quick else config["epochs"]
    run_config_ = {**config, "epochs": epochs}

    print(f"\n===== {name} =====\n{json.dumps(config)}", flush=True)
    fold_rows = []
    started = time.time()

    for k, person in enumerate(fold_people):
        fold_json = out_dir / "folds" / f"{person}.json"
        if fold_json.exists() and not quick:
            with open(fold_json) as file:
                fold_rows.append(json.load(file))
            print(f"{person}: already done, skipped", flush=True)
            continue

        fold_start = time.time()
        person_index = people.index(person)
        test_idx = np.where(use & (data["person"] == person_index))[0]
        train_idx = np.where(use & (data["person"] != person_index))[0]
        assert len(np.intersect1d(test_idx, train_idx)) == 0

        model, errors, epoch_maes = train_fold(
            run_config_, data, train_idx, test_idx, device,
            max_steps_per_epoch=200 if quick else None,
            log_prefix=f"{name} {person}",
        )

        torch.save(model.state_dict(), out_dir / "models" / f"lopo_{person}.pt")
        write_fold_samples(out_dir / "folds" / f"{person}.csv", data, people, test_idx, errors)

        eyes = data["eye"][test_idx]
        row = {
            "person_id": person,
            "n_test_samples": int(len(test_idx)),
            "mae_degrees": float(errors.mean()),
            "mae_right_eye": float(errors[eyes == 0].mean()),
            "mae_left_eye": float(errors[eyes == 1].mean()) if (eyes == 1).any() else None,
            "epoch_maes": epoch_maes,
            "fold_seconds": round(time.time() - fold_start, 1),
        }
        with open(fold_json, "w") as file:
            json.dump(row, file)
        fold_rows.append(row)

        done = k + 1
        remaining = (time.time() - started) / done * (len(fold_people) - done)
        print(f"{person}: MAE {row['mae_degrees']:.2f}° (right eye {row['mae_right_eye']:.2f}°) | "
              f"fold {row['fold_seconds'] / 60:.1f} min | about {remaining / 60:.0f} min left for {name}",
              flush=True)

    if quick:
        total_steps = config["epochs"] * sum(
            int(np.ceil(np.sum(use & (data["person"] != i)) / config["batch_size"]))
            for i in range(len(people))
        )
        quick_steps = 2 * 200
        per_step = (time.time() - started) / quick_steps
        print(f"\nQuick test OK. Estimated full time for '{name}': "
              f"{total_steps * per_step / 3600:.1f} h (rough, includes evaluation overhead)", flush=True)
        return

    summarize(name, out_dir, fold_rows)


def summarize(name, out_dir, fold_rows):
    maes = np.array([r["mae_degrees"] for r in fold_rows])
    right = np.array([r["mae_right_eye"] for r in fold_rows])
    counts = np.array([r["n_test_samples"] for r in fold_rows])

    fields = ["person_id", "n_test_samples", "mae_degrees", "mae_right_eye", "mae_left_eye", "fold_seconds"]
    with open(out_dir / "fold_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields + [f"epoch_{i + 1}_mae" for i in range(len(fold_rows[0]["epoch_maes"]))])
        writer.writeheader()
        for r in fold_rows:
            out = {k: r[k] for k in fields}
            for i, m in enumerate(r["epoch_maes"]):
                out[f"epoch_{i + 1}_mae"] = m
            writer.writerow(out)

    # Merge per-fold sample files into one file for analyze_pose / robustness.
    with open(out_dir / "sample_results.csv", "w", newline="") as merged:
        for i, r in enumerate(fold_rows):
            with open(out_dir / "folds" / f"{r['person_id']}.csv") as part:
                lines = part.readlines()
                merged.writelines(lines if i == 0 else lines[1:])

    summary = (
        f"{name}: person-equal MAE {maes.mean():.2f}° "
        f"(population std {maes.std():.2f}°, SE {maes.std() / np.sqrt(len(maes)):.2f}°) | "
        f"pooled {np.average(maes, weights=counts):.2f}° | "
        f"right eye only, person-equal {right.mean():.2f}° (right-eye baseline: 6.34°)"
    )
    print("\n" + summary, flush=True)
    with open(out_dir / "summary.txt", "w") as file:
        file.write(summary + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), default="final")
    parser.add_argument("--all", action="store_true", help="run final, then the ablation steps")
    parser.add_argument("--quick", action="store_true", help="smoke test and time estimate")
    args = parser.parse_args()

    device = pick_device()
    people = list_people()
    print("Using device:", device, "| people found:", len(people), flush=True)
    if len(people) != 15 and not args.quick:
        raise RuntimeError(f"Expected 15 people in the dataset folder, found {len(people)}")

    t0 = time.time()
    data = load_all(people)
    data["images_t"] = torch.from_numpy(data["images"])
    data["gaze_t"] = torch.from_numpy(data["gaze"])
    data["head_t"] = torch.from_numpy(data["head"])
    print(f"Loaded {len(data['images'])} eye images in {time.time() - t0:.0f} s", flush=True)

    names = RUN_ORDER_ALL if args.all else [args.config]
    if args.quick:
        names = ["final"]

    for name in names:
        run_config(name, CONFIGS[name], data, people, device, quick=args.quick)


if __name__ == "__main__":
    main()
