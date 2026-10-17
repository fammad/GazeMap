"""Evaluate the trained LOPO models: both-eye fusion and per-user calibration.

No training. For every held-out person, the fold model that never saw that
person predicts gaze for both eyes. Then:

  1. single eye   : error of each eye on its own (the training-time number)
  2. both eyes    : the two eyes' predictions are rotated back to camera space
                    and averaged; compared with the average of the two true
                    gaze directions
  3. calibrated   : 9 frames from ONE recording day of that person are used to
                    fit a small correction (a fixed offset in pitch/yaw per eye);
                    it is tested on all OTHER days, where head pose, distance
                    and lighting differ. Repeated 5 times with different days.

Head-pose check: calibrated error is also reported by how far the head is
turned compared with the calibration frames, to see whether the correction
still holds when the head moves.

Usage:
  python evaluate.py                       # uses results/final/models
  python evaluate.py --model-dir results/c_head_pose/models
Outputs go to results/evaluation/.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from gaze_model import BASE, GazeNet, list_people, load_all


ORIGINAL = Path("dataset/Data/Original")
OUT_DIR = Path("results/evaluation")
N_CALIBRATION = 9
N_REPEATS = 5
MIN_DAY_FRAMES = 30
SEED = 42
HEAD_BINS = [0, 5, 10, 20, 180]  # degrees of head rotation away from calibration pose


# ---------- geometry ----------

def angles_to_vectors(angles):
    pitch, yaw = angles[:, 0], angles[:, 1]
    return np.stack([-np.cos(pitch) * np.sin(yaw),
                     -np.sin(pitch),
                     -np.cos(pitch) * np.cos(yaw)], axis=1)


def vectors_to_angles(v):
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.stack([np.arcsin(np.clip(-v[:, 1], -1, 1)), np.arctan2(-v[:, 0], -v[:, 2])], axis=1)


def angle_between(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1, 1)))


def normalization_rotation(head_rotation, center):
    """Same R as used to make the dataset's normalized eye images."""
    forward = center / np.linalg.norm(center)
    down = np.cross(forward, head_rotation[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.stack([right, down, forward])


def attach_geometry(data, people):
    """For every eye sample: the normalization rotation R and the head direction in camera space."""
    n = len(data["eye"])
    R = np.zeros((n, 3, 3))
    head_R = np.zeros((n, 3, 3))
    head_forward = np.zeros((n, 3))
    day = np.empty(n, dtype=object)

    for path in sorted(set(data["file_path"])):
        rows = np.where(data["file_path"] == path)[0]
        path = Path(path)
        person, day_name = path.parent.name, path.stem
        names = loadmat(str(path), variable_names=["filenames"])["filenames"].ravel()
        annotation = np.loadtxt(ORIGINAL / person / day_name / "annotation.txt", ndmin=2)

        for i in rows:
            frame = int(str(np.ravel(names[data["sample_index"][i]])[0])[:4]) - 1
            a = annotation[frame]
            head_rotation = cv2.Rodrigues(a[29:32])[0]
            center = a[35:38] if data["eye"][i] == 0 else a[38:41]
            R[i] = normalization_rotation(head_rotation, center)
            head_R[i] = head_rotation
            head_forward[i] = head_rotation[:, 2]
            day[i] = day_name

    data["R"], data["head_R"], data["head_forward"], data["day"] = R, head_R, head_forward, day


# ---------- prediction ----------

def load_model(model_dir, person, config):
    model = GazeNet(config["conv1"], config["conv2"], config["hidden"], config["head_pose"])
    model.load_state_dict(torch.load(Path(model_dir) / f"lopo_{person}.pt", map_location="cpu"))
    return model.eval()


def predict_person(model, data, idx, device, batch_size=2048):
    model = model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            j = idx[start:start + batch_size]
            x = torch.from_numpy(data["images"][j]).float().div(255).unsqueeze(1).to(device)
            h = torch.from_numpy(data["head"][j]).to(device)
            out.append(model(x, h).cpu().numpy())
    return np.concatenate(out)


# ---------- calibration ----------

def pick_calibration(frames_gaze, rng):
    """9 frames spread over the gaze range: one per cell of a 3x3 pitch/yaw grid."""
    pitch, yaw = frames_gaze[:, 0], frames_gaze[:, 1]
    p_edges = np.quantile(pitch, [1 / 3, 2 / 3])
    y_edges = np.quantile(yaw, [1 / 3, 2 / 3])
    cell = np.digitize(pitch, p_edges) * 3 + np.digitize(yaw, y_edges)
    chosen = []
    for c in range(9):
        options = np.where(cell == c)[0]
        if len(options):
            chosen.append(rng.choice(options))
    remaining = np.setdiff1d(np.arange(len(pitch)), chosen)
    while len(chosen) < N_CALIBRATION:
        chosen.append(rng.choice(remaining))
        remaining = np.setdiff1d(remaining, chosen)
    return np.array(chosen)


def calibrate_offsets(pred, truth):
    """Fixed pitch/yaw offset (normalized space) = mean error on the calibration frames."""
    return (truth - pred).mean(axis=0)


# ---------- main evaluation ----------

def evaluate_person(person, data, pred_all, rng):
    p = data["person_index"][person]
    r = np.where((data["person"] == p) & (data["eye"] == 0))[0]
    l = np.where((data["person"] == p) & (data["eye"] == 1))[0]
    # Pair the two eyes of each frame.
    r = r[np.lexsort((data["sample_index"][r], data["file_path"][r]))]
    l = l[np.lexsort((data["sample_index"][l], data["file_path"][l]))]
    assert np.array_equal(data["file_path"][r], data["file_path"][l])
    assert np.array_equal(data["sample_index"][r], data["sample_index"][l])

    def unmirror(angles, idx):
        a = angles[idx].copy()
        a[data["eye"][idx] == 1, 1] *= -1
        return a

    pred = {e: unmirror(pred_all, idx) for e, idx in (("r", r), ("l", l))}
    true = {e: unmirror(data["gaze"], idx) for e, idx in (("r", r), ("l", l))}
    rot = {"r": data["R"][r], "l": data["R"][l]}

    def to_camera(angles, e):
        v = angles_to_vectors(angles)
        return np.einsum("nji,nj->ni", rot[e], v)  # R^T v

    def fused_error(pr, pl, mask):
        pc = to_camera(pr, "r") + to_camera(pl, "l")
        tc = to_camera(true["r"], "r") + to_camera(true["l"], "l")
        return angle_between(pc[mask], tc[mask])

    all_frames = np.ones(len(r), dtype=bool)
    single_r = angle_between(to_camera(pred["r"], "r"), to_camera(true["r"], "r"))
    single_l = angle_between(to_camera(pred["l"], "l"), to_camera(true["l"], "l"))
    fused = fused_error(pred["r"], pred["l"], all_frames)

    # Calibration: 9 frames from one day, test on all other days.
    days = data["day"][r]
    eligible = [d for d in sorted(set(days)) if (days == d).sum() >= MIN_DAY_FRAMES]
    calib_errors, calib_baseline, head_diffs = [], [], []
    for repeat in range(N_REPEATS):
        if not eligible:
            break
        cal_day = eligible[rng.integers(len(eligible))]
        day_frames = np.where(days == cal_day)[0]
        truth_cam = to_camera(true["r"], "r") + to_camera(true["l"], "l")
        chosen = day_frames[pick_calibration(vectors_to_angles(truth_cam[day_frames]), rng)]
        test = days != cal_day
        if not test.any():
            continue

        corrected = {}
        for e in ("r", "l"):
            offset = calibrate_offsets(pred[e][chosen], true[e][chosen])
            corrected[e] = pred[e] + offset

        calib_errors.append(fused_error(corrected["r"], corrected["l"], test))
        calib_baseline.append(fused_error(pred["r"], pred["l"], test))

        cal_head = data["head_forward"][r][chosen].mean(axis=0)
        head_diffs.append(angle_between(data["head_forward"][r][test],
                                        np.repeat(cal_head[None], test.sum(), axis=0)))

    row = {
        "person_id": person,
        "frames": len(r),
        "single_eye_mae": float(np.concatenate([single_r, single_l]).mean()),
        "right_eye_mae": float(single_r.mean()),
        "left_eye_mae": float(single_l.mean()),
        "both_eyes_mae": float(fused.mean()),
    }
    if calib_errors:
        row["calibrated_other_days_mae"] = float(np.mean([e.mean() for e in calib_errors]))
        row["both_eyes_other_days_mae"] = float(np.mean([e.mean() for e in calib_baseline]))
        row["calibration_repeats"] = len(calib_errors)
    by_head = {
        "calibrated": np.concatenate(calib_errors) if calib_errors else np.array([]),
        "uncalibrated": np.concatenate(calib_baseline) if calib_baseline else np.array([]),
        "head_diff": np.concatenate(head_diffs) if head_diffs else np.array([]),
    }
    return row, by_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/final/models")
    args = parser.parse_args()

    config_path = Path(args.model_dir).parent / "config.json"
    with open(config_path) as file:
        config = json.load(file)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    people = list_people()
    print(f"Device: {device} | people: {len(people)} | models: {args.model_dir}")

    data = load_all(people)
    data["person_index"] = {p: i for i, p in enumerate(people)}
    attach_geometry(data, people)

    # Sanity check: R @ head rotation must reproduce the dataset's normalized head pose.
    rebuilt = Rotation.from_matrix(np.einsum("nij,njk->nik", data["R"], data["head_R"])).as_rotvec()
    pose_diff = np.abs(rebuilt - data["pose_raw"]).max(axis=1)
    mismatched = pose_diff > 1e-3
    print(f"Sanity check: normalized head pose rebuilt from annotations for {len(pose_diff)} eye images, "
          f"median difference {np.median(pose_diff):.1e}, {mismatched.sum()} above 0.001")
    for i in np.where(mismatched)[0]:
        print(f"  mismatch: {data['file_path'][i]} sample {data['sample_index'][i]} "
              f"({'right' if data['eye'][i] == 0 else 'left'} eye), difference {pose_diff[i]:.4f} rad")
    if mismatched.mean() > 0.001:
        raise RuntimeError("Normalization rotation does not match the dataset. Stop.")

    available = [p for p in people if (Path(args.model_dir) / f"lopo_{p}.pt").exists()]

    pred_all = np.zeros_like(data["gaze"])
    for person in available:
        idx = np.where(data["person"] == data["person_index"][person])[0]
        pred_all[idx] = predict_person(load_model(args.model_dir, person, config), data, idx, device)

    rng = np.random.default_rng(SEED)
    rows, head_parts = [], []
    for person in available:
        row, by_head = evaluate_person(person, data, pred_all, rng)
        rows.append(row)
        head_parts.append(by_head)
        print(f"{person}: single eye {row['single_eye_mae']:.2f}° | both eyes {row['both_eyes_mae']:.2f}°"
              + (f" | other days: {row['both_eyes_other_days_mae']:.2f}° -> calibrated "
                 f"{row['calibrated_other_days_mae']:.2f}°" if "calibrated_other_days_mae" in row else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_person.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def pe(key):
        vals = np.array([r[key] for r in rows if key in r])
        return vals.mean(), vals.std() / np.sqrt(len(vals)), len(vals)

    lines = []
    for key, label in [("single_eye_mae", "single eye"),
                       ("both_eyes_mae", "both eyes fused"),
                       ("both_eyes_other_days_mae", "both eyes, other days (no calibration)"),
                       ("calibrated_other_days_mae", "both eyes, other days, 9-frame calibration")]:
        m, se, n = pe(key)
        lines.append(f"{label:45s} {m:.2f}° (SE {se:.2f}, n={n})")

    # Calibration benefit by head-pose change.
    cal = np.concatenate([h["calibrated"] for h in head_parts])
    unc = np.concatenate([h["uncalibrated"] for h in head_parts])
    hd = np.concatenate([h["head_diff"] for h in head_parts])
    lines.append("\nHead turned away from calibration pose (pooled over people and repeats):")
    head_rows = []
    for lo, hi in zip(HEAD_BINS[:-1], HEAD_BINS[1:]):
        m = (hd >= lo) & (hd < hi)
        if m.sum() == 0:
            continue
        head_rows.append({"head_change": f"{lo}-{hi}", "n": int(m.sum()),
                          "uncalibrated_mae": float(unc[m].mean()), "calibrated_mae": float(cal[m].mean())})
        lines.append(f"  {lo:>3}-{hi:<3} deg  n={m.sum():<7} uncalibrated {unc[m].mean():.2f}°  calibrated {cal[m].mean():.2f}°")

    with open(OUT_DIR / "calibration_by_head_change.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(head_rows[0].keys()))
        writer.writeheader()
        writer.writerows(head_rows)

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    plot(rows, head_rows, OUT_DIR / "evaluation.png")
    print(f"\nSaved: {OUT_DIR}/per_person.csv, summary.txt, calibration_by_head_change.csv, evaluation.png")


def plot(rows, head_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    people = [r["person_id"] for r in rows]
    x = np.arange(len(rows))
    series = [("single_eye_mae", "single eye"), ("both_eyes_mae", "both eyes"),
              ("calibrated_other_days_mae", "both eyes + 9-frame calibration")]
    for k, (key, label) in enumerate(series):
        vals = [r.get(key, np.nan) for r in rows]
        axes[0].bar(x + (k - 1) * 0.27, vals, width=0.27, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(people, rotation=0, fontsize=8)
    axes[0].set_ylabel("Mean angular error (degrees)")
    axes[0].set_title("Per held-out person")
    axes[0].legend(fontsize=8, loc="lower right")

    labels = [h["head_change"].replace("-180", "+") + "°" for h in head_rows]
    hx = np.arange(len(head_rows))
    axes[1].plot(hx, [h["uncalibrated_mae"] for h in head_rows], marker="o", label="no calibration")
    axes[1].plot(hx, [h["calibrated_mae"] for h in head_rows], marker="o", label="calibrated")
    axes[1].set_xticks(hx)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Head turned vs calibration pose")
    axes[1].set_title("Does calibration survive head movement?")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
