import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


RESULTS_CSV = "lopo_sample_results.csv"
OUTPUT_CSV = "pose_results.csv"
OUTPUT_PLOT = "pose_error.png"

# Bin edges in degrees of head rotation away from the camera.
BIN_EDGES = [0, 10, 20, 30, 40, 180]
MIN_SAMPLES_PER_PERSON = 50


def load_results(path):
    person_ids = []
    poses = []
    gazes = []
    errors = []

    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            person_ids.append(row["person_id"])
            poses.append(
                [float(row["pose_0"]), float(row["pose_1"]), float(row["pose_2"])]
            )
            gazes.append([float(row["gaze_pitch"]), float(row["gaze_yaw"])])
            errors.append(float(row["error_degrees"]))

    return (
        np.array(person_ids),
        np.array(poses),
        np.array(gazes),
        np.array(errors),
    )


def head_angle_degrees(poses):
    # pose is a rotation vector, not three angles.
    # Column 2 of the rotation matrix is the direction the face points.
    # Its z part is the cosine of the angle away from the camera axis.
    rotations = Rotation.from_rotvec(poses).as_matrix()
    face_z = rotations[:, 2, 2]
    face_z = np.clip(face_z, -1.0, 1.0)
    return np.degrees(np.arccos(face_z))


def gaze_angle_degrees(gazes):
    # Same idea as metrics.py: angle between the gaze and looking straight
    # at the camera. For pitch p and yaw y that cosine is cos(p) * cos(y).
    pitch = gazes[:, 0]
    yaw = gazes[:, 1]
    cosine = np.clip(np.cos(pitch) * np.cos(yaw), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def summarize_bins(person_ids, head_angles, gaze_angles, errors):
    bin_index = np.digitize(head_angles, BIN_EDGES[1:-1])
    people = sorted(set(person_ids))
    rows = []

    for i in range(len(BIN_EDGES) - 1):
        in_bin = bin_index == i

        person_means = []
        for person in people:
            mask = in_bin & (person_ids == person)
            if mask.sum() >= MIN_SAMPLES_PER_PERSON:
                person_means.append(errors[mask].mean())

        rows.append(
            {
                "bin": f"{BIN_EDGES[i]}-{BIN_EDGES[i + 1]}",
                "n_samples": int(in_bin.sum()),
                "n_people": len(person_means),
                "pooled_mae": errors[in_bin].mean(),
                "person_equal_mae": np.mean(person_means),
                "mean_gaze_angle": gaze_angles[in_bin].mean(),
            }
        )

    return rows


def count_people_with_trend(person_ids, head_angles, errors, low=10, high=25):
    # For each person: is error higher when the head is turned (>= high)
    # than when it faces the camera (< low)?
    worse = 0
    checked = 0

    for person in sorted(set(person_ids)):
        mine = person_ids == person
        frontal = mine & (head_angles < low)
        turned = mine & (head_angles >= high)

        if frontal.sum() < MIN_SAMPLES_PER_PERSON or turned.sum() < MIN_SAMPLES_PER_PERSON:
            print(f"{person}: skipped (frontal {frontal.sum()}, turned {turned.sum()})")
            continue

        checked += 1
        difference = errors[turned].mean() - errors[frontal].mean()
        if difference > 0:
            worse += 1
        print(f"{person}: frontal {errors[frontal].mean():.2f}°, turned {errors[turned].mean():.2f}°, diff {difference:+.2f}°")

    return worse, checked


def save_table(rows, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bins(rows, path):
    labels = [row["bin"] + "°" for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [row["person_equal_mae"] for row in rows], marker="o", label="person-equal MAE")
    ax.plot(x, [row["pooled_mae"] for row in rows], marker="s", linestyle="--", label="pooled MAE")

    for i, row in enumerate(rows):
        ax.annotate(f"n={row['n_samples']}", (x[i], row["pooled_mae"]),
                    textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Head rotation away from camera")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title("LOPO error by head pose (15 held-out people)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    person_ids, poses, gazes, errors = load_results(RESULTS_CSV)
    print("Samples loaded:", len(errors))

    head_angles = head_angle_degrees(poses)
    gaze_angles = gaze_angle_degrees(gazes)

    print("Head angle percentiles (5, 50, 95):", np.percentile(head_angles, [5, 50, 95]).round(1))

    rows = summarize_bins(person_ids, head_angles, gaze_angles, errors)

    print("\nbin       n       people  pooled  person-eq  gaze")
    for row in rows:
        print(
            f"{row['bin']:<9} {row['n_samples']:<7} {row['n_people']:<7} "
            f"{row['pooled_mae']:.2f}°  {row['person_equal_mae']:.2f}°     "
            f"{row['mean_gaze_angle']:.1f}°"
        )

    print("\nPer-person check (frontal < 10° vs turned >= 25°):")
    worse, checked = count_people_with_trend(person_ids, head_angles, errors)
    print(f"Error is higher when turned for {worse} of {checked} people")

    save_table(rows, OUTPUT_CSV)
    plot_bins(rows, OUTPUT_PLOT)
    print(f"\nSaved: {OUTPUT_CSV}, {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
