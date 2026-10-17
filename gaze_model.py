"""GazeMap: shared data loading and gaze network.

Following Zhang et al. (MPIIGaze, CVPR 2015 / TPAMI):
- both eyes are used; left-eye images are mirrored so they look like right eyes
- the network can be widened to the paper's size (20 / 50 / 500)
- head pose (2 angles) can be fed into the last layer
- all data is loaded into memory once, so training is fast
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.spatial.transform import Rotation


BASE = Path("./dataset/Data/Normalized")


# ---------- label and pose conversion ----------

def gaze_to_angles(gaze):
    # Same convention as metrics.py: pitch = arcsin(-y), yaw = arctan2(-x, -z).
    gaze = gaze / np.linalg.norm(gaze, axis=1, keepdims=True)
    x, y, z = gaze[:, 0], gaze[:, 1], gaze[:, 2]
    pitch = np.arcsin(np.clip(-y, -1.0, 1.0))
    yaw = np.arctan2(-x, -z)
    return np.stack([pitch, yaw], axis=1)


def pose_to_angles(pose):
    # pose is a rotation vector. Column 2 of its rotation matrix is the
    # direction the face points; turn that into head pitch and yaw.
    forward = Rotation.from_rotvec(pose).as_matrix()[:, :, 2]
    pitch = np.arcsin(np.clip(forward[:, 1], -1.0, 1.0))
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    return np.stack([pitch, yaw], axis=1)


# ---------- data ----------

def list_people():
    return sorted(p.name for p in BASE.iterdir() if p.is_dir() and p.name.startswith("p"))


def load_all(people, eyes=("right", "left")):
    """Load every sample of the given people into memory.

    Returns a dict of numpy arrays, one row per eye image:
      images      uint8   (N, 36, 60)   left eyes already mirrored
      gaze        float32 (N, 2)        pitch, yaw in radians (yaw mirrored for left)
      head        float32 (N, 2)        head pitch, yaw in radians (yaw mirrored for left)
      person      int64   (N,)          index into `people`
      eye         int64   (N,)          0 = right, 1 = left
      pose_raw    float32 (N, 3)        original rotation vector, for later analysis
      file_path   object  (N,)
      sample_index int64  (N,)
    """
    parts = {k: [] for k in ["images", "gaze", "head", "person", "eye",
                             "pose_raw", "file_path", "sample_index"]}

    for person_index, person in enumerate(people):
        for path in sorted((BASE / person).rglob("*.mat")):
            data = loadmat(str(path), struct_as_record=False)["data"][0, 0]

            for eye_index, eye in enumerate(["right", "left"]):
                if eye not in eyes:
                    continue

                side = getattr(data, eye)[0, 0]
                images = np.asarray(side.image, dtype=np.uint8).reshape(-1, 36, 60)
                gaze_raw = np.asarray(side.gaze, dtype=np.float64).reshape(-1, 3)
                pose_raw = np.asarray(side.pose, dtype=np.float64).reshape(-1, 3)
                n = len(images)

                if len(gaze_raw) != n or len(pose_raw) != n:
                    raise ValueError(f"Length mismatch in {path} ({eye})")

                gaze = gaze_to_angles(gaze_raw)
                head = pose_to_angles(pose_raw)

                if eye == "left":
                    # Mirror left to right: flip columns, negate both yaws.
                    images = images[:, :, ::-1]
                    gaze[:, 1] *= -1
                    head[:, 1] *= -1

                parts["images"].append(np.ascontiguousarray(images))
                parts["gaze"].append(gaze.astype(np.float32))
                parts["head"].append(head.astype(np.float32))
                parts["person"].append(np.full(n, person_index, dtype=np.int64))
                parts["eye"].append(np.full(n, eye_index, dtype=np.int64))
                parts["pose_raw"].append(pose_raw.astype(np.float32))
                parts["file_path"].append(np.array([str(path)] * n, dtype=object))
                parts["sample_index"].append(np.arange(n, dtype=np.int64))

    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------- model ----------

class GazeNet(nn.Module):
    """Two conv layers + two fully connected layers.

    conv1=8, conv2=16, hidden=16, use_head_pose=False is the original small baseline network.
    conv1=20, conv2=50, hidden=500, use_head_pose=True is the paper's MnistNet.
    """

    def __init__(self, conv1=8, conv2=16, hidden=16, use_head_pose=False):
        super().__init__()
        self.use_head_pose = use_head_pose
        self.conv1 = nn.Conv2d(1, conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1, conv2, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(conv2 * 6 * 12, hidden)
        self.fc2 = nn.Linear(hidden + (2 if use_head_pose else 0), 2)

    def forward(self, x, head_pose=None):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))

        if self.use_head_pose:
            x = torch.cat([x, head_pose], dim=1)

        return self.fc2(x)


def build_model(config):
    return GazeNet(
        conv1=config["conv1"],
        conv2=config["conv2"],
        hidden=config["hidden"],
        use_head_pose=config["head_pose"],
    )


def to_batch(images_uint8, device):
    # uint8 (B, 36, 60) -> float (B, 1, 36, 60) in 0-1
    return (images_uint8.float() / 255.0).unsqueeze(1).to(device)
