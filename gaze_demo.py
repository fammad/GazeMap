"""GazeMap live demo: webcam -> face landmarks -> normalized eye images -> CNN -> gaze arrows.

Pipeline (same normalization as MPIIGaze, Zhang et al.):
  1. MediaPipe finds 6 face points (4 eye corners, 2 mouth corners).
  2. solvePnP fits the dataset's 6-point 3D face model -> head rotation + position.
  3. Each eye is warped into a virtual camera 600 mm away, looking straight at
     the eye, 60x36 pixels, grey, histogram-equalized. This is exactly how the
     training images were made (checked against the dataset: pixel correlation 0.998).
  4. The trained fold models predict pitch/yaw in that normalized space
     (all models in the folder are averaged).
  5. A 13-dot calibration fits a per-eye offset, then an affine or quadratic
     mapping on the screen (chosen by leave-one-out error).
  6. The prediction is rotated back to the camera and drawn: red = each eye,
     green = both eyes combined.

Usage:
  python gaze_demo.py                         # webcam with the final models
  python gaze_demo.py --calibrate             # run the 9-dot calibration first
  python gaze_demo.py --heatmap               # where you look on the screen, as a heatmap
  python gaze_demo.py --heatmap --stimulus poster.png
  python gaze_demo.py --heatmap --stimulus my_photos/ --seconds 10   # slideshow, one heatmap per image
  python gaze_demo.py --video stimuli/demo_video.mp4   # watch a video, saves a gaze-trail video
  python gaze_demo.py --test                  # 16-dot accuracy map with error arrows
  python gaze_demo.py --selftest              # dataset check, no camera needed
  python gaze_demo.py --image photo.jpg       # one image, saves photo_gaze.jpg

Keys (camera view): q = quit, c = calibrate, x = clear calibration, m = mirror view, s = smoothing.
Keys (heatmap): q = quit, r = reset, p = pause, s = save PNG + CSV, v = start/stop video, c = calibrate.
Keys (video): space = pause, q = stop early (what was recorded is still saved).
"""

import argparse
import bisect
import json
from collections import deque
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from gaze_model import GazeNet, pose_to_angles


FACE_MODEL_PATH = Path("dataset/6 points-based face model.mat")
LANDMARKER_PATH = Path("face_landmarker.task")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
CALIBRATION_PATH = Path("calibration.json")
MODELS_URL = "https://github.com/fammad/GazeMap/releases/download/models/gazemap_models.zip"
HEATMAP_DIR = Path("heatmaps")
HEAT_SCALE = 8  # the heat grid is 1/8 of the window resolution
TRAIL_SECONDS = 1.0  # video mode: how much recent gaze the trail shows
VIDEO_OUT_WIDTH = 1280
PANEL_ASPECT = 2 / 3  # webcam panel next to the video: width / height
PANEL_ZOOM = 0.9      # the panel shows this fraction of the webcam height
PANEL_FACE_UP = 0.18  # face sits this far above the panel centre, so shoulders and desk show

# MediaPipe face-mesh indices in the same order as the dataset's face model columns:
# right eye outer, right eye inner, left eye inner, left eye outer,
# right mouth corner, left mouth corner ("right" = the person's right).
MEDIAPIPE_IDS = [33, 133, 362, 263, 61, 291]

FOCAL_NORM = 960.0
DISTANCE_NORM = 600.0
EYE_SIZE = (60, 36)  # width, height

# 13 calibration dots as fractions of the screen (x from the left, y from the top):
# a 3x3 grid close to the edges plus 4 inner points.
CALIBRATION_DOTS = ([(x, y) for y in (0.06, 0.5, 0.94) for x in (0.06, 0.5, 0.94)]
                    + [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)])
# 5 different dots to measure the real accuracy after calibrating.
VALIDATION_DOTS = [(0.25, 0.15), (0.75, 0.15), (0.5, 0.5), (0.25, 0.85), (0.75, 0.85)]
MAPPING_RIDGE = 0.001  # tiny pull towards "no change", only to keep the fit stable
# 16 test dots, one in the middle of each cell of a 4x4 grid.
TEST_DOTS = [((c + 0.5) / 4, (r + 0.5) / 4) for r in range(4) for c in range(4)]
# Warn when the head has moved this much since calibration.
DRIFT_MM = 50.0
DRIFT_DEG = 12.0
DRIFT_WINDOW = 15  # frames (about 1 s); the warning uses the median, so short wobbles do not trigger it
# A more flexible correction is only used if it beats the simpler one by this much.
SELECT_MARGIN_DEG = 0.2
SETTLE_SECONDS = 1.0   # wait for the eyes to land on the dot
COLLECT_SECONDS = 1.5  # then record frames


# ---------- models ----------

def download_models(model_dir):
    """First run only: fetch the 15 trained fold models (about 100 MB) from the GitHub release."""
    print(f"No trained models in {model_dir}. Downloading them once from\n  {MODELS_URL}")
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / "gazemap_models.zip"
    try:
        urllib.request.urlretrieve(MODELS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if Path(name).name.startswith("lopo_p") and name.endswith(".pt"):
                    (model_dir / Path(name).name).write_bytes(zf.read(name))
    except Exception as error:
        print(f"Download failed ({error}). Get gazemap_models.zip from the repository's Releases "
              f"page and unzip it into {model_dir}, or train your own with train_lopo.py.")
    finally:
        archive.unlink(missing_ok=True)


def load_gaze_models(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir.parent / "config.json") as file:
        config = json.load(file)

    paths = sorted(model_dir.glob("lopo_p*.pt"))
    if not paths:
        download_models(model_dir)
        paths = sorted(model_dir.glob("lopo_p*.pt"))
    if not paths:
        raise FileNotFoundError(f"No lopo_p*.pt files in {model_dir}")

    models = []
    for path in paths:
        model = GazeNet(config["conv1"], config["conv2"], config["hidden"], config["head_pose"])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} models from {model_dir} "
          f"(network {config['conv1']}/{config['conv2']}/{config['hidden']}, head pose: {config['head_pose']})")
    return models


def predict(models, eye_images, head_angles):
    """eye_images: uint8 (N, 36, 60); head_angles: float (N, 2). Returns (N, 2) pitch, yaw."""
    x = torch.from_numpy(eye_images).float().div(255.0).unsqueeze(1)
    h = torch.from_numpy(head_angles.astype(np.float32))
    with torch.no_grad():
        outputs = torch.stack([m(x, h) for m in models])
    return outputs.mean(dim=0).numpy()


# ---------- geometry ----------

def load_face_model():
    return loadmat(str(FACE_MODEL_PATH))["model"].astype(np.float64)  # (3, 6), mm


def estimate_head_pose(points_2d, face_model, camera_matrix):
    object_points = face_model.T.reshape(-1, 1, 3)
    image_points = points_2d.reshape(-1, 1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None,
                                  flags=cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None,
                                  rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def rotation_angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


class HeadPoseTracker:
    """Head pose that stays consistent from frame to frame.

    Fitting 6 almost-flat face points is ambiguous: with a few pixels of
    landmark noise, a single-frame fit sometimes flips to a wrong solution
    30-50 degrees away. The tracker starts each fit from the previous pose,
    rejects sudden jumps, and smooths the result.
    """

    def __init__(self, smoothing=0.5, max_jump_deg=15.0, max_rejects=10):
        self.smoothing = smoothing
        self.max_jump_deg = max_jump_deg
        self.max_rejects = max_rejects
        self.reset()

    def reset(self):
        self.rvec = self.tvec = self.R = self.t = None
        self.rejects = 0

    def update(self, points_2d, face_model, camera_matrix):
        if self.R is None:
            R, t = estimate_head_pose(points_2d, face_model, camera_matrix)
            self.rvec, self.tvec = cv2.Rodrigues(R)[0], t.reshape(3, 1)
            self.R, self.t = R, t
            return R, t

        object_points = face_model.T.reshape(-1, 1, 3)
        image_points = points_2d.reshape(-1, 1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None,
                                      self.rvec.copy(), self.tvec.copy(), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        R_new, t_new = cv2.Rodrigues(rvec)[0], tvec.reshape(3)

        if rotation_angle_deg(self.R, R_new) > self.max_jump_deg and self.rejects < self.max_rejects:
            self.rejects += 1  # most likely a flipped fit: keep the previous rotation
            R_new = self.R
        else:
            self.rejects = 0

        a = self.smoothing
        self.R = orthonormalize(a * self.R + (1 - a) * R_new)
        self.t = a * self.t + (1 - a) * t_new
        self.rvec, self.tvec = cv2.Rodrigues(self.R)[0], self.t.reshape(3, 1).copy()
        return self.R, self.t


def eye_centers(head_rotation, head_translation, face_model):
    face_3d = head_rotation @ face_model + head_translation.reshape(3, 1)
    right = face_3d[:, 0:2].mean(axis=1)
    left = face_3d[:, 2:4].mean(axis=1)
    return right, left


def normalize_eye(image_bgr, camera_matrix, head_rotation, center):
    """Warp one eye into the normalized camera. Returns (60x36 uint8, R)."""
    distance = np.linalg.norm(center)
    forward = center / distance
    down = np.cross(forward, head_rotation[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.stack([right, down, forward])

    S = np.diag([1.0, 1.0, DISTANCE_NORM / distance])
    camera_norm = np.array([[FOCAL_NORM, 0, EYE_SIZE[0] / 2],
                            [0, FOCAL_NORM, EYE_SIZE[1] / 2],
                            [0, 0, 1.0]])
    W = camera_norm @ S @ R @ np.linalg.inv(camera_matrix)

    warped = cv2.warpPerspective(image_bgr, W, EYE_SIZE)
    grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(grey), R


def angles_to_vector(pitch, yaw):
    # Same convention as metrics.py / training labels.
    return np.array([-np.cos(pitch) * np.sin(yaw),
                     -np.sin(pitch),
                     -np.cos(pitch) * np.cos(yaw)])


def vector_to_angles(v):
    v = v / np.linalg.norm(v)
    return np.array([np.arcsin(np.clip(-v[1], -1, 1)), np.arctan2(-v[0], -v[2])])


def estimate_gaze(image_bgr, camera_matrix, head_rotation, centers, models, offsets=None):
    """Returns one dict per eye: name, center, R, raw normalized angles, camera gaze, eye image."""
    eyes, heads, rotations = [], [], []

    for name, center in zip(["right", "left"], centers):
        eye_img, R = normalize_eye(image_bgr, camera_matrix, head_rotation, center)
        head_norm = cv2.Rodrigues(R @ head_rotation)[0].reshape(1, 3)
        head = pose_to_angles(head_norm)[0]

        if name == "left":  # the models only know mirrored left eyes
            eye_img = eye_img[:, ::-1]
            head = head * np.array([1, -1])

        eyes.append(np.ascontiguousarray(eye_img))
        heads.append(head)
        rotations.append(R)

    angles = predict(models, np.stack(eyes), np.stack(heads))

    results = []
    for i, (name, center) in enumerate(zip(["right", "left"], centers)):
        raw = angles[i].copy()
        if name == "left":
            raw[1] = -raw[1]
        corrected = raw + (offsets[name] if offsets else 0.0)
        gaze_camera = rotations[i].T @ angles_to_vector(*corrected)
        results.append({
            "name": name,
            "center": center,
            "R": rotations[i],
            "head_R": head_rotation,
            "raw": raw,
            "gaze": gaze_camera / np.linalg.norm(gaze_camera),
            "eye_img": eyes[i][:, ::-1] if name == "left" else eyes[i],
        })
    return results


def combined_gaze(results):
    """Both eyes together: average direction, starting between the eyes."""
    center = np.mean([r["center"] for r in results], axis=0)
    gaze = np.sum([r["gaze"] for r in results], axis=0)
    return center, gaze / np.linalg.norm(gaze)


# ---------- calibration ----------

class Screen:
    """Maps between window pixels and camera coordinates (mm) for a laptop screen.

    Assumes the camera sits at the top-centre of the screen, in the screen plane.
    Camera x points to the image right, which is the user's LEFT, so the
    horizontal axis is flipped. Millimetres per pixel come from the physical
    screen width, so a fullscreen window that stops below the notch still maps correctly.
    """

    def __init__(self, width_px, height_px, width_mm, camera_offset_mm):
        self.width_px, self.height_px = width_px, height_px
        self.width_mm = width_mm
        self.mm_per_px = width_mm / width_px
        self.camera_offset_mm = camera_offset_mm

    def to_camera(self, px, py):
        x = -(px / self.width_px - 0.5) * self.width_mm
        y = self.camera_offset_mm + py * self.mm_per_px
        return np.array([x, y, 0.0])

    def to_pixels(self, point_mm):
        px = (0.5 - point_mm[0] / self.width_mm) * self.width_px
        py = (point_mm[1] - self.camera_offset_mm) / self.mm_per_px
        return np.array([px, py])


def intersect_screen(center, gaze):
    """Where the gaze ray hits the screen plane (z = 0). None if it points away from the screen."""
    if gaze[2] > -1e-6:
        return None
    t = -center[2] / gaze[2]
    return center + t * gaze


def open_fullscreen(window):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, np.zeros((900, 1440, 3), np.uint8))
    for _ in range(15):  # give macOS time to finish the fullscreen animation
        cv2.waitKey(50)
    rect = cv2.getWindowImageRect(window)
    return (rect[2], rect[3]) if rect[2] > 100 else (1440, 900)


def calibration_target_angles(result, target_mm):
    """True gaze of one eye towards the target, in that eye's normalized space."""
    gaze = target_mm - result["center"]
    return vector_to_angles(result["R"] @ (gaze / np.linalg.norm(gaze)))


def fit_offsets(samples):
    """samples: (dot, eye name, raw angles, true angles).

    Offset per eye = mean over dots of the median error at each dot, so a blink
    or a glance away during one dot cannot pull the result.
    """
    offsets = {}
    for name in ("right", "left"):
        per_dot = {}
        for dot, n, raw, true in samples:
            if n == name:
                per_dot.setdefault(dot, []).append(true - raw)
        if len(per_dot) < 5:
            raise RuntimeError(f"Too few calibration dots for the {name} eye ({len(per_dot)}).")
        offsets[name] = np.mean([np.median(d, axis=0) for d in per_dot.values()], axis=0)
    return offsets


def with_offsets(results, offsets):
    """Recompute each eye's camera-space gaze from its raw prediction plus the offset."""
    out = []
    for r in results:
        corrected = r["raw"] + (offsets[r["name"]] if offsets else 0.0)
        gaze = r["R"].T @ angles_to_vector(*corrected)
        out.append({**r, "gaze": gaze / np.linalg.norm(gaze)})
    return out


def screen_fraction(results, screen):
    """Where the combined gaze hits the screen, as fractions of width and height."""
    center, gaze = combined_gaze(results)
    hit = intersect_screen(center, gaze)
    if hit is None:
        return None, None
    px = screen.to_pixels(hit)
    return np.array([px[0] / screen.width_px, px[1] / screen.height_px]), float(np.linalg.norm(center))


def mapping_features(fractions, method):
    f = np.atleast_2d(fractions)
    x, y, one = f[:, 0], f[:, 1], np.ones(len(f))
    columns = [x, y, one] if method == "affine" else [x, y, one, x * x, y * y, x * y]
    return np.column_stack(columns)


def fit_mapping(estimates, targets, method, ridge=MAPPING_RIDGE):
    """Least-squares screen-space correction, gently pulled towards 'no change'.

    affine    : x, y, 1                 (scale, shift, tilt per axis)
    quadratic : x, y, 1, x², y², xy     (can also bend: fixes a squeezed middle)
    """
    X = mapping_features(estimates, method)
    identity = np.zeros((X.shape[1], 2))
    identity[0, 0] = identity[1, 1] = 1.0
    A = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ targets + ridge * identity)
    return {"method": method, "A": A}


def apply_mapping(mapping, fraction):
    if mapping is None:
        return np.asarray(fraction, dtype=float)
    return (mapping_features(fraction, mapping["method"]) @ mapping["A"])[0]


def gaze_on_screen(results, offsets, mapping, screen):
    """Screen position (fractions, after calibration) and eye distance for one frame."""
    fraction, distance = screen_fraction(with_offsets(results, offsets), screen)
    if fraction is None:
        return None, None
    return apply_mapping(mapping, fraction), distance


def head_reference(frames):
    """Where the eyes were and how the head was oriented during calibration."""
    mids = [np.mean([r["center"] for r in res], axis=0) for res in frames]
    rotation = orthonormalize(np.mean([res[0]["head_R"] for res in frames], axis=0))
    return {"eye_mid_mm": np.median(mids, axis=0), "head_R": rotation}


def head_drift(results, reference):
    """(movement in mm, head turn in degrees) compared with the calibration pose."""
    if reference is None:
        return None
    mid = np.mean([r["center"] for r in results], axis=0)
    moved = float(np.linalg.norm(mid - reference["eye_mid_mm"]))
    if "head_R" in reference:
        turned = rotation_angle_deg(reference["head_R"], results[0]["head_R"])
    else:  # calibration file from an earlier version
        forward = results[0]["head_R"][:, 2]
        turned = float(np.degrees(np.arccos(np.clip(forward @ reference["head_forward"], -1, 1))))
    return moved, turned


class DriftMonitor:
    """Median head drift over the last second, so a short wobble does not trigger the warning."""

    def __init__(self):
        self.values = deque(maxlen=DRIFT_WINDOW)

    def update(self, drift):
        if drift is not None:
            self.values.append(drift)
        return tuple(np.median(self.values, axis=0)) if len(self.values) >= 5 else None


def draw_drift_warning(view, drift):
    if drift is None or (drift[0] < DRIFT_MM and drift[1] < DRIFT_DEG):
        return
    text = f"You moved {drift[0] / 10:.1f} cm / turned {drift[1]:.0f} deg since calibration: press c"
    cv2.rectangle(view, (0, 0), (view.shape[1], 50), (0, 0, 180), -1)
    cv2.putText(view, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def save_calibration(calib):
    mapping = calib.get("mapping")
    reference = calib.get("reference")
    info = {k: v for k, v in calib.get("info", {}).items()
            if k not in ("offsets_rad", "mapping", "reference", "points")}
    payload = {
        **info,
        "offsets_rad": {k: v.tolist() for k, v in calib["offsets"].items()},
        "mapping": {"method": mapping["method"], "A": mapping["A"].tolist()} if mapping else None,
        "reference": {k: np.asarray(v).tolist() for k, v in reference.items()} if reference else None,
        "points": [{"target": [float(v) for v in p["target"]], "estimate": [float(v) for v in p["estimate"]],
                    "distance": float(p["distance"]), "source": p["source"]} for p in calib.get("points") or []],
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))


def load_calibration():
    if not CALIBRATION_PATH.exists():
        return None
    payload = json.loads(CALIBRATION_PATH.read_text())
    mapping = payload.get("mapping")
    if mapping:
        mapping = {"method": mapping["method"], "A": np.array(mapping["A"])}
    elif payload.get("affine") is not None:  # file from the previous version
        mapping = {"method": "affine", "A": np.array(payload["affine"])}
    reference = payload.get("reference")
    points = [{"target": np.array(p["target"]), "estimate": np.array(p["estimate"]),
               "distance": p["distance"], "source": p["source"]} for p in payload.get("points", [])]
    info = {k: v for k, v in payload.items()
            if k not in ("offsets_rad", "mapping", "reference", "points", "affine")}
    return {"offsets": {k: np.array(v) for k, v in payload["offsets_rad"].items()},
            "mapping": mapping,
            "reference": {k: np.array(v) for k, v in reference.items()} if reference else None,
            "points": points, "info": info}


def show_message(window, width, height, lines, wait_keys=(" ",)):
    canvas = np.zeros((height, width, 3), np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(canvas, line, (60, height // 2 - 40 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
    cv2.imshow(window, canvas)
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            return "esc"
        for k in wait_keys:
            if key == ord(k):
                return k


def collect_dot(cap, detect, start_time, face_model, models, args, window, width, height, fx, fy):
    """Show one shrinking dot; return the eye results of every frame after the settle time."""
    frames = []
    dot_start = time.time()
    while time.time() - dot_start < SETTLE_SECONDS + COLLECT_SECONDS:
        elapsed = time.time() - dot_start
        canvas = np.zeros((height, width, 3), np.uint8)
        radius = int(18 - 10 * min(elapsed / (SETTLE_SECONDS + COLLECT_SECONDS), 1))
        center_px = (int(fx * width), int(fy * height))
        cv2.circle(canvas, center_px, radius, (255, 255, 255), -1)
        cv2.circle(canvas, center_px, 3, (0, 0, 255), -1)
        cv2.imshow(window, canvas)
        if (cv2.waitKey(1) & 0xFF) == 27:
            return None
        ok, frame = cap.read()
        if not ok or elapsed < SETTLE_SECONDS:
            continue
        h, w = frame.shape[:2]
        camera_matrix = default_camera_matrix(w, h, args.focal)
        out = process_frame(frame, detect, int((time.time() - start_time) * 1000),
                            face_model, camera_matrix, models, offsets=None)
        if out is not None:
            frames.append(out[0])
    return frames


def dot_estimate(frames, offsets, mapping, screen):
    """Median screen position and eye distance over the frames of one dot."""
    hits = [gaze_on_screen(res, offsets, mapping, screen) for res in frames]
    hits = [(p, d) for p, d in hits if p is not None]
    if len(hits) < 3:
        return None, None
    return np.median([p for p, _ in hits], axis=0), float(np.median([d for _, d in hits]))


METHOD_COLOURS = {"offset": (255, 140, 0), "affine": (0, 220, 0), "quadratic": (255, 0, 255)}
MIN_POINTS = {"offset": 0, "affine": 6, "quadratic": 12}


def choose_method(scores, margin=SELECT_MARGIN_DEG):
    """Simplest correction unless a more flexible one is clearly better (by `margin` degrees)."""
    chosen = "offset"
    for method in ("affine", "quadratic"):
        if method in scores and scores[method] < scores[chosen] - margin:
            chosen = method
    return chosen


def point_error_deg(estimate, target, distance, screen):
    err_mm = np.hypot((estimate[0] - target[0]) * screen.width_px,
                      (estimate[1] - target[1]) * screen.height_px) * screen.mm_per_px
    return float(np.degrees(np.arctan(err_mm / distance))), float(err_mm)


def leave_one_out_scores(points, screen):
    """Mean error of each correction on every point when that point is left out of the fit."""
    est = np.array([p["estimate"] for p in points])
    tgt = np.array([p["target"] for p in points])
    scores = {}
    for method, minimum in MIN_POINTS.items():
        if len(points) - 1 < minimum:
            continue
        errors = []
        for i, p in enumerate(points):
            keep = np.arange(len(points)) != i
            mapping = None if method == "offset" else fit_mapping(est[keep], tgt[keep], method)
            errors.append(point_error_deg(apply_mapping(mapping, est[i]), tgt[i], p["distance"], screen)[0])
        scores[method] = float(np.mean(errors))
    return scores


def refit_with_points(calib, new_points, screen):
    """Add measured dots to the calibration and refit the screen correction on all of them."""
    points = list(calib.get("points") or []) + list(new_points)
    scores = leave_one_out_scores(points, screen)
    method = choose_method(scores)
    est = np.array([p["estimate"] for p in points])
    tgt = np.array([p["target"] for p in points])
    calib = dict(calib)
    calib["mapping"] = None if method == "offset" else fit_mapping(est, tgt, method)
    calib["points"] = points
    info = dict(calib.get("info", {}))
    info.update({"method": method, "n_points": len(points),
                 "leave_one_out_error_deg": scores})
    calib["info"] = info
    save_calibration(calib)
    return calib, scores, method


def run_calibration(cap, detect, start_time, face_model, models, args):
    """13 dots to fit, then 5 new dots to measure the real accuracy.

    Step 1: a pitch/yaw offset per eye (median per dot, then averaged).
    Step 2: a screen-space correction of the combined gaze, either affine
            (scale, shift, tilt) or quadratic (can also bend).
    The 5 validation dots, which none of the fits have seen, decide which of
    offset / affine / quadratic is used.
    """
    window = "GazeMap calibration"
    width, height = open_fullscreen(window)
    screen = Screen(width, height, args.screen_width_mm, args.camera_offset_mm)
    print(f"Calibration window: {width} x {height}")

    while True:
        answer = show_message(window, width, height, [
            "Calibration: look at each white dot until it disappears.",
            "Keep your head still and natural. 13 dots, then 5 test dots.",
            "Press SPACE to start, ESC to cancel."])
        if answer == "esc":
            cv2.destroyWindow(window)
            return None

        dots = []
        for fx, fy in CALIBRATION_DOTS:
            frames = collect_dot(cap, detect, start_time, face_model, models, args, window, width, height, fx, fy)
            if frames is None:
                cv2.destroyWindow(window)
                return None
            dots.append((fx, fy, frames))

        samples = []
        for d, (fx, fy, frames) in enumerate(dots):
            target = screen.to_camera(fx * width, fy * height)
            for results in frames:
                for r in results:
                    samples.append((d, r["name"], r["raw"], calibration_target_angles(r, target)))
        try:
            offsets = fit_offsets(samples)
        except RuntimeError as error:
            if show_message(window, width, height, [str(error), "SPACE = try again, ESC = cancel"]) == "esc":
                cv2.destroyWindow(window)
                return None
            continue

        estimates, targets, points = [], [], []
        for fx, fy, frames in dots:
            estimate, distance = dot_estimate(frames, offsets, None, screen)
            if estimate is not None:
                estimates.append(estimate)
                targets.append([fx, fy])
                points.append({"target": np.array([fx, fy]), "estimate": estimate,
                               "distance": distance, "source": "calibration"})
        estimates, targets = np.array(estimates), np.array(targets)
        mappings = {"offset": None}
        if len(estimates) >= 6:
            mappings["affine"] = fit_mapping(estimates, targets, "affine")
        if len(estimates) >= 10:
            mappings["quadratic"] = fit_mapping(estimates, targets, "quadratic")
        reference = head_reference([res for _, _, frames in dots for res in frames])

        # validation dots: not used for any fit
        errors = {m: [] for m in mappings}
        markers = []
        for fx, fy in VALIDATION_DOTS:
            frames = collect_dot(cap, detect, start_time, face_model, models, args, window, width, height, fx, fy)
            if frames is None:
                break
            estimate, distance = dot_estimate(frames, offsets, None, screen)
            if estimate is None:
                continue
            target = np.array([fx, fy])
            points.append({"target": target, "estimate": estimate, "distance": distance, "source": "validation"})
            shown = {}
            for method, mapping in mappings.items():
                est = apply_mapping(mapping, estimate)
                err_mm = np.hypot((est[0] - target[0]) * width, (est[1] - target[1]) * height) * screen.mm_per_px
                errors[method].append((err_mm, np.degrees(np.arctan(err_mm / distance))))
                shown[method] = est
            markers.append((target, shown))

        mean = {m: (np.mean([e[0] for e in v]), np.mean([e[1] for e in v])) for m, v in errors.items() if v}
        method = choose_method({m: v[1] for m, v in mean.items()})

        canvas = np.zeros((height, width, 3), np.uint8)
        for target, shown in markers:
            cv2.circle(canvas, (int(target[0] * width), int(target[1] * height)), 12, (255, 255, 255), 2)
            for m, est in shown.items():
                cv2.circle(canvas, (int(est[0] * width), int(est[1] * height)), 8, METHOD_COLOURS[m], -1)
        lines = ["Validation (white = dot, blue = offset, green = affine, pink = quadratic):"]
        for m, (mm, deg) in mean.items():
            lines.append(f"  {m:9s} mean error {mm / 10:.1f} cm = {deg:.1f} deg")
        lines.append(f"Using: {method} (a more flexible fit must win by {SELECT_MARGIN_DEG} deg).")
        lines.append("SPACE = accept,  R = redo calibration")
        for k, line in enumerate(lines):
            cv2.putText(canvas, line, (60, 80 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(window, canvas)
        answer = None
        while answer is None:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), 27):
                answer = "accept"
            elif key in (ord("r"), ord("R")):
                answer = "redo"
        if answer == "redo":
            continue

        calib = {
            "offsets": offsets,
            "mapping": mappings.get(method),
            "reference": reference,
            "points": points,
            "info": {"method": method, "screen_width_mm": args.screen_width_mm,
                     "camera_offset_mm": args.camera_offset_mm,
                     "validation_error_deg": {m: float(v[1]) for m, v in mean.items()},
                     "validation_error_cm": {m: float(v[0] / 10) for m, v in mean.items()}},
        }
        save_calibration(calib)
        print(f"Calibration saved to {CALIBRATION_PATH}: method {method}, validation error " +
              ", ".join(f"{m} {v[1]:.1f}° ({v[0] / 10:.1f} cm)" for m, v in mean.items()))
        cv2.destroyWindow(window)
        return calib


def run_test(args, models, face_model):
    """16 dots, one per grid cell; draws an arrow from each dot to where the gaze landed."""
    detect = make_landmarker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the camera. On macOS, allow camera access for Terminal "
                           "in System Settings > Privacy & Security > Camera.")
    start = time.time()
    calib = None if args.no_calibration else load_calibration()
    if args.calibrate or calib is None:
        calib = run_calibration(cap, detect, start, face_model, models, args) or calib
    offsets = calib["offsets"] if calib else None
    mapping = calib["mapping"] if calib else None
    reference = calib["reference"] if calib else None

    window = "GazeMap accuracy test"
    width, height = open_fullscreen(window)
    screen = Screen(width, height, args.screen_width_mm, args.camera_offset_mm)
    if show_message(window, width, height, [
            "Accuracy test: 16 dots, one in each cell. Sit the way you normally do.",
            "Press SPACE to start, ESC to cancel."]) == "esc":
        cap.release()
        cv2.destroyAllWindows()
        return

    rows, drifts = [], []
    for fx, fy in TEST_DOTS:
        frames = collect_dot(cap, detect, start, face_model, models, args, window, width, height, fx, fy)
        if frames is None:
            break
        raw_estimate, distance = dot_estimate(frames, offsets, None, screen)
        if frames and reference is not None:
            drifts.append(np.median([head_drift(res, reference) for res in frames], axis=0))
        if raw_estimate is None:
            continue
        estimate = apply_mapping(mapping, raw_estimate)
        err_mm = np.array([(estimate[0] - fx) * width, (estimate[1] - fy) * height]) * screen.mm_per_px
        err_deg = float(np.degrees(np.arctan(np.linalg.norm(err_mm) / distance)))
        # precision: how much single frames scatter around this dot's median
        frame_points = [gaze_on_screen(res, offsets, mapping, screen)[0] for res in frames]
        frame_points = np.array([q for q in frame_points if q is not None])
        spread_mm = np.sqrt(np.mean(np.sum(((frame_points - estimate) * np.array([width, height])
                                            * screen.mm_per_px) ** 2, axis=1)))
        precision_deg = float(np.degrees(np.arctan(spread_mm / distance)))
        rows.append({"target": np.array([fx, fy]), "estimate": estimate, "raw_estimate": raw_estimate,
                     "distance": distance, "err_mm": err_mm, "err_deg": err_deg, "precision_deg": precision_deg})

    if not rows:
        print("No face found during the test.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # diagnosis
    errs = np.array([r["err_mm"] for r in rows])
    shift = errs.mean(axis=0)
    residual = np.linalg.norm(errs - shift, axis=1).mean()
    centre = np.array([0.5 * width, 0.5 * height]) * screen.mm_per_px
    outward = []
    for r in rows:
        direction = r["target"] * np.array([width, height]) * screen.mm_per_px - centre
        outward.append(r["err_mm"] @ (direction / np.linalg.norm(direction)))
    outward = float(np.mean(outward))
    mean_deg = float(np.mean([r["err_deg"] for r in rows]))
    mean_cm = float(np.linalg.norm(errs, axis=1).mean() / 10)
    precision = float(np.mean([r["precision_deg"] for r in rows]))
    drift = np.mean(drifts, axis=0) if drifts else None

    canvas = make_background(width, height, None)
    for r in rows:
        t = (int(r["target"][0] * width), int(r["target"][1] * height))
        e = (int(np.clip(r["estimate"][0], -0.2, 1.2) * width), int(np.clip(r["estimate"][1], -0.2, 1.2) * height))
        colour = (0, 200, 0) if r["err_deg"] < 1.5 else ((0, 220, 255) if r["err_deg"] < 3 else (0, 0, 255))
        cv2.circle(canvas, t, 10, (255, 255, 255), 2)
        cv2.arrowedLine(canvas, t, e, colour, 3, tipLength=0.15)
        cv2.putText(canvas, f"{r['err_deg']:.1f}", (t[0] + 14, t[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    vertical = "lower" if shift[1] > 0 else "higher"
    horizontal = "right" if shift[0] > 0 else "left"
    lines = [
        f"Mean error {mean_deg:.1f} deg ({mean_cm:.1f} cm), calibration: {calib['mapping']['method'] if mapping else 'offset'}",
        f"Whole map shifted {abs(shift[1]) / 10:.1f} cm {vertical} and {abs(shift[0]) / 10:.1f} cm {horizontal}",
        f"Error left after removing that shift: {residual / 10:.1f} cm",
        f"Points pushed {'outward from' if outward > 0 else 'in towards'} the middle by {abs(outward) / 10:.1f} cm on average",
        f"Precision (frame-to-frame wobble while holding still): {precision:.1f} deg",
    ]
    if drift is not None:
        lines.append(f"Head since calibration: moved {drift[0] / 10:.1f} cm, turned {drift[1]:.0f} deg")
    lines.append("A = add these 16 dots to the calibration and refit,  any other key = close")
    top = height - 40 * len(lines) - 40
    panel = canvas.copy()
    cv2.rectangle(panel, (20, top - 10), (min(width - 20, 1150), height - 15), (0, 0, 0), -1)
    canvas = cv2.addWeighted(panel, 0.75, canvas, 0.25, 0)
    for k, line in enumerate(lines):
        cv2.putText(canvas, line, (40, top + 30 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    HEATMAP_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(HEATMAP_DIR / f"test_{stamp}.png"), canvas)
    with open(HEATMAP_DIR / f"test_{stamp}.csv", "w") as file:
        file.write("target_x,target_y,estimate_x,estimate_y,error_x_mm,error_y_mm,error_deg,precision_deg\n")
        for r in rows:
            file.write(f"{r['target'][0]:.3f},{r['target'][1]:.3f},{r['estimate'][0]:.4f},{r['estimate'][1]:.4f},"
                       f"{r['err_mm'][0]:.1f},{r['err_mm'][1]:.1f},{r['err_deg']:.2f},{r['precision_deg']:.2f}\n")
    print("\n".join(lines[:-1]))
    print(f"Saved {HEATMAP_DIR}/test_{stamp}.png and .csv")

    cv2.imshow(window, canvas)
    key = cv2.waitKey(0) & 0xFF
    if key in (ord("a"), ord("A")) and calib is not None:
        new_points = [{"target": r["target"], "estimate": r["raw_estimate"], "distance": r["distance"],
                       "source": "test"} for r in rows]
        calib, scores, method = refit_with_points(calib, new_points, screen)
        summary = ", ".join(f"{m} {v:.1f} deg" for m, v in scores.items())
        print(f"Refitted on {len(calib['points'])} dots. Leave-one-out error: {summary}. Using: {method}")
        show_message(window, width, height, [
            f"Refitted on {len(calib['points'])} dots (each one tested while left out of the fit):",
            f"  {summary}",
            f"Using: {method}. Run --test again to check.  SPACE = close"])
    cap.release()
    cv2.destroyAllWindows()


# ---------- drawing ----------

def project(camera_matrix, points):
    projected, _ = cv2.projectPoints(np.asarray(points, float), np.zeros(3), np.zeros(3), camera_matrix, None)
    return projected.reshape(-1, 2).astype(int)


def draw(frame, camera_matrix, results, arrow_mm=150.0):
    for r in results:
        start, end = project(camera_matrix, [r["center"], r["center"] + arrow_mm * r["gaze"]])
        cv2.circle(frame, tuple(start), 3, (0, 255, 255), -1)
        cv2.arrowedLine(frame, tuple(start), tuple(end), (0, 0, 255), 2, tipLength=0.2)
    center, gaze = combined_gaze(results)
    start, end = project(camera_matrix, [center, center + 1.5 * arrow_mm * gaze])
    cv2.arrowedLine(frame, tuple(start), tuple(end), (0, 220, 0), 3, tipLength=0.15)


def draw_eye_patches(frame, results):
    y = 10
    for r in results:
        patch = cv2.resize(r["eye_img"], (180, 108), interpolation=cv2.INTER_NEAREST)
        frame[y:y + 108, 10:190] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, r["name"], (14, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 116


def gaze_text(results):
    _, gaze = combined_gaze(results)
    pitch, yaw = np.degrees(vector_to_angles(gaze))
    return f"pitch {pitch:+.0f} deg  yaw {yaw:+.0f} deg"


# ---------- landmarks ----------

def make_landmarker():
    import mediapipe as mp

    if not LANDMARKER_PATH.exists():
        print("Downloading MediaPipe face landmarker model (about 4 MB)...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    last = {"ts": -1}

    def detect(frame_bgr, timestamp_ms):
        timestamp_ms = max(timestamp_ms, last["ts"] + 1)  # must strictly increase
        last["ts"] = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        marks = result.face_landmarks[0]
        return np.array([[marks[i].x * w, marks[i].y * h] for i in MEDIAPIPE_IDS])

    detect.pose_tracker = HeadPoseTracker()
    return detect


def default_camera_matrix(width, height, focal=None):
    f = focal if focal else float(width)  # rough guess for a laptop camera
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


# ---------- modes ----------

def process_frame(frame, detect, timestamp_ms, face_model, camera_matrix, models, offsets=None):
    points = detect(frame, timestamp_ms)
    tracker = getattr(detect, "pose_tracker", None)
    if points is None:
        if tracker is not None:
            tracker.reset()
        return None
    if tracker is not None:
        head_rotation, head_translation = tracker.update(points, face_model, camera_matrix)
    else:
        head_rotation, head_translation = estimate_head_pose(points, face_model, camera_matrix)
    centers = eye_centers(head_rotation, head_translation, face_model)
    return estimate_gaze(frame, camera_matrix, head_rotation, centers, models, offsets), points


def run_camera(args, models, face_model):
    detect = make_landmarker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the camera. On macOS, allow camera access for Terminal "
                           "in System Settings > Privacy & Security > Camera.")

    start = time.time()
    calib = None if args.no_calibration else load_calibration()
    if args.calibrate:
        calib = run_calibration(cap, detect, start, face_model, models, args) or calib
    offsets = calib["offsets"] if calib else None
    if offsets:
        print("Using calibration from", CALIBRATION_PATH)

    mirror, smooth = True, True
    drift_monitor = DriftMonitor()
    smoothed = {}
    last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        camera_matrix = default_camera_matrix(w, h, args.focal)
        out = process_frame(frame, detect, int((time.time() - start) * 1000),
                            face_model, camera_matrix, models, offsets)

        if out is not None:
            results, points = out
            if smooth:
                for r in results:
                    previous = smoothed.get(r["name"], r["gaze"])
                    r["gaze"] = 0.5 * previous + 0.5 * r["gaze"]
                    r["gaze"] /= np.linalg.norm(r["gaze"])
                    smoothed[r["name"]] = r["gaze"]
            for p in points:
                cv2.circle(frame, tuple(p.astype(int)), 2, (255, 200, 0), -1)
            draw(frame, camera_matrix, results)
            text = gaze_text(results)
        else:
            text = "no face"

        view = cv2.flip(frame, 1) if mirror else frame
        if out is not None:
            draw_eye_patches(view, results)
            draw_drift_warning(view, drift_monitor.update(head_drift(results, calib["reference"] if calib else None)))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        status = "calibrated" if offsets else "not calibrated"
        cv2.putText(view, f"{text}  {fps:.0f} fps  {status}  [q]uit [c]alibrate [x]clear [m]irror [s]mooth",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("GazeMap", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mirror = not mirror
        if key == ord("s"):
            smooth = not smooth
            smoothed.clear()
        if key == ord("c"):
            calib = run_calibration(cap, detect, start, face_model, models, args) or calib
            offsets = calib["offsets"] if calib else None
            smoothed.clear()
        if key == ord("x"):
            offsets = None
            smoothed.clear()

    cap.release()
    cv2.destroyAllWindows()


def fit_image(image, width, height, fill=30):
    """Scale an image to fit the window, centred. Returns the canvas and (x0, y0, scale)."""
    canvas = np.full((height, width, 3), fill, np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas, (x0, y0, scale)


def make_background(width, height, stimulus=None):
    """What the user looks at: an image scaled to fit, or a plain labelled grid."""
    if stimulus:
        image = cv2.imread(stimulus)
        if image is None:
            raise FileNotFoundError(stimulus)
        return fit_image(image, width, height)[0]

    canvas = np.full((height, width, 3), 30, np.uint8)

    for k in range(1, 4):
        cv2.line(canvas, (k * width // 4, 0), (k * width // 4, height), (70, 70, 70), 1)
        cv2.line(canvas, (0, k * height // 4), (width, k * height // 4), (70, 70, 70), 1)
    for row in range(4):
        for col in range(4):
            label = "ABCD"[row] + str(col + 1)
            cv2.putText(canvas, label, (col * width // 4 + 20, row * height // 4 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (110, 110, 110), 2)
    return canvas


IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_stimuli(path):
    """One image, or every image in a folder (sorted by name)."""
    if not path:
        return [None]
    path = Path(path)
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_TYPES)
        if not images:
            raise FileNotFoundError(f"No images in {path}")
        return [str(p) for p in images]
    return [str(path)]


def save_heatmap(base, heat, blob_px, points, name):
    HEATMAP_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{name}_{stamp}"
    cv2.imwrite(str(HEATMAP_DIR / f"{stem}.png"), render_heatmap(base, heat, blob_px))
    with open(HEATMAP_DIR / f"{stem}_points.csv", "w") as file:
        file.write("seconds,x_px,y_px\n")
        file.writelines(f"{t:.3f},{x:.1f},{y:.1f}\n" for t, x, y in points)
    print(f"Saved {HEATMAP_DIR}/{stem}.png and _points.csv ({len(points)} samples)")


def render_heatmap(base, heat, blob_px, alpha=0.6):
    """Blur the gaze counts into soft blobs, colour them, and lay them over the background."""
    sigma = max(blob_px / HEAT_SCALE, 1.0)
    blurred = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma)
    if blurred.max() <= 0:
        return base.copy()
    level = (blurred / blurred.max() * 255).astype(np.uint8)
    level = cv2.resize(level, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    weight = level.astype(np.float32)[..., None] / 255.0 * alpha
    weight[level < 25] = 0  # leave cold areas untouched
    return (base * (1 - weight) + colour * weight).astype(np.uint8)


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the camera. On macOS, allow camera access for Terminal "
                           "in System Settings > Privacy & Security > Camera.")
    return cap


def load_or_calibrate(cap, detect, start, face_model, models, args):
    calib = None if args.no_calibration else load_calibration()
    if args.calibrate or calib is None:
        calib = run_calibration(cap, detect, start, face_model, models, args) or calib
    if calib is None:
        print("Warning: running without calibration, gaze on the screen will be less accurate.")
    return calib


def run_heatmap(args, models, face_model):
    detect = make_landmarker()
    cap = open_camera(args.camera)
    start = time.time()

    calib = load_or_calibrate(cap, detect, start, face_model, models, args)
    offsets = calib["offsets"] if calib else None
    mapping = calib["mapping"] if calib else None
    reference = calib["reference"] if calib else None

    window = "GazeMap heatmap"
    width, height = open_fullscreen(window)
    screen = Screen(width, height, args.screen_width_mm, args.camera_offset_mm)
    print(f"Heatmap window: {width} x {height}, {screen.mm_per_px:.3f} mm per pixel")

    stimuli = list_stimuli(args.stimulus)
    index = 0
    base = make_background(width, height, stimuli[index])
    heat = np.zeros((height // HEAT_SCALE + 1, width // HEAT_SCALE + 1), np.float32)
    points = []
    shown_since = time.time()
    distance = None

    def stimulus_name():
        return Path(stimuli[index]).stem if stimuli[index] else "heatmap"
    smoothed = None
    paused = False
    drift_monitor = DriftMonitor()
    writer = None
    blob_px = 50.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        camera_matrix = default_camera_matrix(w, h, args.focal)
        out = process_frame(frame, detect, int((time.time() - start) * 1000),
                            face_model, camera_matrix, models, offsets)

        current = None
        drift = None
        if out is not None:
            fraction, distance = gaze_on_screen(out[0], offsets, mapping, screen)
            drift = drift_monitor.update(head_drift(out[0], reference))
            if fraction is not None:
                p = np.array([fraction[0] * width, fraction[1] * height])
                smoothed = p if smoothed is None else 0.6 * smoothed + 0.4 * p
                current = smoothed
                # blob size = the angular uncertainty seen from the user's distance
                blob_px = distance * np.tan(np.radians(args.blob_deg)) / screen.mm_per_px
                inside = 0 <= current[0] < width and 0 <= current[1] < height
                if inside and not paused:
                    heat[int(current[1]) // HEAT_SCALE, int(current[0]) // HEAT_SCALE] += 1
                    points.append((time.time() - start, float(current[0]), float(current[1])))

        view = render_heatmap(base, heat, blob_px)
        if current is not None:
            x = int(np.clip(current[0], 0, width - 1))
            y = int(np.clip(current[1], 0, height - 1))
            cv2.circle(view, (x, y), 14, (255, 255, 255), 2)
            cv2.circle(view, (x, y), 3, (255, 255, 255), -1)

        preview = cv2.flip(frame, 1)
        pw = min(320, width // 5)
        preview = cv2.resize(preview, (pw, int(preview.shape[0] * pw / preview.shape[1])))
        view[10:10 + preview.shape[0], width - pw - 10:width - 10] = preview

        state = "PAUSED" if paused else ("no face" if out is None else "tracking")
        rec = "  REC" if writer is not None else ""
        dist = f"  eyes {distance / 10:.0f} cm from camera" if distance else ""
        slide = f"  image {index + 1}/{len(stimuli)} [n]ext" if len(stimuli) > 1 else ""
        status = (f"{state}{rec}  {len(points)} samples{dist}{slide}  "
                  f"[r]eset [p]ause [s]ave [v]ideo [c]alibrate [q]uit")
        cv2.putText(view, status, (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(view, status, (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        draw_drift_warning(view, drift)
        cv2.imshow(window, view)

        if writer is not None:
            writer.write(cv2.resize(view, (1280, int(height * 1280 / width))))

        key = cv2.waitKey(1) & 0xFF
        timed_out = args.seconds and not paused and time.time() - shown_since > args.seconds
        if key == ord("n") or timed_out:
            if points:
                save_heatmap(base, heat, blob_px, points, stimulus_name())
            if index + 1 >= len(stimuli):
                if timed_out or len(stimuli) > 1:
                    break  # slideshow finished
            else:
                index += 1
                base = make_background(width, height, stimuli[index])
                heat[:] = 0
                points.clear()
                shown_since = time.time()
                continue
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            heat[:] = 0
            points.clear()
        if key == ord("p"):
            paused = not paused
        if key == ord("c"):
            calib = run_calibration(cap, detect, start, face_model, models, args) or calib
            offsets = calib["offsets"] if calib else None
            mapping = calib["mapping"] if calib else None
            reference = calib["reference"] if calib else None
            width, height = open_fullscreen(window)
            smoothed = None
        if key == ord("s"):
            save_heatmap(base, heat, blob_px, points, stimulus_name())
        if key == ord("v"):
            if writer is None:
                HEATMAP_DIR.mkdir(exist_ok=True)
                path = HEATMAP_DIR / f"heatmap_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                         (1280, int(height * 1280 / width)))
                print("Recording video to", path)
            else:
                writer.release()
                writer = None
                print("Video saved")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


# ---------- video mode ----------

def record_video_gaze(args, models, face_model, video_path):
    """Play the video fullscreen and record where the user looks, in video pixels.

    The video clock follows real time: when the webcam loop is slower than the
    video, frames are skipped rather than played in slow motion, so gaze and
    video stay in sync. Each gaze sample is timestamped when its webcam frame
    was captured.
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    detect = make_landmarker()
    cap = open_camera(args.camera)
    start = time.time()
    calib = load_or_calibrate(cap, detect, start, face_model, models, args)
    offsets = calib["offsets"] if calib else None
    mapping = calib["mapping"] if calib else None
    reference = calib["reference"] if calib else None

    out_height = int(round(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * VIDEO_OUT_WIDTH
                           / video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    panel_size = (int(out_height * PANEL_ASPECT), out_height)

    window = "GazeMap video"
    width, height = open_fullscreen(window)
    screen = Screen(width, height, args.screen_width_mm, args.camera_offset_mm)
    if show_message(window, width, height, [
            f"{Path(video_path).name}: {duration:.0f} s",
            "Watch the video naturally. Keep your head roughly where it was during calibration.",
            "SPACE = start, then space = pause, q = stop and save"], wait_keys=(" ", "q")) != " ":
        cv2.destroyWindow(window)
        cap.release()
        return None
    countdown(window, width, height, cap)

    ok, frame = video.read()
    frame_index = 0
    canvas, (x0, y0, scale) = fit_image(frame, width, height, fill=0)
    samples = []  # (video seconds, x, y in video pixels, eye distance mm)
    insets = []   # (video seconds, JPEG of the webcam panel)
    face_x = face_y = None
    hint_y = height - max(12, (height - int(frame.shape[0] * scale)) // 4)
    drift_monitor = DriftMonitor()
    smoothed = None
    paused = False
    clock = 0.0        # seconds of video played
    last_tick = time.time()

    while True:
        ok_cam, cam = cap.read()
        captured = time.time()
        if not ok_cam:
            break
        if not paused:
            clock += captured - last_tick
        last_tick = captured
        if clock >= duration:
            break

        # advance the video to the frame that matches the clock
        target = int(clock * fps)
        while frame_index < target:
            ok = video.grab()
            frame_index += 1
            if not ok:
                break
        if ok and frame_index == target:
            ok, decoded = video.retrieve()
            if ok:
                canvas, (x0, y0, scale) = fit_image(decoded, width, height, fill=0)
        if not ok:
            break

        h, w = cam.shape[:2]
        camera_matrix = default_camera_matrix(w, h, args.focal)
        out = process_frame(cam, detect, int((captured - start) * 1000), face_model,
                            camera_matrix, models, offsets)
        current = None
        drift = None
        if out is not None and not paused:
            fraction, distance = gaze_on_screen(out[0], offsets, mapping, screen)
            drift = drift_monitor.update(head_drift(out[0], reference))
            if fraction is not None:
                p = np.array([fraction[0] * width, fraction[1] * height])
                smoothed = p if smoothed is None else 0.6 * smoothed + 0.4 * p
                current = smoothed
                samples.append((clock, (current[0] - x0) / scale, (current[1] - y0) / scale, distance))
        if not paused:
            shot = cam.copy()
            if out is not None:
                draw(shot, camera_matrix, out[0], arrow_mm=100.0)
                fx, fy = out[1].mean(axis=0)
                face_x = fx if face_x is None else 0.8 * face_x + 0.2 * fx  # steady crop, no jitter
                face_y = fy if face_y is None else 0.8 * face_y + 0.2 * fy
            panel = webcam_panel(shot, face_x, face_y, panel_size)
            insets.append((clock, cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]))

        view = canvas.copy()
        if args.show_gaze and current is not None:
            cv2.circle(view, (int(current[0]), int(current[1])), 14, (255, 255, 255), 2)
        if out is None:
            cv2.putText(view, "no face", (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if paused:
            cv2.putText(view, "PAUSED: space to continue", (20, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            hint = f"{clock:4.0f} / {duration:.0f} s    space = pause    q = stop and save"
            cv2.putText(view, hint, (width - 560, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 1)
        draw_drift_warning(view, drift)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            smoothed = None

    video.release()
    cap.release()
    cv2.destroyWindow(window)
    cv2.waitKey(1)

    seen = len({int(t * 10) for t, *_ in samples}) / max(clock * 10, 1)
    print(f"Recorded {len(samples)} gaze samples over {clock:.1f} s "
          f"({len(samples) / max(clock, 1e-6):.1f} per second, gaze found in {seen:.0%} of the time)")
    return {"fps": fps, "samples": samples, "insets": insets, "screen": screen,
            "video_scale": scale, "played": clock}


def countdown(window, width, height, cap, seconds=3):
    """3-2-1 in the middle of the screen, so the eyes start from the centre, not the instructions."""
    for n in range(seconds, 0, -1):
        until = time.time() + 1.0
        while time.time() < until:
            cap.read()  # keep the webcam buffer fresh
            canvas = np.zeros((height, width, 3), np.uint8)
            cv2.putText(canvas, str(n), (width // 2 - 30, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (255, 255, 255), 6)
            cv2.imshow(window, canvas)
            cv2.waitKey(1)


def webcam_panel(frame, face_x, face_y, size):
    """Mirrored webcam crop around the face, sized to sit next to the video."""
    h, w = frame.shape[:2]
    crop_h = int(h * PANEL_ZOOM)
    crop_w = min(w, int(crop_h * size[0] / size[1]))
    cx = w / 2 if face_x is None else face_x
    cy = h / 2 if face_y is None else face_y + PANEL_FACE_UP * crop_h
    x0 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
    y0 = int(np.clip(cy - crop_h / 2, 0, h - crop_h))
    crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.flip(cv2.resize(crop, size), 1)


def draw_trail(frame, recent, now, radius):
    """Recent gaze as a fading line, the newest sample as a soft disc of the model's uncertainty."""
    if not recent:
        return frame
    for (t0, x0, y0), (t1, x1, y1) in zip(recent, recent[1:]):
        fade = 1.0 - (now - t1) / TRAIL_SECONDS
        colour = (0, int(140 + 115 * fade), 255)
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), colour, max(1, int(2 + 4 * fade)), cv2.LINE_AA)
    _, x, y = recent[-1]
    disc = frame.copy()
    cv2.circle(disc, (int(x), int(y)), int(radius), (0, 255, 255), -1, cv2.LINE_AA)
    frame = cv2.addWeighted(disc, 0.35, frame, 0.65, 0)
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def render_video_gaze(video_path, record, blob_deg, inset=True):
    """Write the video again with the gaze trail, a summary heatmap and the raw samples."""
    video = cv2.VideoCapture(str(video_path))
    vw = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = VIDEO_OUT_WIDTH / vw
    size = (VIDEO_OUT_WIDTH, int(round(vh * k)))
    panel_w = int(size[1] * PANEL_ASPECT) if inset else 0
    out_size = (size[0] + panel_w, size[1])
    samples = record["samples"]
    times = [s[0] for s in samples]
    inset_times = [t for t, _ in record["insets"]]

    # disc size: the blob angle seen from the user's median distance, in output pixels
    distance = np.median([s[3] for s in samples]) if samples else 600.0
    screen_px = distance * np.tan(np.radians(blob_deg)) / record["screen"].mm_per_px
    radius = max(screen_px / record["video_scale"] * k, 6)

    HEATMAP_DIR.mkdir(exist_ok=True)
    stem = f"{Path(video_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = cv2.VideoWriter(str(HEATMAP_DIR / f"{stem}_gaze.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             record["fps"], out_size)
    heat = np.zeros((size[1] // HEAT_SCALE + 1, size[0] // HEAT_SCALE + 1), np.float32)
    middle = None
    n_out = int(record["played"] * record["fps"])

    for i in range(n_out):
        ok, frame = video.read()
        if not ok:
            break
        now = i / record["fps"]
        frame = cv2.resize(frame, size)
        if i == n_out // 2:
            middle = frame.copy()
        lo, hi = bisect.bisect_left(times, now - TRAIL_SECONDS), bisect.bisect_right(times, now)
        recent = [(t, x * k, y * k) for t, x, y, _ in samples[lo:hi]]
        frame = draw_trail(frame, recent, now, radius)

        combined = np.zeros((out_size[1], out_size[0], 3), np.uint8)
        combined[:, :size[0]] = frame
        j = bisect.bisect_right(inset_times, now) - 1
        if inset and j >= 0:
            panel = cv2.imdecode(record["insets"][j][1], cv2.IMREAD_COLOR)
            combined[:, size[0]:] = cv2.resize(panel, (panel_w, size[1]))
        for text, x in (("", 16), ("webcam eye tracker", size[0] + 16)):
            if x < out_size[0]:
                cv2.putText(combined, text, (x, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(combined, text, (x, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(combined)
    writer.release()
    video.release()

    # whole-clip heatmap over the middle frame: meaningful for a fixed camera only
    for _, x, y, _ in samples:
        if 0 <= x * k < size[0] and 0 <= y * k < size[1]:
            heat[int(y * k) // HEAT_SCALE, int(x * k) // HEAT_SCALE] += 1
    summary = render_heatmap(middle, heat, radius) if middle is not None else None
    if summary is not None:
        cv2.imwrite(str(HEATMAP_DIR / f"{stem}_summary.png"), summary)
    with open(HEATMAP_DIR / f"{stem}_points.csv", "w") as file:
        file.write("video_seconds,x_video_px,y_video_px,eye_distance_mm\n")
        file.writelines(f"{t:.3f},{x:.1f},{y:.1f},{d:.0f}\n" for t, x, y, d in samples)

    folder = HEATMAP_DIR.resolve()
    print(f"Saved in {folder}:\n  {stem}_gaze.mp4\n  {stem}_summary.png\n  {stem}_points.csv")
    print(f"For a GIF: ffmpeg -i {HEATMAP_DIR}/{stem}_gaze.mp4 -vf "
          f"\"fps=12,scale=900:-1:flags=lanczos\" {HEATMAP_DIR}/{stem}.gif")
    return summary, folder, stem


def run_video(args, models, face_model):
    record = record_video_gaze(args, models, face_model, args.video)
    if record is None or not record["samples"]:
        print("No gaze recorded, nothing saved.")
        return
    print("Rendering the gaze video...")
    summary, folder, stem = render_video_gaze(args.video, record, args.blob_deg, inset=not args.no_inset)
    if summary is not None:  # show the result straight away
        view = summary.copy()
        for k, line in enumerate([f"Saved in {folder}", f"{stem}_gaze.mp4 (trail video), _summary.png",
                                  "any key = close"]):
            cv2.putText(view, line, (16, 34 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(view, line, (16, 34 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("GazeMap result", view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_image(args, models, face_model):
    detect = make_landmarker()
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)
    h, w = frame.shape[:2]
    camera_matrix = default_camera_matrix(w, h, args.focal)
    out = process_frame(frame, detect, 0, face_model, camera_matrix, models,
                        (load_calibration() or {}).get("offsets"))
    if out is None:
        print("No face found.")
        return
    results, _ = out
    draw(frame, camera_matrix, results)
    draw_eye_patches(frame, results)
    print(gaze_text(results))
    out_path = str(Path(args.image).with_name(Path(args.image).stem + "_gaze.jpg"))
    cv2.imwrite(out_path, frame)
    print("Saved:", out_path)


def run_selftest(args, models, face_model):
    """Raw MPIIGaze frames with the dataset's own head pose and eye centers.

    Checks normalization + model + back-rotation without a camera or MediaPipe,
    and the calibration maths: 9 frames of the first day are used as 'dots'
    (their real gaze targets), the offset is tested on the remaining frames.
    """
    person = args.selftest_person
    base = Path("dataset/Data/Original") / person
    camera_matrix = loadmat(str(base / "Calibration" / "Camera.mat"))["cameraMatrix"]
    frames = []
    for day in sorted(p for p in base.iterdir() if p.name.startswith("day")):
        annotation = np.loadtxt(day / "annotation.txt", ndmin=2)
        for i, row in enumerate(annotation[: args.selftest_frames]):
            image = cv2.imread(str(day / f"{i + 1:04d}.jpg"))
            if image is not None:
                frames.append((image, row))

    def run(offsets, subset):
        errors = []
        for image, row in subset:
            head_rotation = cv2.Rodrigues(row[29:32])[0]
            centers = (row[35:38], row[38:41])
            results = estimate_gaze(image, camera_matrix, head_rotation, centers, models, offsets)
            for r in results:
                truth = row[26:29] - r["center"]
                errors.append(np.degrees(np.arccos(np.clip(r["gaze"] @ truth / np.linalg.norm(truth), -1, 1))))
        return np.mean(errors)

    print(f"Self-test on {person}: {len(frames)} frames, mean angular error {run(None, frames):.2f} deg")

    calib, test = frames[:9], frames[9:]
    samples = []
    for k, (image, row) in enumerate(calib):
        head_rotation = cv2.Rodrigues(row[29:32])[0]
        for r in estimate_gaze(image, camera_matrix, head_rotation, (row[35:38], row[38:41]), models):
            samples.append((k, r["name"], r["raw"], calibration_target_angles(r, row[26:29])))
    offsets = fit_offsets(samples)
    print(f"Calibration check: 9 frames -> remaining {len(test)} frames: "
          f"{run(None, test):.2f} deg without, {run(offsets, test):.2f} deg with calibration")
    print("Note: if these models were trained with this person included, the error is optimistic.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/final/models")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--focal", type=float, default=None,
                        help="camera focal length in pixels (default: image width)")
    parser.add_argument("--calibrate", action="store_true", help="run the 9-dot calibration at start")
    parser.add_argument("--no-calibration", action="store_true", help="ignore a saved calibration.json")
    parser.add_argument("--screen-width-mm", type=float, default=345.0,
                        help="physical screen width in mm (default: MacBook Pro 16 inch, 2021)")
    parser.add_argument("--camera-offset-mm", type=float, default=5.0,
                        help="distance from the camera down to the top edge of the fullscreen window")
    parser.add_argument("--heatmap", action="store_true", help="screen heatmap mode")
    parser.add_argument("--test", action="store_true", help="16-dot accuracy test with error arrows")
    parser.add_argument("--stimulus", default=None,
                        help="image, or folder of images (slideshow), to show under the heatmap")
    parser.add_argument("--seconds", type=float, default=0,
                        help="slideshow: seconds per image, then save its heatmap and go on (0 = press n)")
    parser.add_argument("--blob-deg", type=float, default=2.0,
                        help="size of each gaze sample on the heatmap, in degrees of visual angle")
    parser.add_argument("--video", default=None,
                        help="video to watch; saves the video again with your gaze trail")
    parser.add_argument("--show-gaze", action="store_true",
                        help="video mode: show your gaze live (off by default, it pulls the eyes)")
    parser.add_argument("--no-inset", action="store_true",
                        help="video mode: leave the webcam picture out of the saved video")
    parser.add_argument("--image", default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-person", default="p12")
    parser.add_argument("--selftest-frames", type=int, default=30)
    args = parser.parse_args()

    models = load_gaze_models(args.model_dir)
    face_model = load_face_model()

    if args.selftest:
        run_selftest(args, models, face_model)
    elif args.test:
        run_test(args, models, face_model)
    elif args.video:
        run_video(args, models, face_model)
    elif args.heatmap:
        run_heatmap(args, models, face_model)
    elif args.image:
        run_image(args, models, face_model)
    else:
        run_camera(args, models, face_model)


if __name__ == "__main__":
    main()
