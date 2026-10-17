"""GazeMap live demo: webcam -> face landmarks -> normalized eye images -> CNN -> gaze arrows.

Pipeline (same normalization as MPIIGaze, Zhang et al.):
  1. MediaPipe finds 6 face points (4 eye corners, 2 mouth corners).
  2. solvePnP fits the dataset's 6-point 3D face model -> head rotation + position.
  3. Each eye is warped into a virtual camera 600 mm away, looking straight at
     the eye, 60x36 pixels, grey, histogram-equalized. This is exactly how the
     training images were made (checked against the dataset: pixel correlation 0.998).
  4. The trained fold models predict pitch/yaw in that normalized space
     (all models in the folder are averaged).
  5. Optional 9-dot calibration adds a per-user offset to each eye's prediction.
  6. The prediction is rotated back to the camera and drawn: red = each eye,
     green = both eyes combined.

Usage:
  python gaze_demo.py                         # webcam with the final models
  python gaze_demo.py --calibrate             # run the 9-dot calibration first
  python gaze_demo.py --selftest              # dataset check, no camera needed
  python gaze_demo.py --image photo.jpg       # one image, saves photo_gaze.jpg

Keys: q = quit, c = calibrate, x = clear calibration, m = mirror view, s = smoothing.
"""

import argparse
import json
import time
import urllib.request
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

# MediaPipe face-mesh indices in the same order as the dataset's face model columns:
# right eye outer, right eye inner, left eye inner, left eye outer,
# right mouth corner, left mouth corner ("right" = the person's right).
MEDIAPIPE_IDS = [33, 133, 362, 263, 61, 291]

FOCAL_NORM = 960.0
DISTANCE_NORM = 600.0
EYE_SIZE = (60, 36)  # width, height

# 9 calibration dots as fractions of the screen (x from the left, y from the top).
CALIBRATION_DOTS = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
SETTLE_SECONDS = 1.0   # wait for the eyes to land on the dot
COLLECT_SECONDS = 1.2  # then record frames


# ---------- models ----------

def load_gaze_models(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir.parent / "config.json") as file:
        config = json.load(file)

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

def screen_point_mm(fx, fy, screen_mm, camera_offset_mm):
    """3D position (camera coordinates, mm) of a point on a laptop screen.

    Assumes the camera sits at the top-centre of the screen, in the screen plane.
    Camera x points to the image right, which is the user's LEFT, so the
    screen's horizontal axis is flipped.
    """
    width, height = screen_mm
    x = -(fx - 0.5) * width
    y = camera_offset_mm + fy * height
    return np.array([x, y, 0.0])


def calibration_target_angles(result, target_mm):
    """True gaze of one eye towards the target, in that eye's normalized space."""
    gaze = target_mm - result["center"]
    return vector_to_angles(result["R"] @ (gaze / np.linalg.norm(gaze)))


def fit_offsets(samples):
    """samples: list of (eye name, raw angles, true angles). Offset = mean difference per eye."""
    offsets = {}
    for name in ("right", "left"):
        diffs = [true - raw for n, raw, true in samples if n == name]
        if len(diffs) < 9:
            raise RuntimeError(f"Too few calibration frames for the {name} eye ({len(diffs)}).")
        offsets[name] = np.mean(diffs, axis=0)
    return offsets


def calibration_report(samples, offsets):
    before, after = [], []
    for name, raw, true in samples:
        t = angles_to_vector(*true)
        before.append(np.degrees(np.arccos(np.clip(angles_to_vector(*raw) @ t, -1, 1))))
        after.append(np.degrees(np.arccos(np.clip(angles_to_vector(*(raw + offsets[name])) @ t, -1, 1))))
    return float(np.mean(before)), float(np.mean(after))


def save_offsets(offsets, info):
    payload = {"offsets_rad": {k: v.tolist() for k, v in offsets.items()}, **info}
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))


def load_offsets():
    if not CALIBRATION_PATH.exists():
        return None
    payload = json.loads(CALIBRATION_PATH.read_text())
    return {k: np.array(v) for k, v in payload["offsets_rad"].items()}


def run_calibration(cap, detect, start_time, face_model, models, args):
    window = "GazeMap calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    blank = np.zeros((900, 1440, 3), np.uint8)
    cv2.imshow(window, blank)
    cv2.waitKey(300)
    rect = cv2.getWindowImageRect(window)
    width, height = (rect[2], rect[3]) if rect[2] > 100 else (1440, 900)

    intro = np.zeros((height, width, 3), np.uint8)
    for k, line in enumerate(["Calibration: look at each white dot until it disappears.",
                              "Keep your head natural. Press SPACE to start, ESC to cancel."]):
        cv2.putText(intro, line, (60, height // 2 - 20 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
    cv2.imshow(window, intro)
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key == ord(" "):
            break

    samples = []
    for fx, fy in CALIBRATION_DOTS:
        target = screen_point_mm(fx, fy, args.screen_mm, args.camera_offset_mm)
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
                cv2.destroyWindow(window)
                return None

            ok, frame = cap.read()
            if not ok or elapsed < SETTLE_SECONDS:
                continue
            h, w = frame.shape[:2]
            camera_matrix = default_camera_matrix(w, h, args.focal)
            out = process_frame(frame, detect, int((time.time() - start_time) * 1000),
                                face_model, camera_matrix, models, offsets=None)
            if out is None:
                continue
            for result in out[0]:
                samples.append((result["name"], result["raw"], calibration_target_angles(result, target)))

    cv2.destroyWindow(window)
    offsets = fit_offsets(samples)
    before, after = calibration_report(samples, offsets)
    save_offsets(offsets, {"screen_mm": list(args.screen_mm), "camera_offset_mm": args.camera_offset_mm,
                           "frames": len(samples) // 2, "error_before_deg": before, "error_after_deg": after})
    print(f"Calibration saved to {CALIBRATION_PATH}: {len(samples) // 2} frames, error on the dots "
          f"{before:.1f}° -> {after:.1f}° (measured on the calibration frames themselves, so optimistic)")
    return offsets


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

    return detect


def default_camera_matrix(width, height, focal=None):
    f = focal if focal else float(width)  # rough guess for a laptop camera
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


# ---------- modes ----------

def process_frame(frame, detect, timestamp_ms, face_model, camera_matrix, models, offsets=None):
    points = detect(frame, timestamp_ms)
    if points is None:
        return None
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
    offsets = None if args.no_calibration else load_offsets()
    if args.calibrate:
        offsets = run_calibration(cap, detect, start, face_model, models, args) or offsets
    if offsets:
        print("Using calibration from", CALIBRATION_PATH)

    mirror, smooth = True, True
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
            offsets = run_calibration(cap, detect, start, face_model, models, args) or offsets
            smoothed.clear()
        if key == ord("x"):
            offsets = None
            smoothed.clear()

    cap.release()
    cv2.destroyAllWindows()


def run_image(args, models, face_model):
    detect = make_landmarker()
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)
    h, w = frame.shape[:2]
    camera_matrix = default_camera_matrix(w, h, args.focal)
    out = process_frame(frame, detect, 0, face_model, camera_matrix, models, load_offsets())
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
    for image, row in calib:
        head_rotation = cv2.Rodrigues(row[29:32])[0]
        for r in estimate_gaze(image, camera_matrix, head_rotation, (row[35:38], row[38:41]), models):
            samples.append((r["name"], r["raw"], calibration_target_angles(r, row[26:29])))
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
    parser.add_argument("--screen-mm", type=float, nargs=2, default=[302.0, 196.0],
                        help="visible screen width and height in mm (default: MacBook Pro 14 inch)")
    parser.add_argument("--camera-offset-mm", type=float, default=8.0,
                        help="distance from the camera down to the top edge of the screen image")
    parser.add_argument("--image", default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-person", default="p12")
    parser.add_argument("--selftest-frames", type=int, default=30)
    args = parser.parse_args()

    models = load_gaze_models(args.model_dir)
    face_model = load_face_model()

    if args.selftest:
        run_selftest(args, models, face_model)
    elif args.image:
        run_image(args, models, face_model)
    else:
        run_camera(args, models, face_model)


if __name__ == "__main__":
    main()
