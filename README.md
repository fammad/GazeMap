# GazeMap

GazeMap is a small CNN that predicts where an eye is looking (gaze pitch and yaw) from a normalized 36×60 eye image, trained and evaluated on MPIIGaze. The focus is the evaluation: how well the model works on people it has never seen, what each design choice actually buys, and where it fails. A webcam demo runs the same pipeline live.

Error is mean angular error in degrees. Unless stated otherwise, every number is person-equal (each of the 15 held-out people counts once) under leave-one-person-out cross-validation.

## Key findings

- **Unseen people: 6.03° per eye** (SE 0.23°). The original small model scored 6.34°; a network at the paper's size is the only change that clearly helped.
- **Both eyes together: 4.97°.** Averaging the two eyes' predictions in camera space lowers error for all 15 people, often a lot, because the two eyes make different mistakes.
- **9-dot calibration: 4.06°**, tested on recording days other than the calibration day. It still helps when the head turns, but the gain shrinks from 0.9° to 0.5° beyond 20° of head movement.
- **Weak spots:** error rises from about 5.8° to 7.3° when the head is turned more than 30°, and covering the top 30% of the eye image adds 2.4°. Darkness costs nothing once the image is histogram-equalized, as the pipeline does.
- **Gaze range:** MPIIGaze only contains people looking at a laptop screen, from 20° below the camera to about 5° above it. The model cannot recognise gaze outside that range, such as looking up at the ceiling.

## Setup

| | |
|---|---|
| Data | MPIIGaze normalized eye images, 15 people, 213,658 frames, both eyes (left eyes mirrored to look like right eyes) |
| Evaluation | Leave-one-person-out: 15 folds, each trains a fresh model on 14 people and tests on the one left out |
| Model | 2 conv layers (20 and 50 filters, 5×5) + fully connected 500 + head pose (2 angles) → pitch, yaw. The same layout as MnistNet in Zhang et al. |
| Training | Adam, learning rate 1e-3 dropped ×0.1 after epoch 10, 15 epochs, batch 128, MSE loss, seed 42 per fold |
| Model selection | None: the last epoch is always reported. Test error per epoch is logged, never used to pick an epoch |
| Hardware | Apple M1, MPS. One full 15-fold run takes about 2 hours |

## What each change bought

Each row adds one change to the row above. All rows use the same 15-fold protocol.

| Step | Change | Both eyes | Right eye only | Change vs previous (right eye) | People improved |
|---|---|---|---|---|---|
| baseline | small network (8/16/16), right eye only, 5 epochs | n/a | 6.34° | | |
| a | train and test on both eyes | 6.19° | 6.28° | −0.06° | 9/15 |
| b | paper-size network (20/50/500) | 6.00° | 5.98° | **−0.30°** | **12/15** |
| c | head pose as input | 6.06° | 5.95° | −0.03° | 8/15 |
| final | 15 epochs, learning-rate drop | **6.03°** | **5.90°** | −0.05° | 8/15 |

Only the wider network made a clear difference (about 2.5 standard errors of the paired difference). Using the left eyes, head pose input and longer training changed less than the run-to-run noise. Between epoch 5 and 15 training loss fell by about 40% while test error stayed flat: the model keeps fitting the 14 training people better, but that does not carry over to a new person. The limit is how different people's eyes look, not training time.

The first five epochs of `c_head_pose` and `final` match to the last digit, so with a fixed seed the runs are deterministic and the comparisons are clean.

## Comparison with the paper

Zhang et al., *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation* (TPAMI, arXiv:1711.09017), report leave-one-person-out errors of 6.3° for MnistNet and 5.5° for GazeNet (a 16-layer VGG network pretrained on ImageNet).

GazeMap's 6.03° per eye sits between the two, but this is **not** a like-for-like win. The paper trained and tested on a fixed subset of 3,000 eye images per person (42,000 per fold); GazeMap uses all frames, about 400,000 per fold. Part of the difference is simply more data. The both-eye number (4.97°) is a different target, one gaze direction from between the eyes, and should not be compared with the paper's per-eye figures.

## Both eyes and calibration

![Per-person error and calibration under head movement](results/evaluation/evaluation.png)

| | Error | People improved |
|---|---|---|
| Single eye | 6.03° | |
| Both eyes combined | **4.97°** | 15/15 |
| Both eyes, tested on other days, no calibration | 4.99° | |
| Both eyes, tested on other days, 9-frame calibration | **4.06°** | 12/15 |

**Both eyes:** each eye's prediction is rotated from its normalized space back to the camera, the two directions are averaged, and the result is compared with the average of the two true directions. p04 drops from 7.10° to 4.02°.

**Calibration:** 9 frames from one recording day, spread over a 3×3 grid of gaze directions, fit a fixed pitch/yaw offset per eye. The offset is tested on all *other* days of that person, where sitting position, distance and light differ. Repeated with 5 different calibration days. For three people (p03, p07, p12) it was slightly worse, by 0.04 to 0.21°.

**Does calibration survive head movement?**

| Head turned vs calibration pose | No calibration | Calibrated | Gain |
|---|---|---|---|
| 0-5° | 4.61° | 3.73° | 0.88° |
| 5-10° | 4.61° | 3.76° | 0.85° |
| 10-20° | 4.77° | 3.98° | 0.79° |
| over 20° | 5.20° | 4.69° | 0.51° |

The correction is applied after head-pose normalization, which is why most of it survives head movement. Beyond 20° the gain drops by about 40%; recalibrating fixes that.

## Head pose

![Error by head rotation](results/pose/pose_error.png)

| Head rotation away from camera | Eye images | People | Person-equal error | Mean gaze angle |
|---|---|---|---|---|
| 0-10° | 107,509 | 15 | 5.84° | 11.5° |
| 10-20° | 155,700 | 15 | 6.00° | 12.5° |
| 20-30° | 114,009 | 15 | 6.11° | 12.9° |
| 30-40° | 41,788 | 15 | 7.28° | 13.2° |
| 40°+ | 8,310 | 13 | 7.03° | 12.5° |

Error is flat up to 30° and then steps up by about 1.2°. Two other explanations were checked. Gaze direction barely changes across bins (11.5° to 13.2°), so it does not explain the step. Within each person, error on turned samples (25° and above) was higher than on frontal samples (below 10°) for 12 of 15 people, so it is not just harder people sitting more turned. Only about 12% of frames have the head turned 30° or more. Feeding head pose into the network did not remove this weakness. The result is observational: it shows where the model is weaker, not exactly why.

## Image conditions

![Error under image corruptions](results/robustness/robustness_error.png)

Each fold model was tested on its own held-out person after a synthetic corruption at five levels, then the image was histogram-equalized again, as the real pipeline does. Nothing was retrained. At level 0 every fold reproduced its training-time error.

| Condition | Simulates | Strongest level | Error | Change |
|---|---|---|---|---|
| Brightness | weaker light | pixel values ×0.2 | 6.03° | none |
| Blur | motion, defocus | σ = 2 px | 6.88° | +0.85° |
| Noise | low-light sensor noise | std 0.2 | 7.56° | +1.53° |
| Occlusion | eyelid, glasses frame | top 30% black | 8.42° | +2.39° |

Equalization removes the effect of darkness completely. An earlier version of this test darkened images *after* equalization, which no real pipeline does, and wrongly concluded that low light doubles the error. Partial occlusion of the eye is the real weakness.

## Live demo

`gaze_demo.py` runs the same pipeline on a webcam: MediaPipe finds 6 face points, a 3D face model gives head pose, each eye is normalized exactly like the training images (pixel correlation 0.998 with the dataset's own images on the same frames), the 15 fold models are averaged, and the result is drawn as one arrow per eye plus a combined arrow. `--calibrate` runs a 9-dot calibration and saves it.

The demo works within the dataset's gaze range: looking at the camera and at the screen. Gaze far above the camera is outside anything the model has seen.

## Limitations

- **Gaze range.** Training gaze runs from about −20° to +5° vertically and ±16° horizontally (99% of frames). Directions outside it, for example looking up, are not recognised.
- **One dataset.** 15 people, laptop webcams, indoor light. No infrared camera, no car cabin.
- **More training data than the paper.** See the comparison above.
- **Head pose from the dataset's calibration.** The evaluation uses MPIIGaze's own head pose and camera calibration. The webcam demo estimates both, with a guessed focal length, so its error will be higher.
- **Synthetic corruptions.** Real occlusion, blur and noise look different from these approximations.
- **One seed per configuration.** Differences below about 0.2° between configurations are within noise.

## Earlier baselines

A within-person split (one person, one day, 80/20 random frames) gives 4.01°, far better than the 6.3° on unseen people, because neighbouring video frames end up in both training and test. Before running leave-one-person-out I predicted about 6.5° for the small model; it scored 6.35° (rerun 6.34°). The code and write-up of that stage are in the git history (commit `7af3654`).

## Reproduce

1. Download MPIIGaze so that `dataset/Data/Normalized/p00` to `p14` and `dataset/Data/Original/` exist.
2. `python3 -m venv .gazemap-env`, activate it, `pip install -r requirements.txt`.
3. `python train_lopo.py --all` trains `final` and the ablation steps (about 4 hours on an M1). Results go to `results/<step>/`.
4. `python evaluate.py` runs the both-eye and calibration evaluation.
5. `python analyze_pose.py` and `python robustness.py` run the head-pose and image-condition analyses.
6. `python gaze_demo.py --selftest`, then `python gaze_demo.py --calibrate` for the webcam demo.

## Files

| File | Purpose |
|---|---|
| `gaze_model.py` | data loading (both eyes, mirroring, head pose) and the network |
| `train_lopo.py` | leave-one-person-out training and the ablation steps |
| `metrics.py` | angular error |
| `evaluate.py` | both-eye fusion and calibration evaluation |
| `analyze_pose.py` | error by head rotation |
| `robustness.py` | error under image corruptions |
| `gaze_demo.py` | webcam demo with calibration |
| `results/` | per-fold results, summaries and figures (models and per-sample files are not committed) |
