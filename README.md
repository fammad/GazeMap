# GazeMap

GazeMap takes normalized eye images, predicts pitch and yaw gaze angles, and evaluates the predictions using mean angular error in degrees.

**Status:** Working single-subject CNN baseline; evaluation is currently limited to a random split of p00/day01, so leave-one-person-out testing is not implemented yet.

## Result

Mean angular error: **4.01°**

| | |
|---|---|
| Data | MPIIGaze, subject p00, day01 only |
| Split | 80/20 random, seed 42 |
| Model | 2 conv layers (1→8→16, k5), fc 1152→16→2, no output activation |
| Training | Adam lr 1e-3, batch 16, MSELoss, 5 epochs |
| Hardware | CPU (M1) |

The frames are sampled from continuous recordings, so neighboring frames often show nearly the same eye appearance and gaze direction. Because the split is random, near-duplicate frames can be placed in both training and validation, allowing the model to be evaluated on images that closely resemble images it has already seen.

The change from 3.62° (unseeded) to 4.01° (seed 42) shows that this estimate has noticeable run-to-run sensitivity, so 4.01° should be treated as a single baseline observation rather than a stable performance claim.

## What this number is not

Published MPIIGaze baselines are roughly 5–6° for person-independent evaluation (e.g., Zhang et al., CVPR 2015 / TPAMI 2017; benchmark tables in later surveys). However, I do not interpret the gap as evidence that this small CNN is better than those models, because my evaluation uses a random split from one subject and one recording day, allowing highly similar frames to appear on both sides of the split. The warning sign is that the error fell to 2.15° after 50 epochs while the training loss reached 0.0003: the model improved as it increasingly memorized the session, which indicates that the split—not necessarily the model’s ability to generalize—was driving the unusually low error.

**Prediction:** Leave-one-person-out evaluation will increase the error to approximately 6.5°, because the model will no longer see that person’s eye shape, skin appearance, camera conditions, or recording-specific visual patterns during training.