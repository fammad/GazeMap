# GazeMap

GazeMap takes normalized eye images, predicts pitch and yaw gaze angles, and evaluates the predictions using mean angular error in degrees.

**Status:** Working CNN baseline with both within-person and leave-one-person-out evaluation. The within-person result is an optimistic baseline; the LOPO result is the primary person-independent evaluation.

## Results

### Within-person baseline

Mean angular error: **4.01°**

| | |
|---|---|
| Data | MPIIGaze, subject p00, day01 only |
| Split | 80/20 random, seed 42 |
| Model | 2 conv layers (1→8→16, k5), fc 1152→16→2, no output activation |
| Training | Adam lr 1e-3, batch 16, MSELoss, 5 epochs |
| Hardware | CPU (M1) |

The frames come from continuous recordings, so neighboring frames often have very similar eye appearance and gaze direction. A random split can therefore place near-duplicate frames in both training and validation. The 4.01° result is a single optimistic within-person baseline, not a person-independent performance claim.

### Leave-one-person-out evaluation

Mean angular error across 15 held-out people: **6.77°**

| | |
|---|---|
| Data | MPIIGaze, subjects p00–p14 |
| Split | Leave one entire person out for testing; train on the other 14 |
| Model | Same 2-convolution-layer CNN |
| Training | Adam lr 1e-3, batch 128, MSELoss, 5 fixed epochs per fold |
| Hardware | Apple Silicon MPS GPU |
| Standard deviation | 1.03° |
| Fold range | 4.12°–8.46° |

The LOPO result is the primary result because the test person is never included in training. The model generalizes to unseen people, but performance varies by person: the easiest fold was p00 at 4.12°, and the hardest was p07 at 8.46°.

## What the baseline means

The difference between 4.01° and 6.77° shows the generalization gap between a random within-person split and a person-independent test. The lower random-split error was helped by repeated visual conditions and similar neighboring frames; it should not be interpreted as evidence that this small CNN outperforms published person-independent methods.

The LOPO result was close to the pre-experiment prediction of approximately 6.5°. The observed mean was 0.27° higher. Some folds reached their lowest test error before epoch 5, but the final comparison reports epoch 5 for every fold because choosing the best epoch using the held-out person would let the test set influence model selection.

## Published comparison

Zhang et al., *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation* (TPAMI, arXiv:1711.09017), report leave-one-person-out MPIIGaze errors ranging from approximately **5.5° for GazeNet**, their best method, to **8.7° for simpler baselines**. Their **MnistNet** result is approximately **6.3°**. These results are not directly comparable in every implementation detail, but GazeMap’s 6.77° LOPO result is within the reported range for person-independent baselines.

## Scope and next analysis

GazeMap estimates gaze direction from an eye image. It does not estimate screen point-of-gaze, perform user calibration, or provide a real-time application.

The next analysis will test whether the error differences are associated with head pose rather than only with gaze angle. It will use the dataset’s pose metadata and compare error across pose groups while checking gaze-angle distributions as a possible confound.
