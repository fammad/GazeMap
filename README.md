# GazeMap

GazeMap predicts gaze direction—pitch and yaw—from normalized eye images. Performance is measured with mean angular error in degrees.

**Status:** The project now has a within-person baseline and a person-independent leave-one-person-out evaluation. The LOPO result is the primary result.

## Results

### Within-person baseline

Mean angular error: **4.01°**

| | |
|---|---|
| Data | MPIIGaze, subject p00, day01 only |
| Split | 80/20 random split, seed 42 |
| Model | Two convolution layers (1→8→16, kernel size 5), fully connected 1152→16→2 |
| Training | Adam, learning rate 1e-3, batch size 16, MSELoss, 5 epochs |
| Hardware | Apple M1 CPU |

This split is optimistic. Frames come from continuous recordings, so neighboring frames can have nearly identical eye appearance and gaze direction. Similar frames can therefore appear in both training and validation.

The 4.01° value is a useful baseline, but it is not a person-independent performance claim.

### Person-independent LOPO evaluation

The final evaluation used leave-one-person-out cross-validation over all 15 subjects, p00–p14. For each fold, one complete subject was held out for testing, a fresh model and optimizer were created, and the model trained on the other 14 subjects.

| | |
|---|---|
| Data | MPIIGaze, all 15 subjects |
| Split | One complete subject held out per fold |
| Model | Same small two-convolution CNN |
| Input | Normalized right-eye images, shape 1×36×60 |
| Training | Adam, learning rate 1e-3, batch size 128, MSELoss, fixed 5-epoch budget |
| Seed | 42 reset at the start of every fold |
| Hardware | Apple Silicon MPS GPU |
| Total test samples | 213,658 |

The primary person-equal mean MAE was **6.35°** with a population standard deviation of **0.93°** and a standard error of approximately **0.24°**.

The pooled per-sample MAE was **6.09°**. The difference between 6.35° and 6.09° reflects the strong imbalance in test-set sizes: p03 has 35,075 test samples while p13 has 1,498, a roughly 23× difference. Person-equal MAE is therefore the primary number for a person-independent claim; the pooled value gives the sample-weighted view.

## What the result means

The model generalizes to unseen people, but the error is higher than the 4.01° within-person baseline. This gap shows why a random frame split is not enough for this task.

The fold extremes are meaningful under the fixed per-fold seed: p00 remains the easiest subject, while p04 and p14 remain among the hardest. However, the complete subject ranking should not be treated as stable. Error can vary more between epochs within one subject than between subjects: p07 spans approximately 2.04° across the five epochs, while the between-subject population standard deviation is 0.93°.

Five epochs were a fixed compute budget, not a convergence claim. Several folds were still improving when the budget ended, so the reported epoch-5 values were not selected by looking for the best epoch on the held-out subject. Selecting the best test epoch would use the test set for model selection and introduce optimistic bias.

MPS results can vary slightly between reruns and hardware because numerical execution is not fully deterministic. The reported 6.35° is the observed result for this fixed configuration, not a universal exact constant.

A rerun on 2026-10-06 with the identical configuration gave a person-equal MAE of 6.34° (pooled 6.05°), within 0.01° of the reported value. Individual folds moved by up to 0.48° (p00: 4.12° to 3.64°), so the aggregate is reproducible but the per-subject ranking is not. The pose and robustness analyses below use this rerun's per-sample errors and saved fold models. The original per-fold results are kept in `lopo_results_2026-09-17.csv`.

## Prediction check

Before running LOPO, I predicted a mean error of approximately **6.50°**, based on the published MPIIGaze LOPO baselines and the limited capacity of this small CNN. The observed mean was **6.35°**, a difference of **0.15°**, or approximately **0.63 standard errors**. The prediction was therefore within the noise of the final estimate.

This result is in the range of published person-independent MPIIGaze baselines. It should not be described as matching MnistNet or any other published model because the architectures, preprocessing, training budget, and input setup are not controlled comparisons.

## Published comparison

Zhang et al., *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation* (TPAMI, arXiv:1711.09017), report leave-one-person-out MPIIGaze errors ranging from approximately **5.5° for GazeNet**, their best method, to **8.7° for simpler baselines**. Their **MnistNet** result is approximately **6.3°**. GazeMap’s 6.35° is within this published LOPO baseline range, but the result is not a direct architecture-to-architecture comparison.

## Head-pose analysis

Does the error depend on how far the head is turned away from the camera? For every LOPO test sample, the head pose (a rotation vector in the normalized data) was converted into one number: the angle between the direction the face points and the camera axis. Samples were grouped into 10° bins. The last bin is 40° and above, because past about 45° only a few people have data.

![LOPO error by head pose](pose_error.png)

| Head rotation | Samples | People | Person-equal MAE | Pooled MAE | Mean gaze angle |
|---|---|---|---|---|---|
| 0-10° | 51,921 | 14 | 6.17° | 5.48° | 11.6° |
| 10-20° | 78,351 | 15 | 6.25° | 5.81° | 12.5° |
| 20-30° | 57,834 | 15 | 6.56° | 6.16° | 12.8° |
| 30-40° | 21,287 | 14 | 7.78° | 7.74° | 13.2° |
| 40°+ | 4,265 | 12 | 7.74° | 7.61° | 12.6° |

"People" counts subjects with at least 50 samples in the bin. Person-equal MAE is the primary number, for the same reason as in the LOPO result.

Error stays close to 6.2° up to 20° of head rotation, rises slightly at 20-30°, and then steps up by about 1.2° beyond 30°.

Two other explanations were checked:

- **Gaze direction.** If turned-head samples also had more extreme gaze, gaze could explain the rise. The mean gaze angle changes by only about 1.6° across bins (11.6° to 13.2°), so this is unlikely to account for it.
- **Subject mix.** Harder subjects might simply sit more turned. Within each person, error on turned samples (25° and above) was compared with error on frontal samples (below 10°). Error was higher when turned for 12 of 14 people. The exceptions were p07 and p12; p11 had too few frontal samples to compare.

Only about 12% of all samples have the head turned 30° or more, so the model sees few such examples during training. That, and the more side-on view of the eye, are plausible reasons for the step, but this analysis does not separate them. It is an observational result within MPIIGaze, not a controlled experiment: it shows where the model is weaker, not exactly why.

Reproduce with `python analyze_pose.py`. It reads `lopo_sample_results.csv` (written by `lopo_train_all_folds.py`, about 30 MB, not committed) and writes `pose_results.csv` and `pose_error.png`.

## Scope and next analysis

GazeMap estimates gaze direction from an eye image. It does not estimate screen point-of-gaze, perform user calibration, or provide a real-time application.

The next analysis tests robustness to cabin-like image conditions (brightness, blur, sensor noise and partial occlusion) by evaluating the saved fold models on corrupted test images, without retraining.

Frame-stride subsampling was considered as a speed optimization but was not needed: the full-data evaluation completed in under half an hour after the cumulative-index lookup was improved. The reported LOPO result therefore uses all available frames.
