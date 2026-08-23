# DopplerWild

In-the-wild micro-Doppler dataset of pedestrians, collected across four indoor/outdoor sites with a Texas Instruments IWR6843ISK FMCW radar, LiDAR, and cameras mounted 1 m above ground. ≈447k radar frames (≈110k labelled), 539 labelled subjects, 19 atomic activities grouped into four kinematic regimes (`normal_walking`, `on-wheels`, `dual-hand load`, `single-hand load`) plus an `others` class.

## Dataset Links

- **Full Dataset**: [https://www.kaggle.com/datasets/dopplerwild/dopplerwild/data](https://www.kaggle.com/datasets/dopplerwild/dopplerwild/data)
- **Preview Dataset**: [https://www.kaggle.com/datasets/dopplerwild/dopplerwild-preview](https://www.kaggle.com/datasets/dopplerwild/dopplerwild-preview)
- **Project Website**: <https://dopplerwild.github.io/DopplerWild-web/>

## DopplerWild dataset release

The dataset is organized into:

- **`labeled_tracks_Doppler/`** — 844 human-annotated `.npz` track files. Each file holds a single array `uD` (`float64`, shape `(256, T)`): the range-collapsed micro-Doppler (Doppler-vs-time) spectrogram for one tracked person. These are exactly the tracks referenced by at least one of the three task CSVs.
- **`unlabeled_tracks_Doppler/`** — 2,237 `.npz` tracks of the same format, produced by a lightweight signal-processing tracker. Intended for self-supervised pretraining. Split: 2,036 train / 201 val (index in `fold_splits/DopplerWild_unlabeled_tracklist.csv`).
- **`fold_splits/`** — the three task CSVs (below), plus `DopplerWild_unlabeled_tracklist.csv` (unlabeled track index) and `labeled_frame_metadata.csv` (per-frame metadata for all labeled subjects). Each task CSV row is a **1-second window** taken with a **0.25 s stride** from a labelled track, with task labels, fold assignment, and scene-context features.
- **`preview_samples/`** — preview package selected randomly with a fixed seed: 6 unlabeled tracks, 22 labeled 1-second sample windows, a `labeled_samples.csv` manifest, and 28 PNG previews. It is used by `visualize_1s_samples.ipynb` when both are kept in the same parent folder.
- **`checkpoints/`** — pre-trained PyTorch model weights (`.pt`). Contains both (a) fully trained per-fold eval checkpoints for every benchmark task / evaluation split / backbone combination (see [Eval Checkpoints](#eval-checkpoints)), and (b) SSL pretraining checkpoints under `ssl_pretraining/` (contrastive and reconstruction objectives, MobileNetV2 and ResNet-18 backbones) intended as starting points for downstream fine-tuning.

---
## Training Files (`fold_splits/*.csv`)

The three CSVs correspond to three tasks defined over the same underlying tracks:

| File | Task | Subjects | Label |
| --- | --- | --- | --- |
| `motionstate_1s_fold3.csv` | 3-class motion-state classification | 302 | `activity_load` ∈ {`normal_walking`, `Dual-hand Load`, `On Wheels`} (fine: `activity_atomic`) |
| `singlehand_constrained_1s_fold3.csv` | Binary single-hand-constrained vs. normal walking | 255 | `activity_load` ∈ {`normal_walking`, `Single-hand Load`} (fine: `activity_atomic`) |
| `velocity_regression_1s_folds3.csv` | Speed / heading regression | 539 | continuous `speed_mean`, `vx_mean`, `vy_mean`, `angle_mean`, `x_mean`, `y_mean`, `v_radial`, `v_lateral` |

All three CSVs share the windowing scheme and a 3-fold **cross-subject** split (`fold` ∈ {0, 1, 2}; no subject crosses folds). The `location` column additionally supports **leave-one-location-out** evaluation across the four sites (`A`, `B`, `C`, `D`).

---

## Track files (`labeled_tracks_Doppler/*.npz`, `unlabeled_tracks_Doppler/*.npz`)

Filenames follow `{recording_id}_track_{track_id}.npz` (e.g. `20250523125036_track_1.npz`), where `recording_id` is the radar session timestamp `YYYYMMDDhhmmss` (= CSV `file_name`) and `track_id` is the per-recording person index or global ID.

Each `.npz` contains a single entry:

- **`uD`** — `float64`, shape `(256, T)`. Axis 0 indexes **256 Doppler bins** (signed velocity, symmetric around zero); axis 1 indexes **time** at 90 bins/second. `T` varies per track. Each CSV row addresses a 1 s sub-window via `(file_name, global_id, t_start, t_end)`.

---

## Common CSV columns

The following columns appear in **all three** CSVs and have the same meaning everywhere. Note that a track is per-subject, while a file may contain multiple subjects.

| Column | Type | Description |
| --- | --- | --- |
| `file_name` | str | Recording session ID (`YYYYMMDDhhmmss`). |
| `global_id` | int | Unique subject ID. |
| `location` | str | Collection site. One of `A`, `B`, `C`, `D`. |
| `t_start_rel`, `t_end_rel` | float (s) | Window bounds relative to the start of the track (seconds). |
| `t_start`, `t_end` | float (s) | Window bounds relative to start of uD file (seconds). |
| `duration` | float (s) | Window length. |
| `fold` | int ∈ {0,1,2} | Cross-subject fold; consistent across all three CSVs. |

The following scene-context columns appear in `motionstate_1s_fold3.csv` and `singlehand_constrained_1s_fold3.csv` only:

| Column | Type | Description |
| --- | --- | --- |
| `occlusion_seconds` | float (s) | Cumulative seconds in the window where the subject is occluded (annotated from the synchronized RGB / LiDAR pass). |
| `close_person_avg_count_5m` | float | Mean count of *other* tracked persons within 5 m of the subject, averaged over the window's frames. |
| `close_person_frames_5m` | int | Number of radar frames in the window with at least one other person within 5 m. |
| `close_person_seconds_5m` | float (s) | Same as above, expressed in seconds. |


---

## `motionstate_1s_fold3.csv`

Three-class motion-state classification (walking vs. dual-hand load vs. on-wheels) training groundtruth. Adds:

| Column | Type | Description |
| --- | --- | --- |
| `filename_id` | str | Per-track key: `{file_name}_{global_id}` |
| `activity_atomic` | str | Fine atomic label ∈ {`normal_walking`, `hands_in_pockets`, `two_hands_carrying`, `two_hands_folded`, `two_hands_texting`, `one_hand_carry_one_hand_pocket`, `one_hand_carry_phone_one_hand_carry`, `one_hand_fold_one_hand_in_pocket`, `one_hand_phone_one_hand_pocket`, `cyclist`, `on_scooter`, `on_a_vehicle`}. |
| `activity_load` | str | 3-class target ∈ {`normal_walking`, `Dual-hand Load`, `On Wheels`}; deterministic function of `activity_atomic`. |

---

## `singlehand_constrained_1s_fold3.csv`

Binary single-hand-constrained vs. normal walking classification training groundtruth. Adds:

| Column | Type | Description |
| --- | --- | --- |
| `filename_id` | str | `{file_name}_{global_id}` track key. |
| `activity_atomic` | str | Fine atomic activity label ∈ {`normal_walking`, `one_hand_carry`, `one_hand_carry_phone`, `one_hand_fold`, `one_hand_in_pocket`, `one_hand_lift`, `one_hand_phone_call`, `shoulder_bag_fold_walking`}. |
| `activity_load` | str | Binary target ∈ {`normal_walking`, `Single-hand Load`}. |

---

## `velocity_regression_1s_folds3.csv`

Speed and heading regression training groundtruth. Contains the [Common CSV columns](#common-csv-columns) and adds:

| Column | Type | Description |
| --- | --- | --- |
| `x_mean`, `y_mean` | float (m) | Mean 2-D position of the subject in the window (radar coordinate frame). |
| `vx_mean`, `vy_mean` | float (m/s) | Mean Cartesian velocity components over the window. |
| `speed_mean` | float (m/s) | Mean scalar speed (`√(vx²+vy²)`) over the window. |
| `angle_mean` | float (rad) | Mean heading angle over the window. |
| `v_radial` | float (m/s) | Mean velocity component along the radar line-of-sight. |
| `v_lateral` | float (m/s) | Mean velocity component perpendicular to the radar line-of-sight. |

This CSV does **not** include the occlusion or proximity columns (see [Common CSV columns](#common-csv-columns) for which files carry those).

---

## `labeled_frame_metadata.csv`

Per-frame metadata for every labeled subject across all recordings (~195 k rows). One row per radar frame per tracked person. Columns:

| Column | Type | Description |
| --- | --- | --- |
| `file_name` | str | Recording session ID (`YYYYMMDDhhmmss`). |
| `radar_frame` | int | Frame index within the recording. |
| `global_id` | int | Unique subject ID (consistent with task CSVs). |
| `x`, `y` | float (m) | Subject position in the radar coordinate frame. |
| `vx`, `vy` | float (m/s) | Per-frame Cartesian velocity (empty when `valid_velocity=0`). |
| `valid_velocity` | int ∈ {0,1} | Whether the velocity estimate is considered reliable for this frame. |
| `location` | str | Anonymized collection site (`A`, `B`, `C`, `D`). |
| `occlusion` | float (s) | Occlusion flag/duration for this frame. |
| `activity` | str | Atomic activity label for this frame (`unclear` when annotation is ambiguous). |
| `nearby_people_count` | float | Number of other tracked persons in the scene for this frame. |

## Eval Checkpoints

The `checkpoints/` folder ships fully trained per-fold evaluation checkpoints for every benchmark task × evaluation split × backbone × adaptation combination, plus the SSL pretraining checkpoints under `ssl_pretraining/` (excluded from the table below).

Folder names follow `{init}_{eval_split}_{task}[_{backbone}]/`. `{init}` is one of `supervised` (supervised training from random initialization), `contrastive` (initialized from contrastive SSL pretraining), or `Reconstruction` (initialized from masked-reconstruction SSL pretraining). For SSL-initialized folders, weights are further split into `frozen_backbone/` (linear/regression head trained on frozen features) and `full_finetune/` (end-to-end fine-tuning) subfolders. The default backbone is **MobileNetV2**; folders with the `_resnet18` suffix use **ResNet-18**. Cross-subject folders contain three fold checkpoints (`fold0.pt`, `fold1.pt`, `fold2.pt`); cross-location folders contain four leave-one-location-out checkpoints (`fold_A.pt`, `fold_B.pt`, `fold_C.pt`, `fold_D.pt`).

| Folder | Init | Task | Eval split | Backbone | Adaptations available |
| --- | --- | --- | --- | --- | --- |
| `supervised_cross-subj_motionstate/` | supervised | 3-class motion state | Cross-subject (3-fold) | MobileNetV2 | — |
| `supervised_cross-subj_motionstate_resnet18/` | supervised | 3-class motion state | Cross-subject (3-fold) | ResNet-18 | — |
| `supervised_cross-subj_singlehand/` | supervised | Single-hand binary | Cross-subject (3-fold) | MobileNetV2 | — |
| `supervised_cross-subj_velocityregression/` | supervised | Velocity regression | Cross-subject (3-fold) | MobileNetV2 | — |
| `supervised_cross-subj_velocityregression_resnet18/` | supervised | Velocity regression | Cross-subject (3-fold) | ResNet-18 | — |
| `supervised_cross-loc_motionstate/` | supervised | 3-class motion state | Leave-one-location-out | MobileNetV2 | — |
| `supervised_cross-loc_velocityregression/` | supervised | Velocity regression | Leave-one-location-out | MobileNetV2 | — |
| `reconstruction_cross-subj_motionstate/` | Reconstruction SSL | 3-class motion state | Cross-subject (3-fold) | MobileNetV2 | `frozen_backbone/`, `full_finetune/` |
| `reconstruction_cross-subj_motionstate_resnet18/` | Reconstruction SSL | 3-class motion state | Cross-subject (3-fold) | ResNet-18 | `frozen_backbone/`, `full_finetune/` |
| `reconstruction_cross-subj_singlehand/` | Reconstruction SSL | Single-hand binary | Cross-subject (3-fold) | MobileNetV2 | `frozen_backbone/`, `full_finetune/` |
| `reconstruction_cross-subj_velocityregression/` | Reconstruction SSL | Velocity regression | Cross-subject (3-fold) | MobileNetV2 | `full_finetune/` |
| `reconstruction_cross-subj_velocityregression_resnet18/` | Reconstruction SSL | Velocity regression | Cross-subject (3-fold) | ResNet-18 | `full_finetune/` |
| `reconstruction_cross-loc_motionstate/` | Reconstruction SSL | 3-class motion state | Leave-one-location-out | MobileNetV2 | `frozen_backbone/`, `full_finetune/` |
| `reconstruction_cross-loc_velocityregression/` | Reconstruction SSL | Velocity regression | Leave-one-location-out | MobileNetV2 | `full_finetune/` |
| `contrastive_cross-subj_motionstate/` | contrastive SSL | 3-class motion state | Cross-subject (3-fold) | MobileNetV2 | `frozen_backbone/`, `full_finetune/` |
| `contrastive_cross-subj_motionstate_resnet18/` | contrastive SSL | 3-class motion state | Cross-subject (3-fold) | ResNet-18 | `frozen_backbone/`, `full_finetune/` |
| `contrastive_cross-subj_singlehand/` | contrastive SSL | Single-hand binary | Cross-subject (3-fold) | MobileNetV2 | `frozen_backbone/`, `full_finetune/` |
| `contrastive_cross-subj_velocityregression/` | contrastive SSL | Velocity regression | Cross-subject (3-fold) | MobileNetV2 | `full_finetune/` |
| `contrastive_cross-subj_velocityregression_resnet18/` | contrastive SSL | Velocity regression | Cross-subject (3-fold) | ResNet-18 | `full_finetune/` |
| `contrastive_cross-loc_motionstate/` | contrastive SSL | 3-class motion state | Leave-one-location-out | MobileNetV2 | `frozen_backbone/`, `full_finetune/` |
| `contrastive_cross-loc_velocityregression/` | contrastive SSL | Velocity regression | Leave-one-location-out | MobileNetV2 | `full_finetune/` |

---

## DopplerWild dataset release
The DopplerWild dataset is released under the CC BY 4.0 license. 
The dataset release is accompanied by documentation describing the dataset structure, labels, metadata fields, intended use, etc.

DopplerWild data collection complied with institutional IRB guidelines. 
Data was collected in outdoor public areas such as sidewalks and passages. 
During collection, signs and flyers describing the project and privacy concerns were provided. 
The benchmark focuses on Doppler signatures that do not include direct visual appearance.
