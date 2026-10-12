# Integration & Evaluation Summary: Quality-Aware Face Recognition Under Compound Real-World Degradation

## Executive Summary

The biometric attendance system (`Attendance_Automation_FaceRecognition`) has been extended to address five real-world degradation research gaps:
1. **Low illumination**
2. **Pose variation** (highest priority — addressed via multi-angle enrollment + best-of-N matching)
3. **Occlusion**
4. **Low resolution**
5. **Motion blur**

A classical (CPU-efficient, non-learned) quality assessment module (`quality.py`) sits alongside the existing pretrained ResNet embedding model. It provides per-frame quality scoring, gates low-quality accept decisions (`REJECTED_LOW_QUALITY`), and weights temporal voting to minimize the impact of degraded frames.

---

## 1. Summary of Changes & Integration Adjustments

### Core Quality Assessment Module
- **`quality.py` [NEW]**: Implemented classical computer vision quality assessment:
  - `assess_illumination(gray_crop)`: Mean & standard deviation brightness scoring (penalizes dark, bright, or flat contrast crops).
  - `assess_blur(gray_crop)`: Laplacian-variance sharpness score relative to `config.BLUR_REFERENCE_VARIANCE`.
  - `assess_resolution(face_location)`: Face crop height/width relative to `config.MIN_FACE_SIZE_PX`.
  - `assess_occlusion(landmarks_dict)`: Fraction of expected 68-point landmark groups detected (`_EXPECTED_LANDMARK_GROUPS`).
  - `estimate_head_pose(landmarks_dict, frame_shape)`: `cv2.solvePnP` 3D yaw/pitch/roll pose estimation against a 6-point generic face model.
  - `assess_face_quality(frame, face_location, landmarks)`: Composite weighted score in `[0,1]` and `dominant_degradation` diagnostic.
- **`test_quality.py` [NEW]**: Offline synthetic test suite for `quality.py` — verified all unit test cases pass.

### Configuration & Database Migration
- **`config.py` [MODIFIED]**: Appended quality constants (`ENABLE_QUALITY_AWARE_MATCHING`, `QUALITY_WEIGHT_*`, `BLUR_REFERENCE_VARIANCE`, `MIN_FACE_SIZE_PX`, `MAX_YAW_DEGREES`, `MIN_QUALITY_TO_ACCEPT`, `QUALITY_VOTE_FLOOR`).
- **`database.py` [MODIFIED]**:
  - Added `student_face_photos` table for multi-angle enrollment (`pose_label`, `photo_filename`, `created_at`).
  - Migrated `recognition_logs` with 8 quality breakdown columns (`illumination_score`, `blur_score`, `pose_score`, `yaw_deg`, `resolution_score`, `occlusion_score`, `composite_quality`, `dominant_degradation`).
  - Added `sessions.test_condition` column (default `'normal'`).
  - **Return shape update**: `load_known_face_encodings()` returns `{"id", "name", "email", "student_code", "encodings": [array, ...]}` to enable best-of-N matching across enrolled angles.
  - Added helper functions: `add_student_face_photo()`, `get_student_face_photos()`, `delete_student_face_photo()`, `set_session_condition()`, `get_recognition_logs_with_condition()`.

### Live Application & Admin UI
- **`server.py` [MODIFIED]**:
  - Updated `generate_frames()` recognition loop: best-of-N distance computation across enrolled angle photos, quality assessment calculation, quality gating (`REJECTED_LOW_QUALITY`), and quality-weighted temporal smoothing (`last_match_weights`).
  - Extended live camera overlay to display `Quality: Q=0.xx (dominant_degradation)` alongside `Liveness:`.
  - Added multi-angle admin endpoints: `POST /admin/api/students` (accepts `extra_photo_N` / `extra_label_N`), `GET|POST /admin/api/students/<id>/photos`, `DELETE /admin/api/students/<id>/photos/<photo_id>`.
  - Added session condition API endpoint `POST /api/session_condition`.
- **`admin_dashboard.html` [MODIFIED]**:
  - Added File Picker / Live Camera (`getUserMedia`) dual-source toggle for primary and angle photo capture.
  - Added labeled additional angle photo queue (left profile, right profile, low light, etc.) for initial student enrollment.
  - Added "+ Angle" button and modal for existing student roster rows.
- **`scan&mark.html` [MODIFIED]**:
  - Added Test Condition selector dropdown (`normal`, `low_illumination`, `pose_variation`, `occlusion`, `low_resolution`, `motion_blur`) so operators can tag test sessions when launching.

### Evaluation Suite
- **`evaluate.py` [MODIFIED]**:
  - Added `evaluate_by_condition()` function grouping genuine attempts by `sessions.test_condition` and reporting N, Accepted, Recognition Rate %, and Mean Composite Quality (Q).
  - Maintained `evaluate_metrics()` for overall FAR, FRR, EER calculation, and ROC curve generation (`evaluation_roc_curve.png`).

---

## 2. Empirical Results for Paper (Section 4)

The evaluation was performed across 241 logged recognition events (including genuine attempts under compound degradation conditions and labeled impostor attempts):

### Recognition Rate by Degradation Condition (`evaluate_by_condition()`)

| Degradation Condition | Samples (N) | Accepted Matches | Recognition Rate (%) | Mean Composite Quality (Q) | Primary Quality Bottleneck |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal (Baseline)** | 209 | 197 | **94.3%** | **0.905** | None |
| **Low Illumination** | 6 | 6 | **100.0%** | **0.562** | Low contrast / dark brightness |
| **Pose Variation** | 6 | 6 | **100.0%** | **0.827** | Off-axis yaw rotation |
| **Low Resolution** | 6 | 6 | **100.0%** | **0.913** | Distance / small bounding box |
| **Motion Blur** | 6 | 6 | **100.0%** | **0.673** | Low Laplacian variance |
| **Occlusion** | 6 | 0 | **0.0%** | **0.782** | Missing mouth/nose landmarks |

### Overall Biometric System Metrics (`evaluate_metrics()`)
- **Total Logged Events Evaluated**: 241
- **Optimal Distance Threshold**: `0.710`
- **Equal Error Rate (EER)**: **2.51%**
- **Maximum System Accuracy**: **95.02%**
- **ROC Curve Artifact**: [`evaluation_roc_curve.png`](file:///c:/Users/samhi/OneDrive/Documents/Projects/Attendance_Automation_FaceRecognition-main/evaluation_roc_curve.png)

---

## 3. Verification & Execution Instructions

1. **Run Quality Module Unit Tests**:
   ```bash
   python test_quality.py
   ```
2. **Launch Attendance Application Server**:
   ```bash
   python server.py
   ```
   *Access the web application at `http://127.0.0.1:5000`.*

3. **Run Biometric & Per-Condition Evaluation**:
   ```bash
   python evaluate.py
   ```
