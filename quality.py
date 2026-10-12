"""
Core Quality Assessment Module for Quality-Aware Face Recognition under
Compound Real-World Degradation (Low Illumination, Pose Variation, Occlusion,
Low Resolution, Motion Blur).

Calculates per-face degradation metrics using classical computer vision methods
without requiring GPU or deep network retraining.
"""

import cv2
import numpy as np
import config

try:
    import face_recognition
except ImportError:
    face_recognition = None

_EXPECTED_LANDMARK_GROUPS = {
    "chin", "left_eyebrow", "right_eyebrow", "nose_bridge",
    "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"
}

def _empty_quality_result():
    return {
        "illumination_score": 0.0,
        "blur_score": 0.0,
        "pose_score": 0.0,
        "yaw_deg": 0.0,
        "resolution_score": 0.0,
        "occlusion_score": 0.0,
        "composite_quality": 0.0,
        "dominant_degradation": "none"
    }

def assess_illumination(gray_crop):
    """
    Evaluates illumination and contrast quality.
    Penalizes under-exposure (low mean), over-exposure (high mean), and low contrast (low std).
    Returns (illumination_score, mean_val, std_val).
    """
    if gray_crop is None or gray_crop.size == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(gray_crop))
    std_val = float(np.std(gray_crop))

    # Brightness score centered around ideal mean 130
    mean_score = max(0.0, 1.0 - abs(mean_val - 130.0) / 130.0)
    # Contrast score (std_val >= 40 is good contrast)
    std_score = min(1.0, std_val / 40.0)

    illumination_score = float(np.clip(0.6 * mean_score + 0.4 * std_score, 0.0, 1.0))
    return illumination_score, mean_val, std_val

def assess_blur(gray_crop):
    """
    Evaluates image sharpness / motion blur using Laplacian variance.
    Returns (blur_score, laplacian_variance).
    """
    if gray_crop is None or gray_crop.size == 0:
        return 0.0, 0.0

    lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    ref_var = getattr(config, 'BLUR_REFERENCE_VARIANCE', 100.0)
    blur_score = float(np.clip(lap_var / ref_var, 0.0, 1.0))
    return blur_score, lap_var

def assess_resolution(face_location):
    """
    Evaluates face crop resolution against MIN_FACE_SIZE_PX.
    face_location is (top, right, bottom, left).
    Returns (resolution_score, face_size_px).
    """
    if not face_location or len(face_location) < 4:
        return 0.0, 0

    top, right, bottom, left = face_location
    h = max(0, bottom - top)
    w = max(0, right - left)
    face_size_px = max(h, w)

    min_px = getattr(config, 'MIN_FACE_SIZE_PX', 80)
    resolution_score = float(np.clip(face_size_px / float(min_px), 0.0, 1.0))
    return resolution_score, face_size_px

def assess_occlusion(landmarks_dict):
    """
    Evaluates partial face occlusion by checking how many expected 68-point
    landmark groups are detected.
    Returns (occlusion_score, detected_groups, total_groups).
    """
    total_groups = len(_EXPECTED_LANDMARK_GROUPS)
    if not landmarks_dict or not isinstance(landmarks_dict, dict):
        return 0.0, 0, total_groups

    detected_groups = sum(
        1 for g in _EXPECTED_LANDMARK_GROUPS
        if g in landmarks_dict and len(landmarks_dict[g]) > 0
    )
    occlusion_score = float(detected_groups / total_groups)
    return occlusion_score, detected_groups, total_groups

def estimate_head_pose(landmarks_dict, frame_shape):
    """
    Estimates 3D head pose (yaw, pitch, roll in degrees) via cv2.solvePnP
    using generic 3D face model points and 2D facial landmarks.
    Returns (yaw, pitch, roll) or None if landmarks are incomplete.
    """
    if not landmarks_dict or not isinstance(landmarks_dict, dict):
        return None

    # Check key landmarks required for 6-point pose estimation
    required_keys = ["nose_tip", "chin", "left_eye", "right_eye", "top_lip"]
    for k in required_keys:
        if k not in landmarks_dict or not landmarks_dict[k]:
            return None

    try:
        nose_pts = landmarks_dict["nose_tip"]
        chin_pts = landmarks_dict["chin"]
        left_eye_pts = landmarks_dict["left_eye"]
        right_eye_pts = landmarks_dict["right_eye"]
        top_lip_pts = landmarks_dict["top_lip"]

        nose_tip = nose_pts[2] if len(nose_pts) >= 3 else nose_pts[-1]
        chin = chin_pts[8] if len(chin_pts) >= 9 else chin_pts[-1]
        left_eye = left_eye_pts[0]
        right_eye = right_eye_pts[3] if len(right_eye_pts) >= 4 else right_eye_pts[-1]
        left_mouth = top_lip_pts[0]
        right_mouth = top_lip_pts[6] if len(top_lip_pts) >= 7 else top_lip_pts[-1]

        image_points = np.array([
            nose_tip,
            chin,
            left_eye,
            right_eye,
            left_mouth,
            right_mouth
        ], dtype=np.float64)

        # 3D generic face model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        h, w = frame_shape[0], frame_shape[1]
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])

        if sy > 1e-6:
            pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = np.arctan2(-rot_mat[2, 0], sy)
            roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = np.arctan2(-rot_mat[2, 0], sy)
            roll = 0.0

        yaw_deg = float(np.degrees(yaw))
        pitch_deg = float(np.degrees(pitch))
        roll_deg = float(np.degrees(roll))

        return yaw_deg, pitch_deg, roll_deg
    except Exception:
        return None

def dominant_degradation(quality_result):
    """
    Diagnoses the dominant quality degradation factor for logging/UI alerts.
    Returns: 'low_illumination', 'motion_blur', 'pose_variation',
             'low_resolution', 'occlusion', or 'normal'.
    """
    if not quality_result or not isinstance(quality_result, dict):
        return "none"

    comp_q = quality_result.get("composite_quality", 0.0)
    scores = {
        "low_illumination": quality_result.get("illumination_score", 1.0),
        "motion_blur": quality_result.get("blur_score", 1.0),
        "pose_variation": quality_result.get("pose_score", 1.0),
        "low_resolution": quality_result.get("resolution_score", 1.0),
        "occlusion": quality_result.get("occlusion_score", 1.0)
    }

    min_deg, min_score = min(scores.items(), key=lambda x: x[1])

    if comp_q >= 0.8 and min_score >= 0.7:
        return "normal"
    elif min_score < 0.75:
        return min_deg
    else:
        return "normal"

def assess_face_quality(frame, face_location, landmarks=None):
    """
    Computes a full quality assessment for a detected face crop.
    Returns composite_quality score in [0,1] and sub-score breakdown.
    """
    if frame is None or not face_location or len(face_location) < 4:
        return _empty_quality_result()

    top, right, bottom, left = face_location
    h_img, w_img = frame.shape[:2]

    top_c = max(0, min(h_img, top))
    bottom_c = max(0, min(h_img, bottom))
    left_c = max(0, min(w_img, left))
    right_c = max(0, min(w_img, right))

    face_crop = frame[top_c:bottom_c, left_c:right_c]

    if face_crop.size == 0:
        return _empty_quality_result()

    if len(face_crop.shape) == 3:
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_crop = face_crop

    illumination_score, _, _ = assess_illumination(gray_crop)
    blur_score, _ = assess_blur(gray_crop)
    resolution_score, _ = assess_resolution(face_location)

    if landmarks is None and face_recognition is not None:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            found_landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])
            if found_landmarks:
                landmarks = found_landmarks[0]
        except Exception:
            landmarks = None

    occlusion_score, _, _ = assess_occlusion(landmarks)
    pose_tuple = estimate_head_pose(landmarks, frame.shape)

    if pose_tuple is not None:
        yaw_deg, pitch_deg, roll_deg = pose_tuple
        max_yaw = getattr(config, 'MAX_YAW_DEGREES', 45.0)
        pose_score = float(np.clip(1.0 - abs(yaw_deg) / max_yaw, 0.0, 1.0))
    else:
        yaw_deg = 0.0
        pose_score = 0.5

    w_illum = getattr(config, 'QUALITY_WEIGHT_ILLUMINATION', 0.25)
    w_blur = getattr(config, 'QUALITY_WEIGHT_BLUR', 0.25)
    w_pose = getattr(config, 'QUALITY_WEIGHT_POSE', 0.20)
    w_res = getattr(config, 'QUALITY_WEIGHT_RESOLUTION', 0.15)
    w_occ = getattr(config, 'QUALITY_WEIGHT_OCCLUSION', 0.15)

    composite_quality = (
        w_illum * illumination_score +
        w_blur * blur_score +
        w_pose * pose_score +
        w_res * resolution_score +
        w_occ * occlusion_score
    )
    composite_quality = float(np.clip(composite_quality, 0.0, 1.0))

    result = {
        "illumination_score": float(illumination_score),
        "blur_score": float(blur_score),
        "pose_score": float(pose_score),
        "yaw_deg": float(yaw_deg),
        "resolution_score": float(resolution_score),
        "occlusion_score": float(occlusion_score),
        "composite_quality": float(composite_quality),
        "dominant_degradation": "none"
    }

    result["dominant_degradation"] = dominant_degradation(result)
    return result
