"""
Liveness Detection Module using Eye Aspect Ratio (EAR).

This module implements real-time anti-spoofing via Eye Aspect Ratio (EAR)
calculation from 68 facial landmarks. It prevents photo-attack and screen-replay
spoofing by verifying dynamic eye blinks over consecutive live video frames.

Mathematical Formulation:
-------------------------
For 6 eye landmark points p1, p2, p3, p4, p5, p6 (ordered horizontally p1-p4 and
vertically p2-p6, p3-p5):

    EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2.0 * ||p1 - p4|| )

Where ||.|| represents Euclidean distance in 2D pixel space.
An eye blink causes EAR to drop below a configured threshold (typically ~0.20).
"""

import numpy as np
import face_recognition
import config

def euclidean_distance(pt1, pt2):
    """Computes the 2D Euclidean distance between two coordinate tuples."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye given 6 facial landmark points.
    
    :param eye_landmarks: List of 6 (x, y) tuples representing eye contour.
    :return: Computed Eye Aspect Ratio (float).
    """
    if len(eye_landmarks) < 6:
        return 0.0

    # Vertical landmark distances
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])

    # Horizontal landmark distance
    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)

class LivenessDetector:
    """
    Tracks facial landmarks across consecutive frames to detect eye blinks
    and establish dynamic proof-of-life.
    """
    def __init__(self, ear_threshold=None, consec_frames=None):
        self.ear_threshold = ear_threshold if ear_threshold is not None else config.EAR_THRESHOLD
        self.consec_frames = consec_frames if consec_frames is not None else config.EAR_CONSEC_FRAMES
        self.face_histories = {}  # Map candidate_id -> list of recent EAR values

    def compute_frame_ear(self, rgb_frame, face_location=None):
        """
        Extracts facial landmarks from an RGB frame and returns the average EAR
        across left and right eyes.
        """
        locations = [face_location] if face_location else None
        landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations=locations)

        if not landmarks_list:
            return None

        landmarks = landmarks_list[0]
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return None

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear

    def verify_liveness(self, subject_id, current_ear):
        """
        Evaluates whether a candidate subject demonstrates dynamic liveness based on EAR history.
        
        :param subject_id: Identifier for the candidate subject (e.g. matched name/ID).
        :param current_ear: Current frame's computed EAR value.
        :return: (is_live: bool, reason: str)
        """
        if not config.ENABLE_LIVENESS_CHECK:
            return True, "Liveness check disabled in configuration."

        if current_ear is None:
            return False, "Facial landmarks for eyes could not be extracted."

        if subject_id not in self.face_histories:
            self.face_histories[subject_id] = []

        history = self.face_histories[subject_id]
        history.append(current_ear)
        if len(history) > 15:
            history.pop(0)

        # Check for blink or dynamic variation in eye aspect ratio
        min_ear = min(history)
        max_ear = max(history)
        ear_range = max_ear - min_ear

        # Dynamic blink detected if EAR dipped below threshold or showed micro-blink variance
        if min_ear < self.ear_threshold or ear_range >= 0.05:
            return True, f"Blink/Liveness confirmed (EAR min: {min_ear:.2f}, range: {ear_range:.2f})"
        
        # If insufficient frames collected so far, allow initial evaluation window
        if len(history) < self.consec_frames:
            return True, f"Evaluating liveness buffer ({len(history)}/{self.consec_frames})"

        return False, f"Potential photo spoof detected: Static EAR ({current_ear:.2f}, range: {ear_range:.2f})"

    def reset(self):
        """Resets tracked face histories."""
        self.face_histories.clear()
