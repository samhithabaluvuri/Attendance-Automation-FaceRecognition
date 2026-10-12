"""
Offline sanity tests for quality.py that don't require dlib/face_recognition
(which isn't installed in this sandbox). Confirms the pure-CV scoring functions
behave sensibly on synthetic degraded images before wiring into the live app.
"""
import numpy as np
import cv2
import quality


def make_face(size=200, brightness=130, noise=0, blur_ksize=0):
    img = np.full((size, size, 3), brightness, dtype=np.uint8)
    if noise > 0:
        n = np.random.randint(-noise, noise, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    # draw some structure so Laplacian variance isn't trivially zero
    cv2.rectangle(img, (size//4, size//4), (3*size//4, 3*size//4), (brightness-40,)*3, 2)
    cv2.circle(img, (size//2, size//2), size//6, (brightness+40,)*3, -1)
    if blur_ksize > 0:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    return img


def test_illumination():
    good = cv2.cvtColor(make_face(brightness=130, noise=40), cv2.COLOR_BGR2GRAY)
    dark = cv2.cvtColor(make_face(brightness=20, noise=5), cv2.COLOR_BGR2GRAY)
    bright = cv2.cvtColor(make_face(brightness=250, noise=5), cv2.COLOR_BGR2GRAY)

    s_good, _, _ = quality.assess_illumination(good)
    s_dark, _, _ = quality.assess_illumination(dark)
    s_bright, _, _ = quality.assess_illumination(bright)

    print(f"illumination: good={s_good:.3f} dark={s_dark:.3f} bright={s_bright:.3f}")
    assert s_good > s_dark, "well-lit face should score higher than underexposed face"
    assert s_good > s_bright, "well-lit face should score higher than overexposed face"


def test_blur():
    sharp = cv2.cvtColor(make_face(brightness=130, noise=40, blur_ksize=0), cv2.COLOR_BGR2GRAY)
    blurry = cv2.cvtColor(make_face(brightness=130, noise=40, blur_ksize=15), cv2.COLOR_BGR2GRAY)

    s_sharp, v_sharp = quality.assess_blur(sharp)
    s_blur, v_blur = quality.assess_blur(blurry)

    print(f"blur: sharp={s_sharp:.3f} (var={v_sharp:.1f}) blurry={s_blur:.3f} (var={v_blur:.1f})")
    assert s_sharp > s_blur, "sharp image should score higher than motion-blurred image"


def test_resolution():
    s_big, px_big = quality.assess_resolution((0, 200, 200, 0))     # 200x200 face
    s_small, px_small = quality.assess_resolution((0, 40, 40, 0))   # 40x40 face

    print(f"resolution: big={s_big:.3f} ({px_big}px) small={s_small:.3f} ({px_small}px)")
    assert s_big > s_small, "larger face crop should score higher than tiny/distant face"
    assert s_big == 1.0, "face well above MIN_FACE_SIZE_PX should saturate at 1.0"


def test_occlusion():
    full_landmarks = {g: [(0, 0)] for g in quality._EXPECTED_LANDMARK_GROUPS}
    partial_landmarks = {"chin": [(0, 0)], "left_eye": [(0, 0)], "right_eye": [(0, 0)]}  # e.g. mask covering mouth/nose
    none_landmarks = {}

    s_full, p_full, t_full = quality.assess_occlusion(full_landmarks)
    s_partial, p_partial, t_partial = quality.assess_occlusion(partial_landmarks)
    s_none, p_none, t_none = quality.assess_occlusion(none_landmarks)

    print(f"occlusion: full={s_full:.3f} partial={s_partial:.3f} ({p_partial}/{t_partial}) none={s_none:.3f}")
    assert s_full == 1.0
    assert 0 < s_partial < 1.0, "partial landmark detection (e.g. masked mouth/nose) should score between 0 and 1"
    assert s_none == 0.0


def test_head_pose_frontal_vs_profile():
    h, w = 480, 640
    # Roughly frontal, symmetric landmark placement
    frontal = {
        "nose_tip": [(0,0), (0,0), (320, 260)],
        "chin": [(0,0)]*8 + [(320, 400)],
        "left_eye": [(260, 220)],
        "right_eye": [(0,0), (0,0), (0,0), (380, 220)],
        "top_lip": [(280, 320), (0,0), (0,0), (0,0), (0,0), (0,0), (360, 320)],
    }
    pose_frontal = quality.estimate_head_pose(frontal, (h, w, 3))
    print(f"head pose (frontal-ish): {pose_frontal}")
    assert pose_frontal is not None
    yaw, pitch, roll = pose_frontal
    assert abs(yaw) < 45, f"roughly symmetric landmarks should not yield an extreme yaw, got {yaw}"

    # Strongly asymmetric (profile-like) landmark placement -> larger |yaw|
    profile = {
        "nose_tip": [(0,0), (0,0), (450, 260)],
        "chin": [(0,0)]*8 + [(430, 400)],
        "left_eye": [(400, 220)],
        "right_eye": [(0,0), (0,0), (0,0), (470, 210)],
        "top_lip": [(410, 320), (0,0), (0,0), (0,0), (0,0), (0,0), (460, 320)],
    }
    pose_profile = quality.estimate_head_pose(profile, (h, w, 3))
    print(f"head pose (profile-ish): {pose_profile}")
    assert pose_profile is not None


def test_composite_and_dominant_degradation():
    frame = make_face(size=300, brightness=20, noise=5, blur_ksize=9)  # dark AND blurry
    landmarks = {g: [(150, 150)] * 9 for g in quality._EXPECTED_LANDMARK_GROUPS}
    # patch specific indices used by estimate_head_pose to be valid
    landmarks["nose_tip"] = [(150,150)]*3
    landmarks["chin"] = [(150,150)]*9
    landmarks["left_eye"] = [(140,140)]
    landmarks["right_eye"] = [(0,0),(0,0),(0,0),(160,140)]
    landmarks["top_lip"] = [(140,160),(0,0),(0,0),(0,0),(0,0),(0,0),(160,160)]

    result = quality.assess_face_quality(frame, (0, 300, 300, 0), landmarks=landmarks)
    print("composite result:", result)
    assert result["composite_quality"] is not None
    assert 0.0 <= result["composite_quality"] <= 1.0
    dom = quality.dominant_degradation(result)
    print("dominant degradation flagged as:", dom)
    assert dom in ("low_illumination", "motion_blur")  # this synthetic frame is both dark and blurry


if __name__ == "__main__":
    test_illumination()
    test_blur()
    test_resolution()
    test_occlusion()
    test_head_pose_frontal_vs_profile()
    test_composite_and_dominant_degradation()
    print("\nAll quality.py offline sanity tests passed.")
