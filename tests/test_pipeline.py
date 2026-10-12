import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpvmd import filters, retarget, tracker  # noqa: E402
from mpvmd.vmd import BoneFrame, read_vmd_bones, write_vmd  # noqa: E402

ARM = 35.0


def rest_skeleton():
    """33 landmarks in MMD axes matching the model rest pose (A-pose, facing -Z)."""
    w = np.zeros((33, 3))
    a = math.radians(ARM)
    w[retarget.L_HIP] = [0.1, 0, 0]
    w[retarget.R_HIP] = [-0.1, 0, 0]
    w[retarget.L_SH] = [0.18, 0.5, 0]
    w[retarget.R_SH] = [-0.18, 0.5, 0]
    for side, sx in ((0, 1), (1, -1)):
        sh = w[retarget.L_SH + side]
        d = np.array([sx * math.cos(a), -math.sin(a), 0])
        w[retarget.L_EL + side] = sh + 0.28 * d
        w[retarget.L_WR + side] = sh + 0.53 * d
        hand = sh + 0.6 * d
        w[retarget.L_INDEX + side] = hand + [0, 0, -0.03]
        w[retarget.L_PINKY + side] = hand + [0, 0, 0.03]
        hip = w[retarget.L_HIP + side]
        w[retarget.L_KNEE + side] = hip + [0, -0.45, 0]
        w[retarget.L_ANK + side] = hip + [0, -0.88, 0]
        w[retarget.L_HEEL + side] = hip + [0, -0.93, 0.05]
        w[retarget.L_TOE + side] = hip + [0, -0.93, -0.15]
    p = math.radians(15)
    w[retarget.L_EAR] = [0.07, 0.68, 0]
    w[retarget.R_EAR] = [-0.07, 0.68, 0]
    w[retarget.NOSE] = [0, 0.68 - 0.1 * math.sin(p), -0.1 * math.cos(p)]
    return w


def angle_of(m):
    return math.degrees(math.acos(max(-1, min(1, (np.trace(m) - 1) / 2))))


def test_rest_pose_is_identity():
    bones = retarget.Retargeter(arm_angle=ARM).compute(rest_skeleton())
    for name, m in bones.items():
        assert angle_of(m) < 1e-3, (name, angle_of(m))


def test_raised_arm_rotates_only_the_arm():
    w = rest_skeleton()
    sh = w[retarget.L_SH]
    w[retarget.L_EL] = sh + [0.28, 0, 0]
    w[retarget.L_WR] = sh + [0.53, 0, 0]
    w[retarget.L_INDEX] = sh + [0.6, 0, -0.03]
    w[retarget.L_PINKY] = sh + [0.6, 0, 0.03]
    bones = retarget.Retargeter(arm_angle=ARM).compute(w)
    assert abs(angle_of(bones["左腕"]) - ARM) < 1e-3
    rest_dir = np.array([math.cos(math.radians(ARM)), -math.sin(math.radians(ARM)), 0])
    assert np.allclose(bones["左腕"] @ rest_dir, [1, 0, 0], atol=1e-6)
    assert angle_of(bones["左ひじ"]) < 1e-3
    assert angle_of(bones["右腕"]) < 1e-3


def test_bent_knee():
    w = rest_skeleton()
    knee = w[retarget.L_KNEE]
    w[retarget.L_ANK] = knee + [0, 0, 0.43]  # shin pointing backwards: 90 degree bend
    bones = retarget.Retargeter(arm_angle=ARM).compute(w)
    assert abs(angle_of(bones["左ひざ"]) - 90) < 1e-3
    # The knee hinge is the left-right axis.
    assert np.allclose(bones["左ひざ"] @ [0, -1, 0], [0, 0, 1], atol=1e-6)
    assert angle_of(bones["左足"]) < 1e-3


def test_turning_body_rotates_upper_and_lower_body():
    w = rest_skeleton()
    c, s = math.cos(math.radians(90)), math.sin(math.radians(90))
    yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    bones = retarget.Retargeter(arm_angle=ARM).compute(w @ yaw.T)
    assert np.allclose(bones["上半身"], yaw, atol=1e-6)
    assert np.allclose(bones["下半身"], yaw, atol=1e-6)
    for name in ("左腕", "右ひじ", "首", "頭", "左足", "右ひざ"):
        assert angle_of(bones[name]) < 1e-3, name


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = retarget.mat_to_quat(retarget.quat_to_mat(q))
        assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)


def test_vmd_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.vmd")
        frames = [BoneFrame("センター", 3, (1, 2, 3)), BoneFrame("左ひじ", 4, rotation=(0, 0, 0.6, 0.8))]
        write_vmd(path, frames, ik_off_bones=retarget.IK_BONES)
        back = read_vmd_bones(path)
        assert [(b.name, b.frame) for b in back] == [("センター", 3), ("左ひじ", 4)]
        assert np.allclose(back[0].position, (1, 2, 3))
        assert np.allclose(back[1].rotation, (0, 0, 0.6, 0.8))


def _det(x, color):
    px = np.zeros((33, 3))
    px[:, 0] = x
    px[:, 1] = np.linspace(100, 500, 33)
    hist = np.zeros(128)
    hist[color] = 1.0
    return {"image_px": px, "hist": hist}


def test_tracker_keeps_identity_when_people_cross():
    # Person A (colour 3) walks right, person B (colour 90) walks left; they
    # cross in the middle and detections come back in random order.
    rng = np.random.default_rng(1)
    frames = []
    for t in range(60):
        a = _det(100 + 10 * t, 3)
        b = _det(700 - 10 * t, 90)
        dets = [a, b]
        rng.shuffle(dets)
        if 28 <= t <= 31:
            dets = dets[:1]  # occlusion: only one detected while crossing
        frames.append(dets)
    out = tracker.assign_tracks(frames, 2)
    for t in range(60):
        if out[t][0] is not None:
            assert out[t][0]["hist"][3] == 1.0, t
        if out[t][1] is not None:
            assert out[t][1]["hist"][90] == 1.0, t
    assert out[0][0]["image_px"][0, 0] < out[0][1]["image_px"][0, 0]  # P1 starts on the left


def test_fill_gaps():
    arr = np.array([0.0, 0, 0, 0, 4.0])
    valid = np.array([True, False, False, False, True])
    filled, _ = filters.fill_gaps(arr, valid, max_gap=5)
    assert np.allclose(filled, [0, 1, 2, 3, 4])
