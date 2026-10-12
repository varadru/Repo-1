"""Turn MediaPipe 3D landmarks into MMD bone rotations.

All maths happens in MMD's coordinate system: +X is screen right (the model's
left when it faces the camera), +Y is up and +Z points away from the camera.
MediaPipe world landmarks use x right, y down, z away from the camera, so the
conversion is just a flip of Y.

MMD bones carry no local axes: a VMD rotation is expressed in global axes
relative to the rest pose, and is local to the parent bone. For each body
segment we therefore build an orthonormal frame from the landmarks and from a
matching frame in the rest pose; the segment's global rotation is
R = B_current * B_rest^T and the value written to the VMD is
R_parent^T * R.
"""
import math

import numpy as np

# MediaPipe pose landmark indices ("left" is the person's left).
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

UP = np.array([0.0, 1.0, 0.0])
DOWN = -UP
RIGHT = np.array([1.0, 0.0, 0.0])  # model's left side
FORWARD = np.array([0.0, 0.0, -1.0])  # the way the model faces
BACK = -FORWARD

IK_BONES = ["左足ＩＫ", "右足ＩＫ", "左つま先ＩＫ", "右つま先ＩＫ"]


def mediapipe_to_mmd(points):
    p = np.array(points, dtype=np.float64)
    p[..., 1] *= -1.0
    return p


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def frame_from(primary, secondary):
    """Orthonormal basis (columns) with exact `primary` and `secondary` in-plane."""
    x = _unit(primary)
    s = secondary - np.dot(secondary, x) * x
    if np.linalg.norm(s) < 1e-6:
        # Degenerate: pick any perpendicular vector.
        s = np.cross(x, UP if abs(x[1]) < 0.9 else RIGHT)
    y = _unit(s)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def rotation_between_frames(current, rest):
    return current @ rest.T


def mat_to_quat(m):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def partial_rotation(m, fraction):
    """The rotation `fraction` of the way from identity to m (same axis)."""
    q = mat_to_quat(m)
    if q[3] < 0:
        q = -q
    angle = 2 * math.acos(min(1.0, q[3]))
    axis = q[:3]
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    axis = axis / n
    half = angle * fraction / 2
    return quat_to_mat(np.concatenate([axis * math.sin(half), [math.cos(half)]]))


class Retargeter:
    """Computes MMD bone rotations for one person, frame by frame.

    arm_angle: how far below horizontal the model's arms point in its rest pose
    (degrees). Standard MMD models are in an A-pose of roughly 30-40 degrees.
    """

    def __init__(self, arm_angle=35.0, head_pitch=15.0, wrists=True, ankles=True):
        a = math.radians(arm_angle)
        self.rest_arm = {
            "左": np.array([math.cos(a), -math.sin(a), 0.0]),
            "右": np.array([-math.cos(a), -math.sin(a), 0.0]),
        }
        p = math.radians(head_pitch)
        self.rest_face = np.array([0.0, -math.sin(p), -math.cos(p)])
        self.wrists = wrists
        self.ankles = ankles

    def _limb(self, root, mid, end, rest_dir, rest_bend_dir, fallback_hint, rest_hint):
        """Two-bone limb (arm/leg). Returns global rotations of both bones and the bend axis."""
        upper = _unit(mid - root)
        lower = _unit(end - mid)
        bend = np.cross(upper, lower)
        # When the limb is almost straight the bend axis is undefined; a small
        # contribution from a hint vector (hand or foot direction) keeps it stable.
        bend = bend + 0.15 * _unit(np.cross(upper, fallback_hint))
        rest_bend = np.cross(rest_dir, rest_bend_dir) + 0.15 * _unit(np.cross(rest_dir, rest_hint))
        rest_frame = frame_from(rest_dir, rest_bend)
        r_upper = rotation_between_frames(frame_from(upper, bend), rest_frame)
        r_lower = rotation_between_frames(frame_from(lower, bend), rest_frame)
        return r_upper, r_lower, bend, rest_bend

    def compute(self, w):
        """w: 33x3 landmarks in MMD axes (metres). Returns {bone: 3x3 local rotation}."""
        hip_mid = (w[L_HIP] + w[R_HIP]) / 2
        sh_mid = (w[L_SH] + w[R_SH]) / 2
        spine_up = sh_mid - hip_mid

        bones = {}
        r_lower_body = rotation_between_frames(frame_from(w[L_HIP] - w[R_HIP], spine_up), frame_from(RIGHT, UP))
        r_upper_body = rotation_between_frames(frame_from(spine_up, w[L_SH] - w[R_SH]), frame_from(UP, RIGHT))
        bones["下半身"] = r_lower_body
        bones["上半身"] = r_upper_body

        # Head, split evenly between neck and head bones.
        ear_mid = (w[L_EAR] + w[R_EAR]) / 2
        r_head_global = rotation_between_frames(
            frame_from(w[L_EAR] - w[R_EAR], w[NOSE] - ear_mid), frame_from(RIGHT, self.rest_face)
        )
        head_local = r_upper_body.T @ r_head_global
        neck = partial_rotation(head_local, 0.5)
        bones["首"] = neck
        bones["頭"] = neck.T @ head_local

        for side, (sh, el, wr, pinky, index) in {
            "左": (L_SH, L_EL, L_WR, L_PINKY, L_INDEX),
            "右": (R_SH, R_EL, R_WR, R_PINKY, R_INDEX),
        }.items():
            thumb_side = w[index] - w[pinky]
            r_arm, r_elbow, bend, rest_bend = self._limb(
                w[sh], w[el], w[wr], self.rest_arm[side], FORWARD, thumb_side, FORWARD
            )
            bones[side + "腕"] = r_upper_body.T @ r_arm
            bones[side + "ひじ"] = r_arm.T @ r_elbow
            if self.wrists:
                hand_dir = (w[index] + w[pinky]) / 2 - w[wr]
                r_hand = rotation_between_frames(
                    frame_from(hand_dir, thumb_side), frame_from(self.rest_arm[side], FORWARD)
                )
                bones[side + "手首"] = r_elbow.T @ r_hand

        for side, (hip, knee, ank, heel, toe) in {
            "左": (L_HIP, L_KNEE, L_ANK, L_HEEL, L_TOE),
            "右": (R_HIP, R_KNEE, R_ANK, R_HEEL, R_TOE),
        }.items():
            foot_dir = w[toe] - w[heel]
            r_thigh, r_shin, bend, rest_bend = self._limb(
                w[hip], w[knee], w[ank], DOWN, BACK, -foot_dir, BACK
            )
            bones[side + "足"] = r_lower_body.T @ r_thigh
            bones[side + "ひざ"] = r_thigh.T @ r_shin
            if self.ankles:
                r_foot = rotation_between_frames(frame_from(foot_dir, bend), frame_from(FORWARD, rest_bend))
                bones[side + "足首"] = r_shin.T @ r_foot
        return bones
