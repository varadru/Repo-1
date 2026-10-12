"""Minimal VMD (Vocaloid Motion Data) writer.

Only bone keyframes and IK on/off frames are written; morph, camera, light and
shadow sections are left empty. Strings are Shift-JIS (cp932) encoded, as MMD
expects.
"""
import struct

# Linear interpolation curve (control points 20,20 / 107,107) for X, Y, Z and
# rotation. Row 0 holds the 16 meaningful bytes; the following rows repeat it
# shifted by one byte each, which is the layout MMD itself writes.
_BASE = [20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107]
LINEAR_INTERPOLATION = bytes(b for r in range(4) for b in _BASE[r:] + [0] * r)

assert len(LINEAR_INTERPOLATION) == 64


def _fixed(text, size):
    raw = text.encode("cp932")
    if len(raw) > size:
        raise ValueError(f"name too long for VMD field ({size} bytes): {text!r}")
    return raw + b"\0" * (size - len(raw))


class BoneFrame:
    __slots__ = ("name", "frame", "position", "rotation")

    def __init__(self, name, frame, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
        self.name = name
        self.frame = int(frame)
        self.position = tuple(float(v) for v in position)
        self.rotation = tuple(float(v) for v in rotation)  # quaternion x, y, z, w


def write_vmd(path, bone_frames, model_name="MultiPersonMocap", ik_off_bones=None):
    """Write a VMD file.

    bone_frames: iterable of BoneFrame.
    ik_off_bones: optional list of IK bone names to disable at frame 0, so the
    FK leg rotations in the motion are used instead of the model's leg IK.
    """
    bone_frames = list(bone_frames)
    out = bytearray()
    out += _fixed("Vocaloid Motion Data 0002", 30)
    out += _fixed(model_name, 20)

    out += struct.pack("<I", len(bone_frames))
    for bf in bone_frames:
        out += _fixed(bf.name, 15)
        out += struct.pack("<I3f4f", bf.frame, *bf.position, *bf.rotation)
        out += LINEAR_INTERPOLATION

    out += struct.pack("<I", 0)  # morphs
    out += struct.pack("<I", 0)  # camera
    out += struct.pack("<I", 0)  # light
    out += struct.pack("<I", 0)  # self shadow

    if ik_off_bones:
        out += struct.pack("<I", 1)
        out += struct.pack("<IBI", 0, 1, len(ik_off_bones))
        for name in ik_off_bones:
            out += _fixed(name, 20)
            out += struct.pack("<B", 0)
    else:
        out += struct.pack("<I", 0)

    with open(path, "wb") as f:
        f.write(out)


def read_vmd_bones(path):
    """Read bone keyframes back (used by the tests and for sanity checks)."""
    with open(path, "rb") as f:
        data = f.read()
    off = 50
    (count,) = struct.unpack_from("<I", data, off)
    off += 4
    frames = []
    for _ in range(count):
        name = data[off:off + 15].split(b"\0", 1)[0].decode("cp932")
        off += 15
        frame, px, py, pz, rx, ry, rz, rw = struct.unpack_from("<I3f4f", data, off)
        off += 32 + 64
        frames.append(BoneFrame(name, frame, (px, py, pz), (rx, ry, rz, rw)))
    return frames
