"""Keeps a stable identity for each person across frames.

MediaPipe returns the detected poses of a frame in no particular order, so a
person can be "person 0" in one frame and "person 1" in the next. This module
matches every frame's detections to persistent tracks using where each person
is (with a constant-velocity prediction), how big they are and the colour of
their clothes, so that crossing or briefly hidden dancers keep their identity.
"""
import itertools

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


def body_center(image_px):
    """Centre of the torso in pixels (mean of shoulders and hips)."""
    return image_px[[L_SH, R_SH, L_HIP, R_HIP], :2].mean(axis=0)


def body_height(image_px):
    """Rough body size in pixels, used to normalise distances."""
    xy = image_px[:, :2]
    return max(float(np.ptp(xy[:, 1])), float(np.ptp(xy[:, 0])), 1.0)


def hist_distance(h1, h2):
    if h1 is None or h2 is None:
        return 0.5
    # Bhattacharyya distance between normalised histograms, in [0, 1].
    bc = float(np.sum(np.sqrt(h1 * h2)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


class Track:
    def __init__(self, tid, det, frame):
        self.id = tid
        self.center = body_center(det["image_px"])
        self.velocity = np.zeros(2)
        self.size = body_height(det["image_px"])
        self.hist = det.get("hist")
        self.last_frame = frame

    def predicted_center(self, frame):
        return self.center + self.velocity * min(abs(frame - self.last_frame), 10)

    def update(self, det, frame):
        c = body_center(det["image_px"])
        dt = max(1, abs(frame - self.last_frame))
        self.velocity = 0.6 * self.velocity + 0.4 * (c - self.center) / dt
        self.center = c
        self.size = 0.8 * self.size + 0.2 * body_height(det["image_px"])
        if det.get("hist") is not None:
            self.hist = det["hist"] if self.hist is None else 0.9 * self.hist + 0.1 * det["hist"]
        self.last_frame = frame


def assign_tracks(frames, n_persons, color_weight=1.0, max_cost=2.5):
    """Assign detections to n_persons persistent tracks.

    frames: list (one entry per video frame) of lists of detection dicts with at
    least "image_px" (33x3 pixel coords). Returns a list of lists: out[t][k] is
    the detection for track k at frame t, or None.

    Tracks are numbered left to right as seen in the first frame where all
    persons are visible.
    """
    out = [[None] * n_persons for _ in frames]
    start = next((t for t, d in enumerate(frames) if len(d) >= n_persons), None)
    if start is None:
        start = next((t for t, d in enumerate(frames) if d), None)
        if start is None:
            return out

    first = sorted(frames[start], key=lambda d: -d.get("score", 1.0))[:n_persons]
    first.sort(key=lambda d: body_center(d["image_px"])[0])
    for k, d in enumerate(first):
        out[start][k] = d

    # Track forwards from the starting frame, then backwards to the beginning.
    _track_range(frames, out, range(start + 1, len(frames)), [Track(k, d, start) for k, d in enumerate(first)],
                 n_persons, color_weight, max_cost)
    _track_range(frames, out, range(start - 1, -1, -1), [Track(k, d, start) for k, d in enumerate(first)],
                 n_persons, color_weight, max_cost)
    return out


def _track_range(frames, out, order, tracks, n_persons, color_weight, max_cost):
    def cost(track, det, frame):
        pos = np.linalg.norm(body_center(det["image_px"]) - track.predicted_center(frame)) / track.size
        size = abs(np.log(body_height(det["image_px"]) / track.size))
        return pos + 0.5 * size + color_weight * hist_distance(track.hist, det.get("hist"))

    for t in order:
        dets = frames[t]
        if not dets:
            continue
        best, best_perm = None, None
        slots = list(range(len(dets))) + [None] * len(tracks)
        for perm in set(itertools.permutations(slots, len(tracks))):
            total = 0.0
            for tr, di in zip(tracks, perm):
                if di is None:
                    total += max_cost  # leaving a track empty costs as much as a bad match
                else:
                    total += min(cost(tr, dets[di], t), max_cost * 2)
            if best is None or total < best:
                best, best_perm = total, perm
        for tr, di in zip(tracks, best_perm):
            if di is None or cost(tr, dets[di], t) > max_cost:
                continue
            tr.update(dets[di], t)
            out[t][tr.id] = dets[di]

        # Late-appearing persons: start their track with an unused detection.
        if len(tracks) < n_persons:
            used = {id(d) for d in out[t] if d is not None}
            for d in dets:
                far = all(np.linalg.norm(body_center(d["image_px"]) - tr.center) > 0.3 * tr.size for tr in tracks)
                if id(d) not in used and far and len(tracks) < n_persons:
                    tr = Track(len(tracks), d, t)
                    tracks.append(tr)
                    out[t][tr.id] = d


def apply_swaps(assigned, swaps):
    """Swap identities from given frames onwards.

    swaps: list of (frame, a, b) - from `frame` on, tracks a and b trade places.
    Used to fix an identity switch spotted in the preview video.
    """
    for frame, a, b in sorted(swaps):
        for t in range(frame, len(assigned)):
            row = assigned[t]
            row[a], row[b] = row[b], row[a]
    return assigned
