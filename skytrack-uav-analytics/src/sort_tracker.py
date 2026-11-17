"""SORT multi-object tracker, implemented from scratch.

Simple Online and Realtime Tracking (Bewley et al., 2016):
- Each track carries a Kalman filter with a constant-velocity model over
  the bounding-box state [u, v, s, r, du, dv, ds] (centre x, centre y,
  scale = area, aspect ratio, and their velocities).
- Frame-to-frame association between predicted track boxes and new
  detections uses IoU cost + the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment).
- Track lifecycle: tentative tracks must be hit `min_hits` times before
  they are reported; tracks unseen for `max_age` frames are deleted.

This module is deliberately dependency-light (numpy + scipy only) so the
Kalman filter and data association are fully auditable — no black-box
tracking library is used.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of boxes in [x1, y1, x2, y2] format."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)))
    xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
    yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
    xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
    yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_t = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_g = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_t[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> measurement [u, v, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([u, v, s, r]).reshape((4, 1))


def _to_bbox(state: np.ndarray) -> np.ndarray:
    """State [u, v, s, r, ...] -> [x1, y1, x2, y2]."""
    u, v, s, r = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
    w = np.sqrt(max(s * r, 1e-6))
    h = s / max(w, 1e-6)
    return np.array([u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0])


class KalmanBoxTracker:
    """One Kalman filter per tracked object (constant-velocity model)."""

    _count = 0

    def __init__(self, bbox: np.ndarray):
        # State: [u, v, s, r, du, dv, ds]; measurement: [u, v, s, r]
        self.x = np.zeros((7, 1))
        self.x[:4] = _to_z(bbox)
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocity
        self.Q = np.eye(7)
        self.Q[6, 6] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.cls = -1
        self.conf = 0.0

    def predict(self) -> np.ndarray:
        # Prevent negative scale
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _to_bbox(self.x)

    def update(self, bbox: np.ndarray, cls: int = -1, conf: float = 0.0):
        z = _to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.cls = cls
        self.conf = conf

    def get_bbox(self) -> np.ndarray:
        return _to_bbox(self.x)


class SortTracker:
    """SORT: Kalman prediction + IoU/Hungarian association."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.total_id_switch_proxies = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """detections: N x 6 array [x1, y1, x2, y2, conf, cls].
        Returns M x 7 array [x1, y1, x2, y2, track_id, cls, conf] for
        confirmed tracks."""
        self.frame_count += 1
        dets = detections if len(detections) else np.empty((0, 6))

        # 1. Predict all existing tracks
        predicted = []
        for t in self.tracks:
            predicted.append(t.predict())
        predicted = np.array(predicted) if predicted else np.empty((0, 4))

        # 2. Associate detections to predicted tracks by IoU
        matched, unmatched_dets, unmatched_trks = [], [], []
        if len(predicted) and len(dets):
            iou = iou_batch(dets[:, :4], predicted)
            cost = 1.0 - iou
            rows, cols = linear_sum_assignment(cost)
            matched_set_d, matched_set_t = set(), set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matched.append((r, c))
                    matched_set_d.add(r)
                    matched_set_t.add(c)
            unmatched_dets = [i for i in range(len(dets)) if i not in matched_set_d]
            unmatched_trks = [i for i in range(len(predicted)) if i not in matched_set_t]
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(predicted)))

        # 3. Update matched tracks
        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(dets[d_idx, :4], cls=int(dets[d_idx, 5]),
                                      conf=float(dets[d_idx, 4]))
        # 4. Spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[d_idx, :4])
            trk.update(dets[d_idx, :4], cls=int(dets[d_idx, 5]), conf=float(dets[d_idx, 4]))
            self.tracks.append(trk)
        # 5. Delete stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6. Emit confirmed tracks
        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bb = t.get_bbox()
                out.append([bb[0], bb[1], bb[2], bb[3], t.id, t.cls, t.conf])
        return np.array(out) if out else np.empty((0, 7))
