"""SORT tracker: Kalman filter + IoU association + Hungarian matching.

Reference: Bewley et al., "Simple Online and Realtime Tracking" (2016).
Implemented in-repo — not a wrapper around Ultralytics ``model.track()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Detection / track row: [x1, y1, x2, y2, score, class_id]
BBox = np.ndarray


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype=np.float64)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - inter
    return inter / np.maximum(union, 1e-9)


def convert_bbox_to_z(bbox: Sequence[float]) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx, cy, s, r]^T where s=area, r=aspect."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([x, y, s, r], dtype=np.float64).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: Optional[float] = None) -> np.ndarray:
    """Kalman state center form -> [x1,y1,x2,y2] (+ optional score)."""
    w = np.sqrt(max(x[2, 0] * x[3, 0], 0.0))
    h = x[2, 0] / max(w, 1e-6)
    x1 = x[0, 0] - w / 2.0
    y1 = x[1, 0] - h / 2.0
    x2 = x[0, 0] + w / 2.0
    y2 = x[1, 0] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2], dtype=np.float64)
    return np.array([x1, y1, x2, y2, score], dtype=np.float64)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over bbox center / scale / aspect."""

    _count = 0

    def __init__(self, bbox: Sequence[float], score: float = 1.0, class_id: int = 0):
        # State: [cx, cy, s, r, vx, vy, vs]
        self.kf_x = np.zeros((7, 1), dtype=np.float64)
        self.kf_P = np.eye(7, dtype=np.float64)
        self.kf_F = np.eye(7, dtype=np.float64)
        self.kf_H = np.zeros((4, 7), dtype=np.float64)
        self.kf_R = np.eye(4, dtype=np.float64)
        self.kf_Q = np.eye(7, dtype=np.float64)

        self.kf_F[0, 4] = 1.0
        self.kf_F[1, 5] = 1.0
        self.kf_F[2, 6] = 1.0
        self.kf_H[0, 0] = 1.0
        self.kf_H[1, 1] = 1.0
        self.kf_H[2, 2] = 1.0
        self.kf_H[3, 3] = 1.0

        self.kf_P[4:, 4:] *= 1000.0
        self.kf_P *= 10.0
        self.kf_R[2:, 2:] *= 10.0
        self.kf_Q[-1, -1] *= 0.01
        self.kf_Q[4:, 4:] *= 0.01

        self.kf_x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.history: List[np.ndarray] = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.score = float(score)
        self.class_id = int(class_id)

    def predict(self) -> np.ndarray:
        if (self.kf_x[2, 0] + self.kf_x[6, 0]) <= 0:
            self.kf_x[6, 0] *= 0.0
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf_x))
        return self.history[-1]

    def update(self, bbox: Sequence[float], score: float = 1.0, class_id: Optional[int] = None) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        if class_id is not None:
            self.class_id = int(class_id)

        z = convert_bbox_to_z(bbox)
        y = z - self.kf_H @ self.kf_x
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(7) - K @ self.kf_H) @ self.kf_P

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf_x)


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match detections to predicted tracks via IoU + Hungarian."""
    if trackers.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )
    if detections.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers)),
        )

    iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
    if iou_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    # Maximize IoU <=> minimize (1 - IoU)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
    matched: List[List[int]] = []
    unmatched_dets = set(range(detections.shape[0]))
    unmatched_trks = set(range(trackers.shape[0]))

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            continue
        matched.append([r, c])
        unmatched_dets.discard(r)
        unmatched_trks.discard(c)

    matches = np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int)
    return matches, np.array(sorted(unmatched_dets), dtype=int), np.array(
        sorted(unmatched_trks), dtype=int
    )


@dataclass
class TrackResult:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    score: float
    class_id: int
    coasted: bool = False  # True when box is Kalman-predicted (no match this frame)


class SortTracker:
    """Online multi-object SORT tracker."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        coast_frames: int = 25,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # Emit predicted boxes for this many frames after the last match (≤ max_age).
        self.coast_frames = max(0, min(int(coast_frames), int(max_age)))
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0

    def reset(self) -> None:
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0

    def update(self, detections: Optional[np.ndarray] = None) -> List[TrackResult]:
        """Update with detections shaped (N, 5+) = xyxy + score [+ class].

        Returns confirmed tracks for the current frame.
        """
        self.frame_count += 1
        if detections is None:
            detections = np.empty((0, 5), dtype=np.float64)
        else:
            detections = np.asarray(detections, dtype=np.float64)
            if detections.ndim == 1:
                detections = detections.reshape(1, -1)
            if detections.size == 0:
                detections = np.empty((0, 5), dtype=np.float64)

        # Predict
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)
        trks_arr = np.array(trks, dtype=np.float64) if trks else np.empty((0, 4))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks_arr, self.iou_threshold
        )

        for m in matched:
            det = detections[m[0]]
            score = float(det[4]) if det.shape[0] > 4 else 1.0
            cls = int(det[5]) if det.shape[0] > 5 else 0
            self.trackers[m[1]].update(det[:4], score=score, class_id=cls)

        for i in unmatched_dets:
            det = detections[i]
            score = float(det[4]) if det.shape[0] > 4 else 1.0
            cls = int(det[5]) if det.shape[0] > 5 else 0
            self.trackers.append(KalmanBoxTracker(det[:4], score=score, class_id=cls))

        results: List[TrackResult] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            matched = trk.time_since_update < 1
            # Matched: same gate as classic SORT (hit_streak / early frames).
            # Coast: only after the track is confirmed (hits), through brief misses.
            emit_matched = matched and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            )
            emit_coast = (
                (not matched)
                and self.coast_frames > 0
                and trk.time_since_update <= self.coast_frames
                and trk.hits >= self.min_hits
            )
            if emit_matched or emit_coast:
                results.append(
                    TrackResult(
                        track_id=trk.id + 1,  # 1-indexed IDs (SORT convention)
                        xyxy=(float(d[0]), float(d[1]), float(d[2]), float(d[3])),
                        score=trk.score,
                        class_id=trk.class_id,
                        coasted=emit_coast,
                    )
                )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return results
