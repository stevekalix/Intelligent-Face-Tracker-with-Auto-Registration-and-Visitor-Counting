"""
Multi-Object Tracking Module for Intelligent Face Tracker & Visitor Counting.
Implements the ByteTrack association algorithm with an 8-state Kalman Filter,
two-stage Hungarian matching, frame-skipping motion extrapolation, and life-cycle exit detection.
"""

import time
import logging
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("Tracker")


class TrackState(Enum):
    """Lifecycle states for a tracked face identity."""
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    REMOVED = 4


class KalmanBoxTracker:
    """Standard 8-dimensional Kalman Filter for bounding box tracking [x, y, a, h, vx, vy, va, vh].

    x, y: bounding box center
    a: aspect ratio (width / height)
    h: height
    vx, vy, va, vh: corresponding velocities
    """

    count = 0

    def __init__(self, bbox: List[float]):
        """Initialize state from [x1, y1, x2, y2]."""
        # State vector: [x, y, a, h, vx, vy, va, vh]^T
        self.dim_x = 8
        self.dim_z = 4

        # Transition matrix F
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0, 10000.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.01])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

        # State initialization
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = self._bbox_to_z(bbox)

    @staticmethod
    def _bbox_to_z(bbox: List[float]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [center_x, center_y, aspect_ratio, height]."""
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        x = float(bbox[0]) + w / 2.0
        y = float(bbox[1]) + h / 2.0
        s = w / h
        return np.array([[x], [y], [s], [h]])

    @staticmethod
    def _z_to_bbox(x: np.ndarray) -> List[int]:
        """Convert [center_x, center_y, aspect_ratio, height] to [x1, y1, x2, y2]."""
        w = x[2, 0] * x[3, 0]
        h = x[3, 0]
        x1 = x[0, 0] - w / 2.0
        y1 = x[1, 0] - h / 2.0
        x2 = x[0, 0] + w / 2.0
        y2 = x[1, 0] + h / 2.0
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    def predict(self) -> List[int]:
        """Advance the state vector and return the predicted bounding box."""
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self._z_to_bbox(self.x)

    def update(self, bbox: List[float]) -> None:
        """Correct the state distribution with a new measurement."""
        z = self._bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

    def get_state(self) -> List[int]:
        """Return current bounding box estimate."""
        return self._z_to_bbox(self.x)


class Track:
    """Represents a single identity tracked across video frames."""

    def __init__(self, bbox: List[int], score: float, face_crop: Optional[np.ndarray] = None):
        KalmanBoxTracker.count += 1
        self.track_id: int = KalmanBoxTracker.count
        self.person_id: Optional[str] = None  # Associated SQLite DB visitor identity (e.g. VISITOR_0001)
        self.kalman = KalmanBoxTracker(bbox)
        self.bbox: List[int] = bbox
        self.score: float = score
        self.best_crop: Optional[np.ndarray] = face_crop
        self.best_score: float = score
        self.embedding: Optional[np.ndarray] = None

        self.state: TrackState = TrackState.TENTATIVE
        self.hits: int = 1
        self.age: int = 0
        self.time_since_update: int = 0

        # Session timestamps
        self.start_time: datetime = datetime.now()
        self.last_seen_time: datetime = datetime.now()

        # Invariant event logging flags (guarantees exactly one entry and one exit log)
        self.entry_logged: bool = False
        self.exit_logged: bool = False

    def update(self, bbox: List[int], score: float, face_crop: Optional[np.ndarray] = None) -> None:
        """Update track with a newly matched detection."""
        self.kalman.update(bbox)
        self.bbox = self.kalman.get_state()
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_time = datetime.now()

        if score > self.best_score and face_crop is not None:
            self.best_score = score
            self.best_crop = face_crop

        if self.state == TrackState.TENTATIVE and self.hits >= 2:
            self.state = TrackState.CONFIRMED

    def predict(self) -> List[int]:
        """Predict next bounding box position."""
        self.bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def mark_lost(self) -> None:
        """Mark track as temporarily lost due to occlusion or missed detection."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Mark track as permanently exited."""
        self.state = TrackState.REMOVED


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Perform optimal bipartite assignment using Hungarian algorithm."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matches, list(unmatched_rows), list(unmatched_cols)


class ByteTracker:
    """ByteTrack Multi-Object Tracker integrating two-stage IoU association."""

    def __init__(
        self,
        track_thresh: float = 0.50,
        high_thresh: float = 0.60,
        match_thresh: float = 0.80,
        max_lost_frames: int = 30
    ):
        """Initialize ByteTracker parameters.

        Args:
            track_thresh: Detection score threshold for first association stage.
            high_thresh: Score threshold above which new tracks can be instantiated.
            match_thresh: Maximum IoU distance threshold (1 - IoU) for matching.
            max_lost_frames: Maximum missed frames before track is declared exited.
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames

        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.exited_tracks: List[Track] = []

    def predict_skipped_frame(self) -> List[Track]:
        """Extrapolate bounding box positions during detection-skipped frames.

        Advances Kalman filters without invoking heavy neural network detection.

        Returns:
            List of active confirmed tracks with predicted bounding boxes.
        """
        for t in self.tracked_tracks:
            t.predict()
        return [t for t in self.tracked_tracks if t.state == TrackState.CONFIRMED]

    def update(
        self,
        detections: List[Tuple[List[int], float, np.ndarray]]
    ) -> List[Track]:
        """Update tracker state with new frame detections.

        Args:
            detections: List of (bbox [x1, y1, x2, y2], score, face_crop).

        Returns:
            List of currently active confirmed tracks in this frame.
        """
        # Step 1: Predict positions of all existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # Step 2: Separate detections into high-score and low-score pools
        high_dets = []
        low_dets = []
        for det in detections:
            bbox, score, crop = det
            if score >= self.track_thresh:
                high_dets.append(det)
            elif score >= 0.10:
                low_dets.append(det)

        # Step 3: First Stage Association (Tracked tracks vs High-score detections)
        active_pool = [t for t in self.tracked_tracks if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)]
        cost_matrix = np.zeros((len(active_pool), len(high_dets)), dtype=np.float32)

        for i, t in enumerate(active_pool):
            for j, d in enumerate(high_dets):
                cost_matrix[i, j] = 1.0 - calculate_iou(t.bbox, d[0])

        matches_a, unmatched_tracks_a, unmatched_dets_a = linear_assignment(cost_matrix, self.match_thresh)

        # Update matched tracks from stage 1
        for t_idx, d_idx in matches_a:
            bbox, score, crop = high_dets[d_idx]
            active_pool[t_idx].update(bbox, score, crop)

        # Step 4: Second Stage Association (Unmatched tracks vs Low-score detections)
        remaining_tracks = [active_pool[i] for i in unmatched_tracks_a if active_pool[i].state == TrackState.CONFIRMED]
        cost_matrix_b = np.zeros((len(remaining_tracks), len(low_dets)), dtype=np.float32)

        for i, t in enumerate(remaining_tracks):
            for j, d in enumerate(low_dets):
                cost_matrix_b[i, j] = 1.0 - calculate_iou(t.bbox, d[0])

        matches_b, unmatched_tracks_b, _ = linear_assignment(cost_matrix_b, 0.70)

        for t_idx, d_idx in matches_b:
            bbox, score, crop = low_dets[d_idx]
            remaining_tracks[t_idx].update(bbox, score, crop)

        # Mark tracks that failed both stages as LOST
        for t_idx in unmatched_tracks_b:
            remaining_tracks[t_idx].mark_lost()
            if remaining_tracks[t_idx] not in self.lost_tracks:
                self.lost_tracks.append(remaining_tracks[t_idx])

        # Step 5: Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_dets_a:
            bbox, score, crop = high_dets[d_idx]
            if score >= self.high_thresh:
                new_track = Track(bbox, score, crop)
                # If high confidence is very strong, confirm immediately
                if score >= 0.75:
                    new_track.state = TrackState.CONFIRMED
                self.tracked_tracks.append(new_track)

        # Step 6: Life-Cycle Clean Up (Identify Exits)
        alive_tracks = []
        for t in self.tracked_tracks:
            if t.time_since_update > self.max_lost_frames:
                t.mark_removed()
                if not t.exit_logged:
                    self.exited_tracks.append(t)
            else:
                alive_tracks.append(t)
        self.tracked_tracks = alive_tracks

        alive_lost = []
        for t in self.lost_tracks:
            if t.time_since_update > self.max_lost_frames:
                t.mark_removed()
                if not t.exit_logged and t not in self.exited_tracks:
                    self.exited_tracks.append(t)
            elif t.state == TrackState.CONFIRMED:
                pass  # Recovered back to tracked
            else:
                alive_lost.append(t)
        self.lost_tracks = alive_lost

        # Return all confirmed tracks in the scene
        return [t for t in self.tracked_tracks if t.state == TrackState.CONFIRMED and t.time_since_update == 0]

    def get_newly_exited_tracks(self) -> List[Track]:
        """Fetch all tracks that have transitioned to exited state and need EXIT event logging.

        Returns:
            List of exited Track objects.
        """
        ready_to_log = [t for t in self.exited_tracks if not t.exit_logged]
        for t in ready_to_log:
            t.exit_logged = True
        return ready_to_log
