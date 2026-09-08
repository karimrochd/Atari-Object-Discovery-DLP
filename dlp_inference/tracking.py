"""Hungarian-matching object tracker for DLPInference outputs.

Port of clean_pipeline's HungarianTracker (V1_models/clean_pipeline/
pipeline.py), operating on this package's per-frame TensorDicts:

    from dlp_inference import DLPInference
    from dlp_inference.tracking import HungarianTracker

    model = DLPInference("Asterix")
    tracker = HungarianTracker(frame_hw=(210, 160))
    for frame in frames:                     # consecutive frames
        tracked = tracker(model(frame))      # TensorDict in -> TensorDict out
        tracked["id"]                        # (N,) persistent int64 ids

The returned TensorDict has exactly the input attributes plus "id".
Association cost per (detection, active track) pair:

    if |dy| > max_dy or |dx| > max_dx:   rejected (position gate, [0,1] coords;
                                          default 0.12 = 25 px / 19 px at 210x160)
    else:  cost = w_feat * cos_dist(embeddings)
                + (1 - w_feat) * L2_centroid_distance / sqrt(2)

solved with scipy's linear_sum_assignment; pairs costing more than
``cost_thresh`` are rejected. Unmatched detections start new tracks; tracks
unmatched for more than ``max_age`` frames die; when a track re-matches
across a gap, the missed frames are back-filled into ``tracker.history``
with linearly interpolated rows (marked there by an "interpolated" key -
live outputs never contain ghosts). ``tracker.reset()`` (or
``new_episode=True``) clears active tracks at an episode boundary; ids keep
growing so they stay globally unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tensordict import TensorDict

_GATE_COST = 10.0

# per-object keys handled by the tracker (anything else in the TensorDict -
# e.g. background_embedding - is per-frame and passed through untouched)
_OBJ_KEYS = ("position", "size", "bbox", "confidence", "depth", "embedding",
             "id")
_ROW_KEYS = _OBJ_KEYS + ("interpolated",)      # history rows carry the marker
_LERP_KEYS = ("position", "size", "bbox", "confidence")


@dataclass
class _Track:
    tid: int
    center: np.ndarray            # (cy, cx) in [0, 1]
    feat: Optional[np.ndarray]    # L2-normed embedding
    row: dict                     # last detection as 1-row tensors
    last_frame: int


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 0.5 * (1.0 - float(np.dot(a, b) / (na * nb)))


class HungarianTracker:
    def __init__(self, frame_hw=(210, 160), cost_thresh: float = 0.5,
                 w_feat: float = 0.3, max_age: int = 3,
                 max_dy: float = 0.12, max_dx: float = 0.12):
        """frame_hw: (H, W) of the frames given to DLPInference - used to
        normalize the pixel positions to [0, 1] for the distance gate."""
        assert 0.0 <= w_feat <= 1.0
        self.frame_hw = frame_hw
        self.cost_thresh = cost_thresh
        self.w_feat = w_feat
        self.max_age = max_age
        self.max_dy = max_dy
        self.max_dx = max_dx
        self.active: Dict[int, _Track] = {}
        self.next_id = 0
        self.history: Dict[int, TensorDict] = {}
        self._t = 0

    def reset(self):
        """Forget active tracks (new episode); ids stay globally unique."""
        self.active = {}

    # ------------------------------------------------------------------ #
    def __call__(self, dets: TensorDict, new_episode: bool = False
                 ) -> TensorDict:
        """Track one frame: TensorDict in, same TensorDict + "id" out.

        Frames must be passed in temporal order; the frame index is kept
        internally. Use ``new_episode=True`` (or ``reset()``) at cuts."""
        out = self.step(self._t, dets, new_episode=new_episode)
        self._t += 1
        return out

    def step(self, frame_idx: int, dets: TensorDict,
             new_episode: bool = False) -> TensorDict:
        """Same as __call__ but with an explicit frame index."""
        if new_episode:
            self.reset()
        for tid in [t for t, tr in self.active.items()
                    if frame_idx - tr.last_frame > self.max_age]:
            del self.active[tid]

        n = len(dets["confidence"])
        ids = torch.full((n,), -1, dtype=torch.int64)

        if n:
            h, w = self.frame_hw
            pos = dets["position"].detach().cpu().numpy()      # (N, 2) px (x, y)
            centers = np.stack([pos[:, 1] / h, pos[:, 0] / w], axis=1)
            feats = dets["embedding"].detach().cpu().numpy().astype(np.float64)
            feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True),
                                1e-8)

            tracks = list(self.active.values())
            matched = set()
            if tracks:
                cost = np.full((n, len(tracks)), _GATE_COST, np.float32)
                for i in range(n):
                    for j, tr in enumerate(tracks):
                        dy = abs(centers[i, 0] - tr.center[0])
                        dx = abs(centers[i, 1] - tr.center[1])
                        if dy > self.max_dy or dx > self.max_dx:
                            continue
                        spatial = float(np.hypot(dy, dx)) / np.sqrt(2.0)
                        feat = (_cos_dist(feats[i], tr.feat)
                                if tr.feat is not None else spatial)
                        cost[i, j] = (self.w_feat * feat
                                      + (1 - self.w_feat) * spatial)
                r_idx, c_idx = linear_sum_assignment(cost)
                for r, c in zip(r_idx, c_idx):
                    if cost[r, c] < self.cost_thresh:
                        ids[r] = tracks[c].tid
                        matched.add(int(r))
            for i in range(n):
                if i not in matched:
                    ids[i] = self.next_id
                    self.next_id += 1

        # live output: input attributes + id, nothing else
        out = dets.clone()
        out["id"] = ids

        # history copy carries the ghost marker; update/create track states
        hist = out.clone()
        hist["interpolated"] = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            tid = int(ids[i])
            row = {k: hist[k][i : i + 1].clone() for k in _ROW_KEYS}
            if tid in self.active:
                self._extend(self.active[tid], frame_idx, row,
                             centers[i], feats[i])
            else:
                self.active[tid] = _Track(tid, centers[i].copy(),
                                          feats[i].copy(), row, frame_idx)
        self.history[frame_idx] = hist
        return out

    # ------------------------------------------------------------------ #
    def _extend(self, tr: _Track, frame_idx: int, row: dict,
                center: np.ndarray, feat: np.ndarray):
        gap = frame_idx - tr.last_frame
        if gap > 1:
            prev = tr.row
            for k in range(1, gap):
                a = k / gap
                f = tr.last_frame + k
                hist = self.history.get(f)
                if hist is None:
                    continue
                ghost = {key: (1 - a) * prev[key] + a * row[key]
                         if key in _LERP_KEYS else prev[key].clone()
                         for key in _ROW_KEYS}
                ghost["id"] = torch.tensor([tr.tid])
                ghost["interpolated"] = torch.tensor([True])
                for key in _ROW_KEYS:
                    hist[key] = torch.cat(
                        [hist[key], ghost[key].to(hist[key].device)])
        tr.center = center.copy()
        tr.feat = feat.copy()
        tr.row = row
        tr.last_frame = frame_idx
