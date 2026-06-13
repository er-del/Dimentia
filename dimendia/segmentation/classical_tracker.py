"""Weight-free object tracker: motion + saliency + depth -> tracked masks.

Builds a foreground probability from optical-flow motion, spectral-residual
saliency, and depth proximity; thresholds it into connected components; optionally
tightens each with GrabCut; then associates components to the previous frame by
IoU to assign stable ids and estimate velocity. This is the fallback used when
SAM2 is not installed, and it keeps the whole pipeline runnable on CPU.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from dimendia.config import Mode, PipelineConfig
from dimendia.imageops import normalize01
from dimendia.logging import get_logger
from dimendia.segmentation.base import ObjectTracker
from dimendia.segmentation.flow import flow_magnitude
from dimendia.segmentation.saliency import spectral_residual_saliency
from dimendia.types import DepthMap, Frame, TrackedObject

log = get_logger(__name__)

_GRABCUT_TIMEOUT = 15  # seconds


def _grabcut_with_timeout(
    bgr, gc, bgd, fgd, iterations: int, timeout: int = _GRABCUT_TIMEOUT
) -> bool:
    """Run cv2.grabCut with a timeout so large masks don't hang the pipeline."""
    exc: list[BaseException | None] = [None]

    def _target() -> None:
        try:
            cv2.grabCut(bgr, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        log.warning("cv2.grabCut timed out after %ds; skipping refinement", timeout)
        return False
    if exc[0] is not None:
        if isinstance(exc[0], cv2.error):
            return False
        raise exc[0]  # type: ignore[misc]
    return True


class ClassicalTracker(ObjectTracker):
    name = "classical"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_objects = 4
        self.use_grabcut = config.mode != Mode.FAST
        # Raised min area to filter noise fragments that become black blobs.
        self.min_area_ratio = 0.01
        # Lowered max area to avoid selecting the entire frame as foreground.
        self.max_area_ratio = 0.70
        self.max_lost = 10  # frames a track survives on Kalman prediction alone
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._diag = 1.0

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 0

    def track(self, frame: Frame, depth: DepthMap, flow: np.ndarray | None) -> list[TrackedObject]:
        h, w = frame.shape[:2]
        self._diag = float(np.hypot(h, w))
        fg = self._foreground_probability(frame, depth, flow)
        components = self._components(fg, h, w)

        objects: list[TrackedObject] = []
        saliency = spectral_residual_saliency(frame)
        for mask in components:
            if self.use_grabcut:
                mask = self._refine_grabcut(frame, mask)
            obj = self._describe(mask, depth, saliency, frame.shape[:2])
            if obj is not None:
                objects.append(obj)

        objects = self._associate(objects)
        return objects

    # -- cue fusion ----------------------------------------------------------

    def _foreground_probability(
        self, frame: Frame, depth: DepthMap, flow: np.ndarray | None
    ) -> np.ndarray:
        saliency = spectral_residual_saliency(frame)
        proximity = normalize01(depth)
        if flow is not None:
            motion = normalize01(flow_magnitude(flow))
            fg = 0.45 * motion + 0.30 * saliency + 0.25 * proximity
        else:  # first frame / static: rely on appearance + depth
            fg = 0.55 * saliency + 0.45 * proximity
        return normalize01(fg)

    def _components(self, fg: np.ndarray, h: int, w: int) -> list[np.ndarray]:
        u8 = (fg * 255).astype(np.uint8)
        _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Larger kernel for cleaner mask boundaries.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Additional smoothing pass: dilate then erode to fill small holes.
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(binary, kernel_sm, iterations=1)
        binary = cv2.erode(binary, kernel_sm, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        frame_area = h * w
        scored: list[tuple[float, np.ndarray]] = []
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            ratio = area / frame_area
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                continue
            mask = labels == label
            scored.append((float(fg[mask].mean()) * ratio, mask))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[: self.max_objects]]

    def _refine_grabcut(self, frame: Frame, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return mask
        pad = 6
        h, w = mask.shape
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w - 1, int(xs.max()) + pad)
        y1 = min(h - 1, int(ys.max()) + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return mask
        gc = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        gc[y0:y1, x0:x1] = cv2.GC_PR_FGD
        gc[mask] = cv2.GC_FGD
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok = _grabcut_with_timeout(bgr, gc, bgd, fgd, 3)
        if not ok:
            return mask
        refined = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        return refined if refined.sum() > 0.2 * mask.sum() else mask

    # -- object description & association ------------------------------------

    def _describe(
        self,
        mask: np.ndarray,
        depth: DepthMap,
        saliency: np.ndarray,
        shape: tuple[int, int],
    ) -> TrackedObject | None:
        h, w = shape
        area = int(mask.sum())
        if area == 0:
            return None
        ys, xs = np.where(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        return TrackedObject(
            object_id=-1,  # assigned during association
            mask=mask,
            centroid=(cx, cy),
            mean_depth=float(depth[mask].mean()),
            area_ratio=area / float(h * w),
            saliency=float(saliency[mask].mean()),
        )

    def _associate(self, objects: list[TrackedObject]) -> list[TrackedObject]:
        # Advance every track's Kalman prediction one step.
        predicted: dict[int, tuple[float, float]] = {}
        for tid, tr in self._tracks.items():
            pred = tr.kalman.predict()
            predicted[tid] = (float(pred[0, 0]), float(pred[1, 0]))

        matched: set[int] = set()
        for obj in objects:
            # Primary association: IoU against tracks seen in the previous frame.
            best_iou = 0.0
            best_id: int | None = None
            for tid, tr in self._tracks.items():
                if tid in matched or tr.lost != 0 or tr.mask is None:
                    continue
                iou = _mask_iou(obj.mask, tr.mask)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None and best_iou > 0.1:
                self._update_track(best_id, obj)
                matched.add(best_id)
                continue

            # Fallback: nearest Kalman-predicted centroid (revives lost tracks).
            best_dist = 0.15 * self._diag
            cand: int | None = None
            for tid, (px, py) in predicted.items():
                if tid in matched:
                    continue
                dist = float(np.hypot(obj.centroid[0] - px, obj.centroid[1] - py))
                if dist < best_dist:
                    best_dist = dist
                    cand = tid
            if cand is not None:
                self._update_track(cand, obj)
                matched.add(cand)
            else:
                matched.add(self._new_track(obj))

        # Age unmatched tracks; keep them alive on prediction for up to max_lost.
        for tid in list(self._tracks):
            if tid in matched:
                continue
            tr = self._tracks[tid]
            tr.lost += 1
            tr.centroid = predicted.get(tid, tr.centroid)
            if tr.lost > self.max_lost:
                del self._tracks[tid]

        for obj in objects:
            obj.velocity_mag = float(np.hypot(*obj.velocity))
        return objects

    def _update_track(self, tid: int, obj: TrackedObject) -> None:
        tr = self._tracks[tid]
        prev_cx, prev_cy = tr.centroid
        obj.object_id = tid
        obj.velocity = (obj.centroid[0] - prev_cx, obj.centroid[1] - prev_cy)
        # Temporal mask smoothing: blend with previous to prevent popping.
        if tr.mask is not None and obj.mask.shape == tr.mask.shape:
            blended = 0.7 * obj.mask.astype(np.float32) + 0.3 * tr.mask.astype(np.float32)
            obj.mask = blended > 0.4
        tr.kalman.correct(np.array([[obj.centroid[0]], [obj.centroid[1]]], dtype=np.float32))
        tr.centroid = obj.centroid
        tr.mask = obj.mask
        tr.lost = 0

    def _new_track(self, obj: TrackedObject) -> int:
        tid = self._next_id
        self._next_id += 1
        obj.object_id = tid
        obj.velocity = (0.0, 0.0)
        self._tracks[tid] = _Track(
            object_id=tid, kalman=_make_kalman(obj.centroid), centroid=obj.centroid, mask=obj.mask
        )
        return tid


@dataclass
class _Track:
    """Persistent per-object state with a constant-velocity Kalman filter."""

    object_id: int
    kalman: cv2.KalmanFilter
    centroid: tuple[float, float]
    mask: np.ndarray | None
    lost: int = 0


def _make_kalman(centroid: tuple[float, float]) -> cv2.KalmanFilter:
    """Constant-velocity Kalman filter on state ``(cx, cy, vx, vy)``."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    cx, cy = centroid
    kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
    return kf


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0
