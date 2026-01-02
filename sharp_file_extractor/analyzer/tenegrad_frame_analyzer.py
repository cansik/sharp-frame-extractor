from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from sharp_file_extractor.analyzer.frame_analyzer_base import (
    FrameAnalyzerBase,
    FrameAnalyzerTask,
    FrameAnalyzerResult,
)


class CenterWeighting(Enum):
    NONE = "none"
    HANN = "hann"
    GAUSSIAN = "gaussian"


class ScoreNormalization(Enum):
    SIGMOID_ROBUST = "sigmoid_robust"
    MINMAX = "minmax"


@dataclass(frozen=True, slots=True)
class TenengradConfig:
    center_weight: CenterWeighting = CenterWeighting.GAUSSIAN
    normalize: ScoreNormalization = ScoreNormalization.SIGMOID_ROBUST
    gaussian_sigma_fraction: float = 0.22
    sobel_ksize: int = 3
    eps: float = 1e-8

    def validate(self) -> None:
        if not (0.0 < self.gaussian_sigma_fraction <= 1.0):
            raise ValueError("gaussian_sigma_fraction must be in (0, 1].")

        if self.sobel_ksize not in {1, 3, 5, 7}:
            raise ValueError("sobel_ksize must be one of {1, 3, 5, 7}.")

        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")


class TenengradFrameAnalyzer(FrameAnalyzerBase):
    """
    Selects the sharpest frame using Tenengrad (Sobel gradient energy).

    - Optional center weighting (none, hann, gaussian)
    - Score normalization to 0..1 (minmax or robust sigmoid)
    """

    def __init__(self, config: TenengradConfig | None = None):
        self._cfg = config or TenengradConfig()
        self._cfg.validate()

        self._cached_weight_key: tuple[int, int, CenterWeighting, float] | None = None
        self._cached_weight: np.ndarray | None = None

    @property
    def config(self) -> TenengradConfig:
        return self._cfg

    def reset_states(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None

    def process(self, task: FrameAnalyzerTask) -> FrameAnalyzerResult:
        frames = task.frames
        if frames.ndim not in (3, 4):
            raise ValueError(f"Expected frames with shape (N,H,W) or (N,H,W,C), got {frames.shape}")

        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        weights = self._center_weights(h, w)

        raw_scores = np.empty((n,), dtype=np.float32)
        for i in range(n):
            gray = self._to_gray(frames[i])
            raw_scores[i] = self._tenengrad(gray, weights)

        best_idx = int(np.argmax(raw_scores))
        best_frame = frames[best_idx]
        score = float(self._score_01(raw_scores, best_idx))

        return FrameAnalyzerResult(frame=best_frame, score=score)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame

        channels = int(frame.shape[-1])
        if channels == 1:
            return frame[..., 0]

        if channels == 3:
            if frame.dtype == np.uint8:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(frame.astype(np.float32, copy=False), cv2.COLOR_BGR2GRAY)

        if channels == 4:
            if frame.dtype == np.uint8:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(frame.astype(np.float32, copy=False), cv2.COLOR_BGRA2GRAY)

        return frame[..., 0]

    def _center_weights(self, h: int, w: int) -> np.ndarray | None:
        mode = self._cfg.center_weight
        if mode is CenterWeighting.NONE:
            return None

        key = (h, w, mode, float(self._cfg.gaussian_sigma_fraction))
        if self._cached_weight is not None and self._cached_weight_key == key:
            return self._cached_weight

        if mode is CenterWeighting.HANN:
            wy = np.hanning(h).astype(np.float32, copy=False)
            wx = np.hanning(w).astype(np.float32, copy=False)
            ww = np.outer(wy, wx).astype(np.float32, copy=False)
        elif mode is CenterWeighting.GAUSSIAN:
            sigma_y = max(1.0, float(h) * float(self._cfg.gaussian_sigma_fraction))
            sigma_x = max(1.0, float(w) * float(self._cfg.gaussian_sigma_fraction))
            ky = cv2.getGaussianKernel(h, sigma_y, ktype=cv2.CV_32F)
            kx = cv2.getGaussianKernel(w, sigma_x, ktype=cv2.CV_32F)
            ww = (ky @ kx.T).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unhandled center_weight: {mode!r}")

        s = float(ww.sum())
        if s <= 0.0:
            return None

        ww = ww / s
        self._cached_weight_key = key
        self._cached_weight = ww
        return ww

    def _tenengrad(self, gray: np.ndarray, weights: np.ndarray | None) -> np.float32:
        g = gray.astype(np.float32, copy=False)

        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=self._cfg.sobel_ksize)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=self._cfg.sobel_ksize)
        g2 = gx * gx + gy * gy

        if weights is None:
            return np.float32(g2.mean())

        return np.float32((g2 * weights).sum())

    def _score_01(self, raw_scores: np.ndarray, best_idx: int) -> float:
        eps = float(self._cfg.eps)
        best = float(raw_scores[best_idx])

        if self._cfg.normalize is ScoreNormalization.MINMAX:
            mn = float(raw_scores.min())
            mx = float(raw_scores.max())
            return float(np.clip((best - mn) / (mx - mn + eps), 0.0, 1.0))

        med = float(np.median(raw_scores))
        mad = float(np.median(np.abs(raw_scores - med)))
        scale = 1.4826 * mad + eps
        z = (best - med) / scale
        return float(1.0 / (1.0 + np.exp(-z)))
