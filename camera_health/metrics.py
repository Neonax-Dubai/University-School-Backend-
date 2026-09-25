"""
Camera-health image measurements. Pure functions over one grey sample.

Every measurement runs on the production sample: the decoded frame the
inference loop already holds, reduced once to 320x180 grey (to_sample). No
second stream, no GPU, a few milliseconds at 1 Hz per camera.

Calibrated 2026-09-22 against the Dubai acceptance fixtures (real CAM-R25
lights-on/off, lens covers, people walking, normal footage) and the synthetic
cases in the camera-health tests:

  structured_share  cells with std >= CELL_STD_MIN (2.0 grey levels). Night and
                    dim views keep 0.74-0.92 of their cells; covered lenses
                    fall to 0.00-0.07; textured covers stay >= 0.76.
  sharpness         var(Laplacian) / var(image): lighting-normalised, so a
                    dimmer view is not a blurrier one (ratio 1.01 at 28% gain)
                    while a real defocus collapses it (<= 0.08 of baseline).
  signature         zero-mean, unit-norm 16x9 thumbnail. Global gain cancels;
                    a moved, covered or re-lit view does not.
  edge_map          strongest 5% of gradients above a noise floor - the
                    structure a lighting change preserves and a cover replaces.
"""

import cv2
import numpy as np

SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 180

CELL_COLS = 16
CELL_ROWS = 9
CELL_STD_MIN = 2.0

SIGNATURE_SIZE = (16, 9)

EDGE_TOP_PERCENT = 5.0
EDGE_NOISE_FLOOR = 4.0
EDGE_BLUR_SIGMA = 1.2


def to_sample(frame):
    """A decoded frame (BGR, BGRA or grey) -> grey SAMPLE_HEIGHT x SAMPLE_WIDTH uint8.

    Raises ValueError for anything that is not an image; callers that must not
    raise (observe) catch it.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("not an image array")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"degenerate frame {frame.shape}")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError(f"unsupported channel count {frame.shape[2]}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    # Resize first: converting 5 MP to grey and then shrinking costs far more.
    if frame.shape[0] != SAMPLE_HEIGHT or frame.shape[1] != SAMPLE_WIDTH:
        frame = cv2.resize(frame, (SAMPLE_WIDTH, SAMPLE_HEIGHT), interpolation=cv2.INTER_AREA)
    if frame.ndim == 3:
        code = cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        frame = cv2.cvtColor(frame, code)
    return np.ascontiguousarray(frame)


def ensure_sample(gray):
    """Accept an already-reduced sample, or reduce anything else."""
    if (isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and gray.shape == (SAMPLE_HEIGHT, SAMPLE_WIDTH)):
        return gray
    return to_sample(gray)


def cell_contrast(gray):
    """Per-cell standard deviation, CELL_ROWS x CELL_COLS."""
    h, w = gray.shape[:2]
    ch, cw = h // CELL_ROWS, w // CELL_COLS
    cells = gray[:CELL_ROWS * ch, :CELL_COLS * cw].astype(np.float32)
    return cells.reshape(CELL_ROWS, ch, CELL_COLS, cw).std(axis=(1, 3))


def structured_share(gray):
    """Fraction of the frame that carries usable structure (0..1)."""
    return float((cell_contrast(gray) >= CELL_STD_MIN).mean())


def sharpness(gray):
    """Lighting-normalised focus measure; 0.0 for a flat image."""
    img = gray.astype(np.float32)
    variance = float(img.var())
    if variance <= 1e-3:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_32F, ksize=3).var()) / variance


def brightness(gray):
    return float(gray.mean())


def signature(gray):
    """Zero-mean, unit-norm thumbnail, or None for a featureless image."""
    thumb = cv2.resize(gray, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    thumb -= thumb.mean()
    norm = float(np.linalg.norm(thumb))
    if norm < 1e-3:
        return None
    return thumb / norm


def distance(a, b):
    """Euclidean distance between two signatures (0 = identical, up to 2)."""
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def edge_map(gray):
    """Boolean map of the strongest structure in the view."""
    img = cv2.GaussianBlur(gray, (0, 0), EDGE_BLUR_SIGMA).astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    threshold = max(float(np.percentile(magnitude, 100.0 - EDGE_TOP_PERCENT)), EDGE_NOISE_FLOOR)
    return magnitude >= threshold


def structural_overlap(edges_a, brightness_a, edges_b, brightness_b, dilate=5):
    """Share of the DIMMER view's strong edges that lie on the brighter view's.

    A lighting change hides or reveals structure but does not move it, so what
    is still visible in the darker view sits on edges of the brighter one. A
    cover replaces the structure instead.
    """
    if brightness_a <= brightness_b:
        dim_edges, bright_edges = edges_a, edges_b
    else:
        dim_edges, bright_edges = edges_b, edges_a
    total = int(dim_edges.sum())
    if total == 0:
        return 0.0
    kernel = np.ones((dilate, dilate), np.uint8)
    bright_wide = cv2.dilate(bright_edges.astype(np.uint8), kernel) > 0
    return float((dim_edges & bright_wide).sum()) / total
