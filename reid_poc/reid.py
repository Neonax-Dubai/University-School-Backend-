"""
OSNet-AIN x1.0 person Re-ID embedding extractor.

Crop -> pad -> validate -> batch -> 512-D L2-normalised embedding. Nothing
here talks to JeztSort, Qdrant, or the camera pipeline - this module's whole
job is "given a person crop, is it usable, and if so what is its appearance
vector."

torchreid.utils.FeatureExtractor does NOT pre-normalise its output (measured
raw L2 norm ~31.7 on this model) - extract()/extract_batch() below normalise
to unit length, so cosine similarity between two returned embeddings reduces
to a plain dot product.
"""
import cv2
import numpy as np
import torchreid

import config


class InvalidCrop(Exception):
    """Raised by validate_crop() with a short, specific reason."""


def pad_bbox(bbox, frame_width, frame_height, padding_frac=None):
    """Expand a bbox by padding_frac of its own size on each side, clamped
    to the frame. Returns (x1, y1, x2, y2) ints."""
    if padding_frac is None:
        padding_frac = config.REID_CROP_PADDING_FRAC

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    pad_x = int(round(w * padding_frac))
    pad_y = int(round(h * padding_frac))

    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(frame_width, int(x2 + pad_x)),
        min(frame_height, int(y2 + pad_y)),
    )


def validate_crop(crop):
    """Raise InvalidCrop with a specific reason, or return the crop unchanged."""
    if crop is None or crop.size == 0:
        raise InvalidCrop("empty")

    h, w = crop.shape[:2]

    if w < config.REID_MIN_CROP_WIDTH or h < config.REID_MIN_CROP_HEIGHT:
        raise InvalidCrop(f"too_small({w}x{h})")

    aspect = w / h if h else float("inf")

    if aspect > config.REID_MAX_ASPECT_RATIO:
        raise InvalidCrop(f"bad_aspect({aspect:.2f})")

    if config.REID_BLUR_CHECK_ENABLED:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < config.REID_BLUR_MIN_VARIANCE:
            raise InvalidCrop(f"blurry(var={variance:.1f})")

    return crop


def crop_person(frame, bbox):
    """
    Full pipeline from a tracked bbox to a validated, padded crop.

    Returns the crop (a view/copy of frame pixels), or None with the reason
    printed nowhere - callers should catch InvalidCrop themselves if they
    want to log rejections; this convenience form just returns None.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = pad_bbox(bbox, frame_width, frame_height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    try:
        return validate_crop(crop)
    except InvalidCrop:
        return None


class OSNetReID:
    """Loads once; extract()/extract_batch() are the only hot-path calls."""

    def __init__(self, model_name=None, device=None):
        model_name = config.REID_MODEL_NAME if model_name is None else model_name
        device = config.REID_DEVICE if device is None else device

        print(f"Loading {model_name} Re-ID (device={device})...")

        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            device=device,
        )

        print(f"{model_name} loaded")

    @staticmethod
    def _normalise(features):
        """(N, 512) raw torchreid output -> (N, 512) unit-norm float32."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (features / norms).astype(np.float32)

    def extract(self, crop):
        """One BGR crop -> one (512,) unit-norm embedding, or None if invalid."""
        try:
            validate_crop(crop)
        except InvalidCrop:
            return None

        features = self.extractor(crop).detach().cpu().numpy()
        return self._normalise(features)[0]

    def extract_batch(self, crops):
        """
        Multiple BGR crops -> one (N, 512) unit-norm embedding batch, in a
        SINGLE forward pass. Crops are assumed already-validated (callers in
        this PoC validate at crop time via crop_person()); an empty list
        returns an empty (0, 512) array rather than erroring, so callers can
        always index the result 1:1 against their input list.
        """
        if not crops:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

        features = self.extractor(list(crops)).detach().cpu().numpy()
        return self._normalise(features)
