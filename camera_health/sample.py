"""The diagnostic frame a camera-health raise was decided on (ZAYED 2026-10-12).

On 2026-10-12 camera_01 raised camera_tamper recording distance 1.0584, brightness ratio
1.545 and sharpness 0.436. Ten minutes of NVR footage spanning that moment, replayed through
this package, measured distance 0.009-0.019 and luma 90.0-90.8 throughout - the recorder never
saw what the detector saw. Nothing kept the picture the detector judged, so a corrupted frame
and a real scene change were indistinguishable after the fact.

This module encodes that frame, and nothing else. It is deliberately tiny and free of storage,
network and detector dependencies so it can be exercised on its own.
"""

#: Bound on the stored diagnostic. The sample is 320x180 greyscale, so a JPEG is a few KB;
#: anything larger means something unexpected was handed over, and it is dropped rather than
#: stored. One image per raise, never per sample.
MAX_BYTES = 256 * 1024

#: Enough to read a corrupted frame, small enough that the bound is never the binding
#: constraint for a real 320x180 sample.
QUALITY = 80


def health_sample_jpeg(sample):
    """(jpeg, width, height) for the frame a detector judged, or None.

    Encodes the array AS GIVEN - no resize, no fresh grab - so what is stored is what was
    judged. Never raises: a diagnostic may not cost an event, so every failure path returns
    None and the caller sends the event without it.
    """
    try:
        import cv2
        import numpy as np

        if sample is None or not isinstance(sample, np.ndarray):
            return None
        if sample.ndim not in (2, 3) or sample.size == 0:
            return None

        ok, buffer = cv2.imencode(".jpg", sample, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
        if not ok:
            return None

        data = buffer.tobytes()
        if not data or len(data) > MAX_BYTES:
            return None

        height, width = sample.shape[:2]
        return data, int(width), int(height)
    except Exception:                                             # noqa: BLE001
        return None
