"""
Camera health for the Zayed University inference stack.

Recreated 2026-09-22 from the proven Dubai contract (the original package was
never committed). See manager.py for the behaviour, state.py for the incident
rules and detectors.py for thresholds.
"""

from . import detectors, metrics, state                                   # noqa: F401
from .manager import (CAMERA_HEALTH_FEATURES, CAMERA_TAMPERING_FEATURES,  # noqa: F401
                      FEATURE_DEFOCUS, FEATURE_OBSTRUCTION, FEATURE_SIGNAL_LOSS,
                      FEATURE_TAMPER, IMAGE_FEATURES, PUBLISH_INTERVAL_SECONDS,
                      REASON_LABELS, TELEMETRY_PATH, UMBRELLA_EVENT_TYPE,
                      CameraHealthManager, HealthTelemetryPublisher)
