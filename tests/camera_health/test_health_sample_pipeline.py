"""The judged frame reaches storage without ever costing the event (ZAYED 2026-10-12).

    python -m pytest tests/camera_health/test_health_sample_pipeline.py

The companion to test_health_sample_capture.py: that one pins WHICH frame is kept, this one
pins what happens to it in EventPipeline.handle_camera_event() and the uploader.

THE RULE THAT MATTERS MOST. A diagnostic is a convenience; the event is not. Whatever storage
does - raises, saturates, is absent entirely - the event is sent exactly once. The
"exactly once" half is not hypothetical: submit() sends the event itself on a saturated queue
and returns False, so treating False as "not sent" would deliver it twice.
"""
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import events as events_mod                                                   # noqa: E402
from events import HEALTH_SAMPLE_KIND                                         # noqa: E402

CAM = "camera_01"
DIAGNOSTIC = (b"\xff\xd8\xff-jpeg-bytes", 320, 180)


class Sender:
    def __init__(self):
        self.sent = []

    def submit(self, event, notable=False):
        self.sent.append((event, notable))


class Evidence:
    """Stands in for EvidenceUploader with the behaviours that matter."""

    def __init__(self, behaviour="ok", sender=None):
        self.behaviour = behaviour
        self.calls = []
        self.skipped = 0
        self.sender = sender

    def submit(self, jpeg, width, height, event, notable, camera_id, track_id,
               observed_at, kind="crop"):
        self.calls.append(dict(jpeg=jpeg, width=width, height=height, kind=kind,
                               camera_id=camera_id, track_id=track_id, event=event))
        if self.behaviour == "raise":
            raise RuntimeError("object store unreachable")
        if self.behaviour == "full":
            # What the real uploader does when its queue is saturated.
            self.sender.submit(event, notable=notable)
            return False
        return True

    def note_skipped(self):
        self.skipped += 1


def pipeline(evidence_behaviour="ok", with_evidence=True):
    p = events_mod.EventPipeline.__new__(events_mod.EventPipeline)
    p.sender = Sender()
    p.evidence = Evidence(evidence_behaviour, p.sender) if with_evidence else None
    p.debouncer = types.SimpleNamespace(allow=lambda *a, **k: True)
    p.suppressed = p.invalid = p.generated = 0
    return p


def emit(p, diagnostic=DIAGNOSTIC, scope="raise"):
    return p.handle_camera_event(
        CAM, "camera_tamper", {"reason": "camera_tamper", "distance": 1.06},
        frame_width=2592, frame_height=1944, observed_at=1791000000.0,
        bbox=[0, 0, 2592, 1944], scope=scope, diagnostic=diagnostic)


# --------------------------------------------------------- the happy path
def test_the_sample_is_stored_under_its_own_evidence_type():
    p = pipeline()
    assert emit(p) == "sent"
    assert len(p.evidence.calls) == 1
    call = p.evidence.calls[0]
    assert call["kind"] == HEALTH_SAMPLE_KIND == "health_sample"
    assert (call["width"], call["height"]) == (320, 180)
    assert call["jpeg"] == DIAGNOSTIC[0]


def test_it_is_not_filed_as_a_crop_or_a_snapshot():
    """So Event.still_evidence can never pick it up as a card image."""
    p = pipeline()
    emit(p)
    assert p.evidence.calls[0]["kind"] not in ("crop", "snapshot", "video", "plate")


def test_the_event_is_sent_exactly_once():
    p = pipeline()
    emit(p)
    # The uploader owns the event in this path and sends it after upload.
    assert p.sender.sent == [], "the pipeline must not send it as well"
    assert len(p.evidence.calls) == 1


def test_the_track_label_marks_it_as_health_not_a_subject():
    p = pipeline()
    emit(p, scope="raise")
    assert p.evidence.calls[0]["track_id"] == "healthraise"


# --------------------------------------------------------- failure paths
def test_a_storage_exception_never_costs_the_event():
    p = pipeline("raise")
    assert emit(p) == "sent"
    assert len(p.sender.sent) == 1, "the event must still be sent"


def test_a_saturated_queue_sends_the_event_exactly_once():
    """THE REGRESSION. submit() sends the event itself and returns False; returning early
    only on True would deliver it a second time."""
    p = pipeline("full")
    assert emit(p) == "sent"
    assert len(p.sender.sent) == 1, f"event delivered {len(p.sender.sent)} times"


def test_no_evidence_uploader_at_all_still_sends():
    p = pipeline(with_evidence=False)
    assert emit(p) == "sent"
    assert len(p.sender.sent) == 1


@pytest.mark.parametrize("bad", [None, (None, 320, 180), (b"", 320, 180)])
def test_a_missing_or_empty_sample_falls_back_to_the_plain_send(bad):
    p = pipeline()
    assert emit(p, diagnostic=bad) == "sent"
    assert len(p.sender.sent) == 1
    assert p.evidence.calls == []


def test_a_malformed_diagnostic_tuple_is_survivable():
    p = pipeline()
    assert emit(p, diagnostic=("only-one-value",)) == "sent"
    assert len(p.sender.sent) == 1


# --------------------------------------------------------- nothing else changed
def test_the_existing_path_is_untouched_without_a_diagnostic():
    """A camera event with no sample behaves exactly as it did before."""
    p = pipeline()
    result = p.handle_camera_event(
        CAM, "camera_tamper", {"reason": "camera_tamper"},
        frame_width=2592, frame_height=1944, observed_at=1791000000.0,
        bbox=[0, 0, 2592, 1944], scope="raise")
    assert result == "sent"
    assert p.evidence.calls == []
    assert len(p.sender.sent) == 1


def test_the_storage_key_names_the_kind():
    from evidence import build_storage_key
    key = build_storage_key(CAM, "healthraise", 1791000000.0, "abc123",
                            kind=HEALTH_SAMPLE_KIND)
    assert key.endswith("/health_sample.jpg")
    assert "/crop.jpg" not in key


def test_the_uploader_reports_the_kind_as_the_evidence_type():
    """The dashboard takes evidence_type straight from this field."""
    from evidence import EvidencePipeline

    class Store:
        name = "test-store"

        def put(self, storage_key, data, content_type="image/jpeg"):
            return storage_key

    uploader = EvidencePipeline.__new__(EvidencePipeline)
    uploader.store = Store()
    uploader.event_sender = Sender()
    uploader._lock = __import__("threading").Lock()
    uploader.uploaded = uploader.failed = 0
    event = {"metadata": {}}
    uploader._process((b"jpeg", 320, 180, event, True, CAM, "healthraise",
                       1791000000.0, HEALTH_SAMPLE_KIND))
    reference = event["metadata"]["evidence"]
    assert reference["type"] == "health_sample"
    assert reference["width"] == 320 and reference["height"] == 180
    assert reference["storage_key"].endswith("/health_sample.jpg")
