"""
Camera-health transitions must produce VALID events.

    aienv/bin/python test_camera_health_event.py
    MULTICAM_PATH=multicam_inf.py.pre-health-bbox aienv/bin/python test_camera_health_event.py

THE DEFECT. handle_camera_health_event() passed no bbox, because a tampered
lens or a lost signal is a property of the whole image rather than of a box
inside it. build_event() turns a missing bbox into [] and validate_event()
requires four values, so every camera-health event was rejected inside the
inference process and none ever reached the dashboard. Measured on the
2026-09-15 production run: 52 transitions, 52 invalid (50 camera_signal_loss,
2 camera_tamper), and the events table held no camera-health event at all.

This runs the REAL handler out of multicam_inf.py against a recording stub and
puts the payload it builds through the REAL validator.
"""
import ast
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import events                                                          # noqa: E402

TARGET = os.path.join(HERE, os.environ.get("MULTICAM_PATH", "multicam_inf.py"))


class Recorder:
    """Stands in for events.EventPipeline, keeping what it was handed."""

    def __init__(self):
        self.calls = []

    def handle_camera_event(self, camera_id, event_type, metadata, **kwargs):
        self.calls.append(dict(camera_id=camera_id, event_type=event_type,
                               metadata=metadata, **kwargs))
        return "sent"


class Stream:
    width, height = 1920, 1080


def load_handler(pipeline, streams):
    source = open(TARGET).read()
    func = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "handle_camera_health_event")
    namespace = {"event_pipeline": pipeline, "camera_streams": streams, "print": lambda *a, **k: None}
    exec(compile(ast.Module([func], []), TARGET, "exec"), namespace)
    return namespace["handle_camera_health_event"]


class CameraHealthEventTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Recorder()
        self.handler = load_handler(self.pipeline, {"CAM-R25": Stream()})

    def call(self, frame_width=1920, frame_height=1080, action="raise"):
        self.handler("CAM-R25", "camera_tamper", 1789542346.0,
                     {"distance": 1.17, "threshold": 0.65}, action,
                     frame_width, frame_height)
        return self.pipeline.calls[-1]

    def built_event(self, call):
        """The payload handle_camera_event() builds from what it was handed -
        including require_track=False, since a camera-level event has no track."""
        return events.build_event(
            camera_id=call["camera_id"], event_type=call["event_type"],
            observed_at=1789542346.0, frame_width=call["frame_width"],
            frame_height=call["frame_height"], track_id="", confidence=1.0,
            bbox=call.get("bbox"), metadata=call["metadata"])

    def validate(self, call):
        return events.validate_event(self.built_event(call), require_track=False)

    def test_the_event_passes_production_validation(self):
        """THE REGRESSION: this returned 'bbox must be 4 values, got []'."""
        problem = self.validate(self.call())
        self.assertIsNone(problem, f"camera-health event rejected: {problem}")

    def test_extent_is_the_whole_frame(self):
        self.assertEqual(self.call()["bbox"], [0, 0, 1920, 1080])

    def test_dimensions_fall_back_to_the_stream_when_the_worker_has_none(self):
        """The health worker is asynchronous and may report no dimensions."""
        call = self.call(frame_width=None, frame_height=None)
        self.assertEqual(call["bbox"], [0, 0, 1920, 1080])
        self.assertIsNone(self.validate(call))

    def test_valid_even_with_no_stream_and_no_dimensions(self):
        handler = load_handler(self.pipeline, {})
        handler("CAM-R99", "camera_signal_loss", 1789542346.0, {"gap": 12},
                "raise", None, None)
        call = self.pipeline.calls[-1]
        self.assertEqual(len(call["bbox"]), 4)
        self.assertIsNone(self.validate(call))

    def test_recovery_keeps_its_scope(self):
        """Scope separates a recovery from its own raise in the debouncer."""
        self.assertEqual(self.call(action="recover")["scope"], "recover")


if __name__ == "__main__":
    print("handler from:", TARGET)
    unittest.main(verbosity=1)
