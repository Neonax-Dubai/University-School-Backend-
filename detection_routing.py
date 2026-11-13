"""
ZAYED: where one detection goes - the tracker, the mobile-phone analytic, both, or nowhere.

Pure (no state, no I/O) so the live loop (zayed_inference.process_result) and the isolated replay
(tests/unattended/dump_unattended_inputs.py) share the one rule.

    person / object  -> the tracker at the camera's floor; an unattended-object class on a camera with
                        abandoned_object armed also at abandoned.OBJECT_CONFIDENCE (0.25)
    cell phone       -> the mobile-phone analytic (phone_use, MOBILE_PHONE_DETECTED) at its own floor,
                        exactly as before; and, ONLY when "cell phone" is an unattended class on a
                        camera with abandoned_object armed, ALSO the tracker - so a phone left behind
                        can become an unattended episode. phone_use consumes raw phone boxes and the
                        person tracks, never phone tracks: its input does not change.
"""
import abandoned
import phone_use


def route(class_id, confidence, *, allowed, min_confidence, abandon_armed, phone_armed):
    """(to_phone_analytic, to_tracker) for one detection."""
    class_id = int(class_id)
    if class_id not in allowed:
        return False, False
    to_phone = phone_armed and class_id == phone_use.PHONE_CLASS_ID and confidence >= phone_use.PHONE_CONFIDENCE
    unattended = abandon_armed and class_id in abandoned.CANDIDATE_CLASS_IDS
    if to_phone and not unattended:
        return True, False                   # a phone in use: the phone analytic only, as before
    to_tracker = confidence >= min_confidence or (unattended and confidence >= abandoned.OBJECT_CONFIDENCE)
    return to_phone, to_tracker
