"""
WHICH boxes belong on an event's evidence still.

The renderer (bbox.py) draws what it is given. This module decides what that
is, per event type, from the data the event already carries - and deliberately
refuses to draw anything else. An evidence frame usually contains people,
vehicles and bags that had nothing to do with the event; outlining them would
turn the picture into a wall of rectangles and hide the thing an operator
opened it for.

    crowd_detected      the people the count was made of, passed in by the
                        caller from the same tracks zones.evaluate_crowd
                        counted - never "the nearest N people"
    abandoned_object    the object left behind
    object_detected     the object
    vehicle_zone_detection / vehicle_detected
                        the vehicle, labelled with its class and, when ANPR
                        has already published one, its plate
    face_watchlist_hit  the recognised person
    ppe_violation / behaviour_violation / police_uniform_detected
                        the person the policy layer judged
    intrusion / perimeter_breach / loitering / line_crossing /
    fall_detected / violence_detected / physical_distancing_violation
                        the subject the event was raised on
    camera_tamper       NOTHING. The subject is the whole view: there is no
                        object to point at, and a box would invent one.

Labels use the same words the dashboard prints (events_log.models
.entity_label_for): the class through the detector's own name, the person's
stored name, "Plate X" only when consensus published it and "(unverified)"
when it did not. Two systems, one vocabulary.
"""
import logging

import bbox as bbox_renderer

LOGGER = logging.getLogger("evidence_overlay")

#: Event types whose still is about the whole frame, not an object in it.
NO_OVERLAY_TYPES = frozenset({"camera_tamper"})

#: The role (and so the colour) a type's primary box is drawn in.
PRIMARY_ROLE = {
    "crowd_detected": "person",
    "face_watchlist_hit": "person",
    "ppe_violation": "person",
    "behaviour_violation": "person",
    "police_uniform_detected": "person",
    "physical_distancing_violation": "person",
    "fall_detected": "person",
    "violence_detected": "person",
    "intrusion": "person",
    "perimeter_breach": "person",
    "loitering": "person",
    "line_crossing": "person",
    "vehicle_zone_detection": "vehicle",
    "vehicle_detected": "vehicle",
    "object_detected": "object",
    "abandoned_object": "object",
    "weapon_detected": "weapon",
}

#: What to call the subject when the row names no class of its own.
FALLBACK_LABEL = {
    "person": "Person",
    "vehicle": "Vehicle",
    "object": "Object",
    "weapon": "Weapon",
}


def _class_label(value):
    """"handbag" -> "Handbag". The detector's own word, tidied, never guessed."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    # "cell phone" -> "Cell phone", matching the dashboard's label helper.
    return value[:1].upper() + value[1:]


def primary_label(event_type, metadata):
    """The words for the subject's own box, or "" when the row names none."""
    meta = metadata or {}

    if event_type in ("face_watchlist_hit", "known_person_absent"):
        return str(meta.get("known_person_name") or "").strip()

    if event_type in ("object_detected", "abandoned_object", "vehicle_detected"):
        return _class_label(meta.get("object_type"))

    if event_type == "vehicle_zone_detection":
        bits = [_class_label(meta.get("object_type")) or "Vehicle"]
        # The same distinction the dashboard makes: a plate is only stated as
        # read when consensus published it.
        if meta.get("plate_number"):
            bits.append(f"Plate {meta['plate_number']}")
        else:
            reading = meta.get("anpr_single_reading") or meta.get("anpr_unverified_reading")
            if reading:
                bits.append(f"Plate {reading} (unverified)")
        return " · ".join(bits)

    if event_type == "weapon_detected":
        return _class_label(meta.get("weapon_class"))

    if event_type == "ppe_violation":
        missing = meta.get("ppe_missing_labels") or []
        if missing:
            return "Missing " + ", ".join(str(m) for m in missing)
        return ""

    if event_type == "behaviour_violation":
        return str(meta.get("behaviour_label") or meta.get("behaviour") or "").strip()

    return ""


def boxes_for_event(event_type, metadata, bbox, contributors=None, confidence=None):
    """The boxes for one event's still, in FRAME coordinates. May be empty.

    contributors: the individual detections that made the event, when the
    caller has them - crowd passes the tracks it counted. When there are
    contributors they ARE the answer to "which ones?", so the event's own
    hull box is not drawn on top of them.
    """
    if event_type in NO_OVERLAY_TYPES:
        return []

    role = PRIMARY_ROLE.get(event_type, "primary")

    if contributors:
        boxes = []
        for index, item in enumerate(contributors, start=1):
            box = getattr(item, "bbox", None) or (item.get("bbox") if isinstance(item, dict) else None)
            if box is None and isinstance(item, (list, tuple)) and len(item) == 4:
                box = item
            if box is None or len(box) != 4:
                continue
            boxes.append(bbox_renderer.Box(
                box[0], box[1], box[2], box[3],
                label=FALLBACK_LABEL.get(role, "Detection"), role=role, index=index))
        if boxes:
            return boxes
        # No usable contributor: fall through to the subject's own box rather
        # than returning a picture with nothing marked on it.

    if not bbox or len(bbox) != 4:
        return []

    label = primary_label(event_type, metadata) or FALLBACK_LABEL.get(role, "")
    return [bbox_renderer.Box(bbox[0], bbox[1], bbox[2], bbox[3],
                              label=label or None, confidence=confidence, role=role)]


def annotate(patch, event_type, metadata, bbox, origin=(0, 0), contributors=None,
             confidence=None):
    """Draw an event's boxes on its evidence patch. Returns boxes drawn.

    origin is the patch's top-left in frame coordinates, because the patch is
    a cut of the frame and every stored box is in frame coordinates.

    Never raises: evidence without an overlay is still evidence.
    """
    try:
        boxes = boxes_for_event(event_type, metadata, bbox, contributors, confidence)
        if not boxes:
            return 0
        return bbox_renderer.draw_boxes(patch, boxes, offset=origin)
    except Exception as exc:                                  # noqa: BLE001
        LOGGER.warning("overlay for %s skipped: %s", event_type, exc)
        return 0
