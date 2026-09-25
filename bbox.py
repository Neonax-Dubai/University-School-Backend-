import cv2
import numpy as np

# ============================================================
# PALETTE  (BGR)
# ============================================================
ACCENT_RED       = (55,  50, 235)
NEAR_WHITE       = (235, 235, 235)
PANEL_BG         = (18,  18,  22)

# ============================================================
# LOW-LEVEL DRAWING HELPERS
# ============================================================

def _alpha_blend(base, overlay, alpha):
    return np.clip(base * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)

def _rounded_rect_filled(image, x1, y1, x2, y2, radius, color, alpha=1.0):
    """Filled rounded rectangle via rect + 4 corner circles, optional alpha."""
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    target = image if alpha >= 1.0 else image.copy()
    cv2.rectangle(target, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(target, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(target, (cx, cy), radius, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        image[:] = _alpha_blend(image, target, alpha)

def _draw_corners(image, bbox, color, thickness, L):
    """L-shaped corners at the four corners of bbox."""
    x1, y1, x2, y2 = bbox
    # top-left
    cv2.line(image, (x1, y1), (x1 + L, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + L), color, thickness, cv2.LINE_AA)
    # top-right
    cv2.line(image, (x2, y1), (x2 - L, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + L), color, thickness, cv2.LINE_AA)
    # bottom-left
    cv2.line(image, (x1, y2), (x1 + L, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1, y2 - L), color, thickness, cv2.LINE_AA)
    # bottom-right
    cv2.line(image, (x2, y2), (x2 - L, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2, y2 - L), color, thickness, cv2.LINE_AA)

# ============================================================
# PROFESSIONAL PRIMITIVES
# ============================================================

def draw_corner_brackets(image, bbox, color=ACCENT_RED, thickness=2, corner_len=0.22, glow=True):
    """
    Corner-bracket reticle — the standard used by professional CCTV UIs.
    Adds a soft outer glow for visibility on any background, plus
    small midpoint ticks for a targeting-reticle feel.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    L = max(int(min(x2 - x1, y2 - y1) * corner_len), thickness * 5)

    # --- soft glow (3 stacked outlines, fading out) ---
    if glow:
        for offset in (3, 2, 1):
            glow_c = tuple(max(0, int(c * 0.35)) for c in color)
            _draw_corners(image, (x1, y1, x2, y2), glow_c,
                          thickness + offset * 2, L + offset)

    # --- primary corners ---
    _draw_corners(image, (x1, y1, x2, y2), color, thickness, L)

    # --- midpoint ticks (targeting reticle) ---
    tick = max(4, thickness + 1)
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    line_t = max(1, thickness // 2)
    cv2.line(image, (mx, y1 - tick), (mx, y1 + tick), color, line_t, cv2.LINE_AA)
    cv2.line(image, (mx, y2 - tick), (mx, y2 + tick), color, line_t, cv2.LINE_AA)
    cv2.line(image, (x1 - tick, my), (x1 + tick, my), color, line_t, cv2.LINE_AA)
    cv2.line(image, (x2 - tick, my), (x2 + tick, my), color, line_t, cv2.LINE_AA)

def draw_label_badge(image, bbox, label_class, confidence, color=ACCENT_RED):
    """
    Accent-barred label badge:
    Dark rounded background, left accent bar in alert color,
    positioned above the bbox (falls below if no room).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = image.shape[:2]

    font_scale = max(0.45, min(1.15, (x2 - x1) / 300.0))
    thick = max(1, int(round(font_scale * 1.8)))

    label = f"{label_class.upper()}   {confidence * 100:.1f}%"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale, thick)
    pad_x, pad_y = 14, 8
    accent_w = 5
    badge_w = tw + pad_x * 2 + accent_w
    badge_h = th + base + pad_y * 2

    # position: prefer above, fall back to below
    bx = max(0, min(x1, w_img - badge_w))
    by = y1 - badge_h - 8
    if by < 0:
        by = y2 + 8
    if by + badge_h > h_img:
        by = max(0, h_img - badge_h)

    # --- dark rounded background (92% opaque) ---
    _rounded_rect_filled(image, bx, by, bx + badge_w, by + badge_h,
                         radius=6, color=PANEL_BG, alpha=0.92)

    # --- left accent bar ---
    cv2.rectangle(image,
                  (bx + 1, by + 4), (bx + accent_w, by + badge_h - 4),
                  color, -1, cv2.LINE_AA)

    # --- subtle top highlight ---
    cv2.line(image,
             (bx + accent_w + 3, by + 1),
             (bx + badge_w - 4, by + 1),
             tuple(min(255, c + 35) for c in PANEL_BG), 1, cv2.LINE_AA)

    # --- text ---
    cv2.putText(image, label,
                (bx + accent_w + pad_x, by + pad_y + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                NEAR_WHITE, thick, cv2.LINE_AA)


# ============================================================
# EVIDENCE OVERLAY - the shared, generalised renderer
#
# The weapon still (weapon_detection/evidence_image.py) is the visual
# baseline: corner brackets rather than a raw rectangle, a dark rounded badge
# with an accent bar, text sized from the box. This section generalises that
# treatment to the other detections, over the evidence crop the existing
# pipeline already produces - one image, one pass, no second encode.
#
# Two rules this module exists to keep:
#   * it draws what the EVENT was about and nothing else - the boxes the
#     caller selected, never "every detection in the frame";
#   * it can never break evidence. Every entry point catches its own errors
#     and returns; a crop with no overlay is a good crop, an exception here
#     would cost the operator the picture AND the event.
# ============================================================
import logging

LOGGER = logging.getLogger("bbox")

#: Roles carry the palette, so a person, a weapon and a vehicle stay
#: distinguishable when one image shows more than one kind.
#: OpenCV is BGR, so these read (blue, green, red).
ROLE_COLOURS = {
    "primary": ACCENT_RED,
    "person": (60, 180, 235),        # amber
    "vehicle": (245, 200, 70),       # cyan
    "object": (120, 220, 120),       # green
    "weapon": ACCENT_RED,
    "plate": (235, 235, 235),
}

#: Above this many boxes the labels are dropped and only the brackets drawn -
#: a dense crowd stays readable as a set of boxes, where 40 badges would be a
#: wall of text. The count on the event says how many; the boxes say which.
LABEL_BUDGET = 12

#: Smaller than this (in the evidence image) a box cannot carry a readable
#: badge, so it is drawn bare rather than covered by its own label.
MIN_LABELLED_SIDE = 44


class Box:
    """One thing to outline, in the coordinates of the image being drawn on.

    label/confidence are optional: a row that does not name its class simply
    gets a bracket. `role` picks the palette. `index` numbers a box in a set
    (Person 1, Person 2 ...) and is set by draw_boxes when it is not given.
    """

    __slots__ = ("x1", "y1", "x2", "y2", "label", "confidence", "role", "index")

    def __init__(self, x1, y1, x2, y2, label=None, confidence=None,
                 role="primary", index=None):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label = label
        self.confidence = confidence
        self.role = role
        self.index = index

    def __repr__(self):                                    # pragma: no cover
        return (f"Box({self.x1},{self.y1},{self.x2},{self.y2},"
                f"label={self.label!r},role={self.role!r})")


def _clean(box, width, height):
    """A drawable (x1, y1, x2, y2) inside the image, or None.

    Rejects what cannot be drawn honestly: non-numeric, NaN, inverted, or a
    box that lands entirely outside the image. Clips the rest to the frame,
    because a detection at the edge is still a true detection.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in (box.x1, box.y1, box.x2, box.y2))
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None                                   # wholly outside the image
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width - 1)))
    y2 = int(max(0, min(y2, height - 1)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None                                   # nothing left to see
    return x1, y1, x2, y2


#: The few non-ASCII characters the dashboard's own labels use.
_ASCII_EQUIVALENT = {"\u00b7": "-", "\u2013": "-", "\u2014": "-", "\u2026": "..."}


def _badge_text(box):
    """What the badge says, or None when the row names nothing."""
    parts = []
    if box.label:
        # cv2.putText draws Hershey fonts, which have no glyph outside ASCII
        # and print "?" instead. The policy keeps the dashboard's own words
        # (including its middle dot); they are transliterated HERE, so the
        # still says the same thing in a form the font can draw.
        text = "".join(_ASCII_EQUIVALENT.get(ch, ch if 32 <= ord(ch) < 127 else " ")
                       for ch in str(box.label))
        text = " ".join(text.split())
        if text:
            parts.append(text if box.index is None else f"{text} {box.index}")
    elif box.index is not None:
        parts.append(str(box.index))
    if not parts:
        return None
    if isinstance(box.confidence, (int, float)) and 0 < box.confidence <= 1:
        parts.append(f"{box.confidence * 100:.0f}%")
    # A label long enough to run off the image helps nobody; the event beside
    # the picture carries the full text.
    text = "  ".join(parts)
    return text if len(text) <= 42 else text[:41] + "…"


def draw_boxes(image, boxes, offset=(0, 0)):
    """Outline `boxes` on `image` in place. Returns how many were drawn.

    offset translates boxes from frame coordinates into the crop's own
    coordinates - the evidence image is a padded cut of the frame, so a box
    recorded against the full frame has to move by the cut's origin.

    Never raises. A box that cannot be drawn is skipped and counted; an
    unexpected failure leaves the image as it was.
    """
    if image is None or not boxes:
        return 0

    try:
        height, width = image.shape[:2]
        dx, dy = offset
        drawn = skipped = 0
        usable = []
        for box in boxes:
            moved = Box(box.x1 - dx, box.y1 - dy, box.x2 - dx, box.y2 - dy,
                        box.label, box.confidence, box.role, box.index)
            cleaned = _clean(moved, width, height)
            if cleaned is None:
                skipped += 1
                continue
            usable.append((cleaned, moved))

        label_all = len(usable) <= LABEL_BUDGET
        for number, (rect, box) in enumerate(usable, start=1):
            if box.index is None and len(usable) > 1:
                box.index = number
            colour = ROLE_COLOURS.get(box.role, ACCENT_RED)
            x1, y1, x2, y2 = rect
            side = min(x2 - x1, y2 - y1)
            thickness = 2 if side < 160 else 3
            draw_corner_brackets(image, rect, color=colour, thickness=thickness)
            text = _badge_text(box) if label_all and side >= MIN_LABELLED_SIDE else None
            if text:
                draw_text_badge(image, rect, text, color=colour)
            drawn += 1

        if skipped:
            LOGGER.debug("overlay skipped %d unusable box(es)", skipped)
        return drawn
    except Exception as exc:                              # noqa: BLE001
        LOGGER.warning("overlay failed, evidence kept unannotated: %s", exc)
        return 0


def draw_text_badge(image, bbox, text, color=ACCENT_RED):
    """draw_label_badge's layout, for text that is already composed.

    Kept separate from draw_label_badge so the weapon still's own
    "CLASS   99.9%" formatting is untouched.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = image.shape[:2]

    pad_x, pad_y, accent_w = 9, 5, 4
    chrome = pad_x * 2 + accent_w

    # FIT THE LABEL TO THE PICTURE. Evidence crops are often small - a person
    # cut is ~130 px wide - and "Missing Face Mask, Gloves" does not fit at a
    # comfortable size. Shrink first, then shorten; dropping the label was the
    # earlier behaviour and it left an operator with a box and no answer.
    font_scale = max(0.38, min(0.72, (x2 - x1) / 420.0))
    while True:
        thick = max(1, int(round(font_scale * 1.7)))
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        if tw + chrome <= w_img or font_scale <= 0.34:
            break
        font_scale -= 0.04
    # ASCII dots, not "\u2026": the badge text has already been transliterated
    # for the Hershey font, so a unicode ellipsis added here would print as
    # "???" - which is exactly what it did before this line said "...".
    while tw + chrome > w_img and len(text) > 4:
        text = (text[:-4] if text.endswith("...") else text[:-1]) + "..."
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

    badge_w = tw + chrome
    badge_h = th + base + pad_y * 2

    # Prefer above the box; fall below when there is no room, and never let
    # the badge leave the image on any side.
    bx = max(0, min(x1, w_img - badge_w))
    by = y1 - badge_h - 6
    if by < 0:
        by = y2 + 6
    if by + badge_h > h_img:
        by = max(0, h_img - badge_h)
    if badge_w > w_img or badge_h >= h_img:
        return                             # even one character will not fit

    _rounded_rect_filled(image, bx, by, bx + badge_w, by + badge_h,
                         radius=5, color=PANEL_BG, alpha=0.9)
    cv2.rectangle(image, (bx + 1, by + 3), (bx + accent_w, by + badge_h - 3),
                  color, -1, cv2.LINE_AA)
    cv2.putText(image, text, (bx + accent_w + pad_x, by + pad_y + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, NEAR_WHITE, thick, cv2.LINE_AA)
