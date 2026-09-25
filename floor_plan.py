"""
Floor-plan coordinates, heatmap and floor-fused occupancy (Zayed).

CALIBRATION lives in the backend (the system of record) and arrives with the camera payload:
  placement.floor_plan_homography = {"image_points":   [[x, y], ...]  >= 4 pixels of the analysed frame,
                                     "floor_points_m": [[x, y], ...]  the same points on the floor, metres}
  classroom.floor_plan            = {"width_m": .., "height_m": .., "grid_m": 0.5}
A camera without a valid homography is simply not mapped: nothing is estimated for it.

For every confirmed person track of a mapped camera the FOOT point (bottom-centre of the box) is
projected onto the floor. Once per FUSION_INTERVAL_SECONDS, per classroom, the latest points of
every mapped camera (no older than FUSION_MAX_AGE_SECONDS) are fused: points closer than
FUSION_RADIUS_M that come from DIFFERENT cameras are the same person - the classroom's cameras
overlap. That yields
  * the floor-fused occupancy ("floor_fusion"), which occupancy.py uses instead of "max_camera"
    once EVERY camera of the classroom is mapped;
  * one person-second per fused person, per second, into the classroom's heatmap grid
    (grid_m cells); a daemon thread POSTs it to /api/ai/heatmap/ every HEATMAP_PUBLISH_SECONDS
    and starts a new window. A failed POST is retried with the next window (bounded).
  * a floor location for per-person events (location_for), carried in metadata["location"].
No individual position is stored anywhere: the heatmap is anonymous and aggregated.
"""
import collections
import math
import os
import threading
import time

import cv2
import numpy as np
import requests

HEATMAP_FEATURE = "occupancy_heatmap"
HEATMAP_PATH = "/api/ai/heatmap/"

FUSION_RADIUS_M = float(os.getenv("FLOOR_FUSION_RADIUS_M", "0.6"))
FUSION_MAX_AGE_SECONDS = float(os.getenv("FLOOR_FUSION_MAX_AGE_SECONDS", "1.0"))
FUSION_INTERVAL_SECONDS = float(os.getenv("FLOOR_FUSION_INTERVAL_SECONDS", "1.0"))
OUTSIDE_MARGIN_M = float(os.getenv("FLOOR_OUTSIDE_MARGIN_M", "0.5"))
HEATMAP_PUBLISH_SECONDS = float(os.getenv("HEATMAP_PUBLISH_SECONDS", "60"))
HEATMAP_PENDING_MAX = 60
DEFAULT_GRID_M = 0.5


def build_homography(spec):
    """image -> floor homography from a calibration dict, or None when it is absent/invalid."""
    try:
        img = np.asarray((spec or {}).get("image_points") or [], dtype=np.float64)
        flr = np.asarray((spec or {}).get("floor_points_m") or [], dtype=np.float64)
        if img.ndim != 2 or img.shape != flr.shape or img.shape[0] < 4 or img.shape[1] != 2:
            return None
        matrix, _ = cv2.findHomography(img, flr, 0)
        if matrix is None or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
            return None
        return matrix
    except Exception:                                    # noqa: BLE001 - bad calibration = unmapped
        return None


def project(matrix, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def foot_point(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def fuse(points, radius=FUSION_RADIUS_M):
    """points: [(camera_id, x, y)] -> [(x, y, cameras)] - one cluster per person.

    Greedy: a point joins the nearest cluster within `radius` that has no point from its own
    camera yet (one camera never sees the same person twice), else it starts a new cluster."""
    clusters = []                                       # [sum_x, sum_y, n, set(cameras)]
    for camera_id, x, y in sorted(points, key=lambda p: (p[1], p[2])):
        best, best_d = None, radius
        for c in clusters:
            if camera_id in c[3]:
                continue
            d = math.hypot(c[0] / c[2] - x, c[1] / c[2] - y)
            if d <= best_d:
                best, best_d = c, d
        if best is None:
            clusters.append([x, y, 1, {camera_id}])
        else:
            best[0] += x
            best[1] += y
            best[2] += 1
            best[3].add(camera_id)
    return [(c[0] / c[2], c[1] / c[2], sorted(c[3])) for c in clusters]


class _Heatmap:
    def __init__(self, width_m, height_m, grid_m, started_at):
        self.width_m, self.height_m, self.grid_m = width_m, height_m, grid_m
        self.cols = max(1, int(math.ceil(width_m / grid_m)))
        self.rows = max(1, int(math.ceil(height_m / grid_m)))
        self.cells = collections.Counter()
        self.started_at = started_at
        self.samples = 0

    def add(self, x, y, seconds):
        col = min(self.cols - 1, max(0, int(x // self.grid_m)))
        row = min(self.rows - 1, max(0, int(y // self.grid_m)))
        self.cells[(col, row)] += seconds


class FloorPlanMapper:
    def __init__(self, log=print):
        self._log = log
        self._lock = threading.Lock()
        self._homography = {}            # camera -> matrix
        self._camera_classroom = {}      # camera -> classroom_id
        self._plans = {}                 # classroom -> (width_m, height_m, grid_m) or None
        self._heatmap_classrooms = set()  # classrooms whose cameras are ALL armed for the heatmap
        self._latest = {}                # camera -> (t, [(track_id, x, y)])
        self._fused = collections.defaultdict(collections.deque)   # classroom -> deque[(t, n)]
        self._heatmaps = {}
        self._last_tick = None
        self.projected = 0
        self.outside = 0

    # ------------------------------------------------------------ configuration
    def configure(self, camera_configs, now=None):
        now = time.time() if now is None else now
        homography, camera_classroom, plans = {}, {}, {}
        armed = set()
        for c in camera_configs:
            classroom = c.classroom or {}
            cid = classroom.get("classroom_id")
            if not cid:
                continue
            camera_classroom[c.camera_id] = cid
            if (getattr(c, "features", None) or {}).get(HEATMAP_FEATURE):
                armed.add(c.camera_id)
            matrix = build_homography((c.placement or {}).get("floor_plan_homography"))
            if matrix is not None:
                homography[c.camera_id] = matrix
            plan = classroom.get("floor_plan") or {}
            try:
                width, height = float(plan.get("width_m") or 0), float(plan.get("height_m") or 0)
                grid = float(plan.get("grid_m") or DEFAULT_GRID_M)
            except (TypeError, ValueError):
                width = height = 0.0
                grid = DEFAULT_GRID_M
            plans[cid] = (width, height, grid) if width > 0 and height > 0 and grid > 0 else None
        # ZAYED: HEATMAP_FEATURE is now authoritative. It was declared here and never read, so
        # the switch on the camera page did nothing - a classroom with a floor plan and
        # calibrated cameras accumulated and published a heatmap whether or not anyone had
        # armed it.
        #
        # The gate is applied to the HEATMAP GRID ONLY, never to `plans`, `_homography` or
        # `_latest`: those also drive the floor-fused occupancy count and `location_for`, and
        # `plans` is what bounds a projected point to the room in observe(). Gating any of
        # them would silently change occupancy, which this switch has no business doing.
        #
        # "Every camera of the classroom" rather than "any": the heatmap is ONE fused grid per
        # classroom, so there is no way to drop one camera's contribution without changing the
        # fusion itself. This is the same rule fused_counts() already applies for calibration
        # (see fully_mapped) - a classroom aggregate is only produced when all of its sources
        # agree. Turning the feature off on any camera of the room therefore stops that room's
        # heatmap, and turning it off everywhere stops all of it.
        classroom_cameras = collections.defaultdict(set)
        for camera_id, cid in camera_classroom.items():
            classroom_cameras[cid].add(camera_id)
        heatmap_classrooms = {cid for cid, cams in classroom_cameras.items()
                              if cams and cams <= armed}

        with self._lock:
            changed = set(homography) != set(self._homography)
            arming_changed = heatmap_classrooms != self._heatmap_classrooms
            self._homography, self._camera_classroom, self._plans = homography, camera_classroom, plans
            self._heatmap_classrooms = heatmap_classrooms
            for cid, plan in plans.items():
                current = self._heatmaps.get(cid)
                if plan is None or cid not in heatmap_classrooms:
                    # Dropping the grid discards the part-window it holds. That is the point:
                    # the switch is off, so those person-seconds must not reach the dashboard.
                    self._heatmaps.pop(cid, None)
                elif current is None or (current.width_m, current.height_m, current.grid_m) != plan:
                    self._heatmaps[cid] = _Heatmap(*plan, started_at=now)
        if changed:
            mapped = ", ".join(sorted(homography)) or "none"
            self._log(f"[FLOOR] mapped cameras: {mapped} (the others have no floor-plan calibration)")
        if arming_changed:
            armed_rooms = ", ".join(sorted(heatmap_classrooms)) or "none"
            self._log(f"[FLOOR] heatmap armed for: {armed_rooms} "
                      f"(a classroom needs {HEATMAP_FEATURE} on every one of its cameras)")

    def mapped(self, camera_id):
        return camera_id in self._homography

    def fully_mapped(self, classroom_id):
        cams = [c for c, cid in self._camera_classroom.items() if cid == classroom_id]
        return bool(cams) and all(c in self._homography for c in cams)

    # ------------------------------------------------------------ per frame
    def observe(self, camera_id, person_tracks, now):
        matrix = self._homography.get(camera_id)
        if matrix is None:
            return
        points = []
        if person_tracks:
            floor = project(matrix, [foot_point(t.bbox) for t in person_tracks])
            plan = self._plans.get(self._camera_classroom.get(camera_id))
            for track, (x, y) in zip(person_tracks, floor):
                if plan is not None and not (-OUTSIDE_MARGIN_M <= x <= plan[0] + OUTSIDE_MARGIN_M
                                             and -OUTSIDE_MARGIN_M <= y <= plan[1] + OUTSIDE_MARGIN_M):
                    self.outside += 1
                    continue
                points.append((track.track_id, float(x), float(y)))
            self.projected += len(points)
        with self._lock:
            self._latest[camera_id] = (now, points)

    def location_for(self, camera_id, bbox):
        matrix = self._homography.get(camera_id)
        if matrix is None or not bbox:
            return None
        x, y = project(matrix, [foot_point(bbox)])[0]
        return {"classroom_id": self._camera_classroom.get(camera_id), "floor_x_m": round(float(x), 2),
                "floor_y_m": round(float(y), 2), "source": "floor_plan_homography"}

    # ------------------------------------------------------------ once a second
    def tick(self, now):
        if self._last_tick is not None and now - self._last_tick < FUSION_INTERVAL_SECONDS:
            return
        dt = min(2.0, now - self._last_tick) if self._last_tick is not None else FUSION_INTERVAL_SECONDS
        self._last_tick = now
        with self._lock:
            by_classroom = collections.defaultdict(list)
            for camera_id, (t, points) in self._latest.items():
                if now - t > FUSION_MAX_AGE_SECONDS:
                    continue
                cid = self._camera_classroom.get(camera_id)
                by_classroom[cid].extend((camera_id, x, y) for _, x, y in points)
            for cid in {c for c in self._camera_classroom.values()}:
                if not any(self._camera_classroom.get(cam) == cid for cam in self._homography):
                    continue
                people = fuse(by_classroom.get(cid, []))
                history = self._fused[cid]
                history.append((now, len(people)))
                while history and now - history[0][0] > 120:
                    history.popleft()
                heatmap = self._heatmaps.get(cid)
                if heatmap is not None:
                    heatmap.samples += 1
                    for x, y, _ in people:
                        heatmap.add(x, y, dt)

    def fused_counts(self, classroom_id, now, window):
        """Fused person counts in the window - only when EVERY camera of the classroom is mapped."""
        if not self.fully_mapped(classroom_id):
            return None
        with self._lock:
            return [n for t, n in self._fused.get(classroom_id, ()) if now - t <= window] or None

    def take_heatmaps(self, now=None):
        """Close the current windows and return them as POST rows (cells sparse)."""
        now = time.time() if now is None else now
        rows = []
        with self._lock:
            for cid, heatmap in list(self._heatmaps.items()):
                if heatmap.samples == 0:
                    continue
                rows.append({"classroom_id": cid, "window_start": heatmap.started_at, "window_end": now,
                             "grid_m": heatmap.grid_m, "cols": heatmap.cols, "rows": heatmap.rows,
                             "width_m": heatmap.width_m, "height_m": heatmap.height_m,
                             "samples": heatmap.samples,
                             "person_seconds": round(sum(heatmap.cells.values()), 1),
                             "cells": [[c, r, round(s, 1)] for (c, r), s in sorted(heatmap.cells.items())],
                             "method": "floor_fusion",
                             "cameras_mapped": sorted(cam for cam in self._homography
                                                      if self._camera_classroom.get(cam) == cid)})
                self._heatmaps[cid] = _Heatmap(heatmap.width_m, heatmap.height_m, heatmap.grid_m, started_at=now)
        return rows

    def forget_camera(self, camera_id):
        with self._lock:
            self._latest.pop(camera_id, None)

    def stats(self):
        return {"mapped_cameras": sorted(self._homography), "projected": self.projected, "outside": self.outside,
                "heatmap_classrooms": sorted(self._heatmaps),
                # Armed but not accumulating means the room has no floor plan or no calibration.
                "heatmap_armed_classrooms": sorted(self._heatmap_classrooms)}


class HeatmapPublisher:
    """POSTs closed heatmap windows every HEATMAP_PUBLISH_SECONDS; keeps failed ones (bounded)."""

    def __init__(self, mapper, dashboard_url, token, interval=HEATMAP_PUBLISH_SECONDS, timeout=5.0,
                 session=None, log=print):
        self.mapper = mapper
        self.url = f"{dashboard_url.rstrip('/')}{HEATMAP_PATH}"
        self.token, self.interval, self.timeout = token, interval, timeout
        self._session = session
        self._log = log
        self._pending = collections.deque(maxlen=HEATMAP_PENDING_MAX)
        self._stop = threading.Event()
        self._thread = None
        self.sent = self.failed = 0

    def publish_once(self, now=None):
        self._pending.extend(self.mapper.take_heatmaps(now))
        if not self._pending:
            return False
        rows = list(self._pending)
        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({"Authorization": f"Token {self.token}"})
            response = self._session.post(self.url, json={"heatmaps": rows}, timeout=self.timeout)
            ok = 200 <= response.status_code < 300
        except Exception:                                # noqa: BLE001
            ok = False
        if ok:
            self._pending.clear()
            self.sent += len(rows)
        else:
            self.failed += 1
        return ok

    def start(self):
        self._thread = threading.Thread(target=self._run, name="heatmap-publisher", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.publish_once()
            except Exception:                            # noqa: BLE001
                self.failed += 1

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)
        try:
            self.publish_once()                          # hand over the last partial window
        except Exception:                                # noqa: BLE001
            pass
