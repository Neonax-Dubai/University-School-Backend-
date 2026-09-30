"""
Classroom occupancy - de-duplicated across a classroom's cameras (Zayed).

A Zayed classroom is watched by three cameras, so summing per-camera counts would count most
people two or three times. Per camera this keeps the confirmed PERSON tracks over a short
window; per classroom it forms ONE occupancy figure:

  method "max_camera"   the best-placed camera's median count - a LOWER BOUND that can never
                        double count (it under-counts people only a different camera sees).
                        Used until the cameras have floor-plan calibration.
  method "floor_fusion" once EVERY camera of the classroom has a floor-plan homography: people
                        projected onto the floor and fused across cameras (floor_plan.py), the
                        median over the window.
  method "zone_ownership" when the operator has drawn OCCUPANCY zones (the existing dashboard
                        Zone, rules["occupancy"]): each such zone is an ownership region, drawn so
                        that no part of the room belongs to two cameras. Per camera, the person
                        tracks whose anchor point (zones.anchor_point, the zone convention) lies
                        inside one of its occupancy zones are counted once each; the classroom is
                        the SUM of those per-camera medians. A camera of that classroom with no
                        occupancy zone (C101's faculty view) is not counted at all. Takes
                        precedence over the other two methods for that classroom. The result is
                        OBSERVED occupancy: people outside every ownership zone are not counted.

Capacity is the backend's (classroom.capacity in the camera payload). With capacity set (and,
for a classroom counted in occupancy zones, only once one of them carries rules["overcrowding"]):

  OVERCROWDING_DETECTED  occupancy above capacity * overcrowding_threshold_pct for
                         OVERCROWD_SECONDS; re-armed only after it has dropped back below for
                         OVERCROWD_CLEAR_SECONDS (hysteresis, no flapping alarm).
  OCCUPANCY_UPDATED      when the occupancy changes by >= OCCUPANCY_EVENT_DELTA, at most once
                         per OCCUPANCY_EVENT_MIN_SECONDS - a change log, not a per-frame stream.

Samples go to POST /api/ai/people-counts/ ("occupancy" list) every PUBLISH_SECONDS on a daemon
thread; a failed cycle is forgotten, never queued.
"""
import collections
import os
import statistics
import threading
import time

import requests

import zones as zone_geometry

WINDOW_SECONDS = float(os.getenv("OCCUPANCY_WINDOW_SECONDS", "10"))
PUBLISH_SECONDS = float(os.getenv("OCCUPANCY_PUBLISH_SECONDS", "10"))
OVERCROWD_SECONDS = float(os.getenv("OVERCROWD_SECONDS", "30"))
OVERCROWD_CLEAR_SECONDS = float(os.getenv("OVERCROWD_CLEAR_SECONDS", "60"))
OCCUPANCY_EVENT_DELTA = int(os.getenv("OCCUPANCY_EVENT_DELTA", "3"))
OCCUPANCY_EVENT_MIN_SECONDS = float(os.getenv("OCCUPANCY_EVENT_MIN_SECONDS", "60"))
PERSON_GROUP = "person"
#: Zone.rules keys (the dashboard's zone modal): an occupancy ownership zone, and whether its
#: classroom raises OVERCROWDING_DETECTED.
OCCUPANCY_RULE, OVERCROWDING_RULE = "occupancy", "overcrowding"


class ClassroomOccupancy:
    def __init__(self):
        self._lock = threading.Lock()
        self._camera_classroom = {}
        self._classrooms = {}                 # classroom_id -> classroom dict from the payload
        self._counts = collections.defaultdict(collections.deque)   # camera -> deque[(t, n)]
        self._over_since = {}
        self._below_since = {}
        self._overcrowded = set()
        self._last_event = {}                 # classroom -> (t, occupancy)
        self.events_emitted = 0
        self._fusion = None                   # floor_plan.FloorPlanMapper, when installed
        self._owner_zones = {}                # camera -> [polygon] of its occupancy zones
        self._zone_classrooms = set()         # classrooms counted in occupancy zones
        self._overcrowding_classrooms = set() # ... of those, the ones that raise overcrowding

    def set_fusion(self, provider):
        """provider.fused_counts(classroom_id, now, window) -> [counts] | None (not fully mapped)."""
        self._fusion = provider

    def configure(self, camera_configs):
        with self._lock:
            self._camera_classroom = {c.camera_id: (c.classroom or {}).get("classroom_id")
                                      for c in camera_configs if (c.classroom or {}).get("classroom_id")}
            self._classrooms = {(c.classroom or {})["classroom_id"]: dict(c.classroom)
                                for c in camera_configs if (c.classroom or {}).get("classroom_id")}
            owners, overcrowding = {}, set()
            for c in camera_configs:
                classroom_id = self._camera_classroom.get(c.camera_id)
                zones = [z for z in (getattr(c, "zones", None) or [])
                         if isinstance(z, dict) and z.get("enabled", True)
                         and (z.get("rules") or {}).get(OCCUPANCY_RULE)
                         and isinstance(z.get("coordinates"), list) and len(z["coordinates"]) >= 3]
                if classroom_id and zones:
                    owners[c.camera_id] = [z["coordinates"] for z in zones]
                    if any((z.get("rules") or {}).get(OVERCROWDING_RULE) for z in zones):
                        overcrowding.add(classroom_id)
            self._owner_zones = owners
            self._zone_classrooms = {self._camera_classroom[c] for c in owners}
            self._overcrowding_classrooms = overcrowding

    def observe(self, camera_id, tracked_objects, now, frame_width=0, frame_height=0):
        classroom_id = self._camera_classroom.get(camera_id)
        if classroom_id is None:
            return
        if classroom_id in self._zone_classrooms:
            polygons = self._owner_zones.get(camera_id)
            if not polygons:
                return                       # no ownership zone on this camera: it is not counted
            count = 0
            for t in tracked_objects:
                if t.group != PERSON_GROUP:
                    continue
                point = zone_geometry.anchor_point(t.bbox, frame_width, frame_height)
                # One track counts once, however many of this camera's occupancy zones hold it.
                if point is not None and any(zone_geometry.point_in_polygon(point[0], point[1], polygon)
                                             for polygon in polygons):
                    count += 1
        else:
            count = sum(1 for t in tracked_objects if t.group == PERSON_GROUP)
        with self._lock:
            window = self._counts[camera_id]
            window.append((now, count))
            while window and now - window[0][0] > WINDOW_SECONDS:
                window.popleft()

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        rows = []
        with self._lock:
            for classroom_id, classroom in self._classrooms.items():
                per_camera, medians, maxima = {}, [], []
                zoned = classroom_id in self._zone_classrooms
                for camera_id, cid in self._camera_classroom.items():
                    if cid != classroom_id:
                        continue
                    if zoned and camera_id not in self._owner_zones:
                        continue                 # excluded (no ownership zone), not "not reporting"
                    window = [n for t, n in self._counts.get(camera_id, ()) if now - t <= WINDOW_SECONDS]
                    if not window:
                        per_camera[camera_id] = None
                        continue
                    median = int(statistics.median(window))
                    # Zones: each camera's share of the sum. Otherwise the camera's latest frame.
                    per_camera[camera_id] = median if zoned else window[-1]
                    medians.append(median)
                    maxima.append(max(window))
                if zoned:
                    # The ownership zones partition the room, so the per-camera shares add up.
                    occupancy = sum(medians) if medians else None
                    occupancy_max = sum(maxima) if maxima else None
                    method = "zone_ownership"
                else:
                    occupancy = max(medians) if medians else None
                    occupancy_max = max(maxima) if maxima else None
                    method = "max_camera"
                    fused = self._fusion.fused_counts(classroom_id, now, WINDOW_SECONDS) if self._fusion else None
                    if fused:
                        occupancy, occupancy_max, method = int(statistics.median(fused)), max(fused), "floor_fusion"
                rows.append({"classroom_id": classroom_id, "measured_at": now, "window_seconds": WINDOW_SECONDS,
                             "occupancy": occupancy,
                             "occupancy_max": occupancy_max,
                             "method": method, "per_camera": per_camera,
                             "capacity": classroom.get("capacity"),
                             "threshold_pct": classroom.get("overcrowding_threshold_pct") or 100.0})
        return rows

    def decisions(self, now=None):
        """(event_type, classroom_id, metadata) for OVERCROWDING / OCCUPANCY_UPDATED transitions."""
        now = time.time() if now is None else now
        out = []
        for row in self.snapshot(now):
            cid, occ, capacity = row["classroom_id"], row["occupancy"], row["capacity"]
            if occ is None:
                self._over_since.pop(cid, None)  # unknown: the hold starts again when counting resumes
                continue
            pct = round(100.0 * occ / capacity, 1) if capacity else None
            last = self._last_event.get(cid)
            if last is None or (abs(occ - last[1]) >= OCCUPANCY_EVENT_DELTA and now - last[0] >= OCCUPANCY_EVENT_MIN_SECONDS):
                self._last_event[cid] = (now, occ)
                out.append(("OCCUPANCY_UPDATED", cid, {"occupancy": occ, "capacity": capacity, "occupancy_pct": pct,
                                                        "method": row["method"], "per_camera": row["per_camera"]}))
            if not capacity:
                continue
            if cid in self._zone_classrooms and cid not in self._overcrowding_classrooms:
                self._over_since.pop(cid, None)  # occupancy zones without the Overcrowding use case
                continue
            over = pct > row["threshold_pct"]
            if over:
                self._below_since.pop(cid, None)
                since = self._over_since.setdefault(cid, now)
                if cid not in self._overcrowded and now - since >= OVERCROWD_SECONDS:
                    self._overcrowded.add(cid)
                    out.append(("OVERCROWDING_DETECTED", cid, {"occupancy": occ, "capacity": capacity,
                                                               "occupancy_pct": pct, "threshold_pct": row["threshold_pct"],
                                                               "over_for_seconds": round(now - since, 1),
                                                               "method": row["method"], "per_camera": row["per_camera"]}))
            else:
                self._over_since.pop(cid, None)
                if cid in self._overcrowded:
                    since = self._below_since.setdefault(cid, now)
                    if now - since >= OVERCROWD_CLEAR_SECONDS:
                        self._overcrowded.discard(cid)
                        self._below_since.pop(cid, None)
        return out

    def classroom_cameras(self, classroom_id):
        return sorted(c for c, cid in self._camera_classroom.items() if cid == classroom_id)


class OccupancyPublisher:
    """Posts classroom occupancy every PUBLISH_SECONDS. Never queues a failed cycle."""

    def __init__(self, occupancy, dashboard_url, token, interval=PUBLISH_SECONDS, timeout=5.0):
        self.occupancy = occupancy
        self.url = f"{dashboard_url.rstrip('/')}/api/ai/people-counts/"
        self.token, self.interval, self.timeout = token, interval, timeout
        self._stop = threading.Event()
        self._thread = None
        self.sent = self.failed = 0
        self._session = None

    def publish_once(self):
        rows = [{k: v for k, v in r.items() if k not in ("capacity", "threshold_pct")}
                for r in self.occupancy.snapshot() if r["occupancy"] is not None]
        if not rows:
            return False
        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({"Authorization": f"Token {self.token}"})
            response = self._session.post(self.url, json={"samples": [], "occupancy": rows}, timeout=self.timeout)
            ok = 200 <= response.status_code < 300
        except Exception:                                        # noqa: BLE001
            ok = False
        if ok:
            self.sent += 1
        else:
            self.failed += 1
        return ok

    def start(self):
        self._thread = threading.Thread(target=self._run, name="occupancy-publisher", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.publish_once()
            except Exception:                                    # noqa: BLE001
                self.failed += 1

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)

    def status_line(self):
        return f"[OCCUPANCY] published={self.sent} failed={self.failed}"
