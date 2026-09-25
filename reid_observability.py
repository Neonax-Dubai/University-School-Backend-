"""
PHASE 3A - minimal, opt-in, decision-level observability for real-world
Re-ID threshold/margin calibration.

Purpose: produce ONE structured JSONL record per Re-ID DECISION BOUNDARY
(never per frame - see ReIDObservabilityLogger.record()'s own docstring),
carrying exactly the fields needed to characterize similarity/margin/
candidate-count distributions per camera group, without touching any
production database and without ever putting image data into a log line
(no crops, no base64, no raw frames - see the module docstring's own
"privacy" section below).

Deliberately NOT a logging framework: no handlers, no formatters, no
levels beyond the one on/off + one coarse REID_OBSERVABILITY_LOG_LEVEL knob
this module defines itself. A single class, a single JSONL sink, reusing
this codebase's existing "_bool_env + os.getenv, module-level constants"
convention (see reid_adapter.py) rather than introducing a new config
pattern.

======================================================================
Privacy / log-volume (see the calibration phase's own explicit rules)
======================================================================
- No image crops, no base64-encoded pixels, no raw frames - ever, in this
  module. Only IDs the system already generates (camera_id, local_track_id,
  global_id) plus numeric/string metadata about the DECISION itself.
- One record per decision boundary (NEW/MATCH/UNCERTAIN/SEARCH_FAILED/
  SKIPPED_SAME_CAMERA/SAME_CAM_BLOCKED/CROSS_CAM_MATCH/SAME_CAM_MATCH/
  SAME_CAM_UNCERTAIN/REFRESH), never per frame - reid_adapter.py's flush()
  calls record() at most once per ObserveResult, exactly where it already
  calls _print_observe()/_print_search_retries().
- REID_OBSERVABILITY_SAMPLE_RATE further bounds volume for a busy
  deployment (default 1.0 - log everything - appropriate for a bounded
  calibration data-collection run; turn down for continuous production use
  if the JSONL file's growth rate ever matters more than the fix's own
  natural per-decision cadence already bounds it to).
- Written to a plain, local JSONL file (REID_OBSERVABILITY_OUTPUT_PATH) -
  explicitly NOT a database, production or otherwise, per this phase's own
  requirement to keep calibration diagnostics outside production databases.

======================================================================
Safety
======================================================================
- REID_OBSERVABILITY_ENABLED defaults to False: constructing a
  ReIDObservabilityLogger and calling record() on it are both meant to be
  unconditionally safe to do from reid_adapter.py regardless of this flag -
  when disabled, record() is a single boolean check, no file is ever
  opened, matching this codebase's established "off means truly inert, not
  just skipped" convention (see reid_adapter.py's own module docstring on
  REID_PRODUCTION_ENABLED's heavy-import avoidance for the same principle).
- record() NEVER raises - a malformed field, a full disk, a permission
  error, anything - is caught, counted (self.errors), and dropped. A
  logging record must never be able to take down the inference tick that
  produced it, exactly like every other Re-ID failure mode in this codebase.
- Read-only with respect to Re-ID state: record() only ever reads its
  camera_id/local_track_id/ObserveResult arguments - it never touches
  GlobalReIDManager, the Store, or anything that could feed back into a
  future decision. Observability cannot change what gets observed.
"""
import json
import os
import random
import time


def _bool_env(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


REID_OBSERVABILITY_ENABLED = _bool_env("REID_OBSERVABILITY_ENABLED", False)

# 1.0 = log every decision (the default - appropriate for a bounded
# calibration run). Lower values further bound volume for continuous use.
REID_OBSERVABILITY_SAMPLE_RATE = float(os.getenv("REID_OBSERVABILITY_SAMPLE_RATE", "1.0"))

# Plain local file, NOT a database - see module docstring.
REID_OBSERVABILITY_OUTPUT_PATH = os.getenv("REID_OBSERVABILITY_OUTPUT_PATH", "reid_observability.jsonl")

# "all"       - every decision boundary, including REFRESH (needed for
#               track_age/margin-drift-over-time analysis - see calibration
#               Part 5's REFRESH-distribution requirement).
# "decisions" - excludes REFRESH, keeping only NEW/MATCH/UNCERTAIN/
#               SEARCH_FAILED/SKIPPED_SAME_CAMERA-family events - lower
#               volume for a deployment that only cares about identity
#               DECISIONS, not gallery-refresh bookkeeping.
REID_OBSERVABILITY_LOG_LEVEL = os.getenv("REID_OBSERVABILITY_LOG_LEVEL", "all").strip().lower()
_VALID_LOG_LEVELS = ("all", "decisions")
if REID_OBSERVABILITY_LOG_LEVEL not in _VALID_LOG_LEVELS:
    raise ValueError(
        f"REID_OBSERVABILITY_LOG_LEVEL={REID_OBSERVABILITY_LOG_LEVEL!r} is not valid - "
        f"must be one of {_VALID_LOG_LEVELS}"
    )


def _search_attempts_and_failed(result):
    """
    Derives (search_attempts, search_failed) from ObserveResult's existing
    status/search_retry_events - no new state needed. retry_events only
    records FAILED attempts (see reid_manager.GlobalReIDManager.
    _search_with_retry()'s own docstring): if the decision ultimately
    succeeded after N failures, there was one more (successful, unlogged)
    attempt on top of those N; if it ultimately gave up (SEARCH_FAILED),
    every attempt - including the last, "exhausted" one - is already in the
    list, so no "+1" is added. REFRESH runs no search at all - None, not 0,
    is the honest answer there (see ObserveResult's own docstring on this
    same distinction for candidate_count).
    """
    if result.status == "REFRESH":
        return None, False
    if result.search_retry_events is None:
        return 1, False
    if result.status == "SEARCH_FAILED":
        return len(result.search_retry_events), True
    return len(result.search_retry_events) + 1, False


class ReIDObservabilityLogger:
    """
    Owns the (optional) JSONL sink. One instance per ReIDAdapter, created
    unconditionally (cheap when disabled - see module docstring), closed
    from ReIDAdapter.close().
    """

    def __init__(self, enabled=None, sample_rate=None, output_path=None, log_level=None):
        self.enabled = REID_OBSERVABILITY_ENABLED if enabled is None else enabled
        self.sample_rate = REID_OBSERVABILITY_SAMPLE_RATE if sample_rate is None else sample_rate
        self.output_path = REID_OBSERVABILITY_OUTPUT_PATH if output_path is None else output_path
        self.log_level = REID_OBSERVABILITY_LOG_LEVEL if log_level is None else log_level

        self.records_written = 0
        self.records_dropped_sampled = 0
        self.records_skipped_log_level = 0
        self.errors = 0

        self._fh = None
        if self.enabled:
            try:
                self._fh = open(self.output_path, "a", buffering=1)  # line-buffered
            except Exception:  # noqa: BLE001 - a bad path must disable, not crash, the adapter
                self.enabled = False
                self._fh = None

    def record(self, camera_id, local_track_id, result):
        """
        One JSONL line for this ObserveResult, or nothing at all - never
        raises. Called from reid_adapter.py's flush() at most once per
        ObserveResult, the exact same call site _print_observe()/
        _print_search_retries() already use - never per frame.
        """
        if not self.enabled or self._fh is None:
            return

        if self.log_level == "decisions" and result.status == "REFRESH":
            self.records_skipped_log_level += 1
            return

        if self.sample_rate < 1.0 and random.random() >= self.sample_rate:
            self.records_dropped_sampled += 1
            return

        try:
            record = self._build_record(camera_id, local_track_id, result)
            self._fh.write(json.dumps(record) + "\n")
            self.records_written += 1
        except Exception:  # noqa: BLE001 - a bad record must never stop the next one
            self.errors += 1

    @staticmethod
    def _build_record(camera_id, local_track_id, result):
        search_attempts, search_failed = _search_attempts_and_failed(result)
        return {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "group": result.group,
            "local_track_id": local_track_id,
            "global_id": result.global_id,
            "decision": result.status,
            "best_similarity": result.similarity,
            "runner_up_similarity": result.second_similarity,
            "margin": result.margin,
            "candidate_count": result.candidate_count,
            "candidate_camera_ids": result.candidate_camera_ids,
            "camera_mode": result.effective_camera_mode,
            "similarity_threshold": result.effective_threshold,
            "min_match_margin": result.effective_min_margin,
            "bootstrap_observations": result.bootstrap_observations,
            "track_age_seconds": result.track_age_seconds,
            "search_attempts": search_attempts,
            "search_failed": search_failed,
            # PHASE 3B WORKSTREAM D - already computed by observe() on every
            # ObserveResult, simply not read by this module before now. No
            # new computation, no new call site - see reid_manager.py's own
            # ObserveResult docstring for what each represents.
            "skipped_candidate": result.skipped_candidate,
            "possible_id_swap": result.possible_id_swap,
            "best_candidate_id": result.best_candidate_id,
            "best_candidate_scope": result.best_candidate_scope,
            "candidate_active": result.candidate_active,
        }

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
