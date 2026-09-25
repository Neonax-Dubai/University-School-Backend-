"""
Identity supervisor (Zayed) - brings face identification and Re-ID back after a failed start.

THE GAP. face_id_adapter (Dubai, unchanged here) latches Face-ID OFF for the life of the
process when its processor cannot be built - for example Qdrant unreachable while the models
load - and reid_adapter disables Re-ID the same way when its store cannot be opened. On a cold
boot where the inference container comes up before Qdrant, identification, and with it
classroom presence, would stay dead until someone restarted the process by hand.

HERE. A background thread notices either condition and rebuilds that component OFF the
inference thread, with exponential backoff (IDENTITY_RETRY_MIN_SECONDS doubling up to
IDENTITY_RETRY_MAX_SECONDS). When the rebuilt component is healthy it is swapped in through the
setter the caller supplied and the failed one is closed. Detection, tracking, events and every
other analytic keep running throughout; the process is never restarted for this.

Runtime outages after a successful start are NOT this module's business: the Dubai modules
already contain those (search failures, a circuit breaker in known_person, retried upserts).
"""
import os
import threading
import time

RETRY_MIN_SECONDS = float(os.getenv("IDENTITY_RETRY_MIN_SECONDS", "30"))
RETRY_MAX_SECONDS = float(os.getenv("IDENTITY_RETRY_MAX_SECONDS", "600"))
CHECK_SECONDS = 5.0


def face_failed(adapter):
    """True when face_id_adapter latched its processor build as failed."""
    return bool(getattr(adapter, "_processor_failed", False))


def face_ready(adapter):
    return getattr(adapter, "_processor", None) is not None and not face_failed(adapter)


def reid_failed(adapter):
    """Re-ID disabled although cameras ARE authorised = its initialisation failed.

    (enabled=False with no authorised camera is the legitimate 'nothing to do' state.)"""
    return bool(getattr(adapter, "_authorised", None)) and not getattr(adapter, "enabled", False)


class _Component:
    def __init__(self, name, get, set_, build, failed, arm=None, ready=None):
        self.name = name
        self.get, self.set, self.build = get, set_, build
        self.failed, self.arm = failed, arm
        self.ready = ready or (lambda adapter: not failed(adapter))
        self.delay = None
        self.next_attempt = 0.0
        self.attempts = 0
        self.recoveries = 0
        self.last_error = None


class IdentitySupervisor:
    """Watches Face-ID and Re-ID and rebuilds whichever failed to start."""

    def __init__(self, log=print, clock=time.monotonic, min_delay=RETRY_MIN_SECONDS,
                 max_delay=RETRY_MAX_SECONDS):
        self._log = log
        self._clock = clock
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._components = []
        self._stop = threading.Event()
        self._thread = None

    def watch(self, name, get, set_, build, failed, arm=None, ready=None):
        """get()/set_(new): the live instance. build(): a fresh, unarmed instance.
        arm(new): make it start work (Face-ID loads its models here). failed(x)/ready(x)."""
        self._components.append(_Component(name, get, set_, build, failed, arm, ready))
        return self

    # ------------------------------------------------------------------ one pass
    def check_once(self):
        """Returns the names of components that were recovered in this pass."""
        recovered = []
        now = self._clock()
        for comp in self._components:
            try:
                current = comp.get()
                if current is None or not comp.failed(current):
                    comp.delay = None                     # healthy: reset the backoff
                    continue
                if comp.delay is None:                    # first sight of the failure
                    comp.delay = self.min_delay
                    comp.next_attempt = now + comp.delay
                    self._log(f"[IDENTITY] {comp.name} failed to start - retrying in the background "
                              f"every {self.min_delay:.0f}-{self.max_delay:.0f}s; inference continues")
                    continue
                if now < comp.next_attempt:
                    continue
                comp.attempts += 1
                fresh = comp.build()
                if comp.arm is not None:
                    comp.arm(fresh)
                if comp.ready(fresh):
                    comp.set(fresh)
                    comp.recoveries += 1
                    comp.delay = None
                    comp.last_error = None
                    recovered.append(comp.name)
                    self._log(f"[IDENTITY] {comp.name} RECOVERED on attempt {comp.attempts} - swapped in")
                    self._close(current)
                else:
                    self._close(fresh)
                    comp.delay = min(self.max_delay, comp.delay * 2)
                    comp.next_attempt = self._clock() + comp.delay
                    self._log(f"[IDENTITY] {comp.name} still unavailable (attempt {comp.attempts}) - "
                              f"next try in {comp.delay:.0f}s")
            except Exception as exc:                      # noqa: BLE001 - never kill the thread
                comp.last_error = f"{type(exc).__name__}: {exc}"
                comp.delay = min(self.max_delay, (comp.delay or self.min_delay) * 2)
                comp.next_attempt = self._clock() + comp.delay
                self._log(f"[IDENTITY] {comp.name} rebuild error ({comp.last_error}) - "
                          f"next try in {comp.delay:.0f}s")
        return recovered

    @staticmethod
    def _close(adapter):
        try:
            adapter.close()
        except Exception:                                 # noqa: BLE001
            pass

    # ------------------------------------------------------------------ thread
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="identity-supervisor", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(CHECK_SECONDS):
            self.check_once()

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)

    def stats(self):
        return {c.name: {"failed_now": bool(c.get() is not None and c.failed(c.get())),
                         "attempts": c.attempts, "recoveries": c.recoveries,
                         "next_delay_s": c.delay, "last_error": c.last_error}
                for c in self._components}
