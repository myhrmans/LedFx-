"""Fork addition: low-latency tick bus from LedFx to local sibling processes.

Publishes high-frequency audio-analysis events (tempo, beat, onset, optional
per-frame bands) over a ZeroMQ PUB socket on a Unix domain socket. Consumed by
the ledfx-controller's TickRelay, which fans the same data out to browsers via
WebSocket. The control plane (REST POST/PUT to /api/virtuals/...) is unchanged.

Why a separate channel instead of the existing WebSocket event bus:
  - Audio-thread emitters need a non-blocking, drop-on-overflow path. The
    aiohttp WS server's send_event is fine for ~Hz-rate events but stacking
    per-audio-frame data on it would couple the audio thread to socket-write
    backpressure.
  - PUB/SUB lets the controller die and reconnect without LedFx noticing.
  - UDS + msgpack keeps end-to-end latency in the tens of microseconds, which
    leaves room for downstream timer jitter (modulator 20Hz, sACN 30Hz) before
    the laser/DMX side notices.

Message framing is ZMQ multipart, two frames per send:
    frame 0: topic bytes (b"tempo", b"beat", b"onset", b"bands")
    frame 1: msgpack-encoded dict; all timestamps are `time.time()` floats
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Mapping

import msgpack
import zmq

from ledfx.events import Event

_LOGGER = logging.getLogger(__name__)

DEFAULT_IPC = "ipc:///tmp/ledfx-ticks.sock"
QUEUE_MAXSIZE = 1024
DROP_LOG_INTERVAL_S = 1.0


class TickPublisher:
    """ZMQ PUB publisher with a dedicated I/O thread.

    pyzmq Sockets are not safe to share across threads, so all sends go through
    a single internal thread that drains a bounded Queue. Callers on the audio
    thread (or anywhere else) push dicts in with `publish()` and never block —
    a full queue drops the message and rate-limits a warning.
    """

    def __init__(self, ledfx, endpoint: str | None = None):
        self._ledfx = ledfx
        self._endpoint = endpoint or os.environ.get(
            "LEDFX_TICK_IPC", DEFAULT_IPC
        )
        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None
        self._queue: queue.Queue[tuple[bytes, dict[str, Any]]] = queue.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._drops_since_log = 0
        self._last_drop_log = 0.0
        self._shutdown_listener = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        # SNDHWM caps how many messages ZMQ buffers per subscriber before
        # dropping. With PUB sockets that haven't connected anyone the buffer
        # backpressures on the publisher thread; we'd rather drop than block
        # the queue drain loop.
        self._sock.setsockopt(zmq.SNDHWM, 256)
        # LINGER=0 means close() doesn't block waiting for unsent messages on
        # shutdown — appropriate for a fire-and-forget tick stream.
        self._sock.setsockopt(zmq.LINGER, 0)
        self._bind()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="TickPublisher", daemon=True
        )
        self._thread.start()
        # Self-manage shutdown so callers don't have to remember.
        self._shutdown_listener = self._ledfx.events.add_listener(
            lambda _e: self.stop(), Event.LEDFX_SHUTDOWN
        )
        _LOGGER.info("TickPublisher bound to %s", self._endpoint)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        # Wake the queue.get() so the thread can observe _stop.
        try:
            self._queue.put_nowait((b"__stop__", {}))
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        # Don't terminate the shared Context.instance() — other LedFx
        # components may use it. Just drop our reference.
        self._ctx = None
        if self._shutdown_listener is not None:
            try:
                self._shutdown_listener()
            except Exception:
                pass
            self._shutdown_listener = None
        self._unlink_ipc_path()
        _LOGGER.info("TickPublisher stopped")

    def publish(self, topic: bytes, payload: Mapping[str, Any]) -> None:
        """Enqueue a message for publication. Safe from any thread.

        On a full queue the message is dropped and a rate-limited warning is
        logged. Callers must not assume the message was delivered.
        """
        if self._thread is None:
            return
        try:
            self._queue.put_nowait((topic, dict(payload)))
        except queue.Full:
            self._drops_since_log += 1
            now = time.time()
            if now - self._last_drop_log >= DROP_LOG_INTERVAL_S:
                _LOGGER.warning(
                    "TickPublisher dropped %d messages (queue full)",
                    self._drops_since_log,
                )
                self._drops_since_log = 0
                self._last_drop_log = now

    def _bind(self) -> None:
        assert self._sock is not None
        self._unlink_ipc_path()
        self._sock.bind(self._endpoint)

    def _unlink_ipc_path(self) -> None:
        # Only meaningful for ipc:// endpoints. tcp:// has nothing to unlink.
        if not self._endpoint.startswith("ipc://"):
            return
        path = self._endpoint[len("ipc://"):]
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            _LOGGER.warning(
                "Could not unlink stale IPC socket %s: %s", path, exc
            )

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                topic, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if topic == b"__stop__":
                break
            try:
                data = msgpack.packb(payload, use_bin_type=True)
                self._sock.send_multipart([topic, data])
            except zmq.ZMQError as exc:
                _LOGGER.warning(
                    "TickPublisher ZMQ send failed (topic=%r): %s",
                    topic,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "TickPublisher unexpected send error (topic=%r): %s",
                    topic,
                    exc,
                )
