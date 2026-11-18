"""Sensor and communication models.

* ``Radar``      - forward range / range-rate sensor (on-board, every car with ACC).
* ``V2VChannel`` - vehicle-to-vehicle wireless link (IEEE 802.11p / C-V2X style)
                   with a fixed broadcast rate, latency and random packet loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RadarTarget:
    target_id: int
    gap: float          # measured bumper-to-bumper distance [m]
    rel_speed: float    # measured v_target - v_ego [m/s]  (negative = closing)


def true_predecessor(ego, vehicles, t):
    """Nearest *active* vehicle ahead of ``ego`` (ground truth), or None."""
    best, best_gap = None, np.inf
    for other in vehicles:
        if other is ego or not other.is_active(t):
            continue
        gap = other.x - other.p.length - ego.x
        if other.x > ego.x and gap < best_gap:
            best, best_gap = other, gap
    return best, best_gap


class Radar:
    def __init__(self, max_range: float = 150.0, sigma_range: float = 0.05,
                 sigma_rate: float = 0.02, rng: np.random.Generator | None = None):
        self.max_range = max_range
        self.sigma_range = sigma_range
        self.sigma_rate = sigma_rate
        self.rng = rng or np.random.default_rng(0)

    def measure(self, ego, vehicles, t) -> RadarTarget | None:
        pred, gap = true_predecessor(ego, vehicles, t)
        if pred is None or gap > self.max_range:
            return None
        return RadarTarget(
            target_id=pred.id,
            gap=gap + self.rng.normal(0.0, self.sigma_range),
            rel_speed=(pred.v - ego.v) + self.rng.normal(0.0, self.sigma_rate),
        )


@dataclass
class V2VMessage:
    sender: int
    stamp: float        # time the message was generated
    u: float            # sender's desired acceleration (feed-forward signal)
    a: float
    v: float


class V2VChannel:
    """Broadcast link: every ``1/rate_hz`` s each vehicle sends a message to the
    vehicle directly behind it. Each packet is lost with probability ``loss``
    and otherwise arrives ``delay`` seconds later."""

    def __init__(self, rate_hz: float = 10.0, delay: float = 0.1, loss: float = 0.0,
                 rng: np.random.Generator | None = None, enabled: bool = True,
                 outage: tuple[float, float] | None = None):
        self.period = 1.0 / rate_hz
        self.delay = delay
        self.loss = loss
        self.enabled = enabled
        self.outage = outage            # (t_start, t_end): total link blackout
        self.rng = rng or np.random.default_rng(1)
        self._next_tx = 0.0
        self._pending: list[tuple[float, int, V2VMessage]] = []
        self.inbox: dict[int, V2VMessage] = {}
        self.sent = 0
        self.dropped = 0

    def transmit(self, t, vehicles):
        if not self.enabled or t + 1e-9 < self._next_tx:
            return
        self._next_tx += self.period
        for rx in vehicles:
            if rx.controller is None or not getattr(rx.controller, "uses_v2v", False):
                continue
            pred, _ = true_predecessor(rx, vehicles, t)
            if pred is None or not getattr(pred, "connected", True):
                continue                         # unconnected car ahead: nothing to receive
            self.sent += 1
            blackout = self.outage is not None and self.outage[0] <= t < self.outage[1]
            if blackout or self.rng.random() < self.loss:
                self.dropped += 1
                continue
            msg = V2VMessage(pred.id, t, pred.u, pred.a, pred.v)
            self._pending.append((t + self.delay, rx.id, msg))

    def deliver(self, t):
        keep = []
        for (t_arr, rx_id, msg) in self._pending:
            if t_arr <= t + 1e-9:
                self.inbox[rx_id] = msg
            else:
                keep.append((t_arr, rx_id, msg))
        self._pending = keep
