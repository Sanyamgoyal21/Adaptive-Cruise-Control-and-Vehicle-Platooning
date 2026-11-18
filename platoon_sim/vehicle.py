"""Longitudinal vehicle model.

Each vehicle is modelled as a point mass whose low-level (engine/brake)
controller tracks a desired acceleration ``u`` with a first-order lag:

    x_dot = v
    v_dot = a
    a_dot = (u - a) / tau

This "third-order" model is the standard model in the ACC/CACC literature
(Rajamani 2012, Ploeg et al. 2011). ``tau`` lumps together engine, driveline
and brake actuator dynamics (typically 0.1 - 0.6 s for passenger cars).
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class VehicleParams:
    length: float = 4.5     # [m] bumper-to-bumper length
    tau: float = 0.3        # [s] actuator / powertrain time constant
    a_min: float = -6.0     # [m/s^2] maximum braking (negative)
    a_max: float = 2.5      # [m/s^2] maximum acceleration


class Vehicle:
    """A single vehicle on a straight single-lane road.

    ``x`` is the position of the FRONT bumper. The gap to a vehicle ahead is
    therefore ``x_ahead - length_ahead - x``.
    """

    def __init__(self, vid: int, x0: float, v0: float,
                 params: VehicleParams | None = None,
                 controller=None, name: str | None = None,
                 active_from: float = 0.0, active_until: float = float("inf"),
                 spawn_ref: int | None = None, spawn_gap: float | None = None,
                 connected: bool = True):
        self.id = vid
        self.name = name or f"veh{vid}"
        self.p = params or VehicleParams()
        self.controller = controller
        # Before ``active_from`` the vehicle exists but is in another lane, so
        # radars in our lane cannot see it (used for cut-in scenarios).
        self.active_from = active_from
        self.active_until = active_until          # lane change OUT of our lane
        # Cut-in: at ``active_from`` the vehicle appears ``spawn_gap`` metres
        # in front of vehicle ``spawn_ref`` (it was driving alongside it).
        self.spawn_ref = spawn_ref
        self.spawn_gap = spawn_gap
        self.spawned = spawn_ref is None
        self.connected = connected                 # has a V2V radio?
        self.x = float(x0)
        self.v = float(v0)
        self.a = 0.0
        self.u = 0.0

    def is_active(self, t: float) -> bool:
        return self.active_from <= t < self.active_until

    def step(self, u: float, dt: float) -> None:
        """Advance the vehicle state by ``dt`` given desired acceleration ``u``."""
        u = min(max(u, self.p.a_min), self.p.a_max)
        self.u = u
        # Exact discretisation of the first-order lag under zero-order hold.
        alpha = 1.0 - math.exp(-dt / self.p.tau)
        a_new = self.a + alpha * (u - self.a)
        v_new = self.v + 0.5 * (self.a + a_new) * dt
        if v_new <= 0.0:              # vehicles do not reverse: brakes hold at standstill
            v_new = 0.0
            a_new = max(a_new, 0.0) if u > 0 else 0.0
        self.x += 0.5 * (self.v + v_new) * dt
        self.v = v_new
        self.a = a_new
