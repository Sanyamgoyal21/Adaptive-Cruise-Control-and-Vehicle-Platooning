"""Fixed-step, discrete-time simulation engine for a single-lane road."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .controllers import MODES
from .sensors import Radar, V2VChannel, true_predecessor


@dataclass
class SimResult:
    t: np.ndarray
    names: list[str]
    x: np.ndarray           # [N, K] front-bumper position
    v: np.ndarray           # [N, K]
    a: np.ndarray           # [N, K]
    u: np.ndarray           # [N, K] commanded acceleration
    gap: np.ndarray         # [N, K] true gap to vehicle ahead (nan if none)
    gap_ref: np.ndarray     # [N, K] desired gap r + h v (nan if n/a)
    mode: np.ndarray        # [N, K] int index into MODES
    collisions: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def spacing_error(self):
        return self.gap - self.gap_ref

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])


class Simulation:
    def __init__(self, vehicles, dt: float = 0.01, radar: Radar | None = None,
                 v2v: V2VChannel | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vehicles = vehicles
        self.dt = dt
        self.radar = radar or Radar(rng=np.random.default_rng(rng.integers(1 << 31)))
        self.v2v = v2v or V2VChannel(rng=np.random.default_rng(rng.integers(1 << 31)))

    # ------------------------------------------------------------ stepping API
    def reset_state(self):
        self.t = 0.0
        self.collisions: list = []
        self.crashed: set = set()

    def step(self):
        """Advance the whole scene by one time step ``dt``.

        Returns a snapshot of the state at the START of the step (what the
        controllers saw) as a dict of 1-D arrays indexed like ``self.vehicles``.
        The vehicle list may grow between calls (live mode adds cut-in cars).
        """
        if not hasattr(self, "t"):
            self.reset_state()
        t, dt, vehs = self.t, self.dt, self.vehicles
        n = len(vehs)
        snap = {k: np.full(n, np.nan) for k in ("x", "v", "a", "u", "gap", "gap_ref")}
        snap["mode"] = np.zeros(n, dtype=int)

        self.v2v.deliver(t)
        for veh in vehs:                       # handle cut-in spawns
            if not veh.spawned and t >= veh.active_from:
                ref = next(o for o in vehs if o.id == veh.spawn_ref)
                veh.x = ref.x + veh.spawn_gap + veh.p.length
                veh.spawned = True
        cmds = []
        for i, veh in enumerate(vehs):
            ctrl = veh.controller
            target = self.radar.measure(veh, vehs, t)
            msg = self.v2v.inbox.get(veh.id)
            u = ctrl.command(t, dt, veh, target, msg) if ctrl else 0.0
            if veh.id in self.crashed:         # wrecked cars brake to standstill
                u = veh.p.a_min
            cmds.append(u)

            pred, gap = true_predecessor(veh, vehs, t)
            snap["x"][i], snap["v"][i], snap["a"][i], snap["u"][i] = veh.x, veh.v, veh.a, veh.u
            snap["mode"][i] = MODES.index(ctrl.mode) if ctrl else 0
            if pred is not None and veh.is_active(t):
                snap["gap"][i] = gap
                h = getattr(ctrl, "h_cur", getattr(ctrl, "h", None))
                if h is not None and hasattr(ctrl, "r"):
                    snap["gap_ref"][i] = ctrl.r + h * veh.v
                if gap < 0:
                    # rear-end collision: perfectly inelastic impact of equal
                    # masses; the cars stay bumper-to-bumper and both brake.
                    if veh.id not in self.crashed:
                        self.collisions.append((float(t), veh.name, pred.name))
                    self.crashed.update({veh.id, pred.id})
                    veh.x = pred.x - pred.p.length
                    v_common = 0.5 * (veh.v + pred.v)
                    veh.v = pred.v = v_common
                    snap["gap"][i] = 0.0
        for veh, u in zip(vehs, cmds):
            veh.step(u, dt)
        self.v2v.transmit(t, vehs)
        self.t = t + dt
        return snap

    # ------------------------------------------------------------ batch run
    def run(self, T: float) -> SimResult:
        self.reset_state()
        K = int(round(T / self.dt)) + 1
        N = len(self.vehicles)
        t_arr = np.arange(K) * self.dt
        out = {k: np.full((N, K), np.nan) for k in ("x", "v", "a", "u", "gap", "gap_ref")}
        M = np.zeros((N, K), dtype=int)
        for k, t in enumerate(t_arr):
            self.t = t                         # avoid floating-point drift
            snap = self.step()
            for key in out:
                out[key][:, k] = snap[key]
            M[:, k] = snap["mode"]
        return SimResult(t_arr, [v.name for v in self.vehicles], out["x"], out["v"], out["a"],
                         out["u"], out["gap"], out["gap_ref"], M, self.collisions,
                         meta={"v2v_sent": self.v2v.sent, "v2v_dropped": self.v2v.dropped})
