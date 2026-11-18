"""Longitudinal controllers.

All controllers share one interface::

    u = ctrl.command(t, dt, ego, radar_target, v2v_msg)

where ``u`` is the desired acceleration sent to the vehicle's low-level
controller, ``radar_target`` is a :class:`RadarTarget` or ``None`` and
``v2v_msg`` the latest V2V message received (or ``None``). After each call
``ctrl.mode`` names the active mode (for logging / plots).

Spacing policy (all following controllers)
------------------------------------------
Constant-time-headway (CTH) policy: desired gap  d_ref = r + h * v
    r : standstill distance [m],  h : time headway [s]
Spacing error:  e = d - d_ref
"""
from __future__ import annotations

import numpy as np

MODES = ["SCRIPTED", "CRUISE", "FOLLOW", "CACC", "ACC-FALLBACK", "AEB", "HUMAN"]


class _Base:
    uses_v2v = False
    label = "base"

    def __init__(self):
        self.mode = "CRUISE"
        self.aeb_events = 0

    def reset(self):
        pass


class SpeedProfile(_Base):
    """Scripted vehicle (platoon leader / traffic) tracking a piecewise-linear
    speed profile ``v_ref(t)`` with feed-forward + proportional feedback."""

    label = "scripted"

    def __init__(self, t_pts, v_pts, k: float = 1.5):
        super().__init__()
        self.t_pts = np.asarray(t_pts, float)
        self.v_pts = np.asarray(v_pts, float)
        self.k = k
        self.mode = "SCRIPTED"

    def v_ref(self, t):
        return float(np.interp(t, self.t_pts, self.v_pts))

    def a_ref(self, t):
        i = np.searchsorted(self.t_pts, t, side="right") - 1
        if i < 0 or i >= len(self.t_pts) - 1:
            return 0.0
        dt = self.t_pts[i + 1] - self.t_pts[i]
        return float((self.v_pts[i + 1] - self.v_pts[i]) / dt) if dt > 0 else 0.0

    def command(self, t, dt, ego, radar, msg):
        return self.a_ref(t) + self.k * (self.v_ref(t) - ego.v)


class ACC(_Base):
    """Adaptive Cruise Control (radar only).

    * CRUISE : u_cc = k_v (v_set - v)
    * FOLLOW : u_f  = k_p e + k_d (v_pred - v)        (Milanes & Shladover 2014)
    * u = min(u_cc, u_f)  -> smooth, automatic switch between the two modes.
    * Optional comfort limit ``a_comfort_min`` on ACC braking (ISO 15622
      suggests about -3.5 m/s^2); harder braking is left to the AEB layer.
    * Optional AEB layer: full braking if time-to-collision < ``ttc_aeb``.
    """

    label = "ACC"

    def __init__(self, v_set: float = 30.0, h: float = 1.6, r: float = 3.0,
                 kp: float = 0.25, kd: float = 0.8, k_v: float = 0.4,
                 ttc_aeb: float | None = 1.5, a_aeb: float = -6.0,
                 a_comfort_min: float | None = None):
        super().__init__()
        self.a_comfort_min = a_comfort_min
        self.v_set, self.h, self.r = v_set, h, r
        self.kp, self.kd, self.k_v = kp, kd, k_v
        self.ttc_aeb, self.a_aeb = ttc_aeb, a_aeb

    def cruise(self, ego):
        return self.k_v * (self.v_set - ego.v)

    def follow(self, ego, radar, h=None):
        h = self.h if h is None else h
        e = radar.gap - self.r - h * ego.v
        return self.kp * e + self.kd * radar.rel_speed

    def _aeb(self, radar):
        if self.ttc_aeb is None or radar is None or radar.rel_speed >= -1e-3:
            return False
        ttc = radar.gap / -radar.rel_speed
        return ttc < self.ttc_aeb or radar.gap < 0.5

    def command(self, t, dt, ego, radar, msg):
        u_cc = self.cruise(ego)
        if radar is None:
            self.mode = "CRUISE"
            return u_cc
        if self._aeb(radar):
            self.mode = "AEB"
            self.aeb_events += 1
            return self.a_aeb
        u_f = self.follow(ego, radar)
        if self.a_comfort_min is not None:
            u_f = max(u_f, self.a_comfort_min)
        self.mode = "FOLLOW" if u_f < u_cc else "CRUISE"
        return min(u_cc, u_f)


class CACC(ACC):
    """Cooperative ACC (Ploeg et al. 2011, 2014).

    The predecessor's desired acceleration u_{i-1} is received over V2V and
    used as a feed-forward term; the controller state u_i obeys

        h * du_i/dt = -u_i + k_p e_i + k_d de_i/dt + u_{i-1}(t - theta)
        de_i/dt     = (v_{i-1} - v_i) - h a_i

    which is string stable for much smaller headways than ACC.

    Graceful degradation: if no fresh message (age > ``timeout``) is available,
    the controller falls back to ACC and ramps its headway to ``h_fallback`` at
    ``h_rate`` s/s (avoids the hard braking an instant headway jump causes).
    """

    uses_v2v = True
    label = "CACC"

    def __init__(self, v_set: float = 30.0, h: float = 0.6, r: float = 3.0,
                 kp: float = 0.2, kd: float = 0.7, k_v: float = 0.4,
                 timeout: float = 0.5, h_fallback: float = 1.6, h_rate: float = 0.2,
                 acc_kp: float = 0.25, acc_kd: float = 0.8,
                 ttc_aeb: float | None = 1.5, a_aeb: float = -6.0):
        super().__init__(v_set, h, r, kp, kd, k_v, ttc_aeb, a_aeb)
        self.h_cacc = h
        self.h_fallback = h_fallback
        self.h_rate = h_rate
        self.timeout = timeout
        self.acc_kp, self.acc_kd = acc_kp, acc_kd
        self.h_cur = h
        self.uf = 0.0           # controller state (filtered command)

    def _safe_stop_accel(self, ego, radar, msg, trigger=-4.0, margin=1.15):
        """Kinematic safety supervisor (uses the V2V message).

        When the predecessor reports emergency braking (intended or actual
        acceleration below ``trigger``), return the deceleration that lets the
        ego car stop ``r`` metres behind it, assuming it keeps braking that
        hard. Covers the case the linear law misses: an ego that is still
        closing a large gap at higher speed when the platoon brakes."""
        a_p = min(msg.u, msg.a)
        if a_p >= trigger or ego.v < 0.5:
            return np.inf
        v_p = max(ego.v + radar.rel_speed, 0.0)
        room = radar.gap - self.r + v_p ** 2 / (2.0 * -a_p)
        if room <= 0.1:
            return -np.inf
        return -margin * ego.v ** 2 / (2.0 * room)

    def command(self, t, dt, ego, radar, msg):
        u_cc = self.cruise(ego)
        if radar is None:
            self.mode = "CRUISE"
            self.uf = u_cc
            return u_cc
        if self._aeb(radar):
            self.mode = "AEB"
            self.aeb_events += 1
            self.uf = self.a_aeb
            return self.a_aeb

        fresh = (msg is not None and msg.sender == radar.target_id
                 and (t - msg.stamp) <= self.timeout)
        h_target = self.h_cacc if fresh else self.h_fallback
        step = self.h_rate * dt
        self.h_cur += float(np.clip(h_target - self.h_cur, -step, step))
        h = self.h_cur

        e = radar.gap - self.r - h * ego.v
        if fresh:
            e_dot = radar.rel_speed - h * ego.a
            self.uf += dt / h * (-self.uf + self.kp * e + self.kd * e_dot + msg.u)
            self.uf = float(np.clip(self.uf, ego.p.a_min, ego.p.a_max))
            u_f = min(self.uf, self._safe_stop_accel(ego, radar, msg))
            mode = "CACC"
        else:
            u_f = self.acc_kp * e + self.acc_kd * radar.rel_speed
            self.uf = float(np.clip(u_f, ego.p.a_min, ego.p.a_max))   # bumpless re-entry
            mode = "ACC-FALLBACK"
        self.mode = mode if u_f < u_cc else "CRUISE"
        return min(u_cc, u_f)


class IDM(_Base):
    """Intelligent Driver Model (Treiber et al. 2000) - a human-driver baseline
    with an optional perception-reaction delay ``t_react``."""

    label = "Human (IDM)"

    def __init__(self, v0: float = 30.0, T: float = 1.5, s0: float = 2.0,
                 a: float = 1.5, b: float = 2.0, t_react: float = 0.6):
        super().__init__()
        self.v0, self.T, self.s0, self.a, self.b = v0, T, s0, a, b
        self.t_react = t_react
        self._buf: list[tuple[float, float]] = []
        self.mode = "HUMAN"

    def command(self, t, dt, ego, radar, msg):
        free = 1.0 - (ego.v / self.v0) ** 4
        if radar is None:
            acc = self.a * free
        else:
            dv = -radar.rel_speed                        # approach rate
            s_star = self.s0 + max(0.0, ego.v * self.T + ego.v * dv / (2 * np.sqrt(self.a * self.b)))
            acc = self.a * (free - (s_star / max(radar.gap, 0.1)) ** 2)
        # reaction delay: the driver acts on what they decided t_react ago
        self._buf.append((t, acc))
        while len(self._buf) > 1 and self._buf[1][0] <= t - self.t_react:
            self._buf.pop(0)
        return self._buf[0][1] if self._buf[0][0] <= t - self.t_react + 1e-9 else 0.0


class ManualDriver(_Base):
    """Leader driven interactively (real-time mode): tracks a target speed set
    by the user with comfortable acceleration limits; ``emergency()`` brakes at
    the vehicle limit until standstill."""

    label = "manual"

    def __init__(self, v_target: float = 25.0, k: float = 0.8,
                 a_acc: float = 2.0, a_dec: float = -3.5):
        super().__init__()
        self.v_target, self.k, self.a_acc, self.a_dec = v_target, k, a_acc, a_dec
        self.braking = False
        self.mode = "SCRIPTED"

    def emergency(self):
        self.braking = True
        self.v_target = 0.0

    def command(self, t, dt, ego, radar, msg):
        if self.braking:
            if ego.v <= 0.05:
                self.braking = False
            return ego.p.a_min
        return float(np.clip(self.k * (self.v_target - ego.v), self.a_dec, self.a_acc))
