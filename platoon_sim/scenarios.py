"""Scenario builders: leader speed profiles, platoon construction, ACC test drive."""
from __future__ import annotations

import numpy as np

from .controllers import ACC, CACC, IDM, SpeedProfile
from .sensors import Radar, V2VChannel
from .simulation import Simulation
from .vehicle import Vehicle, VehicleParams

# ------------------------------------------------------------------ leader profiles
# (time points [s], speed points [m/s]) - piecewise linear.
LEADER_PROFILES = {
    # moderate braking 25 -> 15 m/s at -2.5 m/s^2, then recovery at +1 m/s^2
    "speed_dip": ([0, 10, 14, 19, 29, 200], [25, 25, 15, 15, 25, 25]),
    # emergency stop from 25 m/s at -6 m/s^2 (vehicle limit)
    "emergency_stop": ([0, 10, 10 + 25 / 6, 200], [25, 25, 0, 0]),
    # urban stop-and-go traffic
    "stop_and_go": ([0, 5, 12, 20, 26, 34, 42, 50, 60, 70, 200],
                    [15, 15, 3, 3, 14, 14, 0, 0, 12, 12, 12]),
}


def equilibrium_gap(ctrl, v):
    if isinstance(ctrl, IDM):
        return (ctrl.s0 + v * ctrl.T) / np.sqrt(max(1 - (v / ctrl.v0) ** 4, 1e-3))
    h = getattr(ctrl, "h", 1.0)
    return ctrl.r + h * v


def controller_factory(kind: str, **kw):
    """Return a zero-arg factory for a follower controller."""
    kinds = {"ACC": ACC, "CACC": CACC, "IDM": IDM}
    cls = kinds[kind]
    return lambda: cls(**kw)


def build_platoon(n_followers: int, factory, profile="speed_dip", v0=None,
                  params: VehicleParams | list | None = None,
                  v_set_followers: float = 33.0):
    t_pts, v_pts = LEADER_PROFILES[profile] if isinstance(profile, str) else profile
    v0 = v_pts[0] if v0 is None else v0
    vehicles = [Vehicle(0, 0.0, v0, params=_param(params, 0),
                        controller=SpeedProfile(t_pts, v_pts), name="Leader")]
    x = 0.0
    for i in range(1, n_followers + 1):
        ctrl = factory()
        if hasattr(ctrl, "v_set"):
            ctrl.v_set = v_set_followers
        if isinstance(ctrl, IDM):
            ctrl.v0 = v_set_followers
        p = _param(params, i)
        x -= vehicles[-1].p.length + equilibrium_gap(ctrl, v0)
        vehicles.append(Vehicle(i, x, v0, params=p, controller=ctrl, name=f"F{i}"))
    return vehicles


def _param(params, i):
    if params is None:
        return VehicleParams()
    if isinstance(params, (list, tuple)):
        return params[i]
    return params


def run_platoon(n_followers=8, kind="CACC", ctrl_kw=None, profile="speed_dip", T=60.0,
                v2v_delay=0.05, v2v_rate=20.0, v2v_loss=0.0, v2v_enabled=True,
                radar_noise=True, params=None, seed=0, dt=0.01, v2v_outage=None):
    vehicles = build_platoon(n_followers, controller_factory(kind, **(ctrl_kw or {})),
                             profile=profile, params=params)
    rng = np.random.default_rng(seed)
    radar = Radar(sigma_range=0.05 if radar_noise else 0.0,
                  sigma_rate=0.02 if radar_noise else 0.0, rng=rng)
    v2v = V2VChannel(rate_hz=v2v_rate, delay=v2v_delay, loss=v2v_loss,
                     enabled=v2v_enabled, outage=v2v_outage, rng=np.random.default_rng(seed + 1))
    return Simulation(vehicles, dt=dt, radar=radar, v2v=v2v).run(T), vehicles


def acc_test_drive(T=90.0, seed=0, acc_kw=None):
    """Single ACC car on a highway:
    0 s   cruising at the 30 m/s set speed; a truck at 22 m/s is 200 m ahead.
    ~6 s  truck enters radar range (150 m) -> ACC slows and follows it.
    30 s  truck changes lane out -> ACC resumes the set speed.
    50 s  a car cuts in 12 m ahead at 25 m/s, then accelerates to 33 m/s.
    """
    ego = Vehicle(0, 0.0, 30.0, controller=ACC(v_set=30.0, **({"a_comfort_min": -3.5} | (acc_kw or {}))), name="Ego (ACC)")
    truck = Vehicle(1, 200.0, 22.0, params=VehicleParams(length=16.0, tau=0.5),
                    controller=SpeedProfile([0, 200], [22, 22]), name="Truck",
                    active_until=30.0)
    cutin = Vehicle(2, -1e4, 25.0, controller=SpeedProfile([0, 52, 60, 200], [25, 25, 33, 33]),
                    name="Cut-in car", active_from=50.0, spawn_ref=0, spawn_gap=12.0)
    sim = Simulation([ego, truck, cutin], radar=Radar(rng=np.random.default_rng(seed)))
    return sim.run(T)
