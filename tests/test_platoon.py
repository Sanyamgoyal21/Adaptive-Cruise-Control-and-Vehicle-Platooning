"""Unit and validation tests.   Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platoon_sim import ACC, CACC, Radar, Simulation, SpeedProfile, V2VChannel, Vehicle  # noqa: E402
from platoon_sim import analysis as an  # noqa: E402
from platoon_sim.scenarios import acc_test_drive, run_platoon  # noqa: E402
from platoon_sim.vehicle import VehicleParams  # noqa: E402
from platoon_sim.sensors import V2VMessage  # noqa: E402


class TestVehicle(unittest.TestCase):
    def test_first_order_lag(self):
        v = Vehicle(0, 0, 10, VehicleParams(tau=0.3))
        for _ in range(int(0.3 / 0.01)):
            v.step(1.0, 0.01)
        self.assertAlmostEqual(v.a, 1 - np.exp(-1), places=3)    # 63 % after one tau

    def test_saturation_and_no_reverse(self):
        v = Vehicle(0, 0, 1.0)
        for _ in range(500):
            v.step(-100.0, 0.01)
        self.assertEqual(v.u, v.p.a_min)
        self.assertEqual(v.v, 0.0)


class TestSensors(unittest.TestCase):
    def test_radar_range_and_predecessor(self):
        ego, near, far = Vehicle(0, 0, 20), Vehicle(1, 50, 20), Vehicle(2, 400, 20)
        r = Radar(sigma_range=0, sigma_rate=0)
        m = r.measure(ego, [ego, near, far], 0.0)
        self.assertEqual(m.target_id, 1)
        self.assertAlmostEqual(m.gap, 50 - near.p.length)
        self.assertIsNone(r.measure(near, [ego, near, far], 0.0))   # 400 m is out of range

    def test_v2v_delay_and_loss(self):
        a, b = Vehicle(0, 50, 20), Vehicle(1, 0, 20, controller=CACC())
        ch = V2VChannel(rate_hz=10, delay=0.1, loss=0.0)
        ch.transmit(0.0, [a, b])
        ch.deliver(0.05)
        self.assertNotIn(1, ch.inbox)
        ch.deliver(0.1)
        self.assertEqual(ch.inbox[1].sender, 0)
        lossy = V2VChannel(loss=1.0)
        for k in range(20):
            lossy.transmit(k * 0.1, [a, b])
            lossy.deliver(k * 0.1)
        self.assertEqual(lossy.inbox, {})


class TestControllers(unittest.TestCase):
    def test_cacc_falls_back_without_messages(self):
        c = CACC()
        ego = Vehicle(1, 0, 20)
        from platoon_sim.sensors import RadarTarget
        tgt = RadarTarget(0, 20.0, 0.0)
        c.command(0.0, 0.01, ego, tgt, None)
        self.assertEqual(c.mode, "ACC-FALLBACK")
        c.command(0.01, 0.01, ego, tgt, V2VMessage(0, 0.0, 0.0, 0.0, 20.0))
        self.assertIn(c.mode, ("CACC", "CRUISE"))


class TestStringStabilityTheory(unittest.TestCase):
    def test_frequency_domain(self):
        w = an.W_GRID
        self.assertLessEqual(an.peak_gain(an.gamma_acc(w, 0.25, 0.8, 1.6, 0.3)), 1 + 1e-6)
        self.assertGreater(an.peak_gain(an.gamma_acc(w, 0.25, 0.8, 0.6, 0.3)), 1.01)
        self.assertLessEqual(an.peak_gain(an.gamma_cacc(w, 0.2, 0.7, 0.6, 0.3, 0.075)), 1 + 1e-6)
        self.assertGreater(an.peak_gain(an.gamma_cacc(w, 0.2, 0.7, 0.6, 0.3, 0.3)), 1.01)

    def test_simulator_matches_transfer_function(self):
        """Linear regime: follower accel from the simulator == lsim(Gamma, leader accel)."""
        kp, kd, h, tau, dt = 0.25, 0.8, 1.6, 0.3, 0.005
        lead = Vehicle(0, 0.0, 20.0, VehicleParams(tau=tau),
                       controller=SpeedProfile([0, 5, 7, 12, 14, 60], [20, 20, 18, 18, 20, 20]))
        ctrl = ACC(v_set=40.0, h=h, kp=kp, kd=kd, ttc_aeb=None)
        fol = Vehicle(1, -(lead.p.length + ctrl.r + h * 20.0), 20.0, VehicleParams(tau=tau),
                      controller=ctrl)
        res = Simulation([lead, fol], dt=dt, radar=Radar(sigma_range=0, sigma_rate=0),
                         v2v=V2VChannel(enabled=False)).run(40.0)
        tf = signal.TransferFunction([kd, kp], [tau, 1.0, kd + kp * h, kp])
        _, a_pred, _ = signal.lsim(tf, res.a[0], res.t)
        err = np.max(np.abs(a_pred - res.a[1]))
        self.assertLess(err, 0.02 * np.max(np.abs(res.a[1])) + 1e-3)


class TestScenarios(unittest.TestCase):
    def test_cacc_platoon_string_stable(self):
        r, _ = run_platoon(6, "CACC", {}, profile="speed_dip", T=45)
        _, peaks = an.string_amplification(r, "a")
        self.assertEqual(r.collisions, [])
        self.assertLess(peaks[-1], peaks[0])

    def test_acc_short_headway_string_unstable(self):
        r, _ = run_platoon(6, "ACC", {"h": 0.6}, profile="speed_dip", T=45)
        _, peaks = an.string_amplification(r, "a")
        self.assertGreater(peaks[-1], peaks[0])

    def test_acc_test_drive_safe_and_comfortable(self):
        r = acc_test_drive()
        self.assertEqual(r.collisions, [])
        self.assertLessEqual(-r.a[0].min(), 3.5 + 1e-6)
        self.assertGreater(np.nanmin(r.gap[0]), 5.0)


class TestRealtimeApp(unittest.TestCase):
    """Headless smoke test of the interactive simulator (no window)."""

    def test_live_app_actions(self):
        import matplotlib
        matplotlib.use("Agg")
        import realtime as rt
        app = rt.LiveApp(rt.parse_args(["--followers", "6"]), interactive=False)
        run = lambda secs: [app.advance() for _ in range(int(secs / rt.DT))]
        run(3)
        app.set_target_speed(18); run(8)
        app.cut_in(); run(4)
        self.assertTrue(any(not v.connected for v in app.sim.vehicles))
        app.set_kind("ACC"); app.set_headway(1.4); run(4)
        app.set_kind("CACC"); app.set_loss(0.5); run(4)
        app.emergency_brake(); run(12)
        self.assertLess(app.sim.vehicles[0].v, 0.1)
        self.assertEqual(app.sim.collisions, [])
        app.update_artists()                                   # drawing code runs
        app.reset()
        self.assertLess(app.sim.t, 1e-9)


if __name__ == "__main__":
    unittest.main()
