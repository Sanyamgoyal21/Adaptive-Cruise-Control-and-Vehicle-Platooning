"""Real-time interactive ACC / CACC platoon simulator.

    python realtime.py                       # interactive window
    python realtime.py --demo                # auto-plays a scripted scenario
    python realtime.py --controller ACC --headway 0.6 --followers 6
    python realtime.py --demo --snapshot shot.png --snapshot-time 50   # headless test

The simulation runs at wall-clock speed (x0.5 / x1 / x2 / x4) and you drive the
leader live. Every follower runs the same controller code as the offline
experiments (platoon_sim/), so behaviour seen here matches the report.

Controls (panel on the right, or keyboard)
    up / down      leader target speed +-2 m/s
    space          leader emergency brake (-6 m/s^2)
    c              a non-connected car cuts into the middle of the platoon
    p              pause / resume
    r              reset

Rendering uses blitting: the static parts of the window are drawn once and only
the moving artists are redrawn each frame, which keeps it smooth on Tk.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DT = 0.01            # simulation step [s]
HIST = 30.0          # seconds of history in the live plots
MAX_FOLLOWERS = 12
LEAD_WIN = 45.0      # leader window for the live amplification check [s]
SAMPLE = 0.1         # history sampling period [s]
V0 = 25.0            # initial speed [m/s]
LANE_W = 3.5
AHEAD = 30.0         # road shown in front of the leader [m]
SPEEDS = [0.5, 1.0, 2.0, 4.0]
KINDS = ["CACC", "ACC", "Human (IDM)"]
FRAME_MS = 8         # Tk re-arms the timer AFTER each frame, so this is the idle gap between frames

# scripted events for --demo:  (time [s], action, value)
DEMO = [(4, "speed", 15), (16, "speed", 25), (24, "cutin", None), (40, "speed", 30),
        (50, "ebrake", None), (58, "speed", 25), (70, "kind", "ACC"), (72, "speed", 18),
        (78, "speed", 25), (92, "kind", "CACC"), (100, "loss", 0.9),
        (104, "speed", 15), (114, "speed", 25), (124, "loss", 0.0)]


CAR_MIN_IN = 0.4     # minimum on-screen car length [inch]
CAR_PART_COLORS = {"wheel": "#1f1f1f", "glass": "#d9e6ee"}


def car_shape(x_front, y, L, W):
    """Top-down car pointing +x as (part, polygon) pairs, in draw order.

    Wheels first (under the body), then the body with a tapered nose and
    rounded tail, then windshield and rear window."""
    xb, xf, h = x_front - L, x_front, W / 2
    parts = []
    for x0 in (xb + 0.14 * L, xf - 0.32 * L):                   # 4 wheels
        for s in (-1, 1):
            y_in, y_out = y + s * 0.80 * h, y + s * 1.10 * h
            parts.append(("wheel", [(x0, y_in), (x0 + 0.17 * L, y_in),
                                    (x0 + 0.17 * L, y_out), (x0, y_out)]))
    parts.append(("body", [
        (xb, y - 0.80 * h), (xb + 0.05 * L, y - h), (xf - 0.22 * L, y - h),
        (xf - 0.06 * L, y - 0.80 * h), (xf, y - 0.40 * h), (xf, y + 0.40 * h),
        (xf - 0.06 * L, y + 0.80 * h), (xf - 0.22 * L, y + h), (xb + 0.05 * L, y + h),
        (xb, y + 0.80 * h)]))
    parts.append(("glass", [(xf - 0.40 * L, y - 0.70 * h), (xf - 0.27 * L, y - 0.80 * h),
                            (xf - 0.27 * L, y + 0.80 * h), (xf - 0.40 * L, y + 0.70 * h)]))
    parts.append(("glass", [(xb + 0.10 * L, y - 0.72 * h), (xb + 0.19 * L, y - 0.66 * h),
                            (xb + 0.19 * L, y + 0.66 * h), (xb + 0.10 * L, y + 0.72 * h)]))
    return parts


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--followers", type=int, default=8, choices=range(1, 13), metavar="1-12")
    ap.add_argument("--controller", choices=["CACC", "ACC", "IDM"], default="CACC")
    ap.add_argument("--headway", type=float, default=0.6)
    ap.add_argument("--speed", type=float, default=1.0, choices=SPEEDS, help="real-time factor")
    ap.add_argument("--demo", action="store_true", help="auto-play a scripted scenario")
    ap.add_argument("--snapshot", help="run headless and save a PNG instead of opening a window")
    ap.add_argument("--snapshot-time", type=float, default=40.0)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    else:
        for k in ("keymap.pan", "keymap.home", "keymap.back", "keymap.forward",
                  "keymap.fullscreen", "keymap.save", "keymap.zoom", "keymap.grid"):
            matplotlib.rcParams[k] = []
    app = LiveApp(args, interactive=not args.snapshot)
    if args.snapshot:
        app.run_headless(args.snapshot_time, args.snapshot)
    else:
        app.show()


class LiveApp:
    def __init__(self, args, interactive=True):
        from experiments import style
        from platoon_sim.controllers import MODES
        self.style, self.MODES, self.plt = style, MODES, style.plt
        self.interactive = interactive          # blitting only in a real window
        self.args = args
        self.n = args.followers
        self.kind = {"IDM": "Human (IDM)"}.get(args.controller, args.controller)
        self.h = args.headway
        self.loss, self.latency, self.v2v_on = 0.0, 0.05, True
        self.speed_factor = args.speed
        self.paused = False
        self.demo = list(DEMO) if args.demo else []
        self.message, self.message_until = "", 0.0
        self._quiet = True                      # suppress slider callbacks while syncing
        self.bg = None
        self.fps = 0.0
        self.build_world()
        self.build_figure()
        self._quiet = False

    # ================================================================ world
    def make_ctrl(self, kind, veh=None):
        from platoon_sim.controllers import ACC, CACC, IDM
        if kind == "CACC":
            c = CACC(v_set=33.0, h=self.h, h_fallback=max(1.6, self.h))
            if veh is not None:
                c.uf, c.h_cur = veh.a, self.h
        elif kind == "ACC":
            c = ACC(v_set=33.0, h=self.h)
        else:
            c = IDM(v0=33.0, T=self.h)
        return c

    def build_world(self):
        from platoon_sim import ManualDriver, Radar, Simulation, V2VChannel, Vehicle
        from platoon_sim.scenarios import equilibrium_gap
        self.leader_ctrl = ManualDriver(v_target=V0)
        vehs = [Vehicle(0, 0.0, V0, controller=self.leader_ctrl, name="L")]
        x = 0.0
        for i in range(1, self.n + 1):
            c = self.make_ctrl(self.kind)
            x -= vehs[-1].p.length + equilibrium_gap(c, V0)
            vehs.append(Vehicle(i, x, V0, controller=c, name=f"F{i}"))
        self.v2v = V2VChannel(rate_hz=20, delay=self.latency, loss=self.loss,
                              enabled=self.v2v_on, rng=np.random.default_rng(1))
        self.sim = Simulation(vehs, dt=DT, radar=Radar(rng=np.random.default_rng(0)), v2v=self.v2v)
        self.sim.reset_state()
        self.hist = deque(maxlen=int(HIST / SAMPLE) + 1)
        # longer buffer for the live amplification check: the leader's peak
        # must still be in the window when the tail of the platoon reacts
        self.metric = deque(maxlen=int(LEAD_WIN / SAMPLE) + 1)
        self.next_sample = 0.0
        self.next_id = self.n + 1
        self.span = None
        self.carry = 0.0
        self.last_wall = None
        self.rtf = 1.0
        self.snap = None

    # ---------------------------------------------------------------- actions
    def set_target_speed(self, v):
        self.leader_ctrl.braking = False
        self.leader_ctrl.v_target = float(np.clip(v, 0, 35))

    def emergency_brake(self):
        self.leader_ctrl.emergency()
        self._sync_slider("speed", 0)
        self.notify("Leader: EMERGENCY BRAKE")

    def set_kind(self, kind):
        self.kind = kind
        for veh in self.sim.vehicles[1:self.n + 1]:
            veh.controller = self.make_ctrl(kind, veh)
        self.notify(f"Followers switched to {kind}")

    def set_headway(self, h):
        self.h = float(h)
        for veh in self.sim.vehicles[1:self.n + 1]:
            c = veh.controller
            if hasattr(c, "h_cacc"):
                c.h_cacc, c.h_fallback = self.h, max(1.6, self.h)
            elif hasattr(c, "T"):
                c.T = self.h
            else:
                c.h = self.h

    def set_loss(self, p):
        self.loss = self.v2v.loss = float(p)

    def set_latency(self, d):
        self.latency = self.v2v.delay = float(d)

    def set_v2v(self, on):
        self.v2v_on = self.v2v.enabled = bool(on)
        self.notify("V2V radio " + ("ON" if on else "OFF -> followers fall back to ACC"))

    def cut_in(self):
        """A non-connected car changes into the middle of the platoon."""
        from platoon_sim import ACC, Vehicle
        if any(not v.connected for v in self.sim.vehicles):
            self.notify("A cut-in is already in progress")
            return
        j = max(1, (self.n + 1) // 2)
        tgt, pred = self.sim.vehicles[j], self.sim.vehicles[j - 1]
        gap = pred.x - pred.p.length - tgt.x
        if gap < 9.0:
            self.notify(f"Gap in front of {tgt.name} too small for a cut-in")
            return
        t = self.sim.t
        car = Vehicle(self.next_id, 0.0, tgt.v, controller=ACC(v_set=33.0, h=0.8),
                      name="Cut-in", active_from=t + 1.5, active_until=t + 16.5, connected=False)
        car.x = tgt.x + 0.4 * gap + car.p.length
        self.next_id += 1
        self.sim.vehicles.append(car)
        self.notify(f"Non-connected car cuts in ahead of {tgt.name} ({0.4 * gap:.1f} m gap)")

    def toggle_pause(self):
        self.paused = not self.paused
        self.last_wall = None
        if "pause" in getattr(self, "ui", {}):
            self.ui["pause"].label.set_text("Resume" if self.paused else "Pause")
            self.full_redraw()

    def cycle_speed(self):
        i = (SPEEDS.index(self.speed_factor) + 1) % len(SPEEDS) if self.speed_factor in SPEEDS else 1
        self.speed_factor = SPEEDS[i]
        self.ui["simspeed"].label.set_text(f"Sim speed x{self.speed_factor:g}")
        self.full_redraw()

    def reset(self):
        self.build_world()
        self._sync_slider("speed", V0)
        self.notify("Reset")
        self.full_redraw()

    def notify(self, msg, secs=4.0):
        self.message, self.message_until = msg, self.sim.t + secs

    def _sync_slider(self, key, val):
        ui = getattr(self, "ui", {})
        if key in ui:
            self._quiet = True
            ui[key].set_val(val)
            self._quiet = False

    # ---------------------------------------------------------------- stepping
    def advance(self):
        sim = self.sim
        snap = sim.step()
        t = sim.t
        while self.demo and t >= self.demo[0][0]:
            self.apply_demo(*self.demo.pop(0)[1:])
        # remove cut-in cars a few seconds after they leave the lane
        sim.vehicles[:] = [v for v in sim.vehicles if t < v.active_until + 3.0]
        if t >= self.next_sample:
            m = self.n + 1
            self.hist.append((t, snap["v"][:m].copy(), snap["gap"][:m].copy(),
                              snap["a"][:m].copy()))
            self.metric.append((t, np.abs(snap["a"][:m])))
            self.next_sample += SAMPLE
        self.snap = snap

    def apply_demo(self, action, val):
        ui = getattr(self, "ui", {})
        if action == "speed":
            self.set_target_speed(val)
            self._sync_slider("speed", val)
        elif action == "ebrake":
            self.emergency_brake()
        elif action == "cutin":
            self.cut_in()
        elif action == "kind":
            if "kind" in ui:
                ui["kind"].set_active(KINDS.index(val))     # fires set_kind
            else:
                self.set_kind(val)
        elif action == "headway":
            self.set_headway(val)
            self._sync_slider("h", val)
        elif action == "loss":
            self.set_loss(val)
            self._sync_slider("loss", val * 100)
            self.notify(f"Demo: V2V packet loss set to {val:.0%}")

    def tick(self):
        now = time.perf_counter()
        if self.last_wall is None:
            self.last_wall = now
        wall = min(now - self.last_wall, 0.25)
        self.last_wall = now
        if wall > 0:
            self.fps = 0.9 * self.fps + 0.1 / wall
        if not self.paused and wall > 0:
            want = wall * self.speed_factor + self.carry
            steps = int(want / DT)
            self.carry = want - steps * DT
            for _ in range(steps):
                self.advance()
            self.rtf = 0.9 * self.rtf + 0.1 * (steps * DT / wall)
        self.update_artists()
        self.blit()

    # ================================================================ figure
    def build_figure(self):
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Patch
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
        st, plt = self.style, self.plt
        fig = self.fig = plt.figure(figsize=(15.5, 8.8))
        try:
            fig.canvas.manager.set_window_title("Real-time ACC / CACC platoon simulator")
        except Exception:
            pass
        fig.text(0.075, 0.965, "Real-time ACC / CACC platoon simulator", fontsize=14, color=st.TEXT)
        self.animated = []

        def anim(artist):
            artist.set_animated(self.interactive)
            self.animated.append(artist)
            return artist

        # --- road view (coordinates relative to the leader -> fixed limits)
        ax = self.ax_road = fig.add_axes([0.075, 0.70, 0.625, 0.17])
        ax.set_ylim(-LANE_W / 2 - 0.4, LANE_W * 1.5 + 0.4)
        ax.set_yticks([0, LANE_W], ["platoon lane", "adjacent lane"])
        ax.grid(False)
        ax.set_facecolor("#efeee9")
        for y in (-LANE_W / 2, LANE_W * 1.5):
            ax.axhline(y, color=st.TEXT2, lw=1.5)
        ax.set_xlabel("position relative to the leader [m]   (car symbols enlarged up to 2.5x when zoomed out)",
                      labelpad=1)
        self.lane_dashes = anim(LineCollection([], colors=st.MUTED, lw=1.2))
        self.posts = anim(LineCollection([], colors=st.TEXT2, lw=1.0))
        self.v2v_links = anim(LineCollection([], colors=st.MODE_COLORS["CACC"], lw=1.4, linestyles=":"))
        for lc in (self.lane_dashes, self.posts, self.v2v_links):
            ax.add_collection(lc)
        legend = [Patch(color=st.LEADER, label="Leader (you)")] + \
                 [Patch(color=st.MODE_COLORS[m], label=lab) for m, lab in
                  [("CACC", "CACC"), ("FOLLOW", "ACC following"), ("CRUISE", "cruise"),
                   ("ACC-FALLBACK", "CACC->ACC fallback"), ("AEB", "AEB"), ("HUMAN", "human (IDM)")]] + \
                 [Patch(color=st.TEXT2, label="non-connected car")]
        ax.legend(handles=legend, ncol=8, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8,
                  handlelength=1.2, columnspacing=1.0)
        from matplotlib.collections import PolyCollection
        self.car_bodies = anim(PolyCollection([], zorder=3))
        ax.add_collection(self.car_bodies)
        self.lbl_lead = anim(ax.text(0, 0, "Leader", ha="center", va="bottom", fontsize=7.5,
                                     color=st.TEXT, clip_on=True))
        self.lbl_cut = anim(ax.text(0, 0, "cut-in", ha="center", va="bottom", fontsize=7.5,
                                    color=st.TEXT2, clip_on=True))

        # --- live time series (x = seconds before now -> fixed limits)
        self.ax_v = fig.add_axes([0.075, 0.47, 0.625, 0.15])
        self.ax_g = fig.add_axes([0.075, 0.27, 0.625, 0.15], sharex=self.ax_v)
        self.ax_a = fig.add_axes([0.075, 0.07, 0.625, 0.15], sharex=self.ax_v)
        self.ax_v.set_ylabel("speed [m/s]")
        self.ax_g.set_ylabel("gap [m]")
        self.ax_a.set_ylabel("accel [m/s²]")
        self.ax_a.set_xlabel("time relative to now [s]")
        self.ax_v.set_xlim(-HIST, 0)
        self.ax_v.set_ylim(0, 37)
        self.ax_g.set_ylim(-2, 30)
        self.ax_a.set_ylim(-6.8, 3.2)
        for a in (self.ax_v, self.ax_g):
            a.tick_params(labelbottom=False)
        self.lines = {"v": [], "g": [], "a": []}
        self.make_lines()
        self.ax_g.axhline(0, color=st.CRITICAL, lw=1)

        # --- status panel (inside an invisible axes so it can be blitted)
        px = 0.755
        fig.text(px, 0.905, "STATUS", fontsize=9, color=st.TEXT2, weight="bold")
        self.ax_status = fig.add_axes([px, 0.675, 0.24, 0.22])
        self.ax_status.set_axis_off()
        self.status = anim(self.ax_status.text(0, 1, "", fontsize=8.6, family="monospace", va="top",
                                               color=st.TEXT, linespacing=1.45, clip_on=True,
                                               transform=self.ax_status.transAxes))
        fig.text(px, 0.655, "CONTROLS", fontsize=9, color=st.TEXT2, weight="bold")
        ui = self.ui = {}

        def slider(key, y, title, lo, hi, val, step, fmt, cb):
            a = fig.add_axes([px, y, 0.16, 0.022])
            a.set_title(title, fontsize=8.5, loc="left", color=st.TEXT2, pad=3)
            s = Slider(a, "", lo, hi, valinit=val, valstep=step, valfmt=fmt,
                       color=st.CAT[0], initcolor="none")
            s.valtext.set_fontsize(8.5)
            s.on_changed(lambda v: None if self._quiet else cb(v))
            ui[key] = s

        slider("speed", 0.610, "Leader target speed [m/s]", 0, 35, V0, 1, "%d", self.set_target_speed)
        slider("n", 0.555, "Number of followers (rebuilds platoon)", 1, MAX_FOLLOWERS, self.n, 1, "%d",
               self.set_followers)
        slider("h", 0.500, "Time gap h (IDM: T) [s]", 0.3, 2.0, self.h, 0.05, "%.2f", self.set_headway)
        slider("loss", 0.445, "V2V packet loss [%]", 0, 100, 0, 5, "%d", lambda v: self.set_loss(v / 100))
        slider("lat", 0.390, "V2V latency [ms]", 0, 500, self.latency * 1000, 10, "%d",
               lambda v: self.set_latency(v / 1000))

        a = fig.add_axes([px, 0.245, 0.11, 0.10])
        a.set_title("Follower controller", fontsize=8.5, loc="left", color=st.TEXT2, pad=3)
        ui["kind"] = RadioButtons(a, KINDS, active=KINDS.index(self.kind), activecolor=st.CAT[0])
        for lbl in ui["kind"].labels:
            lbl.set_fontsize(8.5)
        ui["kind"].on_clicked(self.set_kind)
        b = fig.add_axes([px + 0.115, 0.28, 0.10, 0.05])
        ui["v2v"] = CheckButtons(b, ["V2V radio"], [True])
        ui["v2v"].labels[0].set_fontsize(8.5)
        ui["v2v"].on_clicked(lambda _l: self.set_v2v(ui["v2v"].get_status()[0]))
        for a_ in (a, b):
            a_.set_facecolor(st.SURFACE)
            for sp in a_.spines.values():
                sp.set_visible(False)

        def button(key, rect, label, cb, color="#ecebe6"):
            a = fig.add_axes(rect)
            btn = Button(a, label, color=color, hovercolor="#dcdad3")
            btn.label.set_fontsize(8.8)
            btn.on_clicked(lambda _e: cb())
            ui[key] = btn

        bw = 0.105
        button("ebrake", [px, 0.180, bw, 0.045], "Emergency brake", self.emergency_brake, "#f6d5d5")
        button("cutin", [px + bw + 0.01, 0.180, bw, 0.045], "Cut-in car", self.cut_in)
        button("pause", [px, 0.127, bw, 0.045], "Pause", self.toggle_pause)
        button("simspeed", [px + bw + 0.01, 0.127, bw, 0.045], f"Sim speed x{self.speed_factor:g}",
               self.cycle_speed)
        button("reset", [px, 0.074, bw, 0.045], "Reset", self.reset)
        fig.text(px, 0.045, "Keys:  up/down leader speed    space emergency brake\n"
                            "       c cut-in    p pause    r reset",
                 fontsize=8, color=st.TEXT2, family="monospace", va="top")
        if self.interactive:
            fig.canvas.mpl_connect("key_press_event", self.on_key)
            fig.canvas.mpl_connect("draw_event", self.on_draw)

    def make_lines(self):
        """(Re)create one line per vehicle in the three live plots + legend."""
        from matplotlib.colors import LinearSegmentedColormap
        st = self.style
        for ln in [x for k in self.lines for x in self.lines[k]]:
            self.animated.remove(ln)
            ln.remove()
        self.lines = {"v": [], "g": [], "a": []}
        ramp = LinearSegmentedColormap.from_list("ramp", st.RAMP)
        for i in range(self.n + 1):
            c = st.LEADER if i == 0 else ramp((i - 1) / max(self.n - 1, 1))
            ls, lw = ("--", 1.8) if i == 0 else ("-", 1.3)
            lab = "Leader" if i == 0 else f"F{i}"
            new = {"v": self.ax_v.plot([], [], color=c, ls=ls, lw=lw, label=lab)[0],
                   "a": self.ax_a.plot([], [], color=c, ls=ls, lw=lw)[0]}
            if i:
                new["g"] = self.ax_g.plot([], [], color=c, lw=lw)[0]
            for k, ln in new.items():
                ln.set_animated(self.interactive)
                self.animated.append(ln)
                self.lines[k].append(ln)
        self.ax_v.legend(loc="lower left", ncol=self.n + 1, fontsize=7.5 if self.n <= 8 else 6.5,
                         handlelength=1.2, columnspacing=0.7)

    def set_followers(self, n):
        """Rebuild the platoon with ``n`` followers (restarts the drive)."""
        n = int(n)
        if n == self.n:
            return
        self.n = n
        self.build_world()
        self.make_lines()
        self._sync_slider("speed", V0)
        self.notify(f"Platoon rebuilt with {n} followers")
        self.full_redraw()

    def on_key(self, ev):
        k = ev.key
        if k in ("up", "down"):
            v = self.leader_ctrl.v_target + (2 if k == "up" else -2)
            self.set_target_speed(v)
            self._sync_slider("speed", self.leader_ctrl.v_target)
        elif k == " ":
            self.emergency_brake()
        elif k == "c":
            self.cut_in()
        elif k == "p":
            self.toggle_pause()
        elif k == "r":
            self.reset()

    # ---------------------------------------------------------------- blitting
    # Each region has its own cached background and refresh period, so most
    # frames only redraw and copy the small road strip to the screen.
    REGION_PERIOD = {"road": 0.0, "plots": 0.1, "status": 0.25}   # seconds (wall clock)

    def regions(self):
        return {"road": [self.ax_road], "plots": [self.ax_v, self.ax_g, self.ax_a],
                "status": [self.ax_status]}

    def on_draw(self, _event):
        """After every full redraw: cache each region's static background."""
        c = self.fig.canvas
        self.bg = {ax: c.copy_from_bbox(ax.bbox) for axs in self.regions().values() for ax in axs}
        self._last_region = {}
        for axs in self.regions().values():
            for a in self.animated:
                if a.axes in axs:
                    self.fig.draw_artist(a)

    def full_redraw(self):
        self.bg = None
        if self.interactive:
            self.fig.canvas.draw_idle()

    def blit(self):
        if not self.interactive:
            return
        c = self.fig.canvas
        if self.bg is None:
            c.draw_idle()
            return
        from matplotlib.transforms import Bbox
        now = time.perf_counter()
        dirty = {"left": [], "right": []}          # blit each screen column separately
        for name, axs in self.regions().items():
            if now - self._last_region.get(name, 0.0) < self.REGION_PERIOD[name]:
                continue
            self._last_region[name] = now
            for ax in axs:
                c.restore_region(self.bg[ax])
            for a in self.animated:
                if a.axes in axs:
                    self.fig.draw_artist(a)
            dirty["right" if name == "status" else "left"] += [ax.bbox for ax in axs]
        for boxes in dirty.values():   # union per column: few, tight screen copies
            if boxes:
                c.blit(Bbox.union(boxes))

    # ---------------------------------------------------------------- drawing
    @staticmethod
    def lane_y(veh, t):
        """Lateral position: animate lane changes over 1.5 s."""
        if t < veh.active_from:
            return LANE_W * float(np.clip((veh.active_from - t) / 1.5, 0, 1))
        if t >= veh.active_until:
            return LANE_W * float(np.clip((t - veh.active_until) / 1.5, 0, 1))
        return 0.0

    def update_artists(self):
        from platoon_sim.sensors import true_predecessor
        st, sim, t = self.style, self.sim, self.sim.t
        vehs = sim.vehicles
        lead = vehs[0]
        x0 = lead.x
        ax = self.ax_road

        # zoom only when the platoon length changes a lot (each change = full redraw)
        need = x0 - min(v.x - v.p.length for v in vehs[:self.n + 1]) + 12
        if self.span is None or need > self.span or need < 0.55 * self.span:
            self.span = max(60.0, 1.2 * need)
            ax.set_xlim(-self.span, AHEAD)
            self.full_redraw()

        # moving road markings
        lo = x0 - self.span
        xs = np.arange(np.floor(lo / 12) * 12, x0 + AHEAD, 12) - x0
        self.lane_dashes.set_segments([[(x, LANE_W / 2), (x + 5, LANE_W / 2)] for x in xs])
        ps = np.arange(np.floor(lo / 50) * 50, x0 + AHEAD, 50) - x0
        self.posts.set_segments([[(x, -LANE_W / 2 - 0.4), (x, -LANE_W / 2)] for x in ps])

        # cars: one PolyCollection for all bodies (much cheaper than patches);
        # text labels only for the leader and a cut-in car
        modes = {} if self.snap is None else {
            v.id: self.snap["mode"][i] for i, v in enumerate(vehs) if i < len(self.snap["mode"])}
        verts, faces, edges, widths, links = [], [], [], [], []
        # Car symbols: at least CAR_MIN_IN inches long on screen (up to 2.5x the
        # true length when zoomed out), front anchored at the true front bumper.
        # The road axes stretch y more than x, so the drawn width (in metres) is
        # chosen to give a real car's ~2.2:1 shape on screen.
        sx = ax.bbox.width / (AHEAD + self.span)          # px per metre along the road
        sy = ax.bbox.height / np.ptp(ax.get_ylim())       # px per metre across the road
        scale = float(np.clip(CAR_MIN_IN * self.fig.dpi / (4.5 * sx), 1.0, 2.5))
        car_w = float(np.clip(4.5 * scale * sx / 2.2 / sy, 0.5, 2.6))
        self.lbl_lead.set_position((-lead.p.length * scale / 2, car_w / 2 + 0.15))
        self.lbl_cut.set_visible(False)
        for v in vehs:
            y = self.lane_y(v, t)
            xr = v.x - x0
            if v.id == 0:
                color = st.LEADER
            elif not v.connected:
                color = st.TEXT2
                self.lbl_cut.set_position((xr - v.p.length * scale / 2, y + car_w / 2 + 0.15))
                self.lbl_cut.set_visible(True)
            else:
                color = st.MODE_COLORS[self.MODES[modes.get(v.id, 1)]]
            crashed = v.id in sim.crashed
            for part, poly in car_shape(xr, y, v.p.length * scale, car_w):
                verts.append(poly)
                if part == "body":
                    faces.append(color)
                    edges.append(st.CRITICAL if crashed else color)
                    widths.append(2.5 if crashed else 0.0)
                else:
                    faces.append(CAR_PART_COLORS[part])
                    edges.append(CAR_PART_COLORS[part])
                    widths.append(0.0)
            if modes.get(v.id) == self.MODES.index("CACC"):
                pred, _ = true_predecessor(v, vehs, t)
                if pred is not None and pred.connected:
                    links.append([(xr - v.p.length / 2, 1.15), (pred.x - x0 - pred.p.length / 2, 1.15)])
        self.car_bodies.set_verts(verts)
        self.car_bodies.set_facecolors(faces)
        self.car_bodies.set_edgecolors(edges)
        self.car_bodies.set_linewidths(widths)
        self.v2v_links.set_segments(links)

        # time series
        if self.hist:
            T = np.array([h[0] for h in self.hist]) - t
            V = np.array([h[1] for h in self.hist])
            G = np.array([h[2] for h in self.hist])
            A = np.array([h[3] for h in self.hist])
            for i in range(self.n + 1):
                self.lines["v"][i].set_data(T, V[:, i])
                self.lines["a"][i].set_data(T, A[:, i])
                if i:
                    self.lines["g"][i - 1].set_data(T, G[:, i])
            gmax = np.nanmax(G[:, 1:]) if np.isfinite(G[:, 1:]).any() else 30
            top = self.ax_g.get_ylim()[1]
            want = max(30.0, np.ceil(gmax * 1.15 / 10) * 10)
            if want > top or want < 0.5 * top:
                self.ax_g.set_ylim(-2, want)
                self.full_redraw()
        self.update_status(t)

    def update_status(self, t):
        snap, MODES = self.snap, self.MODES
        lead = self.sim.vehicles[0]
        lines = [f"t = {t:6.1f} s   sim x{self.speed_factor:g} (actual x{self.rtf:.2f})"
                 + (f"  {self.fps:4.0f} fps" if self.interactive else "")
                 + ("   PAUSED" if self.paused else "")]
        lines.append(f"Leader   {lead.v:5.1f} m/s  -> target {self.leader_ctrl.v_target:4.0f}"
                     + ("   E-BRAKE" if self.leader_ctrl.braking else ""))
        lines.append(f"Control  {self.kind}, h = {self.h:.2f} s")
        lines.append(f"V2V      {'on ' if self.v2v_on else 'OFF'}  {self.latency * 1000:.0f} ms, "
                     f"{self.loss:.0%} loss")
        if snap is not None:
            m = self.n + 1
            gaps = snap["gap"][1:m]
            modes = [MODES[k] for k in snap["mode"][1:m]]
            counts = {k: modes.count(k) for k in dict.fromkeys(modes)}
            lines.append(f"Min gap  {np.nanmin(gaps):5.1f} m" if np.isfinite(gaps).any() else "Min gap   -")
            lines.append("Modes    " + "  ".join(f"{k}x{c}" for k, c in counts.items()))
        if len(self.metric) > 20:
            T = np.array([m[0] for m in self.metric])
            A = np.array([m[1] for m in self.metric])
            lead_pk = A[:, 0].max()                         # leader over the last 45 s
            tail_pk = A[T >= t - HIST, -1].max()            # last car over the last 30 s
            if lead_pk > 0.4:
                r = tail_pk / lead_pk
                verdict = "damped" if r <= 1.02 else "AMPLIFIED"
                lines.append(f"F{self.n}/leader peak accel: x{r:.2f} {verdict}")
            else:
                lines.append("Disturbance: change the leader speed")
        col = len(self.sim.collisions)
        lines.append(f"Collisions {col}" + ("  <- press Reset" if col else ""))
        if self.message and t < self.message_until:
            lines.append(">> " + self.message)
        self.status.set_text("\n".join(lines))
        self.status.set_color(self.style.CRITICAL if col else self.style.TEXT)

    # ---------------------------------------------------------------- run modes
    def show(self):
        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.plt.show()

    def run_headless(self, T, path):
        t0 = time.perf_counter()
        while self.sim.t < T:
            self.advance()
        wall = time.perf_counter() - t0
        self.rtf = T / wall
        self.update_artists()
        self.fig.savefig(path, dpi=100, facecolor=self.style.SURFACE)
        print(f"simulated {T:.0f} s in {wall:.1f} s ({T / wall:.1f}x faster than real time) -> {path}")


if __name__ == "__main__":
    main()
