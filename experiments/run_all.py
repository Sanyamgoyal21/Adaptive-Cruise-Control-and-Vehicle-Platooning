"""Run every experiment and write figures + metrics to ``results/``.

    python experiments/run_all.py            # everything (~3-4 min)
    python experiments/run_all.py --quick    # shorter sweeps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")   # file output only, no window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from platoon_sim import analysis as an                         # noqa: E402
from platoon_sim.controllers import MODES                      # noqa: E402
from platoon_sim.scenarios import (acc_test_drive, run_platoon,  # noqa: E402
                                   equilibrium_gap, LEADER_PROFILES)
from platoon_sim.controllers import ACC, CACC, IDM             # noqa: E402
from platoon_sim.vehicle import VehicleParams                  # noqa: E402
from experiments import style                                  # noqa: E402

OUT = os.path.join(ROOT, "results")
N_FOL = 8

# The four follower strategies compared throughout (name, kind, kwargs)
STRATEGIES = [
    ("Human (IDM)", "IDM", {}),
    ("ACC  h=0.6 s", "ACC", {"h": 0.6}),
    ("ACC  h=1.6 s", "ACC", {"h": 1.6}),
    ("CACC h=0.6 s", "CACC", {"h": 0.6}),
]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=style.SURFACE)
    style.plt.close(fig)
    print("  wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- 1. single ACC car
def exp_acc_test_drive():
    print("[1] ACC test drive (cruise -> follow -> cut-out -> cut-in)")
    r = acc_test_drive()
    plt = style.plt
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 2.2, 0.7]})
    c_ego, c_truck, c_cut = style.CAT[0], style.CAT[1], style.CAT[2]
    lane = [np.ones_like(r.t, bool), r.t < 30.0, r.t >= 50.0]   # in ego's lane?
    for i, c in zip(range(3), [c_ego, c_truck, c_cut]):
        ax[0].plot(r.t, np.where(lane[i], r.v[i], np.nan), color=c, lw=2, label=r.names[i])
    ax[0].axhline(30, color=style.MUTED, lw=1, ls=":")
    ax[0].text(r.t[-1], 30.3, "set speed", ha="right", va="bottom", color=style.TEXT2, fontsize=9)
    ax[0].set_ylabel("speed [m/s]")
    ax[0].legend(loc="lower right", ncol=1)

    ax[1].plot(r.t, r.gap[0], color=c_ego, lw=2, label="measured gap")
    ax[1].plot(r.t, r.gap_ref[0], color=style.TEXT2, lw=1.5, ls="--", label="desired gap r + h·v")
    ax[1].set_ylabel("gap to target [m]")
    ax[1].set_ylim(0, 160)
    ax[1].legend(loc="upper right")

    ax[2].plot(r.t, r.a[0], color=c_ego, lw=2, label="actual")
    ax[2].plot(r.t, r.u[0], color=style.TEXT2, lw=1, ls="--", label="commanded")
    ax[2].set_ylabel("accel [m/s²]")
    ax[2].legend(loc="lower right", ncol=2)

    style.mode_strip(ax[3], r.t, r.mode[0])
    ax[3].set_xlabel("time [s]")
    for x, txt in [(6.3, "truck in radar range"), (30, "truck changes lane"), (50, "car cuts in")]:
        for a in ax[:3]:
            a.axvline(x, color=style.MUTED, lw=0.8, ls=":")
        ax[0].text(x + 0.5, 1.02, txt, color=style.TEXT2, fontsize=8.5,
                   transform=ax[0].get_xaxis_transform(), va="bottom")
    fig.suptitle("Single-vehicle ACC: speed control, gap control and cut-in handling",
                 x=0.06, ha="left", color=style.TEXT, fontsize=12)
    save(fig, "fig1_acc_test_drive.png")
    return {"collisions": len(r.collisions),
            **{k: v for k, v in an.vehicle_metrics(r, 0).items()
               if k in ("min_gap_m", "max_decel", "rms_jerk", "min_speed")},
            "aeb_used": bool((r.mode[0] == MODES.index("AEB")).any())}


# ---------------------------------------------------------------- 2. frequency domain
def exp_frequency_domain():
    print("[2] Frequency-domain string stability")
    plt = style.plt
    w = an.W_GRID
    tau, theta = 0.3, 0.075          # theta: 50 ms latency + 25 ms mean ZOH at 20 Hz
    curves = [
        ("ACC  h=0.6 s", an.gamma_acc(w, 0.25, 0.8, 0.6, tau)),
        ("ACC  h=1.6 s", an.gamma_acc(w, 0.25, 0.8, 1.6, tau)),
        ("CACC h=0.6 s", an.gamma_cacc(w, 0.2, 0.7, 0.6, tau, theta)),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for (lab, g), c in zip(curves, [style.CAT[1], style.CAT[2], style.CAT[3]]):
        ax[0].semilogx(w, 20 * np.log10(np.abs(g)), color=c, lw=2, label=f"{lab}  (peak {an.peak_gain(g):.2f})")
    ax[0].axhline(0, color=style.TEXT2, lw=1)
    ax[0].text(1.2e-3, 1.2, "string-stability limit |Γ| = 1", color=style.TEXT2, fontsize=9)
    ax[0].set_xlabel("frequency ω [rad/s]")
    ax[0].set_ylabel("|Γ(jω)| [dB]")
    ax[0].set_ylim(-30, 6)
    ax[0].set_title("Acceleration transfer between consecutive vehicles", loc="left")
    ax[0].legend(loc="lower left")

    thetas = np.linspace(0.0, 0.4, 21)
    hmin_cacc = [an.min_string_stable_headway(lambda h: an.gamma_cacc(w, 0.2, 0.7, h, tau, th))
                 for th in thetas]
    hmin_acc = an.min_string_stable_headway(lambda h: an.gamma_acc(w, 0.25, 0.8, h, tau))
    ax[1].plot(thetas * 1000, hmin_cacc, color=style.CAT[3], lw=2, marker="o", ms=4, label="CACC")
    ax[1].axhline(hmin_acc, color=style.CAT[2], lw=2, label="ACC (no communication)")
    ax[1].axvline(theta * 1000, color=style.MUTED, lw=0.8, ls=":")
    ax[1].text(theta * 1000 + 5, 0.1, "delay used in\nsimulations", color=style.TEXT2, fontsize=8.5)
    ax[1].set_xlabel("effective V2V delay θ [ms]")
    ax[1].set_ylabel("minimum string-stable headway h [s]")
    ax[1].set_ylim(0, 1.4)
    ax[1].set_title("Communication delay erodes the CACC advantage", loc="left")
    ax[1].legend(loc="lower right")

    taus = np.linspace(0.1, 0.8, 15)
    hmin_tau_acc = [an.min_string_stable_headway(lambda h: an.gamma_acc(w, 0.25, 0.8, h, t)) for t in taus]
    hmin_tau_cacc = [an.min_string_stable_headway(lambda h: an.gamma_cacc(w, 0.2, 0.7, h, t, theta)) for t in taus]
    save(fig, "fig2_string_stability_frequency.png")
    return {"theta_s": theta, "tau_s": tau,
            "peak_gain": {lab: round(an.peak_gain(g), 3) for lab, g in curves},
            "h_min_acc_s": round(hmin_acc, 3),
            "h_min_cacc_s_at_theta": round(float(np.interp(theta, thetas, hmin_cacc)), 3),
            "h_min_vs_tau": {"tau": taus.round(3).tolist(),
                             "acc": np.round(hmin_tau_acc, 3).tolist(),
                             "cacc": np.round(hmin_tau_cacc, 3).tolist()}}


# ---------------------------------------------------------------- 3. platoon comparison
def exp_platoon(profile, T, fig_name, title):
    print(f"[3] Platoon of {N_FOL} followers, leader profile '{profile}'")
    plt = style.plt
    results = {}
    for lab, kind, kw in STRATEGIES:
        r, _ = run_platoon(N_FOL, kind, kw, profile=profile, T=T)
        results[lab] = r
    fig, ax = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True, sharey="row")
    for j, (lab, r) in enumerate(results.items()):
        for i in range(N_FOL + 1):
            c = style.LEADER if i == 0 else style.RAMP[i - 1]
            lw = 2.2 if i == 0 else 1.4
            ax[0, j].plot(r.t, r.v[i], color=c, lw=lw, ls="--" if i == 0 else "-",
                          label=r.names[i])
            ax[1, j].plot(r.t, r.a[i], color=c, lw=lw, ls="--" if i == 0 else "-")
            if i > 0:
                ax[2, j].plot(r.t, r.gap[i], color=c, lw=lw)
        ax[0, j].set_title(lab + ("  — collision!" if r.collisions else ""), loc="left",
                           color=style.CRITICAL if r.collisions else style.TEXT)
        ax[2, j].axhline(0, color=style.CRITICAL, lw=1)
        ax[2, j].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("speed [m/s]")
    ax[1, 0].set_ylabel("acceleration [m/s²]")
    ax[2, 0].set_ylabel("gap to predecessor [m]")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=N_FOL + 1, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, x=0.06, y=1.04, ha="left", color=style.TEXT, fontsize=12)
    save(fig, fig_name)
    return results


def platoon_summary(results):
    rows = {}
    for lab, r in results.items():
        ratio, peaks = an.string_amplification(r, "a")
        l2, _ = an.l2_amplification(r)
        m = [an.vehicle_metrics(r, i) for i in range(1, N_FOL + 1)]
        crashed = sorted({c[1] for c in r.collisions})
        v_eq = r.v[0, 0]
        ctrl_gap = np.nanmean(r.gap[1:, :50])
        rows[lab] = {
            "peak_acc_leader": round(float(peaks[0]), 2),
            "peak_acc_last": round(float(peaks[-1]), 2),
            "amplification_last_over_leader": round(float(peaks[-1] / peaks[0]), 2),
            "max_consecutive_ratio": round(float(ratio.max()), 3),
            "string_stable": bool(ratio.max() <= 1.0 + 1e-2),
            "min_gap_m": round(min(x["min_gap_m"] for x in m), 2),
            "min_ttc_s": round(min(x["min_ttc_s"] for x in m), 2),
            "max_decel": round(max(x["max_decel"] for x in m), 2),
            "rms_jerk_mean": round(float(np.mean([x["rms_jerk"] for x in m])), 3),
            "collided_vehicles": crashed,
            "initial_gap_m": round(float(ctrl_gap), 1),
            "lane_capacity_veh_per_h": int(3600 * v_eq / (ctrl_gap + 4.5)) if v_eq > 0 else None,
            "aeb_activations": int(np.sum(np.diff((r.mode == MODES.index("AEB")).astype(int), axis=1) == 1)),
        }
    return rows


def fig_peak_vs_index(sets, name):
    plt = style.plt
    fig, axs = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, (title, results) in zip(axs, sets.items()):
        for (lab, r), c in zip(results.items(), style.CAT):
            _, peaks = an.string_amplification(r, "a")
            ax.plot(range(N_FOL + 1), peaks, color=c, lw=2, marker="o", ms=6, label=lab)
        ax.set_xticks(range(N_FOL + 1), ["L"] + [f"F{i}" for i in range(1, N_FOL + 1)])
        ax.set_xlabel("vehicle position in platoon")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 6.5)
    axs[0].set_ylabel("peak |acceleration| [m/s²]")
    axs[0].legend(loc="upper left")
    fig.suptitle("Disturbance propagation: growing = string unstable, shrinking = string stable",
                 x=0.06, y=1.02, ha="left", color=style.TEXT, fontsize=12)
    save(fig, name)


def fig_space_time(results, name):
    plt = style.plt
    from matplotlib.collections import LineCollection
    labs = list(results)
    fig, axs = plt.subplots(1, len(labs), figsize=(4 * len(labs), 4.5), sharey=True)
    norm = plt.Normalize(0, 16)
    for ax, lab in zip(axs, labs):
        r = results[lab]
        for i in range(N_FOL + 1):
            pts = np.column_stack([r.t, r.x[i] - r.x[0, 0]])[::10]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=style.SPEED_CMAP, norm=norm, lw=2)
            lc.set_array(r.v[i, ::10][:-1])
            ax.add_collection(lc)
        ax.autoscale()
        ax.set_title(lab + ("  — collision!" if r.collisions else ""), loc="left",
                     color=style.CRITICAL if r.collisions else style.TEXT)
        ax.set_xlabel("time [s]")
    axs[0].set_ylabel("position [m]")
    cb = fig.colorbar(lc, ax=axs, shrink=0.85, pad=0.01)
    cb.set_label("speed [m/s]  (darker = slower)")
    fig.suptitle("Stop-and-go traffic, space-time diagram (each line is one vehicle)",
                 x=0.06, y=1.0, ha="left", color=style.TEXT, fontsize=12)
    save(fig, name)


# ---------------------------------------------------------------- 4. V2V degradation
def exp_v2v(quick):
    print("[4] V2V communication degradation")
    plt = style.plt
    delays = [0.02, 0.05, 0.1, 0.2, 0.3] if not quick else [0.05, 0.2]
    losses = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95] if not quick else [0.0, 0.6]
    amp_d, amp_l, gap_l = [], [], []
    for d in delays:
        r, _ = run_platoon(N_FOL, "CACC", {}, profile="speed_dip", T=45, v2v_delay=d)
        ratio, peaks = an.string_amplification(r, "a")
        amp_d.append(peaks[-1] / peaks[0])
    for p in losses:
        r, _ = run_platoon(N_FOL, "CACC", {}, profile="speed_dip", T=45, v2v_loss=p, seed=3)
        ratio, peaks = an.string_amplification(r, "a")
        amp_l.append(peaks[-1] / peaks[0])
        gap_l.append(np.nanmin(r.gap[1:]))

    # total blackout between 8 s and 40 s -> fallback to ACC with ramped headway
    r_out, _ = run_platoon(N_FOL, "CACC", {}, profile="speed_dip", T=70, v2v_outage=(8.0, 40.0))

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[0, 2])
    a4 = fig.add_subplot(gs[1, :2])
    a5 = fig.add_subplot(gs[1, 2])
    a1.plot(np.array(delays) * 1000, amp_d, color=style.CAT[0], lw=2, marker="o", ms=7)
    a1.axhline(1, color=style.TEXT2, lw=1)
    a1.set_xlabel("V2V latency [ms]")
    a1.set_ylabel("peak accel  F8 / leader")
    a1.set_title("Latency (h = 0.6 s)", loc="left")
    a2.plot(np.array(losses) * 100, amp_l, color=style.CAT[0], lw=2, marker="o", ms=7)
    a2.axhline(1, color=style.TEXT2, lw=1)
    a2.set_xlabel("packet loss [%]")
    a2.set_ylabel("peak accel  F8 / leader")
    a2.set_title("Random packet loss", loc="left")
    a2.annotate("heavy loss → mostly ACC\nfallback with longer headway", xy=(80, amp_l[-2] if len(amp_l) > 2 else amp_l[-1]),
                xytext=(5, 0.95), textcoords="data", fontsize=8.5, color=style.TEXT2,
                arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.8))
    a3.plot(np.array(losses) * 100, gap_l, color=style.CAT[0], lw=2, marker="o", ms=7)
    a3.set_xlabel("packet loss [%]")
    a3.set_ylabel("minimum gap in platoon [m]")
    a3.set_ylim(0, None)
    a3.set_title("Safety margin", loc="left")
    for i in range(1, N_FOL + 1):
        a4.plot(r_out.t, r_out.gap[i], color=style.RAMP[i - 1], lw=1.4, label=f"F{i}")
    a4.axvspan(8, 40, color=style.MUTED, alpha=0.15, lw=0)
    a4.text(8.5, 0.97, "V2V blackout: ACC fallback,\nheadway ramps 0.6 → 1.6 s", va="top",
            transform=a4.get_xaxis_transform(), color=style.TEXT2, fontsize=9)
    a4.set_xlabel("time [s]")
    a4.set_ylabel("gap to predecessor [m]")
    a4.legend(ncol=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    a4.set_title("Graceful degradation when communication is lost", loc="left", pad=22)
    style.mode_strip(a5, r_out.t, r_out.mode[N_FOL], legend_y=-0.16)
    a5.set_title("F8 controller mode", loc="left")
    a5.set_xlabel("time [s]")
    fig.tight_layout()
    save(fig, "fig6_v2v_degradation.png")
    return {"latency_ms": (np.array(delays) * 1000).tolist(), "amp_vs_latency": np.round(amp_d, 3).tolist(),
            "loss_pct": (np.array(losses) * 100).tolist(), "amp_vs_loss": np.round(amp_l, 3).tolist(),
            "min_gap_vs_loss": np.round(gap_l, 2).tolist(),
            "blackout_collisions": len(r_out.collisions),
            "blackout_min_gap": round(float(np.nanmin(r_out.gap[1:])), 2)}


# ---------------------------------------------------------------- 5. heterogeneous platoon
def exp_heterogeneous():
    print("[5] Heterogeneous platoon (random actuator lags 0.2-0.6 s, mixed lengths)")
    rng = np.random.default_rng(42)
    params = [VehicleParams(length=float(rng.choice([4.0, 4.5, 5.0, 12.0])),
                            tau=float(rng.uniform(0.2, 0.6)),
                            a_min=float(rng.uniform(-6.5, -5.0))) for _ in range(N_FOL + 1)]
    out = {"tau": [round(p.tau, 3) for p in params], "length": [p.length for p in params]}
    res = {}
    for lab, kind, kw in STRATEGIES[1:]:
        r, _ = run_platoon(N_FOL, kind, kw, profile="speed_dip", T=50, params=params)
        res[lab] = r
    out["summary"] = platoon_summary(res)
    return out, res


def write_markdown(summary):
    lines = ["# Results summary (auto-generated by experiments/run_all.py)\n"]
    for scen in ["speed_dip", "emergency_stop", "stop_and_go", "heterogeneous"]:
        tab = summary[scen] if scen != "heterogeneous" else summary["heterogeneous"]["summary"]
        lines.append(f"\n## Scenario: {scen}\n")
        if scen == "emergency_stop":
            lines.append("_Every vehicle brakes at its limit here, so acceleration ratios are "
                         "saturated and say nothing about string stability; judge this scenario "
                         "by min gap / TTC / collisions._\n")
        cols = ["peak_acc_last", "amplification_last_over_leader", "max_consecutive_ratio",
                "string_stable", "min_gap_m", "min_ttc_s", "max_decel", "rms_jerk_mean",
                "collided_vehicles", "lane_capacity_veh_per_h"]
        lines.append("| strategy | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for lab, row in tab.items():
            vals = [", ".join(row[c]) or "none" if isinstance(row[c], list) else str(row[c]) for c in cols]
            if scen == "emergency_stop":
                vals[cols.index("string_stable")] = "n/a"
            lines.append(f"| {lab} | " + " | ".join(vals) + " |")
    with open(os.path.join(OUT, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    summary = {"acc_test_drive": exp_acc_test_drive(), "frequency": exp_frequency_domain()}
    dip = exp_platoon("speed_dip", 50, "fig3_platoon_speed_dip.png",
                      "Leader brakes 25 → 15 m/s at −2.5 m/s², then recovers (8 followers)")
    stop = exp_platoon("emergency_stop", 40, "fig4_platoon_emergency_stop.png",
                       "Leader performs an emergency stop from 25 m/s at −6 m/s²")
    sng = exp_platoon("stop_and_go", 80, "fig5_platoon_stop_and_go.png",
                      "Urban stop-and-go traffic")
    fig_peak_vs_index({"Speed dip": dip, "Stop-and-go": sng}, "fig3b_peak_acceleration_vs_position.png")
    fig_space_time(sng, "fig5b_space_time_stop_and_go.png")
    summary["speed_dip"] = platoon_summary(dip)
    summary["emergency_stop"] = platoon_summary(stop)
    summary["stop_and_go"] = platoon_summary(sng)
    summary["v2v"] = exp_v2v(args.quick)
    summary["heterogeneous"], _ = exp_heterogeneous()
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    write_markdown(summary)
    print(f"done in {time.time() - t0:.0f} s -> results/metrics.json, results/metrics.md")


if __name__ == "__main__":
    main()
