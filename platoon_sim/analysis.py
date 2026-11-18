"""Performance metrics and frequency-domain string-stability analysis.

String stability
----------------
A platoon is *string stable* if disturbances are attenuated as they travel
backwards along the string. For a homogeneous platoon this is checked with the
transfer function from the predecessor's acceleration to the ego's,

    Gamma(jw) = A_i(jw) / A_{i-1}(jw),     string stable  <=>  sup_w |Gamma(jw)| <= 1.

Vehicle:   G(s) = 1 / (s^2 (tau s + 1))           (desired accel -> position)
Spacing:   H(s) = h s + 1
ACC        u = kp e + kd (v_{i-1} - v_i)
           Gamma_ACC  = (kd s + kp) / (tau s^3 + s^2 + (kd + kp h) s + kp)
CACC       h u' = -u + kp e + kd e' + u_{i-1}(t - theta)
           Gamma_CACC = (D(s) + G K) / (H (1 + G K)),   K = kp + kd s,  D = e^{-theta s}
"""
from __future__ import annotations

import numpy as np
from scipy import signal, optimize

W_GRID = np.logspace(-3, 2, 4000)


def gamma_acc(w, kp, kd, h, tau):
    """|Gamma| of radar-only ACC, evaluated with scipy.signal."""
    tf = signal.TransferFunction([kd, kp], [tau, 1.0, kd + kp * h, kp])
    _, H = signal.freqresp(tf, w)
    return H


def gamma_cacc(w, kp, kd, h, tau, theta):
    s = 1j * np.asarray(w)
    G = 1.0 / (s ** 2 * (tau * s + 1.0))
    K = kp + kd * s
    Hs = h * s + 1.0
    D = np.exp(-theta * s)
    return (D + G * K) / (Hs * (1.0 + G * K))


def peak_gain(gamma_vals):
    return float(np.max(np.abs(gamma_vals)))


def min_string_stable_headway(gain_fn, h_lo=0.05, h_hi=6.0, tol=1e-3):
    """Smallest headway h with sup|Gamma| <= 1, via scipy's Brent root finder.
    ``gain_fn(h)`` must return Gamma on ``W_GRID``."""
    f = lambda h: peak_gain(gain_fn(h)) - 1.0 - 1e-6
    if f(h_hi) > 0:
        return np.nan
    if f(h_lo) <= 0:
        return h_lo
    return optimize.brentq(f, h_lo, h_hi, xtol=tol)


# ---------------------------------------------------------------- time domain
def vehicle_metrics(res, i, t_from=0.0):
    sl = res.t >= t_from
    gap = res.gap[i, sl]
    e = res.spacing_error[i, sl]
    v = res.v[i, sl]
    a = res.a[i, sl]
    jerk = np.gradient(a, res.dt)
    # time-to-collision (only while closing in)
    if i > 0:
        closing = res.v[i, sl] - res.v[i - 1, sl]
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing > 0.05, np.maximum(gap, 0.0) / closing, np.inf)
        ttc_min = float(np.nanmin(ttc)) if ttc.size else np.inf   # 0 => collision
    else:
        ttc_min = np.inf
    return {
        "vehicle": res.names[i],
        "min_gap_m": float(np.nanmin(gap)) if np.isfinite(gap).any() else np.nan,
        "max_abs_e_m": float(np.nanmax(np.abs(e))) if np.isfinite(e).any() else np.nan,
        "min_ttc_s": ttc_min,
        "max_decel": float(-np.min(a)),
        "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))),
        "min_speed": float(np.min(v)),
    }


def string_amplification(res, signal_name="a", t_from=0.0):
    """Peak |signal| of each vehicle divided by that of its predecessor.
    All ratios <= 1 means the disturbance is damped along the string."""
    sl = res.t >= t_from
    data = getattr(res, signal_name)[:, sl]
    peaks = np.nanmax(np.abs(data - np.nanmean(data[:, :10], axis=1, keepdims=True)), axis=1)
    return peaks[1:] / peaks[:-1], peaks


def l2_amplification(res, t_from=0.0):
    """||a_i||_2 / ||a_{i-1}||_2 for consecutive vehicles."""
    sl = res.t >= t_from
    norms = np.sqrt(np.nansum(res.a[:, sl] ** 2, axis=1) * res.dt)
    return norms[1:] / norms[:-1], norms
