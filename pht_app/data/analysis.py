"""Signal-processing helpers shared by the dashboard panels.

Kept separate from the panels themselves so the math/analysis logic is
testable without Streamlit or Plotly in the loop.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle


def window_lightcurve(lc, xrange):
    """Return the subset of a light curve within (xmin, xmax) time bounds, or the full lc if xrange is None."""
    if xrange is None:
        return lc
    xmin, xmax = xrange
    time_vals = lc.time.value
    mask = (time_vals >= xmin) & (time_vals <= xmax)
    return lc[mask]


def run_bls(lc, min_period=0.5, max_period=20.0, n_periods=5000, duration_grid=None):
    """
    Run a Box Least Squares search over the given light curve.
    Returns (periods, power, best_period, best_t0, best_duration).

    BoxLeastSquares requires every trial duration to be strictly shorter than
    the minimum trial period, so the duration grid is always clamped well
    below min_period regardless of what's passed in.
    """
    time_vals = lc.time.value
    flux_vals = lc.flux.value
    finite = np.isfinite(time_vals) & np.isfinite(flux_vals)
    time_vals, flux_vals = time_vals[finite], flux_vals[finite]

    if len(time_vals) < 10:
        return None

    if max_period <= min_period:
        max_period = min_period + 1.0

    # Cap the longest trial duration at 40% of the shortest trial period,
    # so it's always safely below min_period no matter what was requested.
    max_allowed_duration = min_period * 0.4
    if duration_grid is None:
        duration_grid = np.linspace(0.01, max_allowed_duration, 10)
    else:
        duration_grid = np.asarray(duration_grid)
        duration_grid = duration_grid[duration_grid < min_period]
        if duration_grid.size == 0:
            duration_grid = np.linspace(0.01, max_allowed_duration, 10)

    periods = np.linspace(min_period, max_period, n_periods)
    bls = BoxLeastSquares(time_vals, flux_vals)
    result = bls.power(periods, duration_grid)

    best_idx = np.argmax(result.power)
    return {
        "periods": result.period,
        "power": result.power,
        "best_period": float(result.period[best_idx]),
        "best_t0": float(result.transit_time[best_idx]),
        "best_duration": float(result.duration[best_idx]),
        "best_depth": float(result.depth[best_idx]),
    }


def run_lomb_scargle(lc, min_period=0.1, max_period=27.0, n_periods=5000):
    """Run a Lomb-Scargle periodogram (useful for variable stars / rotational modulation)."""
    time_vals = lc.time.value
    flux_vals = lc.flux.value
    finite = np.isfinite(time_vals) & np.isfinite(flux_vals)
    time_vals, flux_vals = time_vals[finite], flux_vals[finite]

    if len(time_vals) < 10:
        return None

    if max_period <= min_period:
        max_period = min_period + 1.0

    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period
    ls = LombScargle(time_vals, flux_vals)
    frequency, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)
    periods = 1.0 / frequency

    best_idx = np.argmax(power)
    return {
        "periods": periods,
        "power": power,
        "best_period": float(periods[best_idx]),
    }


def phase_fold(lc, period, epoch):
    """
    Fold a light curve on the given period/epoch.
    Returns (phase in [-0.5, 0.5), flux, flux_err_or_None), sorted by phase.
    """
    time_vals = lc.time.value
    flux_vals = lc.flux.value
    flux_err = lc.flux_err.value if hasattr(lc, "flux_err") and lc.flux_err is not None else None

    phase = ((time_vals - epoch) / period + 0.5) % 1.0 - 0.5
    order = np.argsort(phase)

    if flux_err is not None:
        return phase[order], flux_vals[order], flux_err[order]
    return phase[order], flux_vals[order], None
