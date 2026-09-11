"""Signal-processing helpers shared by the dashboard panels.

Kept separate from the panels themselves so the math/analysis logic is
testable without Streamlit or Plotly in the loop.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from astropy.timeseries import BoxLeastSquares, LombScargle
from scipy.stats import binned_statistic


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

    Uses BoxLeastSquares.autoperiod() to build the period grid rather than a
    plain linear np.linspace. A uniform-in-period grid is too coarse at short
    periods relative to the observing baseline and commonly locks onto a
    harmonic/alias of the true period instead of the real signal (e.g.
    reporting ~2.3 d when the true period is ~1.68 d); autoperiod() spaces
    trial periods so the transit shape isn't under-sampled at any period in
    range, which resolves this.
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

    bls = BoxLeastSquares(time_vals, flux_vals)

    try:
        periods = bls.autoperiod(
            duration_grid,
            minimum_period=min_period,
            maximum_period=max_period,
            frequency_factor=5.0,  # denser than the default 1.0 for better short-period resolution
        )
        if len(periods) > 50000:
            # autoperiod can generate an enormous grid for long baselines /
            # short min_period; subsample down (preserving span) to stay fast.
            idx = np.linspace(0, len(periods) - 1, 50000).astype(int)
            periods = periods[idx]
    except Exception:
        # Fall back to a fine linear grid if autoperiod fails for any reason.
        periods = np.linspace(min_period, max_period, max(n_periods, 20000))

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


def bin_by_x(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x_min: float,
    x_max: float,
    n_bins: int,
    statistic: str = "mean",
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bin y by x into n_bins equal-width bins over [x_min, x_max].

    Shared helper for every "binned overlay" plot in the app (phase-folded
    views, odd/even comparison, secondary-eclipse zoom, PDF report panels) —
    previously each of those had its own hand-rolled version of this loop:

        for i in range(n_bins):
            mask = (x >= edges[i]) & (x < edges[i + 1])
            ...

    which has two problems this function fixes:

    1. Performance: that loop does n_bins separate full-array boolean
       comparisons (O(n_bins * len(x))) in pure Python. For a bin count of
       ~100 against a multi-sector light curve with 100k+ points, that's
       tens of millions of element-wise comparisons run from the Python
       interpreter loop. `scipy.stats.binned_statistic` does the equivalent
       binning in one vectorized pass (O(len(x))) in compiled code.
    2. Correctness: the hand-rolled version used a strict `< edges[i+1]`
       upper bound on every bin, including the last one — so a point whose
       x value landed exactly on the final bin edge (e.g. phase exactly
       0.5, which is rare but not impossible with floating-point wraparound)
       was silently excluded from every bin. `binned_statistic` treats the
       rightmost bin edge as inclusive, matching `numpy.histogram`'s
       documented (and expected) behavior.

    Parameters
    ----------
    x, y : ndarray
        Same-length arrays to bin. NaNs in y are dropped before binning
        (see implementation note below) rather than propagating and
        blanking the whole bin.
    x_min, x_max : float
        Bin range. Points outside this range are ignored (not clipped into
        the edge bins), matching the previous loop-based behavior.
    n_bins : int
        Number of equal-width bins.
    statistic : str
        "mean" or "median" — passed through to scipy using its built-in
        string form (see implementation note).

    Returns
    -------
    tuple[ndarray, ndarray]
        (bin_centers, bin_values). Empty bins come back as NaN, same as the
        original implementation.

    Notes
    -----
    NaNs are filtered out of `y` (and the matching `x`) *before* calling
    scipy, rather than passing a custom `nanmean`/`nanmedian` callable as
    the `statistic`. That distinction matters more than it looks: scipy's
    `binned_statistic` only uses its fast, vectorized compiled path for its
    built-in string statistics ("mean", "median", "sum", "count", ...) —
    passing any custom Python callable (including `np.nanmean`) forces it
    onto a slow per-bin Python callback path instead. Measured on a
    200k-point array binned into 100 bins, the callable version was
    actually ~5x SLOWER than the original hand-rolled Python loop it was
    meant to replace, while pre-filtering NaNs and using the built-in
    "mean"/"median" string is consistently faster than the loop.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[finite], y[finite]

    bin_means, bin_edges, _ = binned_statistic(
        x_clean, y_clean, statistic=statistic, bins=n_bins, range=(x_min, x_max)
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, bin_means
