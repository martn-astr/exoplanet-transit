"""
False Positive diagnostic module.

Implements three classic vetting checks used to distinguish genuine planet
transits from eclipsing-binary (EB) false positives:

  1. Odd vs Even transit depth comparison — a significant depth mismatch
     between odd- and even-numbered transits is the classic EB signature
     (grazing/blended eclipsing binary at half the true period).
  2. Transit shape — V-shaped (grazing EB) vs U-shaped (planet), measured via
     a simple shape parameter comparing the flatness of the transit bottom.
  3. Secondary eclipse search — out-of-transit flux modulation at phase 0.5,
     which for a planet should be at or below the noise floor, but for an EB
     often shows a detectable secondary dip (occultation of the companion).
"""

import numpy as np

from pht_app.data.analysis import phase_fold


def odd_even_folded_curves(lc, period, epoch, duration_days, window_factor=3.0, n_bins=40):
    """
    Build binned, phase-folded curves for odd- and even-numbered transits
    separately, zoomed to a window around the transit (± window_factor *
    duration), for the classic side-by-side "odd vs even" visual comparison.

    Returns dict with phase_bins and binned flux (+ error) arrays for each,
    or a status flag if there isn't enough data.
    """
    time_vals = lc.time.value
    flux_vals = lc.flux.value

    transit_number = np.round((time_vals - epoch) / period)
    phase_days = ((time_vals - epoch + 0.5 * period) % period) - 0.5 * period
    half_window = duration_days * window_factor / 2.0
    near_transit = np.abs(phase_days) <= half_window

    odd_mask = near_transit & (transit_number % 2 != 0)
    even_mask = near_transit & (transit_number % 2 == 0)

    if odd_mask.sum() < 5 or even_mask.sum() < 5:
        return {"status": "insufficient_data"}

    bin_edges = np.linspace(-half_window, half_window, n_bins + 1)
    bin_centers_hours = 0.5 * (bin_edges[:-1] + bin_edges[1:]) * 24.0

    def _bin(mask):
        means = np.full(n_bins, np.nan)
        for i in range(n_bins):
            sel = mask & (phase_days >= bin_edges[i]) & (phase_days < bin_edges[i + 1])
            if sel.any():
                means[i] = np.nanmean(flux_vals[sel])
        return means

    return {
        "status": "ok",
        "phase_hours": bin_centers_hours,
        "odd_flux": _bin(odd_mask),
        "even_flux": _bin(even_mask),
    }


def secondary_zoom_curves(lc, period, epoch, duration_days, window_factor=3.0, n_bins=60):
    """
    Build binned phase-folded curves zoomed on the primary transit (phase 0)
    and on the expected secondary-eclipse location (phase 0.5) side by side,
    for visual comparison of depth/presence of a secondary dip.
    """
    phase, flux, _ = phase_fold(lc, period, epoch)
    half_window = (duration_days * window_factor / 2.0) / period  # in phase units

    def _zoom(center_phase):
        rel_phase = phase - center_phase
        rel_phase = (rel_phase + 0.5) % 1.0 - 0.5  # wrap into [-0.5, 0.5)
        near = np.abs(rel_phase) <= half_window
        if near.sum() < 10:
            return None
        bin_edges = np.linspace(-half_window, half_window, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        means = np.full(n_bins, np.nan)
        for i in range(n_bins):
            sel = near & (rel_phase >= bin_edges[i]) & (rel_phase < bin_edges[i + 1])
            if sel.any():
                means[i] = np.nanmean(flux[sel])
        return {"phase": bin_centers, "flux": means}

    primary = _zoom(0.0)
    secondary = _zoom(0.5)

    if primary is None or secondary is None:
        return {"status": "insufficient_data"}

    return {"status": "ok", "primary": primary, "secondary": secondary}


def odd_even_test(lc, period, epoch, duration_days, n_bins=20):
    """
    Compare the mean in-transit depth of odd-numbered vs even-numbered transits.
    Returns dict with depths, their difference in sigma, and a flag.
    """
    time_vals = lc.time.value
    flux_vals = lc.flux.value

    transit_number = np.round((time_vals - epoch) / period)
    phase = ((time_vals - epoch + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) <= duration_days / 2.0

    odd_mask = in_transit & (transit_number % 2 != 0)
    even_mask = in_transit & (transit_number % 2 == 0)

    if odd_mask.sum() < 3 or even_mask.sum() < 3:
        return {"status": "insufficient_data", "message": "Not enough odd/even transits in this baseline."}

    odd_depth = 1.0 - np.nanmedian(flux_vals[odd_mask])
    even_depth = 1.0 - np.nanmedian(flux_vals[even_mask])
    odd_err = np.nanstd(flux_vals[odd_mask]) / np.sqrt(odd_mask.sum())
    even_err = np.nanstd(flux_vals[even_mask]) / np.sqrt(even_mask.sum())

    combined_err = np.sqrt(odd_err ** 2 + even_err ** 2)
    sigma_diff = abs(odd_depth - even_depth) / combined_err if combined_err > 0 else 0.0

    flag = sigma_diff > 3.0  # >3-sigma mismatch flags likely EB

    return {
        "status": "ok",
        "odd_depth": float(odd_depth),
        "even_depth": float(even_depth),
        "sigma_diff": float(sigma_diff),
        "likely_eb": bool(flag),
        "message": (
            f"Odd/even depth mismatch of {sigma_diff:.1f}σ — "
            + ("consistent with an eclipsing binary at half the true period."
               if flag else "consistent with a genuine planetary transit.")
        ),
    }


def transit_shape_test(lc, period, epoch, duration_days):
    """
    Classify transit shape as V-shaped (grazing EB) or U-shaped (planet) using
    a normalized "flatness" statistic: the ratio of the flux variance near the
    bottom quartile of the transit vs. near its edges. A flat bottom (U-shape,
    low variance ratio) suggests a planet; a sharply peaked bottom (V-shape,
    high ratio, no flat minimum) suggests a grazing eclipsing binary.
    """
    phase, flux, _ = phase_fold(lc, period, epoch)
    half_dur_phase = (duration_days / 2.0) / period

    in_transit = np.abs(phase) <= half_dur_phase
    if in_transit.sum() < 10:
        return {"status": "insufficient_data", "message": "Not enough in-transit points to assess shape."}

    t_phase = phase[in_transit]
    t_flux = flux[in_transit]

    # Split into "core" (middle third) vs "edges" (outer thirds) of the transit
    core_mask = np.abs(t_phase) <= half_dur_phase / 3.0
    edge_mask = ~core_mask

    if core_mask.sum() < 3 or edge_mask.sum() < 3:
        return {"status": "insufficient_data", "message": "Not enough resolution across the transit to assess shape."}

    core_flux_std = np.nanstd(t_flux[core_mask])
    core_depth_range = np.nanmax(t_flux[core_mask]) - np.nanmin(t_flux[core_mask])
    full_depth = 1.0 - np.nanmin(t_flux)

    # A flat-bottomed (U-shaped) transit has small core depth range relative to full depth.
    flatness_ratio = core_depth_range / full_depth if full_depth > 0 else np.nan

    is_v_shaped = flatness_ratio > 0.5  # core still varies almost as much as full depth => no flat bottom

    return {
        "status": "ok",
        "flatness_ratio": float(flatness_ratio) if not np.isnan(flatness_ratio) else None,
        "shape": "V-shaped (grazing EB-like)" if is_v_shaped else "U-shaped (planet-like)",
        "likely_eb": bool(is_v_shaped),
        "message": (
            f"Flatness ratio {flatness_ratio:.2f} — "
            + ("no clear flat bottom, suggests a grazing eclipsing binary."
               if is_v_shaped else "flat-bottomed transit, consistent with a planet.")
        ) if not np.isnan(flatness_ratio) else "Could not compute a reliable shape statistic.",
    }


def secondary_eclipse_test(lc, period, epoch, duration_days, n_bins=50):
    """
    Search for flux modulation at phase 0.5 (the classic secondary-eclipse
    location for a circular orbit). A significant dip there — beyond what's
    expected from noise — suggests an eclipsing binary rather than a planet
    (planets' secondary eclipses are typically far below TESS's noise floor).
    """
    phase, flux, _ = phase_fold(lc, period, epoch)
    half_dur_phase = (duration_days / 2.0) / period

    secondary_mask = np.abs(np.abs(phase) - 0.5) <= half_dur_phase
    baseline_mask = (np.abs(phase) > 3 * half_dur_phase) & (np.abs(np.abs(phase) - 0.5) > 3 * half_dur_phase)

    if secondary_mask.sum() < 3 or baseline_mask.sum() < 10:
        return {"status": "insufficient_data", "message": "Not enough phase coverage near 0.5 to test for a secondary eclipse."}

    secondary_flux = np.nanmedian(flux[secondary_mask])
    baseline_flux = np.nanmedian(flux[baseline_mask])
    baseline_std = np.nanstd(flux[baseline_mask])
    secondary_err = baseline_std / np.sqrt(secondary_mask.sum())

    depth = baseline_flux - secondary_flux
    sigma = depth / secondary_err if secondary_err > 0 else 0.0

    flag = sigma > 3.0

    return {
        "status": "ok",
        "secondary_depth": float(depth),
        "significance_sigma": float(sigma),
        "likely_eb": bool(flag),
        "message": (
            f"Secondary eclipse detected at {sigma:.1f}σ significance — "
            + ("likely an eclipsing binary." if flag else "consistent with a planet (no significant secondary).")
        ),
    }


def run_all_diagnostics(lc, period, epoch, duration_days):
    """Convenience wrapper running all three FP diagnostics and returning a combined verdict."""
    odd_even = odd_even_test(lc, period, epoch, duration_days)
    shape = transit_shape_test(lc, period, epoch, duration_days)
    secondary = secondary_eclipse_test(lc, period, epoch, duration_days)

    flags = [d.get("likely_eb", False) for d in (odd_even, shape, secondary) if d.get("status") == "ok"]
    n_flagged = sum(flags)

    if n_flagged == 0:
        verdict = "No false-positive indicators triggered — candidate remains viable."
    elif n_flagged == 1:
        verdict = "One diagnostic flagged a possible false positive — worth closer inspection."
    else:
        verdict = f"{n_flagged} diagnostics flagged false-positive indicators — likely an eclipsing binary."

    return {
        "odd_even": odd_even,
        "shape": shape,
        "secondary_eclipse": secondary,
        "n_flagged": n_flagged,
        "verdict": verdict,
    }
