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


def odd_even_test(lc, period, epoch, duration_days, n_bins=20,
                   sigma_threshold=5.0, min_relative_mismatch=0.20,
                   local_baseline_factor=3.0):
    """
    Compare the in-transit depth of odd-numbered vs even-numbered transits.
    Returns dict with depths, their difference in sigma, and a flag.

    Depth is measured against each group's OWN local out-of-transit baseline
    (points within local_baseline_factor x duration of each transit, minus
    the in-transit window itself) rather than an assumed global baseline of
    exactly 1.0. This matters a lot in practice: each sector is normalized
    independently during stitching, so tiny per-sector calibration offsets
    (routinely 0.1-0.5%, and totally normal) get misread as a real depth
    difference whenever odd- and even-numbered transits happen to fall in
    different sectors — which is common. Using a local baseline per group
    cancels that out, since both the in-transit and baseline points for a
    given group come from the same nearby (same-sector) data.

    Flags also require BOTH statistical significance (sigma_threshold) AND a
    practically meaningful relative depth mismatch (min_relative_mismatch) —
    with the tens of thousands of cadences in a stitched multi-sector light
    curve, even a trivial asymmetry becomes ">3 sigma" from sample size alone.
    """
    time_vals = lc.time.value
    flux_vals = lc.flux.value

    transit_number = np.round((time_vals - epoch) / period)
    phase = ((time_vals - epoch + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) <= duration_days / 2.0
    near_transit = np.abs(phase) <= duration_days * local_baseline_factor / 2.0
    local_baseline = near_transit & ~in_transit

    def _group_depth(parity_mask):
        in_grp = in_transit & parity_mask
        base_grp = local_baseline & parity_mask
        if in_grp.sum() < 10 or base_grp.sum() < 10:
            return None
        baseline_flux = np.nanmedian(flux_vals[base_grp])
        transit_flux = np.nanmedian(flux_vals[in_grp])
        depth = baseline_flux - transit_flux
        # Combine in-transit and local-baseline scatter for the error estimate.
        err = np.sqrt(
            (np.nanstd(flux_vals[in_grp]) / np.sqrt(in_grp.sum())) ** 2
            + (np.nanstd(flux_vals[base_grp]) / np.sqrt(base_grp.sum())) ** 2
        )
        return depth, err, int(in_grp.sum())

    odd_result = _group_depth(transit_number % 2 != 0)
    even_result = _group_depth(transit_number % 2 == 0)

    if odd_result is None or even_result is None:
        return {"status": "insufficient_data", "message": "Not enough odd/even transits (with local baseline coverage) in this baseline."}

    odd_depth, odd_err, n_odd = odd_result
    even_depth, even_err, n_even = even_result

    combined_err = np.sqrt(odd_err ** 2 + even_err ** 2)
    sigma_diff = abs(odd_depth - even_depth) / combined_err if combined_err > 0 else 0.0

    mean_depth = 0.5 * (abs(odd_depth) + abs(even_depth))
    relative_mismatch = abs(odd_depth - even_depth) / mean_depth if mean_depth > 0 else 0.0

    flag = (sigma_diff > sigma_threshold) and (relative_mismatch > min_relative_mismatch)

    return {
        "status": "ok",
        "odd_depth": float(odd_depth),
        "even_depth": float(even_depth),
        "sigma_diff": float(sigma_diff),
        "relative_mismatch": float(relative_mismatch),
        "likely_eb": bool(flag),
        "message": (
            f"Odd/even depth mismatch of {sigma_diff:.1f}σ ({relative_mismatch*100:.0f}% relative) — "
            + ("consistent with an eclipsing binary at half the true period."
               if flag else "consistent with a genuine planetary transit.")
        ),
    }


def transit_shape_test(lc, period, epoch, duration_days, flatness_threshold=0.65):
    """
    Classify transit shape as V-shaped (grazing EB) or U-shaped (planet) using
    a normalized "flatness" statistic: the ratio of the flux variance near the
    bottom quartile of the transit vs. near its edges. A flat bottom (U-shape,
    low variance ratio) suggests a planet; a sharply peaked bottom (V-shape,
    high ratio, no flat minimum) suggests a grazing eclipsing binary.

    Uses percentile-based depth (5th percentile, not the raw minimum) so a
    single noisy outlier point can't dominate the full-depth estimate and
    artificially inflate the flatness ratio — the raw-minimum version was
    prone to misclassifying clean, genuinely flat-bottomed transits as
    V-shaped whenever one low-noise point dipped a bit further than the rest.
    """
    phase, flux, _ = phase_fold(lc, period, epoch)
    half_dur_phase = (duration_days / 2.0) / period

    in_transit = np.abs(phase) <= half_dur_phase
    if in_transit.sum() < 20:
        return {"status": "insufficient_data", "message": "Not enough in-transit points to assess shape."}

    t_phase = phase[in_transit]
    t_flux = flux[in_transit]

    # Core = central 50% of the transit width (by phase), edges = the rest.
    core_mask = np.abs(t_phase) <= half_dur_phase * 0.5
    edge_mask = ~core_mask

    if core_mask.sum() < 8 or edge_mask.sum() < 8:
        return {"status": "insufficient_data", "message": "Not enough resolution across the transit to assess shape."}

    # Robust (percentile-based) depth estimates instead of raw min/max, so a
    # single noisy point can't dominate the statistic.
    core_p10 = np.nanpercentile(t_flux[core_mask], 10)
    core_p90 = np.nanpercentile(t_flux[core_mask], 90)
    core_depth_range = core_p90 - core_p10
    full_depth = 1.0 - np.nanpercentile(t_flux, 5)

    flatness_ratio = core_depth_range / full_depth if full_depth > 0 else np.nan

    is_v_shaped = flatness_ratio is not None and not np.isnan(flatness_ratio) and flatness_ratio > flatness_threshold

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


def secondary_eclipse_test(lc, period, epoch, duration_days, n_bins=50,
                            sigma_threshold=5.0, min_relative_depth=0.15):
    """
    Search for flux modulation at phase 0.5 (the classic secondary-eclipse
    location for a circular orbit). A significant dip there — beyond what's
    expected from noise — suggests an eclipsing binary rather than a planet
    (planets' secondary eclipses are typically far below TESS's noise floor).

    Flags only require BOTH statistical significance (sigma_threshold) AND
    the secondary depth being a meaningful fraction of the primary transit
    depth (min_relative_depth) — otherwise, with a large enough sample, even
    a real but astrophysically tiny bump (or correlated instrumental
    systematics) reads as ">3 sigma" without being a genuine secondary
    eclipse.
    """
    phase, flux, _ = phase_fold(lc, period, epoch)
    half_dur_phase = (duration_days / 2.0) / period

    primary_mask = np.abs(phase) <= half_dur_phase
    secondary_mask = np.abs(np.abs(phase) - 0.5) <= half_dur_phase
    baseline_mask = (np.abs(phase) > 3 * half_dur_phase) & (np.abs(np.abs(phase) - 0.5) > 3 * half_dur_phase)

    if secondary_mask.sum() < 10 or baseline_mask.sum() < 20 or primary_mask.sum() < 10:
        return {"status": "insufficient_data", "message": "Not enough phase coverage near 0.5 to test for a secondary eclipse."}

    secondary_flux = np.nanmedian(flux[secondary_mask])
    baseline_flux = np.nanmedian(flux[baseline_mask])
    baseline_std = np.nanstd(flux[baseline_mask])
    secondary_err = baseline_std / np.sqrt(secondary_mask.sum())

    primary_depth = baseline_flux - np.nanmedian(flux[primary_mask])
    depth = baseline_flux - secondary_flux
    sigma = depth / secondary_err if secondary_err > 0 else 0.0

    relative_depth = depth / primary_depth if primary_depth > 0 else 0.0

    flag = (sigma > sigma_threshold) and (relative_depth > min_relative_depth)

    return {
        "status": "ok",
        "secondary_depth": float(depth),
        "relative_depth": float(relative_depth),
        "significance_sigma": float(sigma),
        "likely_eb": bool(flag),
        "message": (
            f"Secondary eclipse detected at {sigma:.1f}σ significance ({relative_depth*100:.0f}% of primary depth) — "
            + ("likely an eclipsing binary." if flag else "consistent with a planet (no significant secondary).")
        ),
    }


def suggest_corrected_period(period, odd_even_result, secondary_result):
    """
    When odd/even depths mismatch and/or a significant secondary eclipse is
    found, the classic explanation is that the search locked onto HALF the
    true orbital period (alternating primary/secondary eclipses of an
    eclipsing binary being mistaken for a regular sequence of transits).

    Returns a suggestion dict if either indicator is flagged, else None.
    """
    odd_even_flag = odd_even_result.get("status") == "ok" and odd_even_result.get("likely_eb")
    secondary_flag = secondary_result.get("status") == "ok" and secondary_result.get("likely_eb")

    if not (odd_even_flag or secondary_flag):
        return None

    reasons = []
    if odd_even_flag:
        reasons.append("odd/even transit depths differ significantly")
    if secondary_flag:
        reasons.append("a significant secondary eclipse was detected at phase 0.5")

    corrected_period = period * 2.0
    return {
        "original_period": float(period),
        "corrected_period": float(corrected_period),
        "reason": " and ".join(reasons),
        "label": f"{period:.4g} x 2 = {corrected_period:.4g} days",
    }


def run_all_diagnostics(lc, period, epoch, duration_days):
    """Convenience wrapper running all three FP diagnostics and returning a combined verdict."""
    odd_even = odd_even_test(lc, period, epoch, duration_days)
    shape = transit_shape_test(lc, period, epoch, duration_days)
    secondary = secondary_eclipse_test(lc, period, epoch, duration_days)

    checks = (odd_even, shape, secondary)
    flags = [d.get("likely_eb", False) for d in checks if d.get("status") == "ok"]
    n_flagged = sum(flags)
    n_inconclusive = sum(1 for d in checks if d.get("status") != "ok")

    if n_flagged == 0 and n_inconclusive == 0:
        verdict = "No false-positive indicators triggered — candidate remains viable."
    elif n_flagged == 0 and n_inconclusive > 0:
        verdict = (
            f"No false-positive indicators triggered, but {n_inconclusive} of 3 diagnostic(s) could not "
            f"run (insufficient data in the assumed transit window) — this is inconclusive, not a clean pass."
        )
    elif n_flagged == 1:
        verdict = "One diagnostic flagged a possible false positive — worth closer inspection."
    else:
        verdict = f"{n_flagged} diagnostics flagged false-positive indicators — likely an eclipsing binary."

    period_suggestion = suggest_corrected_period(period, odd_even, secondary)

    return {
        "odd_even": odd_even,
        "shape": shape,
        "secondary_eclipse": secondary,
        "n_flagged": n_flagged,
        "n_inconclusive": n_inconclusive,
        "verdict": verdict,
        "period_suggestion": period_suggestion,
    }
