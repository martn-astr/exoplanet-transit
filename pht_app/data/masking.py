"""Signal masking: remove known periodic signals from a light curve so the
residuals can be searched for additional, smaller planets."""

import numpy as np


def apply_masks(lc, masks: list):
    """
    Given a light curve and a list of {"period": P, "epoch": T0, "duration_days": D}
    dicts, return a boolean array (True = keep, False = masked out) over lc.time.
    A conservative default duration of 0.2 days is used if not specified.
    """
    if not masks:
        return np.ones(len(lc.time), dtype=bool)

    time_vals = lc.time.value
    keep = np.ones(len(time_vals), dtype=bool)

    for m in masks:
        period = m["period"]
        epoch = m["epoch"]
        half_dur = m.get("duration_days", 0.2) / 2.0
        if period <= 0:
            continue
        phase = ((time_vals - epoch + 0.5 * period) % period) - 0.5 * period
        in_transit = np.abs(phase) <= half_dur
        keep &= ~in_transit

    return keep


def masked_lightcurve(lc, masks: list):
    """Return a new light curve with masked-out points removed."""
    keep = apply_masks(lc, masks)
    return lc[keep]
