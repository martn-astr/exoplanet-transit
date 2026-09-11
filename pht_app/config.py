"""App-wide constants and session-state defaults."""

from __future__ import annotations

import copy
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import streamlit as st  # noqa: F401 — hint-only import, real module passed in by callers

APP_TITLE = "🪐 PHT Candidate Validator"
APP_CAPTION = "Deterministic-physics exoplanet & variable-star vetting for Planet Hunters TESS"

SPOC_2MIN_LABEL = "SPOC (2-min)"
FFI_QLP_LABEL = "FFI / QLP"
FFI_FALLBACK_AUTHORS = ("TESS-SPOC", "QLP")

FLUX_COLUMNS = ["PDCSAP_FLUX", "SAP_FLUX"]

CACHE_TTL_SECONDS = 3600

# Every key the app relies on in st.session_state, with its default value.
SESSION_DEFAULTS: dict[str, Any] = {
    # Target identity
    "tic_id": None,
    "stellar_params": None,
    "exofop_flags": None,

    # Sector discovery / selection
    "sector_list": [],
    "selected_sectors": [],
    "data_source": SPOC_2MIN_LABEL,

    # Loaded light curve data
    "lc_collection": None,     # dict[sector] -> per-sector LightCurve
    "stitched_lc": None,       # single stitched, normalized LightCurve
    "flux_column": "PDCSAP_FLUX",

    # Panel 1 (timeline) interaction state
    "timeline_xrange": None,   # (xmin, xmax) currently zoomed/selected, or None = full range

    # Panel 2 (phase-fold) state
    "fold_period": None,
    "fold_epoch": None,

    # Panel 3 (periodogram) state
    "periodogram_method": "BLS",
    "bls_result": None,
    "ls_result": None,

    # Single-transit Keplerian estimator (Step 3)
    "click_t0": None,
    "click_t14_hours": None,
    "single_transit_estimate": None,   # {"t0", "period", "duration_days", "b_curve", "p_curve"}

    # Signal masking (Step 3)
    "signal_masks": [],   # list of {"period", "epoch", "duration_days"}

    # False positive diagnostics + TPF centroid (Step 4)
    "fp_diagnostics_result": None,
    "tpf_data": None,             # downloaded TPF object for the flagged sector
    "tpf_diff_image": None,
    "gaia_sources": None,
    "gaia_error": None,
    "centroid_result": None,

    # Export (Step: CSV/PDF)
    "pdf_export_bytes": None,
}

# Keys that identify the *target* and its raw loaded light curve data —
# cleared on a brand new TIC search (a new target invalidates all of this).
_TARGET_SCOPED_KEYS: list[str] = [
    "lc_collection", "stitched_lc", "selected_sectors",
    "timeline_xrange", "bls_result", "ls_result",
    "fold_period", "fold_epoch",
]

# Keys holding derived analysis/results — cleared whenever the underlying
# light curve changes (new search OR sectors re-downloaded/stitched), since
# a result computed against one light curve is meaningless against another.
_ANALYSIS_SCOPED_KEYS: list[str] = [
    "fp_diagnostics_result", "single_transit_estimate",
    "click_t0", "click_t14_hours", "signal_masks",
    "tpf_data", "tpf_diff_image", "gaia_sources", "gaia_error", "centroid_result",
    "pdf_export_bytes",
]


def _default_for(key: str) -> Any:
    """Return a fresh copy of a session-state default.

    SESSION_DEFAULTS contains mutable objects (empty lists) shared at module
    scope — assigning them directly (``st.session_state[key] = SESSION_DEFAULTS[key]``)
    would hand out the *same* list object every time. Since several call
    sites mutate session-state lists in place (e.g. ``signal_masks.append(...)``),
    that would silently corrupt the shared module-level default forever after
    the first mutation. Deep-copying on every assignment avoids that.
    """
    return copy.deepcopy(SESSION_DEFAULTS[key])


def init_session_state(st) -> None:
    """Populate st.session_state with defaults for any missing keys."""
    for key in SESSION_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = _default_for(key)


def _clear_dynamic_sector_checkboxes(st) -> None:
    """Clear ad-hoc `sector_<N>` checkbox keys from session_state.

    These keys are created dynamically by the sidebar (one per sector
    number, e.g. "sector_14") and are NOT part of SESSION_DEFAULTS, so
    reset_for_new_target()'s normal key-by-key reset never touches them.
    Left alone, they leak across different targets: if TIC A had sector 14
    and the user deselected it, then a NEW target TIC B also has a sector
    numbered 14, the checkbox would silently start pre-deselected for B too
    — not because of anything about B's data, just because the key number
    happens to match. Only called from reset_for_new_target(), not from
    reset_for_sector_reload() (same target — the user's current checkbox
    selections should survive a sector reload, not get wiped).
    """
    stale_keys = [
        key for key in list(st.session_state.keys())
        if key.startswith("sector_") and key[len("sector_"):].isdigit()
    ]
    for key in stale_keys:
        del st.session_state[key]


def reset_for_new_target(st) -> None:
    """Call when the user searches a new TIC ID.

    Clears everything scoped to the previous target, including derived
    analysis results (odd/even, secondary-eclipse, TPF centroid, signal
    masks, single-transit estimate, prepared PDF export, etc.) — all of it
    was computed against a light curve that's about to be replaced, so
    leaving it in session_state would render stale results against the new
    target. Also clears dynamic per-sector checkbox state (see
    _clear_dynamic_sector_checkboxes) so a coincidentally-matching sector
    number from the previous target can't leak its selection state into
    the new one.
    """
    for key in _TARGET_SCOPED_KEYS + _ANALYSIS_SCOPED_KEYS:
        st.session_state[key] = _default_for(key)
    _clear_dynamic_sector_checkboxes(st)


def reset_for_sector_reload(st) -> None:
    """Call when sectors are re-downloaded/stitched for the SAME target.

    Keeps ``stellar_params`` / ``exofop_flags`` / ``sector_list`` (still
    valid — same target), but clears the light curve itself and everything
    derived from it.
    """
    keys_to_reset = [
        "timeline_xrange", "bls_result", "ls_result",
        "fold_period", "fold_epoch",
    ] + _ANALYSIS_SCOPED_KEYS
    for key in keys_to_reset:
        st.session_state[key] = _default_for(key)
