"""App-wide constants and session-state defaults."""

APP_TITLE = "🪐 PHT Candidate Validator"
APP_CAPTION = "Deterministic-physics exoplanet & variable-star vetting for Planet Hunters TESS"

SPOC_2MIN_LABEL = "SPOC (2-min)"
FFI_QLP_LABEL = "FFI / QLP"
FFI_FALLBACK_AUTHORS = ("TESS-SPOC", "QLP")

FLUX_COLUMNS = ["PDCSAP_FLUX", "SAP_FLUX"]

CACHE_TTL_SECONDS = 3600

# Every key the app relies on in st.session_state, with its default value.
SESSION_DEFAULTS = {
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
    "centroid_result": None,

    # Export (Step: CSV/PDF)
    "pdf_export_bytes": None,
}


def init_session_state(st):
    """Populate st.session_state with defaults for any missing keys."""
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val
