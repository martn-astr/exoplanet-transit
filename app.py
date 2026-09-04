"""
Planet Hunters TESS (PHT) Candidate Validator
==============================================
Entry point. Run with:  streamlit run app.py

Structure:
    pht_app/config.py           constants + session-state defaults
    pht_app/data/lookup.py      TIC catalog + ExoFOP queries
    pht_app/data/lightcurves.py sector discovery + multi-sector download/stitch
    pht_app/data/analysis.py    windowing, BLS/Lomb-Scargle, phase-folding
    pht_app/ui/sidebar.py       sidebar controls
    pht_app/ui/header.py        stellar-params + ExoFOP header card
    pht_app/panels/timeline.py    Panel 1 — stitched timeline
    pht_app/panels/per_sector.py  per-sector breakdown (each sector, own subplot)
    pht_app/panels/phasefold.py   Panel 2 — phase-folded view
    pht_app/panels/periodogram.py Panel 3 — BLS / Lomb-Scargle
"""

import streamlit as st

from pht_app.config import init_session_state
from pht_app.data import resolve_tic, query_exofop, search_available_sectors, download_and_stitch, run_bls
from pht_app.ui.sidebar import render_sidebar
from pht_app.ui.header import render_header
from pht_app.panels.timeline import render_timeline_panel
from pht_app.panels.per_sector import render_per_sector_panel
from pht_app.panels.phasefold import render_phasefold_panel
from pht_app.panels.periodogram import render_periodogram_panel
from pht_app.panels.single_transit import render_single_transit_panel
from pht_app.panels.fp_and_tpf import render_fp_diagnostics_panel, render_tpf_centroid_panel
from pht_app.panels.eb_style_view import render_eb_style_panel

st.set_page_config(page_title="PHT Candidate Validator", page_icon="🪐", layout="wide",
                    initial_sidebar_state="expanded")

# Extra breathing room between panel sections — Streamlit's default spacing
# packs consecutive elements (subheaders, charts, dividers) tightly together.
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] > div:has(> div.pht-section-gap) {
    margin-top: 2.5rem;
}
div[data-testid="stPlotlyChart"] {
    margin-bottom: 1.25rem;
}
h3 {
    margin-top: 1rem !important;
    padding-top: 0.5rem;
}
hr {
    margin: 2rem 0 !important;
}
</style>
""", unsafe_allow_html=True)


def _section_gap():
    """Visible vertical gap between major panel sections (beyond a plain divider)."""
    st.markdown('<div class="pht-section-gap" style="height:1.5rem;"></div>', unsafe_allow_html=True)


init_session_state(st)

# --------------------------------------------------------------------------
# Sidebar + action handling
# --------------------------------------------------------------------------
actions = render_sidebar()

if actions["search_clicked"] and actions["tic_input"].strip():
    st.session_state.tic_id = actions["tic_input"].strip()
    with st.spinner("Resolving TIC catalog parameters..."):
        st.session_state.stellar_params = resolve_tic(st.session_state.tic_id)
    with st.spinner("Cross-referencing ExoFOP..."):
        st.session_state.exofop_flags = query_exofop(st.session_state.tic_id)
    with st.spinner("Querying MAST for available sectors..."):
        st.session_state.sector_list = search_available_sectors(st.session_state.tic_id)
    # Reset anything tied to the previous target
    st.session_state.lc_collection = None
    st.session_state.stitched_lc = None
    st.session_state.timeline_xrange = None
    st.session_state.bls_result = None
    st.session_state.ls_result = None
    st.session_state.fold_period = None
    st.session_state.fold_epoch = None

if actions["load_clicked"]:
    if not actions["selected_sectors"]:
        st.warning("Select at least one sector.")
    else:
        with st.spinner(f"Downloading and stitching {len(actions['selected_sectors'])} sector(s)..."):
            stitched, per_sector, logs = download_and_stitch(
                st.session_state.tic_id, actions["selected_sectors"], st.session_state.data_source
            )
        st.session_state.stitched_lc = stitched
        st.session_state.lc_collection = per_sector
        st.session_state.selected_sectors = actions["selected_sectors"]
        st.session_state.timeline_xrange = None
        st.session_state.bls_result = None
        st.session_state.ls_result = None
        for msg in logs:
            st.toast(msg)

        # Automatically find the period right away (BLS over a broad default
        # range) so a period is available immediately — no need to open the
        # Periodogram panel and click "Recompute" just to see it.
        if stitched is not None:
            with st.spinner("Automatically searching for the period (BLS)..."):
                auto_bls = run_bls(stitched, min_period=0.5, max_period=20.0)
            if auto_bls is not None:
                st.session_state.bls_result = auto_bls
                st.session_state.fold_period = auto_bls["best_period"]
                st.session_state.fold_epoch = auto_bls["best_t0"]

if actions["add_mask"]:
    st.session_state.signal_masks.append({
        "period": actions["mask_period"],
        "epoch": actions["mask_epoch"],
        "duration_days": 0.2,
    })
    st.session_state.bls_result = None
    st.session_state.ls_result = None
    st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
render_header()

# Prominent, always-visible detected-period readout — computed automatically
# as soon as sectors are loaded (see the auto-BLS call above), so the period
# is immediately visible here without needing to open the Periodogram panel.
if st.session_state.bls_result is not None:
    bls = st.session_state.bls_result
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("🔎 Detected Period", f"{bls['best_period']:.4f} d")
    pc2.metric("Epoch T0", f"{bls['best_t0']:.4f} BTJD")
    pc3.metric("Duration", f"{bls['best_duration']*24:.2f} h")
    pc4.metric("Depth", f"{bls['best_depth']*100:.3f} %")
    st.caption(
        "Automatically detected via BLS as soon as sectors were loaded. "
        "Fine-tune it (or search a different range) in the Periodogram panel below."
    )
elif st.session_state.stitched_lc is not None:
    st.caption("Automatic period search did not converge on this baseline — try adjusting the range in the Periodogram panel below.")

st.divider()

# --------------------------------------------------------------------------
# Synchronized dashboard — laid out so more than one panel is visible at
# once instead of a single long vertical scroll:
#   Row 1: Timeline (full width — the shared zoom/selection source)
#   Row 2: Phase-fold | Periodogram, side by side
# --------------------------------------------------------------------------
render_timeline_panel()
_section_gap()

fold_col, periodogram_col = st.columns(2)
with fold_col:
    render_phasefold_panel()
with periodogram_col:
    render_periodogram_panel()

st.divider()
_section_gap()

# --------------------------------------------------------------------------
# Per-sector breakdown — each sector plotted separately, stacked vertically,
# so individual sectors can be compared without zooming into the merged
# stitched timeline above.
# --------------------------------------------------------------------------
render_per_sector_panel()

st.divider()
_section_gap()

# --------------------------------------------------------------------------
# Analysis tools — separated into their own tabs so each is a clean, fully
# visible panel rather than everything stacked one below the other.
# --------------------------------------------------------------------------
tab_single_transit, tab_fp, tab_tpf, tab_eb_style = st.tabs([
    "🎯 Single-Transit Estimator",
    "🕵️ False Positive Diagnostics",
    "📍 TPF Centroid Check",
    "🌑 EB Signal View",
])

with tab_single_transit:
    render_single_transit_panel()

with tab_fp:
    render_fp_diagnostics_panel()

with tab_tpf:
    render_tpf_centroid_panel()

with tab_eb_style:
    render_eb_style_panel()

st.divider()
st.caption(
    "PHT Candidate Validator — all 4 build steps complete: multi-sector download/stitch, "
    "synchronized dashboard, deterministic Keplerian single-transit estimator with "
    "signal masking, and False Positive diagnostics with TPF centroid vetting."
)
