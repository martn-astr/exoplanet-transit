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
    pht_app/panels/phasefold.py   Panel 2 — phase-folded view
    pht_app/panels/periodogram.py Panel 3 — BLS / Lomb-Scargle
"""

import streamlit as st

from pht_app.config import init_session_state
from pht_app.data import resolve_tic, query_exofop, search_available_sectors, download_and_stitch
from pht_app.ui.sidebar import render_sidebar
from pht_app.ui.header import render_header
from pht_app.panels.timeline import render_timeline_panel
from pht_app.panels.phasefold import render_phasefold_panel
from pht_app.panels.periodogram import render_periodogram_panel
from pht_app.panels.single_transit import render_single_transit_panel
from pht_app.panels.fp_and_tpf import render_fp_diagnostics_panel, render_tpf_centroid_panel

st.set_page_config(page_title="PHT Candidate Validator", page_icon="🪐", layout="wide",
                    initial_sidebar_state="expanded")

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
st.divider()

# --------------------------------------------------------------------------
# Synchronized dashboard — laid out so more than one panel is visible at
# once instead of a single long vertical scroll:
#   Row 1: Timeline (full width — the shared zoom/selection source)
#   Row 2: Phase-fold | Periodogram, side by side
# --------------------------------------------------------------------------
render_timeline_panel()

fold_col, periodogram_col = st.columns(2)
with fold_col:
    render_phasefold_panel()
with periodogram_col:
    render_periodogram_panel()

st.divider()

# --------------------------------------------------------------------------
# Analysis tools — separated into their own tabs so each is a clean, fully
# visible panel rather than everything stacked one below the other.
# --------------------------------------------------------------------------
tab_single_transit, tab_fp, tab_tpf = st.tabs([
    "🎯 Single-Transit Estimator",
    "🕵️ False Positive Diagnostics",
    "📍 TPF Centroid Check",
])

with tab_single_transit:
    render_single_transit_panel()

with tab_fp:
    render_fp_diagnostics_panel()

with tab_tpf:
    render_tpf_centroid_panel()

st.divider()
st.caption(
    "PHT Candidate Validator — all 4 build steps complete: multi-sector download/stitch, "
    "synchronized dashboard, deterministic Keplerian single-transit estimator with "
    "signal masking, and False Positive diagnostics with TPF centroid vetting."
)
