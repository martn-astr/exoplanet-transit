"""Panel 1 (top): full stitched multi-sector light curve, SAP/PDCSAP toggle."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pht_app.config import FLUX_COLUMNS


def _flux_series(lc, flux_column):
    """
    Pull the requested flux column off a stitched light curve, falling back
    to the default `flux` column point-by-point wherever the requested
    column is missing/NaN.

    This matters because FFI/QLP-derived sectors often don't carry a
    separate sap_flux/pdcsap_flux column at all — after stitching, those
    sectors' rows are simply NaN for that column. Without backfilling,
    selecting PDCSAP_FLUX on a mixed SPOC+FFI target would only plot the
    SPOC sector's points and silently drop the rest of the baseline.

    Note: lightkurve's PDCSAP_FLUX/SAP_FLUX convenience properties raise
    KeyError (not AttributeError) when the underlying column is missing, so
    hasattr() is not a safe way to probe for them — check lc.colnames
    (lowercased) directly instead.

    Returns (flux_values, n_backfilled).
    """
    col_lower = flux_column.lower()
    flux_fallback = lc.flux.value

    if col_lower not in lc.colnames:
        return flux_fallback, len(flux_fallback)

    raw = lc[col_lower]
    vals = np.asarray(raw.value if hasattr(raw, "value") else raw, dtype=float)

    missing = ~np.isfinite(vals)
    n_backfilled = int(missing.sum())
    if n_backfilled:
        vals = vals.copy()
        vals[missing] = flux_fallback[missing]

    return vals, n_backfilled


def _available_flux_columns(lc, candidates):
    """Return only the flux columns actually present on this stitched light curve, plus a note if narrowed."""
    present = [c for c in candidates if c.lower() in lc.colnames]
    return present if present else ["flux"]


def render_timeline_panel():
    st.subheader("📈 Panel 1 — Stitched Timeline")

    lc = st.session_state.stitched_lc
    if lc is None:
        st.caption("Load a stitched light curve from the sidebar to see the timeline.")
        return

    available_cols = _available_flux_columns(lc, FLUX_COLUMNS)

    col1, col2 = st.columns([3, 1])
    with col2:
        flux_col = st.selectbox(
            "Flux column",
            options=available_cols,
            index=available_cols.index(st.session_state.flux_column)
            if st.session_state.flux_column in available_cols else 0,
            help="PDCSAP = systematics-removed. SAP = raw aperture photometry. "
                 "Only columns present in this stitched light curve are shown.",
        )
        st.session_state.flux_column = flux_col
        if len(available_cols) < len(FLUX_COLUMNS):
            st.caption("⚠ Some sectors (likely FFI/QLP) only provide a single flux column.")

    time_vals = lc.time.value
    flux_vals, n_backfilled = _flux_series(lc, flux_col)
    if n_backfilled:
        pct = 100.0 * n_backfilled / len(flux_vals)
        st.caption(
            f"ℹ {n_backfilled} point(s) ({pct:.0f}%) lack a {flux_col} value (likely FFI/QLP sectors) "
            f"and are filled in from the normalized `flux` column so the baseline stays continuous."
        )

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time_vals, y=flux_vals,
        mode="markers", marker=dict(size=3, opacity=0.6),
        name=flux_col,
    ))

    # Overlay predictive transit windows if a single-transit period estimate exists
    est = st.session_state.get("single_transit_estimate")
    if est and est.get("period"):
        t0 = est["t0"]
        period = est["period"]
        n_start = int(np.floor((time_vals.min() - t0) / period))
        n_end = int(np.ceil((time_vals.max() - t0) / period))
        for n in range(n_start, n_end + 1):
            center = t0 + n * period
            fig.add_vrect(
                x0=center - est["duration_days"] / 2,
                x1=center + est["duration_days"] / 2,
                fillcolor="orange", opacity=0.15, line_width=0,
            )

    fig.update_layout(
        xaxis_title="Time (BTJD)",
        yaxis_title="Normalized Flux",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="zoom",
    )

    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="timeline_chart")

    # Capture the user's zoom/selection so Panel 3 can recompute over that window.
    if event and event.get("selection", {}).get("box"):
        box = event["selection"]["box"][0]
        st.session_state.timeline_xrange = (box["x"][0], box["x"][1])

    reset_col, _ = st.columns([1, 4])
    with reset_col:
        if st.button("Reset zoom / use full range"):
            st.session_state.timeline_xrange = None
