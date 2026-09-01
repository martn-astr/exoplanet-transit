"""Panel 1 (top): full stitched multi-sector light curve, SAP/PDCSAP toggle."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pht_app.config import FLUX_COLUMNS


def _flux_series(lc, flux_column):
    """Pull the requested flux column off a stitched light curve, falling back to default flux if missing."""
    if hasattr(lc, flux_column):
        col = getattr(lc, flux_column)
        return col.value
    return lc.flux.value


def render_timeline_panel():
    st.subheader("📈 Panel 1 — Stitched Timeline")

    lc = st.session_state.stitched_lc
    if lc is None:
        st.caption("Load a stitched light curve from the sidebar to see the timeline.")
        return

    col1, col2 = st.columns([3, 1])
    with col2:
        flux_col = st.selectbox(
            "Flux column",
            options=FLUX_COLUMNS,
            index=FLUX_COLUMNS.index(st.session_state.flux_column)
            if st.session_state.flux_column in FLUX_COLUMNS else 0,
            help="PDCSAP = systematics-removed. SAP = raw aperture photometry.",
        )
        st.session_state.flux_column = flux_col

    time_vals = lc.time.value
    flux_vals = _flux_series(lc, flux_col)

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
