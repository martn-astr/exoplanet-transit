"""
PHT-style Phase-Folded EB Signal view.

Replicates the Planet Hunters TESS website's own vetting plot: dark theme,
phase expressed as hours from the folded center (rather than a phase
fraction), a smoothed trend line through the scatter, and a mini
navigator/overview strip beneath the main plot (Plotly's built-in
xaxis range-slider gives exactly that).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pht_app.data.analysis import phase_fold


def _smoothed_trend(phase_hours, flux, frac=0.02, min_window=15):
    """
    Rolling-median smooth curve through the phase-sorted flux, as a simple
    stand-in for the LOWESS-style trend line on the reference plot.
    """
    order = np.argsort(phase_hours)
    x_sorted = phase_hours[order]
    y_sorted = flux[order]

    window = max(min_window, int(len(y_sorted) * frac))
    if window % 2 == 0:
        window += 1

    series = pd.Series(y_sorted)
    smoothed = series.rolling(window=window, center=True, min_periods=max(3, window // 4)).median()
    return x_sorted, smoothed.to_numpy()


def render_eb_style_panel():
    st.subheader("🌑 Phase-Folded EB Signal View")
    st.caption("PHT-style dark vetting plot: phase in hours from center, smoothed trend, mini navigator strip below.")

    lc = st.session_state.stitched_lc
    period = st.session_state.fold_period
    epoch = st.session_state.fold_epoch
    tic_id = st.session_state.tic_id

    if lc is None:
        st.caption("Load a stitched light curve first.")
        return
    if not period or not epoch:
        st.caption("Set a period/epoch in Panel 2 (or via the single-transit estimator / BLS) first.")
        return

    phase, flux, _ = phase_fold(lc, period, epoch)
    phase_hours = phase * period * 24.0

    x_smooth, y_smooth = _smoothed_trend(phase_hours, flux)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=phase_hours, y=flux, mode="markers",
        marker=dict(size=4, color="#2ecc71", opacity=0.75, line=dict(width=0)),
        name="Flux",
    ))
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_smooth, mode="lines",
        line=dict(color="#aef7e0", width=2.5),
        name="Smoothed trend",
    ))

    title = f"Phase Folded EB Signal for {tic_id} (Star {tic_id})"

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color="orange", size=14)),
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(
            title="Time From Phased Center (Hrs.)",
            rangeslider=dict(visible=True, bgcolor="black", thickness=0.12),
            gridcolor="#333333",
        ),
        yaxis=dict(title="Normalized Flux", gridcolor="#333333"),
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key="eb_style_chart")
    st.caption(
        "Drag the handles on the navigator strip at the bottom to zoom into any part of the phase — "
        "this mirrors the PHT website's own EB-signal vetting view."
    )
