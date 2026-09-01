"""Panel 2 (middle): light curve folded on a period and epoch (T0)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pht_app.data.analysis import phase_fold


def render_phasefold_panel():
    st.subheader("🔁 Panel 2 — Phase-Folded View")

    lc = st.session_state.stitched_lc
    if lc is None:
        st.caption("Load a stitched light curve to fold it on a period.")
        return

    default_period = st.session_state.fold_period or (
        st.session_state.bls_result["best_period"] if st.session_state.bls_result else 3.0
    )
    default_epoch = st.session_state.fold_epoch or (
        st.session_state.bls_result["best_t0"] if st.session_state.bls_result else float(lc.time.value[0])
    )

    c1, c2, c3 = st.columns(3)
    period = c1.number_input("Period (days)", min_value=0.001, value=float(default_period), step=0.01, format="%.4f")
    epoch = c2.number_input("Epoch T0 (BTJD)", value=float(default_epoch), step=0.01, format="%.4f")
    bin_toggle = c3.checkbox("Show binned overlay", value=True)

    st.session_state.fold_period = period
    st.session_state.fold_epoch = epoch

    phase, flux, flux_err = phase_fold(lc, period, epoch)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=phase, y=flux, mode="markers",
        marker=dict(size=3, opacity=0.4, color="steelblue"),
        name="Data",
    ))

    if bin_toggle and len(phase) > 20:
        n_bins = 100
        bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if mask.any():
                bin_means[i] = np.nanmean(flux[mask])
        fig.add_trace(go.Scatter(
            x=bin_centers, y=bin_means, mode="lines+markers",
            line=dict(color="darkorange", width=2),
            marker=dict(size=5),
            name="Binned",
        ))

    fig.update_layout(
        xaxis_title="Phase",
        yaxis_title="Normalized Flux",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="phasefold_chart")
