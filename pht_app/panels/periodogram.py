"""Panel 3 (bottom): BLS / Lomb-Scargle periodogram over the current timeline window."""

import plotly.graph_objects as go
import streamlit as st

from pht_app.data.analysis import window_lightcurve, run_bls, run_lomb_scargle
from pht_app.data.masking import masked_lightcurve


def render_periodogram_panel():
    st.subheader("📊 Panel 3 — Periodogram")

    lc = st.session_state.stitched_lc
    if lc is None:
        st.caption("Load a stitched light curve to compute a periodogram.")
        return

    windowed = window_lightcurve(lc, st.session_state.timeline_xrange)

    if st.session_state.signal_masks:
        windowed = masked_lightcurve(windowed, st.session_state.signal_masks)
        st.caption(f"🎭 {len(st.session_state.signal_masks)} known signal(s) masked out for this search.")

    if st.session_state.timeline_xrange:
        st.caption(
            f"Computed over zoomed window: "
            f"{st.session_state.timeline_xrange[0]:.2f} – {st.session_state.timeline_xrange[1]:.2f} BTJD "
            f"({len(windowed)} points). Reset zoom in Panel 1 to use the full baseline."
        )
    else:
        st.caption(f"Computed over full stitched baseline ({len(windowed)} points).")

    c1, c2, c3 = st.columns(3)
    method = c1.radio("Method", options=["BLS", "Lomb-Scargle"], horizontal=True,
                       index=0 if st.session_state.periodogram_method == "BLS" else 1)
    st.session_state.periodogram_method = method
    min_p = c2.number_input("Min period (days)", min_value=0.05, value=0.5, step=0.1)
    max_p = c3.number_input("Max period (days)", min_value=0.1, value=20.0, step=0.5)

    # Guard against an invalid period range instead of letting np.linspace /
    # BoxLeastSquares crash on it.
    if max_p <= min_p:
        st.warning(
            f"Max period ({max_p:g} d) must be greater than min period ({min_p:g} d) — "
            f"adjusted max period to {min_p + 1:g} d for this run."
        )
        max_p = min_p + 1.0

    recompute = st.button("Recompute Periodogram", type="primary")

    result_key = "bls_result" if method == "BLS" else "ls_result"

    if recompute or st.session_state[result_key] is None:
        with st.spinner(f"Running {method}..."):
            if method == "BLS":
                st.session_state.bls_result = run_bls(windowed, min_period=min_p, max_period=max_p)
            else:
                st.session_state.ls_result = run_lomb_scargle(windowed, min_period=min_p, max_period=max_p)

    result = st.session_state[result_key]
    if result is None:
        st.warning("Not enough data points in the current window to compute a periodogram.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["periods"], y=result["power"], mode="lines", line=dict(width=1.3)))
    fig.add_vline(x=result["best_period"], line_dash="dash", line_color="red",
                  annotation_text=f"P = {result['best_period']:.4f} d")
    fig.update_layout(
        xaxis_title="Period (days)",
        yaxis_title="Power",
        xaxis_type="log",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="periodogram_chart")

    if method == "BLS":
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Period (d)", f"{result['best_period']:.4f}")
        c2.metric("T0 (BTJD)", f"{result['best_t0']:.4f}")
        c3.metric("Depth", f"{result['best_depth']:.5f}")
        if st.button("Use this period/epoch for Panel 2 fold"):
            st.session_state.fold_period = result["best_period"]
            st.session_state.fold_epoch = result["best_t0"]
            st.rerun()
    else:
        st.metric("Best Period (d)", f"{result['best_period']:.4f}")
