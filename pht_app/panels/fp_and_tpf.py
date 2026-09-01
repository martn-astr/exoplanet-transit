"""False Positive diagnostics + TPF spatial centroid checker (Step 4)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pht_app.data.fp_diagnostics import run_all_diagnostics
from pht_app.data.tpf_centroid import (
    download_tpf, build_difference_image, query_gaia_sources, centroid_shift_estimate,
)


def _verdict_badge(likely_eb):
    return "🔴 FP indicator" if likely_eb else "🟢 Passes"


def render_fp_diagnostics_panel():
    st.subheader("🕵️ False Positive Diagnostics")

    lc = st.session_state.stitched_lc
    period = st.session_state.fold_period
    epoch = st.session_state.fold_epoch

    if lc is None:
        st.caption("Load a stitched light curve first.")
        return
    if not period or not epoch:
        st.caption("Set a period/epoch in Panel 2 (or via the single-transit estimator) before running diagnostics.")
        return

    duration_days = st.number_input(
        "Assumed transit duration for diagnostics (hours)",
        min_value=0.1, value=3.0, step=0.1,
    ) / 24.0

    if st.button("Run False Positive Diagnostics", type="primary"):
        with st.spinner("Running odd/even, shape, and secondary-eclipse checks..."):
            st.session_state.fp_diagnostics_result = run_all_diagnostics(lc, period, epoch, duration_days)

    result = st.session_state.fp_diagnostics_result
    if result is None:
        return

    st.markdown(f"**Verdict:** {result['verdict']}")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Odd vs Even Depth**")
        oe = result["odd_even"]
        if oe["status"] == "ok":
            st.write(_verdict_badge(oe["likely_eb"]))
            st.caption(oe["message"])
            st.write(f"Odd depth: {oe['odd_depth']:.5f}")
            st.write(f"Even depth: {oe['even_depth']:.5f}")
            st.write(f"Mismatch: {oe['sigma_diff']:.1f}σ")
        else:
            st.caption(oe["message"])

    with c2:
        st.markdown("**Transit Shape**")
        sh = result["shape"]
        if sh["status"] == "ok":
            st.write(_verdict_badge(sh["likely_eb"]))
            st.caption(sh["message"])
            if sh["flatness_ratio"] is not None:
                st.write(f"Flatness ratio: {sh['flatness_ratio']:.2f}")
        else:
            st.caption(sh["message"])

    with c3:
        st.markdown("**Secondary Eclipse**")
        se = result["secondary_eclipse"]
        if se["status"] == "ok":
            st.write(_verdict_badge(se["likely_eb"]))
            st.caption(se["message"])
            st.write(f"Depth: {se['secondary_depth']:.5f}")
            st.write(f"Significance: {se['significance_sigma']:.1f}σ")
        else:
            st.caption(se["message"])


def render_tpf_centroid_panel():
    st.subheader("📍 TPF Spatial Centroid Check")
    st.caption("Confirms the photometric centroid doesn't shift onto a background star during transit.")

    if st.session_state.tic_id is None or st.session_state.stellar_params is None:
        st.caption("Search a target first.")
        return
    if not st.session_state.sector_list:
        st.caption("No sectors available for this target.")
        return

    sector_options = [s["sector"] for s in st.session_state.sector_list]
    c1, c2 = st.columns(2)
    with c1:
        sector = st.selectbox("Sector for TPF", options=sector_options)
    with c2:
        launch = st.button("Load TPF & Compute Difference Image", type="primary")

    t0 = st.session_state.fold_epoch or st.session_state.click_t0
    duration_days = (st.session_state.click_t14_hours or 3.0) / 24.0

    if t0 is None:
        st.caption("Set a transit epoch (Panel 2 fold or single-transit estimator) first.")
        return

    if launch:
        with st.spinner(f"Downloading TPF for sector {sector}..."):
            tpf = download_tpf(st.session_state.stellar_params["TIC_ID"], sector)
        if tpf is None:
            st.error("No TPF available for this sector.")
            return
        st.session_state.tpf_data = tpf

        with st.spinner("Building in-transit minus out-of-transit difference image..."):
            diff = build_difference_image(tpf, t0, duration_days)
        if diff is None:
            st.error("Not enough in/out-of-transit cadences in this TPF window to build a difference image.")
            return
        st.session_state.tpf_diff_image = diff

        with st.spinner("Querying Gaia DR3 for nearby sources..."):
            gaia = query_gaia_sources(diff["ra"], diff["dec"])
        st.session_state.gaia_sources = gaia

        st.session_state.centroid_result = centroid_shift_estimate(diff["diff_image"], diff["wcs"])

    diff = st.session_state.tpf_diff_image
    if diff is None:
        return

    fig = go.Figure(data=go.Heatmap(z=diff["diff_image"], colorscale="RdBu_r", colorbar=dict(title="Δ Flux")))

    centroid = st.session_state.centroid_result
    if centroid:
        fig.add_trace(go.Scatter(
            x=[centroid["x"]], y=[centroid["y"]],
            mode="markers", marker=dict(symbol="x", size=14, color="lime", line=dict(width=2)),
            name="Difference centroid",
        ))

    gaia = st.session_state.gaia_sources
    if gaia is not None and len(gaia) > 0 and diff["wcs"] is not None:
        try:
            px, py = diff["wcs"].world_to_pixel_values(gaia["ra"].values, gaia["dec"].values)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers",
                marker=dict(symbol="circle-open", size=10, color="yellow", line=dict(width=1.5)),
                name="Gaia DR3 sources",
                text=[f"G={m:.2f}" for m in gaia["phot_g_mean_mag"].values],
            ))
        except Exception:
            pass

    fig.update_layout(
        title="Difference image (out-of-transit − in-transit)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(scaleanchor="x"),
    )
    st.plotly_chart(fig, use_container_width=True, key="tpf_diff_chart")

    if centroid:
        if centroid["ra"] is not None:
            st.write(
                f"**Difference-image centroid:** RA={centroid['ra']:.5f}°, Dec={centroid['dec']:.5f}° "
                f"— compare against the target's catalog position and any overlaid Gaia sources. "
                f"A centroid that sits on top of a neighboring source (yellow circle) rather than "
                f"the target indicates the eclipse likely originates from a blended background star."
            )
        else:
            st.write(f"**Difference-image centroid (pixel coords):** x={centroid['x']:.2f}, y={centroid['y']:.2f}")

    if gaia is None:
        st.caption("Gaia DR3 query failed or returned no results (network-restricted environments may block this).")
    elif len(gaia) == 0:
        st.caption("No Gaia DR3 sources found within the search radius.")
