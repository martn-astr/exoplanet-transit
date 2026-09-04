"""False Positive diagnostics + TPF spatial centroid checker (Step 4)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pht_app.data.fp_diagnostics import (
    run_all_diagnostics, odd_even_folded_curves, secondary_zoom_curves,
)
from pht_app.data.tpf_centroid import (
    download_tpf, build_difference_image, query_gaia_sources,
    centroid_shift_estimate, centroid_offset_arcsec,
)


def _verdict_badge(likely_eb):
    return "🔴 FP indicator" if likely_eb else "🟢 Passes"


def _render_odd_even_plot(lc, period, epoch, duration_days):
    curves = odd_even_folded_curves(lc, period, epoch, duration_days)
    if curves["status"] != "ok":
        st.caption("Not enough odd/even transits in this baseline to plot a comparison.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curves["phase_hours"], y=curves["odd_flux"],
        mode="lines+markers", name="Odd transits",
        line=dict(color="steelblue", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=curves["phase_hours"], y=curves["even_flux"],
        mode="lines+markers", name="Even transits",
        line=dict(color="darkorange", width=2), marker=dict(size=4),
    ))
    fig.update_layout(
        title="Odd vs Even Transits (folded & binned)",
        xaxis_title="Phase (hours)", yaxis_title="Relative flux",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True, key="odd_even_plot")
    st.caption("Mismatched depths between the two curves are the classic eclipsing-binary signature "
               "(a blended EB at half the true period).")


def _render_secondary_zoom_plot(lc, period, epoch, duration_days):
    curves = secondary_zoom_curves(lc, period, epoch, duration_days)
    if curves["status"] != "ok":
        st.caption("Not enough phase coverage to plot the primary/secondary comparison.")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Primary (phase 0)", "Secondary (phase 0.5)"))
    fig.add_trace(go.Scatter(
        x=curves["primary"]["phase"], y=curves["primary"]["flux"],
        mode="lines+markers", line=dict(color="steelblue", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=curves["secondary"]["phase"], y=curves["secondary"]["flux"],
        mode="lines+markers", line=dict(color="firebrick", width=2), marker=dict(size=4),
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Phase", row=1, col=1)
    fig.update_xaxes(title_text="Phase", row=1, col=2)
    fig.update_yaxes(title_text="Relative flux", row=1, col=1)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                       title_text="Primary vs Secondary Eclipse (zoomed, same flux scale)")
    # Match y-axis ranges across both subplots so depth is visually comparable.
    all_flux = np.concatenate([curves["primary"]["flux"], curves["secondary"]["flux"]])
    finite = all_flux[np.isfinite(all_flux)]
    if finite.size:
        y_pad = 0.1 * (finite.max() - finite.min() + 1e-9)
        y_range = [finite.min() - y_pad, finite.max() + y_pad]
        fig.update_yaxes(range=y_range, row=1, col=1)
        fig.update_yaxes(range=y_range, row=1, col=2)

    st.plotly_chart(fig, use_container_width=True, key="secondary_zoom_plot")
    st.caption("A visible dip in the right-hand (phase 0.5) panel at comparable depth to the left "
               "is a secondary-eclipse detection — a strong eclipsing-binary indicator.")


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

    # Default to the BLS-fitted transit duration when available, rather than
    # an arbitrary fixed guess — using the wrong window width is the main
    # reason these tests can silently miss a real eclipsing binary signal
    # (too narrow a window catches only baseline, not the actual eclipse).
    bls = st.session_state.bls_result
    default_duration_hours = bls["best_duration"] * 24.0 if bls else 3.0

    duration_days = st.number_input(
        "Assumed transit duration for diagnostics (hours)",
        min_value=0.1, value=float(default_duration_hours), step=0.1,
        help="Defaults to the BLS-fitted duration when a periodogram has been run. "
             "If this doesn't match the actual eclipse width, the tests below can miss a real signal.",
    ) / 24.0

    if st.button("Run False Positive Diagnostics", type="primary"):
        with st.spinner("Running odd/even, shape, and secondary-eclipse checks..."):
            st.session_state.fp_diagnostics_result = run_all_diagnostics(lc, period, epoch, duration_days)

    result = st.session_state.fp_diagnostics_result
    if result is None:
        return

    n_inconclusive = sum(
        1 for d in (result["odd_even"], result["shape"], result["secondary_eclipse"])
        if d.get("status") != "ok"
    )
    if n_inconclusive:
        st.warning(
            f"⚠ {n_inconclusive} of 3 diagnostic(s) could not run (insufficient data in the assumed "
            f"transit window) and are **not** reflected in the verdict below — a clean verdict does not "
            f"mean those specific checks passed. Try adjusting the assumed transit duration above, "
            f"or confirm the period/epoch are accurate."
        )

    st.markdown(f"**Verdict:** {result['verdict']}")

    suggestion = result.get("period_suggestion")
    if suggestion:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            st.error(
                f"⚠ Incorrect period suspected — {suggestion['reason']}.\n\n"
                f"**Period: {suggestion['label']}**"
            )
        with sc2:
            st.write("")  # vertical spacer to align button with the message box
            if st.button("Use corrected period", use_container_width=True):
                st.session_state.fold_period = suggestion["corrected_period"]
                st.session_state.fp_diagnostics_result = None
                st.rerun()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Odd vs Even Depth**")
        oe = result["odd_even"]
        if oe["status"] == "ok":
            st.write(_verdict_badge(oe["likely_eb"]))
            st.caption(oe["message"])
            st.write(f"Odd depth: {oe['odd_depth']:.5f}")
            st.write(f"Even depth: {oe['even_depth']:.5f}")
            st.write(f"Mismatch: {oe['sigma_diff']:.1f}σ ({oe['relative_mismatch']*100:.0f}% relative)")
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
            st.write(f"Depth: {se['secondary_depth']:.5f} ({se['relative_depth']*100:.0f}% of primary)")
            st.write(f"Significance: {se['significance_sigma']:.1f}σ")
        else:
            st.caption(se["message"])

    st.divider()

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        _render_odd_even_plot(lc, period, epoch, duration_days)
    with plot_col2:
        _render_secondary_zoom_plot(lc, period, epoch, duration_days)


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

    t0 = st.session_state.fold_epoch or st.session_state.click_t0 or (
        st.session_state.bls_result["best_t0"] if st.session_state.bls_result else None
    )
    # Prefer the BLS-fitted duration (from the graph/periodogram) over a
    # manually clicked or guessed value, so this stays in sync with whatever
    # signal the periodogram actually found.
    if st.session_state.bls_result:
        duration_hours = st.session_state.bls_result["best_duration"] * 24.0
    elif st.session_state.click_t14_hours:
        duration_hours = st.session_state.click_t14_hours
    else:
        duration_hours = 3.0
    duration_days = duration_hours / 24.0

    st.caption(
        f"Using T0 = {t0:.4f} BTJD, duration = {duration_hours:.2f} h "
        f"({'from BLS fit' if st.session_state.bls_result else 'from manual entry'})."
        if t0 is not None else ""
    )

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

    img_col, offset_col = st.columns(2)

    with img_col:
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
                    f"— a centroid sitting on a neighboring source (yellow circle) rather than the target "
                    f"indicates the eclipse likely originates from a blended background star."
                )
            else:
                st.write(f"**Difference-image centroid (pixel coords):** x={centroid['x']:.2f}, y={centroid['y']:.2f}")

        if gaia is None:
            st.caption("Gaia DR3 query failed or returned no results (network-restricted environments may block this).")
        elif len(gaia) == 0:
            st.caption("No Gaia DR3 sources found within the search radius.")

    with offset_col:
        st.markdown("**TIC Position Centroid Offset**")
        sp = st.session_state.stellar_params
        centroid = st.session_state.centroid_result
        offset = centroid_offset_arcsec(centroid, sp.get("ra"), sp.get("dec")) if sp else None

        if offset is None:
            st.caption("Offset requires a resolved WCS centroid and a valid target RA/Dec — "
                       "run the difference image above first.")
        else:
            fig2 = go.Figure()
            # Target position at the origin, matching the DV-report convention.
            fig2.add_trace(go.Scatter(
                x=[0], y=[0], mode="markers",
                marker=dict(symbol="star", size=16, color="black"),
                name="TIC catalog position",
            ))
            fig2.add_trace(go.Scatter(
                x=[offset["d_ra_arcsec"]], y=[offset["d_dec_arcsec"]],
                mode="markers", marker=dict(symbol="x", size=14, color="crimson", line=dict(width=2)),
                name="Difference-image centroid",
            ))
            # A rough 1-pixel (~21") reference circle, TESS's approximate pixel scale,
            # as a visual sense of scale rather than a formal uncertainty ellipse.
            theta = np.linspace(0, 2 * np.pi, 100)
            pixel_scale_arcsec = 21.0
            fig2.add_trace(go.Scatter(
                x=pixel_scale_arcsec * np.cos(theta), y=pixel_scale_arcsec * np.sin(theta),
                mode="lines", line=dict(color="gray", dash="dot"),
                name="~1 TESS pixel (21″)",
            ))
            fig2.update_layout(
                xaxis_title="ΔRA (arcsec)", yaxis_title="ΔDec (arcsec)",
                height=420, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(scaleanchor="x"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True, key="centroid_offset_plot")
            st.metric("Total offset", f"{offset['offset_arcsec']:.2f}″")
            if offset["offset_arcsec"] > pixel_scale_arcsec:
                st.warning("Centroid offset exceeds ~1 TESS pixel — signal may originate from a "
                           "different (blended) source than the target.")
            else:
                st.success("Centroid offset is within ~1 TESS pixel of the target — consistent "
                            "with the signal originating on-target.")
