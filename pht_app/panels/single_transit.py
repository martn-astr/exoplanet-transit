"""
Deterministic Single-Transit Estimator (Step 3).

No AI/ML — pure Keplerian orbital physics. The user marks a single transit's
epoch (T0) and duration (T14) manually (via numeric inputs, since Plotly
click-to-select coordinates aren't reliably available from Streamlit's
plotly_chart return value across versions); the app then computes the
maximum period assuming a circular, central-transit orbit and projects
predictive transit windows back onto Panel 1's timeline.
"""

import streamlit as st

from pht_app.data.single_transit import (
    estimate_max_period,
    period_vs_impact_parameter,
    FORMULA_LATEX,
)
import plotly.graph_objects as go


def render_single_transit_panel():
    st.subheader("🎯 Single-Transit Estimator (Deterministic Physics)")
    st.caption("No AI/ML — pure Keplerian orbital mechanics. For sectors where only one transit dip is visible.")

    lc = st.session_state.stitched_lc
    sp = st.session_state.stellar_params

    if lc is None:
        st.caption("Load a stitched light curve first.")
        return
    if sp is None or sp.get("M_star") is None:
        st.caption("Stellar mass/radius unavailable for this target — cannot run the estimator.")
        return

    st.latex(FORMULA_LATEX)

    c1, c2, c3 = st.columns(3)
    t0 = c1.number_input(
        "Transit epoch T0 (BTJD)",
        value=float(st.session_state.click_t0) if st.session_state.click_t0 else float(lc.time.value[0]),
        step=0.001, format="%.4f",
        help="Time of the single observed transit's minimum. Read this off Panel 1.",
    )
    t14 = c2.number_input(
        "Transit duration T14 (hours)",
        min_value=0.01,
        value=float(st.session_state.click_t14_hours) if st.session_state.click_t14_hours else 3.0,
        step=0.1,
        help="Full transit duration, ingress to egress, in hours. Read this off Panel 1.",
    )
    c3.metric("M★ (M☉)", f"{sp['M_star']:.3f}")

    compute = st.button("Compute Maximum Period", type="primary")

    if compute or st.session_state.single_transit_estimate:
        p_max = estimate_max_period(sp["M_star"], sp["R_star"], t14)
        if p_max is None:
            st.error("Could not compute — check stellar radius is valid and non-zero.")
            return

        duration_days = t14 / 24.0
        st.session_state.click_t0 = t0
        st.session_state.click_t14_hours = t14
        st.session_state.single_transit_estimate = {
            "t0": t0,
            "period": p_max,
            "duration_days": duration_days,
        }

        st.success(
            f"**Maximum period (circular, b=0): P_max = {p_max:.3f} days.** "
            f"This is an upper bound — the true period is ≤ P_max, shrinking toward 0 as the "
            f"impact parameter b → 1 (grazing)."
        )

        b_vals, p_vals = period_vs_impact_parameter(sp["M_star"], sp["R_star"], t14)
        if b_vals is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=b_vals, y=p_vals, mode="lines", line=dict(width=2)))
            fig.update_layout(
                xaxis_title="Impact parameter b",
                yaxis_title="Allowed period P (days)",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                title="Allowed period range as a function of impact parameter",
            )
            st.plotly_chart(fig, use_container_width=True, key="impact_param_chart")

        st.info(
            "Predictive transit windows (P_max) are now overlaid as shaded bands on "
            "**Panel 1's timeline** above — check whether any of them line up with other dips."
        )

        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("Use P_max for Panel 2 fold"):
                st.session_state.fold_period = p_max
                st.session_state.fold_epoch = t0
                st.rerun()
        with cta2:
            if st.button("Add this signal as a mask"):
                st.session_state.signal_masks.append({
                    "period": p_max, "epoch": t0, "duration_days": duration_days,
                })
                st.rerun()
