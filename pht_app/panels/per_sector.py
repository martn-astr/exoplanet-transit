"""Per-sector light curves — each sector plotted in its own stacked subplot,
so sectors can be visually compared without zooming into the merged stitch."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def _sector_flux(lc, flux_column):
    """Same NA-safe column pull used by the main timeline panel."""
    col_lower = flux_column.lower()
    fallback = lc.flux.value
    if col_lower not in lc.colnames:
        return fallback
    raw = lc[col_lower]
    vals = np.asarray(raw.value if hasattr(raw, "value") else raw, dtype=float)
    missing = ~np.isfinite(vals)
    if missing.any():
        vals = vals.copy()
        vals[missing] = fallback[missing]
    return vals


def render_per_sector_panel():
    st.subheader("🗂️ Per-Sector Light Curves")
    st.caption("Each downloaded sector shown separately in a compact grid — no zooming required to tell sectors apart.")

    collection = st.session_state.lc_collection
    if not collection:
        st.caption("Load sectors from the sidebar to see them broken out individually.")
        return

    sectors = sorted(collection.keys())
    n = len(sectors)
    flux_col = st.session_state.flux_column or "flux"

    n_cols = st.radio(
        "Sectors per row", options=[2, 3, 4], index=1 if n > 4 else 0,
        horizontal=True, key="per_sector_cols",
    )
    n_rows = int(np.ceil(n / n_cols))
    row_height = 240

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=min(0.5 / max(n_rows, 1), 0.12),
        horizontal_spacing=0.06,
        subplot_titles=[f"Sector {s}" for s in sectors],
    )

    for i, sector in enumerate(sectors):
        row = i // n_cols + 1
        col = i % n_cols + 1
        lc = collection[sector]
        time_vals = lc.time.value
        flux_vals = _sector_flux(lc, flux_col)
        fig.add_trace(
            go.Scattergl(
                x=time_vals, y=flux_vals, mode="markers",
                marker=dict(size=2.5, opacity=0.6, color="steelblue"),
                name=f"Sector {sector}", showlegend=False,
            ),
            row=row, col=col,
        )
        # Only label axes on the outer edges to save space in the compact grid.
        if col == 1:
            fig.update_yaxes(title_text="Flux", row=row, col=col)
        if row == n_rows:
            fig.update_xaxes(title_text="Time (BTJD)", row=row, col=col)

    fig.update_layout(
        height=row_height * n_rows,
        margin=dict(l=10, r=10, t=40, b=10),
        title_text=f"{n} sector(s) — {flux_col}",
        font=dict(size=10),
    )
    fig.update_annotations(font_size=11)  # subplot titles

    st.plotly_chart(fig, use_container_width=True, key="per_sector_chart")
