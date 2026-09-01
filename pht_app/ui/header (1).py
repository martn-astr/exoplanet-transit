"""Header card: TIC stellar parameters, ExoFOP flags, sector/stitch status."""

import numpy as np
import pandas as pd
import streamlit as st

from pht_app.config import APP_TITLE, APP_CAPTION


def _metric(col, label, value, fmt):
    col.metric(label, fmt.format(value) if not (value is None or np.isnan(value)) else "—")


def render_header():
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    if st.session_state.tic_id is None:
        st.info("Enter a TIC ID in the sidebar and click **Search Target** to begin.")
        return

    sp = st.session_state.stellar_params
    st.subheader(f"TIC {st.session_state.tic_id}")

    if sp is None:
        st.error("Could not resolve this TIC ID in the MAST catalog. Check the ID and try again.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        _metric(c1, "R★ (R☉)", sp["R_star"], "{:.3f}")
        _metric(c2, "M★ (M☉)", sp["M_star"], "{:.3f}")
        _metric(c3, "Teff (K)", sp["Teff"], "{:.0f}")
        _metric(c4, "Tmag", sp["Tmag"], "{:.2f}")
        _metric(c5, "logg", sp["logg"], "{:.2f}")

    exo = st.session_state.exofop_flags
    if exo is not None:
        if exo["status"] == "found":
            st.warning(f"⚠ ExoFOP: {exo['message']}")
            with st.expander("View ExoFOP TOI entries"):
                st.dataframe(pd.DataFrame(exo["rows"]), use_container_width=True)
        elif exo["status"] == "not_found":
            st.success("✅ No existing ExoFOP TOI entry found for this target.")
        else:
            st.caption(f"ExoFOP lookup issue: {exo['message']}")

    st.divider()

    if st.session_state.sector_list:
        n_sectors = len(st.session_state.sector_list)
        n_2min = sum(1 for s in st.session_state.sector_list if s["has_2min_spoc"])
        st.write(
            f"**{n_sectors} sector(s) available** — {n_2min} with 2-min SPOC, "
            f"{n_sectors - n_2min} FFI-only. Select sectors in the sidebar and click "
            f"**Download & Stitch Selected Sectors**."
        )

    if st.session_state.stitched_lc is not None:
        lc = st.session_state.stitched_lc
        st.success(
            f"Stitched light curve ready: {len(lc)} points across "
            f"{len(st.session_state.selected_sectors)} sector(s), normalized to baseline flux = 1.0."
        )
    else:
        st.caption("No stitched light curve loaded yet.")
