"""Sidebar: TIC search bar, sector selector, data-source toggle, export buttons."""

import streamlit as st

from pht_app.config import SPOC_2MIN_LABEL, FFI_QLP_LABEL


def render_sidebar():
    """
    Render the sidebar and return a dict of user actions taken this run:
        {
            "search_clicked": bool,
            "tic_input": str,
            "load_clicked": bool,
            "selected_sectors": list[int],
            "export_csv": bool,
            "export_pdf": bool,
        }
    """
    with st.sidebar:
        st.header("🔭 Target Search")

        tic_input = st.text_input(
            "TIC ID",
            placeholder="e.g. 260647166 or TIC 260647166",
            value=st.session_state.tic_id or "",
        )
        search_clicked = st.button("Search Target", type="primary", use_container_width=True)

        st.divider()

        st.subheader("Data Source")
        st.session_state.data_source = st.radio(
            "Preferred cadence",
            options=[SPOC_2MIN_LABEL, FFI_QLP_LABEL],
            index=0 if st.session_state.data_source == SPOC_2MIN_LABEL else 1,
            help="Falls back automatically if the preferred source is unavailable for a sector.",
        )

        st.subheader("Sectors")
        sector_checkboxes = {}
        if st.session_state.sector_list:
            n_sectors = len(st.session_state.sector_list)
            st.caption(f"{n_sectors} sector(s) available")

            sel_a, sel_b = st.columns(2)
            with sel_a:
                select_all = st.button("Select all", use_container_width=True)
            with sel_b:
                select_none = st.button("Select none", use_container_width=True)

            # Initialize / bulk-update the underlying checkbox state before the
            # widgets are instantiated below, so the click takes effect this run.
            for s in st.session_state.sector_list:
                key = f"sector_{s['sector']}"
                if select_all:
                    st.session_state[key] = True
                elif select_none:
                    st.session_state[key] = False
                elif key not in st.session_state:
                    st.session_state[key] = True

            # Compact grid (3 per row) inside a scrollable container so many
            # sectors are visible at once instead of one-per-row scrolling.
            with st.container(height=260, border=True):
                n_cols = 3
                cols = st.columns(n_cols)
                for i, s in enumerate(st.session_state.sector_list):
                    tags = []
                    if s["has_2min_spoc"]:
                        tags.append("2m")
                    if s["has_ffi_fallback"]:
                        tags.append("FFI")
                    tag_str = "/".join(tags) if tags else "?"
                    key = f"sector_{s['sector']}"
                    with cols[i % n_cols]:
                        sector_checkboxes[s["sector"]] = st.checkbox(
                            f"S{s['sector']}",
                            key=key,
                            help=f"Sector {s['sector']} — {tag_str} — authors: {', '.join(s['authors'])}",
                        )

            n_selected = sum(sector_checkboxes.values())
            st.caption(f"{n_selected} of {n_sectors} selected")
        else:
            st.caption("Search a target to list available sectors.")

        load_clicked = False
        if st.session_state.sector_list:
            load_clicked = st.button("Download & Stitch Selected Sectors", use_container_width=True)

        st.divider()
        st.subheader("Signal Masking")
        mask_period = st.number_input("Mask known period (days)", min_value=0.0, value=0.0, step=0.1,
                                       help="Enter a known planet's period to remove it from Panel 3's search.")
        mask_epoch = st.number_input("Mask epoch T0 (BTJD)", value=0.0, step=0.1)
        add_mask = st.button("Add Mask", use_container_width=True, disabled=mask_period <= 0)

        if st.session_state.signal_masks:
            st.caption(f"{len(st.session_state.signal_masks)} active mask(s):")
            for i, m in enumerate(st.session_state.signal_masks):
                st.caption(f"  • P={m['period']:.4f}d, T0={m['epoch']:.4f}")
            if st.button("Clear all masks", use_container_width=True):
                st.session_state.signal_masks = []
                st.rerun()

        st.divider()
        st.subheader("Export")
        col_a, col_b = st.columns(2)
        with col_a:
            export_csv = st.button("⬇ CSV", use_container_width=True,
                                    disabled=st.session_state.stitched_lc is None)
        with col_b:
            export_pdf = st.button("⬇ PDF", use_container_width=True,
                                    disabled=st.session_state.stitched_lc is None)

    return {
        "search_clicked": search_clicked,
        "tic_input": tic_input,
        "load_clicked": load_clicked,
        "selected_sectors": [s for s, checked in sector_checkboxes.items() if checked],
        "add_mask": add_mask,
        "mask_period": mask_period,
        "mask_epoch": mask_epoch,
        "export_csv": export_csv,
        "export_pdf": export_pdf,
    }
