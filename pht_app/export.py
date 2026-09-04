"""
Export helpers: CSV of the stitched light curve, and a PDF summary report
(light curve, phase-fold, periodogram, stellar params, FP diagnostics
verdict) in the spirit of a TESS SPOC Data Validation report page.

Kept dependency-light (matplotlib only) and free of Streamlit imports so it
can be unit tested without a running app.
"""

import io
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend — required outside a GUI session
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_csv_bytes(lc) -> bytes:
    """
    Serialize a (stitched) light curve to CSV bytes: time, flux, and any of
    flux_err / sap_flux / pdcsap_flux / quality that are present.
    """
    time_vals = lc.time.value
    data = {"time_btjd": time_vals, "flux": lc.flux.value}

    for col in ("flux_err", "sap_flux", "pdcsap_flux", "quality"):
        if col in lc.colnames:
            raw = lc[col]
            data[col] = raw.value if hasattr(raw, "value") else np.asarray(raw)

    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _fmt(value, fmt="{:.3f}", default="—"):
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except TypeError:
        pass
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def build_pdf_report_bytes(
    tic_id,
    stellar_params,
    exofop_flags,
    lc,
    flux_column,
    fold_period,
    fold_epoch,
    bls_result,
    fp_result,
    duration_days=None,
    centroid_offset=None,
) -> bytes:
    """
    Build a single-page, TESS-SPOC-DV-report-style PDF summary and return it
    as bytes.

    Layout (one page):
      Row 1: header text — TIC ID, period, stellar params, ExoFOP status
      Row 2: full stitched timeline
      Row 3: phase-fold (days, binned overlay)         | secondary-eclipse zoom
      Row 4: phase-fold (hours, zoomed, binned overlay) | BLS periodogram
      Row 5: odd/even transit-shape comparison          | TIC position centroid offset
      Row 6: Fit Results / Diagnostic Results text blocks
    """
    from pht_app.data.analysis import phase_fold

    buf = io.BytesIO()
    sp = stellar_params or {}
    fp_result = fp_result or {}

    if duration_days is None:
        duration_days = bls_result["best_duration"] if bls_result else 3.0 / 24.0
    duration_hours = duration_days * 24.0

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(15, 13))
        gs = fig.add_gridspec(
            6, 2,
            height_ratios=[0.6, 1.6, 1.6, 1.6, 1.6, 1.5],
            width_ratios=[2, 1],
            hspace=1.1, wspace=0.22,
        )

        # ---- Row 1: header --------------------------------------------------
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis("off")
        title_line = f"TIC: {tic_id}" + (f"    Period: {fold_period:.4f} d" if fold_period else "    Period: —")
        stellar_line = (
            f"Tmag: {_fmt(sp.get('Tmag'), '{:.2f}')}    R★: {_fmt(sp.get('R_star'))} R☉    "
            f"Teff: {_fmt(sp.get('Teff'), '{:.0f}')} K    M★: {_fmt(sp.get('M_star'))} M☉    "
            f"logg: {_fmt(sp.get('logg'), '{:.2f}')}"
        )
        exofop_line = ""
        if exofop_flags:
            if exofop_flags.get("status") == "found":
                exofop_line = f"[!] ExoFOP: {exofop_flags.get('message', '')}"
            elif exofop_flags.get("status") == "not_found":
                exofop_line = "[OK] No existing ExoFOP TOI entry found for this target."
        ax_header.text(0.5, 0.85, title_line, ha="center", va="top", fontsize=15, fontweight="bold")
        ax_header.text(0.5, 0.45, stellar_line, ha="center", va="top", fontsize=10.5)
        if exofop_line:
            ax_header.text(0.5, 0.08, exofop_line, ha="center", va="top", fontsize=9.5, color="darkred")

        # ---- Row 2: full timeline -------------------------------------------
        ax_timeline = fig.add_subplot(gs[1, :])
        if lc is not None:
            time_vals = lc.time.value
            col_lower = (flux_column or "flux").lower()
            if col_lower in lc.colnames:
                raw = lc[col_lower]
                flux_vals = raw.value if hasattr(raw, "value") else np.asarray(raw)
            else:
                flux_vals = lc.flux.value
            ax_timeline.scatter(time_vals, flux_vals - 1.0, s=1.5, alpha=0.5, color="black")
            if fold_epoch is not None and fold_period:
                n_start = int(np.floor((time_vals.min() - fold_epoch) / fold_period))
                n_end = int(np.ceil((time_vals.max() - fold_epoch) / fold_period))
                for n in range(n_start, n_end + 1):
                    ax_timeline.axvline(fold_epoch + n * fold_period, color="green", linestyle="--",
                                         linewidth=0.6, alpha=0.6)
            ax_timeline.set_xlabel("Time [BJD - 2457000]")
            ax_timeline.set_ylabel("Relative Flux")
            ax_timeline.set_title("Stitched Timeline (green dashed = predicted transit times)", fontsize=10, pad=10)
        else:
            ax_timeline.axis("off")
            ax_timeline.text(0.5, 0.5, "No light curve loaded", ha="center", va="center")

        have_fold = lc is not None and fold_period and fold_epoch is not None

        # ---- Row 3: phase-fold (days) | secondary-eclipse zoom ---------------
        ax_fold_days = fig.add_subplot(gs[2, 0])
        ax_secondary = fig.add_subplot(gs[2, 1])

        if have_fold:
            phase, flux, _ = phase_fold(lc, fold_period, fold_epoch)
            ax_fold_days.scatter(phase, flux - 1.0, s=1.5, alpha=0.35, color="black")

            bin_edges = np.linspace(-0.5, 0.5, 101)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            binned = np.full(100, np.nan)
            for i in range(100):
                sel = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
                if sel.any():
                    binned[i] = np.nanmean(flux[sel]) - 1.0
            ax_fold_days.plot(bin_centers, binned, color="cyan", linewidth=1.2)
            ax_fold_days.set_xlabel("Phase [Days]" if False else "Phase")
            ax_fold_days.set_ylabel("Relative Flux")
            ax_fold_days.set_title(f"Phase-Folded (P = {fold_period:.5f} d)", fontsize=10, pad=10)

            # Secondary-eclipse zoom around phase 0.5
            from pht_app.data.fp_diagnostics import secondary_eclipse_test
            se = fp_result.get("secondary_eclipse") or secondary_eclipse_test(lc, fold_period, fold_epoch, duration_days)
            sec_window = max(duration_days / fold_period * 4, 0.05)
            sec_phase = phase - 0.5
            sec_phase = (sec_phase + 0.5) % 1.0 - 0.5
            near_sec = np.abs(sec_phase) <= sec_window
            if near_sec.sum() > 5:
                ax_secondary.scatter(sec_phase[near_sec] + 0.5, flux[near_sec] - 1.0, s=2, alpha=0.4, color="black")
            depth_txt = f"Sec Depth: {se['secondary_depth']*1e6:.1f} ppm" if se.get("status") == "ok" else "Sec Depth: N/A"
            sig_txt = f"Sig: {se['significance_sigma']:.1f}σ" if se.get("status") == "ok" else ""
            ax_secondary.set_title(f"{depth_txt}   {sig_txt}", fontsize=9, pad=10)
            ax_secondary.set_xlabel("Phase")
        else:
            for ax in (ax_fold_days, ax_secondary):
                ax.axis("off")
            ax_fold_days.text(0.5, 0.5, "No period/epoch set", ha="center", va="center")

        # ---- Row 4: phase-fold (hours, zoomed) | BLS periodogram -------------
        ax_fold_hours = fig.add_subplot(gs[3, 0])
        ax_periodogram = fig.add_subplot(gs[3, 1])

        if have_fold:
            phase_hours = phase * fold_period * 24.0
            zoom_hw = max(duration_hours * 4, 2.0)
            near = np.abs(phase_hours) <= zoom_hw
            ax_fold_hours.scatter(phase_hours[near], flux[near] - 1.0, s=2, alpha=0.35, color="black")

            bin_edges_h = np.linspace(-zoom_hw, zoom_hw, 41)
            bin_centers_h = 0.5 * (bin_edges_h[:-1] + bin_edges_h[1:])
            binned_h = np.full(40, np.nan)
            for i in range(40):
                sel = near & (phase_hours >= bin_edges_h[i]) & (phase_hours < bin_edges_h[i + 1])
                if sel.any():
                    binned_h[i] = np.nanmean(flux[sel]) - 1.0
            ax_fold_hours.plot(bin_centers_h, binned_h, color="red", linewidth=1.5)
            ax_fold_hours.set_xlabel("Phase [Hours]")
            ax_fold_hours.set_ylabel("Relative Flux")
            ax_fold_hours.set_title(f"Zoomed Transit (T14 ≈ {duration_hours:.2f} h)", fontsize=10, pad=10)
        else:
            ax_fold_hours.axis("off")

        if bls_result:
            ax_periodogram.plot(bls_result["periods"], bls_result["power"], linewidth=0.7, color="darkorange")
            ax_periodogram.axvline(bls_result["best_period"], color="red", linestyle="--", linewidth=1,
                                    label=f"P={bls_result['best_period']:.4f}d")
            ax_periodogram.set_xscale("log")
            ax_periodogram.set_xlabel("Period (days)")
            ax_periodogram.set_ylabel("BLS Power")
            ax_periodogram.set_title(
                f"MES-like: BLS Power   Dur: {duration_hours:.1f} h", fontsize=9, pad=10)
            ax_periodogram.legend(fontsize=8)
        else:
            ax_periodogram.axis("off")
            ax_periodogram.text(0.5, 0.5, "No periodogram computed", ha="center", va="center")

        # ---- Row 5: odd/even comparison | TIC position centroid offset -------
        ax_odd_even = fig.add_subplot(gs[4, 0])
        ax_centroid = fig.add_subplot(gs[4, 1])

        oe = fp_result.get("odd_even")
        if have_fold and oe and oe.get("status") == "ok":
            from pht_app.data.fp_diagnostics import odd_even_folded_curves
            curves = odd_even_folded_curves(lc, fold_period, fold_epoch, duration_days)
            if curves.get("status") == "ok":
                offset = max(duration_hours * 5, 8.0)
                ax_odd_even.plot(curves["phase_hours"] - offset, curves["odd_flux"] - 1.0,
                                  color="red", linewidth=1.3)
                ax_odd_even.plot(curves["phase_hours"] + offset, curves["even_flux"] - 1.0,
                                  color="red", linewidth=1.3)
                ax_odd_even.axvline(0, color="black", linestyle="--", linewidth=0.8)
                ymin = np.nanmin(np.concatenate([curves["odd_flux"], curves["even_flux"]])) - 1.0
                ax_odd_even.text(-offset, ymin * 0.15, "Odd", ha="center", fontsize=9, fontweight="bold")
                ax_odd_even.text(offset, ymin * 0.15, "Even", ha="center", fontsize=9, fontweight="bold")
                mismatch_pct = oe.get("relative_mismatch", 0) * 100
                ax_odd_even.set_title(
                    f"Depth mismatch: {mismatch_pct:.0f}% [{oe['sigma_diff']:.1f}σ] "
                    f"— {'FLAGGED' if oe['likely_eb'] else 'OK'}", fontsize=9, pad=10,
                )
                ax_odd_even.set_xlabel("Phase [Hours] (odd/even offset for display)")
                ax_odd_even.set_ylabel("Relative Flux")
            else:
                ax_odd_even.axis("off")
                ax_odd_even.text(0.5, 0.5, "Not enough odd/even transits", ha="center", va="center")
        else:
            ax_odd_even.axis("off")
            ax_odd_even.text(0.5, 0.5, "Odd/even test not run", ha="center", va="center")

        if centroid_offset:
            theta = np.linspace(0, 2 * np.pi, 100)
            pixel_scale = 21.0
            ax_centroid.plot(pixel_scale * np.cos(theta), pixel_scale * np.sin(theta),
                              color="blue", linewidth=1)
            ax_centroid.scatter([0], [0], marker="*", s=140, color="black", label="TIC position")
            ax_centroid.scatter([centroid_offset["d_ra_arcsec"]], [centroid_offset["d_dec_arcsec"]],
                                 marker="x", s=90, color="crimson", label="Centroid")
            ax_centroid.set_xlabel("RA Offset (arcsec)")
            ax_centroid.set_ylabel("Dec Offset (arcsec)")
            ax_centroid.set_title(
                f"TIC Position Centroid Offset: {centroid_offset['offset_arcsec']:.2f}\"", fontsize=9, pad=10)
            ax_centroid.set_aspect("equal")
            ax_centroid.legend(fontsize=7, loc="upper right")
        else:
            ax_centroid.axis("off")
            ax_centroid.text(
                0.5, 0.5,
                "TPF centroid not computed\n(run the TPF Centroid Check tab first)",
                ha="center", va="center", fontsize=9,
            )

        # ---- Row 6: Fit Results / Diagnostic Results text blocks -------------
        ax_fit = fig.add_subplot(gs[5, 0])
        ax_diag = fig.add_subplot(gs[5, 1])
        ax_fit.axis("off")
        ax_diag.axis("off")

        fit_lines = ["DV Fit Results", ""]
        if fold_period:
            fit_lines.append(f"Period = {fold_period:.5f} d")
        if fold_epoch is not None:
            fit_lines.append(f"Epoch = {fold_epoch:.4f} BTJD")
        fit_lines.append(f"Duration = {duration_hours:.2f} h")
        if bls_result:
            fit_lines.append(f"Depth = {bls_result['best_depth']*1e6:.1f} ppm")
        ax_fit.text(0, 1.0, "\n".join(fit_lines), va="top", ha="left", fontsize=10, family="monospace")

        diag_lines = ["DV Diagnostic Results", ""]
        diag_lines += textwrap.wrap(f"Verdict: {fp_result.get('verdict', 'N/A')}", width=42)
        suggestion = fp_result.get("period_suggestion")
        if suggestion:
            diag_lines += ["", "** Incorrect period suspected **", f"Period: {suggestion['label']}"]
        sh = fp_result.get("shape")
        if sh and sh.get("status") == "ok":
            diag_lines.append(f"Shape: {sh.get('shape', 'N/A')}")
        ax_diag.text(0, 1.0, "\n".join(diag_lines), va="top", ha="left", fontsize=10, family="monospace")

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        fig.text(0.01, 0.005, f"Generated {generated} — PHT Candidate Validator", fontsize=7.5, color="gray")

        fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.98)
        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()
