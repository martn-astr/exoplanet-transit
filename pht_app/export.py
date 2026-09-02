"""
Export helpers: CSV of the stitched light curve, and a PDF summary report
(light curve, phase-fold, periodogram, stellar params, FP diagnostics
verdict) in the spirit of a TESS SPOC Data Validation report page.

Kept dependency-light (matplotlib only) and free of Streamlit imports so it
can be unit tested without a running app.
"""

import io
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
) -> bytes:
    """
    Build a one-page-per-section PDF summary report and return it as bytes.

    Sections:
      1. Header + stellar params + ExoFOP status
      2. Full stitched timeline
      3. Phase-folded view (if a period/epoch is set)
      4. BLS periodogram (if computed)
      5. False Positive diagnostics verdict (if computed)
    """
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ---- Page 1: header + timeline -----------------------------------
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.9, 0.15, 2, 1])

        ax_header = fig.add_subplot(gs[0])
        ax_header.axis("off")
        sp = stellar_params or {}
        header_lines = [
            f"PHT Candidate Validator — Summary Report",
            f"TIC {tic_id}" + (f"    Period: {fold_period:.4f} d" if fold_period else ""),
            f"R★ = {_fmt(sp.get('R_star'))} R☉    M★ = {_fmt(sp.get('M_star'))} M☉    "
            f"Teff = {_fmt(sp.get('Teff'), '{:.0f}')} K    Tmag = {_fmt(sp.get('Tmag'), '{:.2f}')}",
        ]
        if exofop_flags:
            if exofop_flags.get("status") == "found":
                header_lines.append(f"[!] ExoFOP: {exofop_flags.get('message', '')}")
            elif exofop_flags.get("status") == "not_found":
                header_lines.append("[OK] No existing ExoFOP TOI entry found for this target.")
        ax_header.text(0, 1.0, "\n".join(header_lines), va="top", ha="left", fontsize=11, family="monospace")

        ax_timeline = fig.add_subplot(gs[2])
        if lc is not None:
            time_vals = lc.time.value
            col_lower = (flux_column or "flux").lower()
            if col_lower in lc.colnames:
                raw = lc[col_lower]
                flux_vals = raw.value if hasattr(raw, "value") else np.asarray(raw)
            else:
                flux_vals = lc.flux.value
            ax_timeline.scatter(time_vals, flux_vals, s=2, alpha=0.5, color="steelblue")
            ax_timeline.set_xlabel("Time (BTJD)")
            ax_timeline.set_ylabel(f"Normalized Flux ({flux_column or 'flux'})")
            ax_timeline.set_title("Stitched Timeline")
        else:
            ax_timeline.axis("off")
            ax_timeline.text(0.5, 0.5, "No light curve loaded", ha="center", va="center")

        ax_footer = fig.add_subplot(gs[3])
        ax_footer.axis("off")
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ax_footer.text(0, 1.0, f"Generated {generated} — PHT Candidate Validator", fontsize=8, color="gray")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: phase-fold + periodogram -----------------------------
        if lc is not None and fold_period and fold_epoch:
            from pht_app.data.analysis import phase_fold

            fig2, axes = plt.subplots(2, 1, figsize=(11, 8.5))

            phase, flux, _ = phase_fold(lc, fold_period, fold_epoch)
            axes[0].scatter(phase, flux, s=2, alpha=0.4, color="steelblue")
            axes[0].set_xlabel("Phase")
            axes[0].set_ylabel("Normalized Flux")
            axes[0].set_title(f"Phase-Folded (P = {fold_period:.5f} d, T0 = {fold_epoch:.5f} BTJD)")

            if bls_result:
                axes[1].plot(bls_result["periods"], bls_result["power"], linewidth=0.8, color="darkorange")
                axes[1].axvline(bls_result["best_period"], color="red", linestyle="--",
                                 label=f"P = {bls_result['best_period']:.4f} d")
                axes[1].set_xscale("log")
                axes[1].set_xlabel("Period (days)")
                axes[1].set_ylabel("BLS Power")
                axes[1].set_title("BLS Periodogram")
                axes[1].legend()
            else:
                axes[1].axis("off")
                axes[1].text(0.5, 0.5, "No periodogram computed", ha="center", va="center")

            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

        # ---- Page 3: False Positive diagnostics summary --------------------
        if fp_result:
            fig3 = plt.figure(figsize=(11, 8.5))
            ax = fig3.add_subplot(111)
            ax.axis("off")

            lines = [f"False Positive Diagnostics", "", f"Verdict: {fp_result.get('verdict', 'N/A')}", ""]

            oe = fp_result.get("odd_even", {})
            if oe.get("status") == "ok":
                lines += [
                    "Odd vs Even Depth:",
                    f"  Odd depth:  {oe['odd_depth']:.5f}",
                    f"  Even depth: {oe['even_depth']:.5f}",
                    f"  Mismatch:   {oe['sigma_diff']:.1f}σ  →  {'FLAGGED' if oe['likely_eb'] else 'OK'}",
                    "",
                ]

            sh = fp_result.get("shape", {})
            if sh.get("status") == "ok":
                lines += [
                    "Transit Shape:",
                    f"  {sh.get('shape', 'N/A')}",
                    f"  Flatness ratio: {_fmt(sh.get('flatness_ratio'), '{:.2f}')}  →  "
                    f"{'FLAGGED' if sh['likely_eb'] else 'OK'}",
                    "",
                ]

            se = fp_result.get("secondary_eclipse", {})
            if se.get("status") == "ok":
                lines += [
                    "Secondary Eclipse Search:",
                    f"  Depth: {se['secondary_depth']:.5f}",
                    f"  Significance: {se['significance_sigma']:.1f}σ  →  "
                    f"{'FLAGGED' if se['likely_eb'] else 'OK'}",
                ]

            ax.text(0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
            pdf.savefig(fig3)
            plt.close(fig3)

    return buf.getvalue()
