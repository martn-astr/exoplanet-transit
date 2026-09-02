"""Sector discovery and multi-sector light-curve download / normalization / stitching."""

import numpy as np
import pandas as pd
import streamlit as st
import lightkurve as lk

from pht_app.config import CACHE_TTL_SECONDS, SPOC_2MIN_LABEL, FFI_FALLBACK_AUTHORS
from pht_app.data.lookup import clean_tic_id


def _safe_int(value, default):
    """Coerce a possibly-missing/NA value (pandas NA, NaN, None) to int, or return default."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=None):
    """Coerce a possibly-missing/NA value to float, or return default."""
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def search_available_sectors(tic_id: str):
    """
    Query MAST for every available light-curve product for this TIC across
    SPOC 2-min and FFI-derived pipelines (TESS-SPOC, QLP), grouped by sector.
    """
    clean_id = clean_tic_id(tic_id)
    target = f"TIC {clean_id}"

    search_result = lk.search_lightcurve(target, mission="TESS")
    if len(search_result) == 0:
        return []

    table = search_result.table.to_pandas()
    sectors = {}
    for _, row in table.iterrows():
        sector = _safe_int(row.get("sequence_number"), -1)
        if sector == -1:
            # No usable sector number for this product row — skip rather than
            # collapsing every unresolvable row into a fake "sector -1" bucket.
            continue
        author = str(row.get("author", "unknown"))
        exptime = _safe_float(row.get("exptime"))
        is_2min = author == "SPOC" and exptime is not None and exptime <= 200

        entry = sectors.setdefault(sector, {
            "sector": sector,
            "authors": set(),
            "has_2min_spoc": False,
            "has_ffi_fallback": False,
        })
        entry["authors"].add(author)
        if is_2min:
            entry["has_2min_spoc"] = True
        if author in FFI_FALLBACK_AUTHORS:
            entry["has_ffi_fallback"] = True

    out = []
    for sector in sorted(sectors.keys()):
        e = sectors[sector]
        out.append({
            "sector": sector,
            "authors": sorted(e["authors"]),
            "has_2min_spoc": e["has_2min_spoc"],
            "has_ffi_fallback": e["has_ffi_fallback"],
        })
    return out


def _download_one_sector(target: str, sector: int, prefer_source: str, logs: list):
    """Try preferred source first, then fall back through FFI authors. Returns (lc, source_label) or (None, None)."""
    lc = None
    used_source = None

    if prefer_source == SPOC_2MIN_LABEL:
        sr = lk.search_lightcurve(target, mission="TESS", sector=sector, author="SPOC")
        if len(sr):
            sr = sr[[e <= 200 for e in sr.exptime.value]]
        if len(sr) > 0:
            try:
                lc = sr[0].download()
                used_source = "SPOC 2-min"
            except Exception as e:
                logs.append(f"Sector {sector}: SPOC download failed ({e}); trying FFI fallback.")

    if lc is None:
        for ffi_author in FFI_FALLBACK_AUTHORS:
            sr = lk.search_lightcurve(target, mission="TESS", sector=sector, author=ffi_author)
            if len(sr) > 0:
                try:
                    lc = sr[0].download()
                    used_source = f"FFI fallback ({ffi_author})"
                    break
                except Exception as e:
                    logs.append(f"Sector {sector}: {ffi_author} download failed ({e}).")

    return lc, used_source


def _normalize(lc):
    """
    Median-divide a light curve to a baseline flux of 1.0, dropping NaNs first.
    Also independently normalizes the raw sap_flux/pdcsap_flux columns (if
    present) to their own per-sector baseline of 1.0 — otherwise those raw
    columns stay at their original per-sector count scale (tens of thousands
    of electrons/s, different per sector), which makes them effectively
    unplottable once multiple sectors are stitched together.
    """
    lc = lc.remove_nans()
    try:
        lc = lc.normalize()
    except Exception:
        median_flux = np.nanmedian(lc.flux.value)
        lc.flux = lc.flux / median_flux

    for col in ("sap_flux", "pdcsap_flux"):
        if col in lc.colnames:
            try:
                raw = lc[col]
                raw_vals = raw.value if hasattr(raw, "value") else np.asarray(raw)
                median = np.nanmedian(raw_vals)
                if median and np.isfinite(median) and median != 0:
                    lc[col] = raw / median
            except Exception:
                pass

    return lc


def download_and_stitch(tic_id: str, sectors: list, prefer_source: str):
    """
    Download the requested sectors (preferring SPOC 2-min, falling back to
    FFI/QLP), normalize each to baseline flux 1.0, and stitch into one
    outlier-cleaned LightCurve.

    Returns (stitched_lightcurve_or_None, per_sector_dict, log_messages).
    """
    clean_id = clean_tic_id(tic_id)
    target = f"TIC {clean_id}"
    logs = []
    per_sector_lcs = {}

    for sector in sectors:
        lc, used_source = _download_one_sector(target, sector, prefer_source, logs)

        if lc is None:
            logs.append(f"Sector {sector}: no usable data product found — skipped.")
            continue

        lc = _normalize(lc)
        per_sector_lcs[sector] = lc
        logs.append(f"Sector {sector}: loaded via {used_source} ({len(lc)} points).")

    if not per_sector_lcs:
        return None, {}, logs

    collection = lk.LightCurveCollection(list(per_sector_lcs.values()))
    stitched = collection.stitch(corrector_func=lambda x: x.remove_outliers(sigma=6))

    return stitched, per_sector_lcs, logs
