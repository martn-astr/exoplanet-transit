"""Sector discovery and multi-sector light-curve download / normalization / stitching."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
import lightkurve as lk
from astroquery.mast import Conf as MastConf

from pht_app.config import CACHE_TTL_SECONDS, SPOC_2MIN_LABEL, FFI_FALLBACK_AUTHORS
from pht_app.data.lookup import clean_tic_id

if TYPE_CHECKING:
    from lightkurve import LightCurve  # hint-only import — real module already
                                        # imported unconditionally above via `lightkurve as lk`

# MAST queries (both lightkurve's own search_lightcurve/search_targetpixelfile
# and astroquery.mast.Catalogs elsewhere) share this global astroquery config.
# Without an explicit timeout, a hung connection blocks the Streamlit run
# indefinitely with no feedback to the user.
MastConf.timeout = 30


class SectorInfo(TypedDict):
    """One entry in the sector list returned by search_available_sectors()."""
    sector: int
    authors: list[str]
    has_2min_spoc: bool
    has_ffi_fallback: bool


def _safe_int(value: Any, default: int) -> int:
    """Coerce a possibly-missing/NA value (pandas NA, NaN, None) to int, or return default."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce a possibly-missing/NA value to float, or return default."""
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _search_safe(search_fn, *args: Any, logs: list[str], context: str, **kwargs: Any):
    """
    Run a lightkurve search call (search_lightcurve / search_targetpixelfile)
    defensively: a MAST outage, rate-limit, or response-schema change would
    otherwise raise uncaught here and crash the entire Streamlit run with a
    raw traceback. On failure, logs the issue and returns an empty
    SearchResult-like list (safe for `len(...) == 0` checks downstream)
    instead.
    """
    try:
        return search_fn(*args, **kwargs)
    except Exception as e:
        logs.append(f"{context}: search failed ({type(e).__name__}: {e}).")
        return []


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def search_available_sectors(tic_id: str) -> list[SectorInfo]:
    """Query MAST for every available light-curve product for a TIC.

    Groups results by sector across SPOC 2-minute and FFI-derived pipelines
    (TESS-SPOC, QLP).

    Parameters
    ----------
    tic_id : str
        TIC identifier, with or without a "TIC" prefix (cleaned internally).

    Returns
    -------
    list[SectorInfo]
        One entry per sector with usable data, sorted by sector number.
        Empty list if the target has no light-curve products or the MAST
        search itself fails.
    """
    clean_id = clean_tic_id(tic_id)
    target = f"TIC {clean_id}"
    logs: list[str] = []

    search_result = _search_safe(lk.search_lightcurve, target, mission="TESS",
                                  logs=logs, context=f"TIC {clean_id} sector search")
    if len(search_result) == 0:
        return []

    table = search_result.table.to_pandas()
    sectors: dict[int, dict[str, Any]] = {}
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

    out: list[SectorInfo] = []
    for sector in sorted(sectors.keys()):
        e = sectors[sector]
        out.append(SectorInfo(
            sector=sector,
            authors=sorted(e["authors"]),
            has_2min_spoc=e["has_2min_spoc"],
            has_ffi_fallback=e["has_ffi_fallback"],
        ))
    return out


def _download_one_sector(
    target: str, sector: int, prefer_source: str, logs: list[str]
) -> tuple["LightCurve | None", str | None]:
    """Try the preferred source first, then fall back through FFI authors.

    Returns
    -------
    tuple[LightCurve | None, str | None]
        The downloaded light curve and a label describing which source it
        came from, or (None, None) if nothing was available/downloadable.
    """
    lc = None
    used_source = None

    if prefer_source == SPOC_2MIN_LABEL:
        sr = _search_safe(lk.search_lightcurve, target, mission="TESS", sector=sector, author="SPOC",
                           logs=logs, context=f"Sector {sector} SPOC search")
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
            sr = _search_safe(lk.search_lightcurve, target, mission="TESS", sector=sector, author=ffi_author,
                               logs=logs, context=f"Sector {sector} {ffi_author} search")
            if len(sr) > 0:
                try:
                    lc = sr[0].download()
                    used_source = f"FFI fallback ({ffi_author})"
                    break
                except Exception as e:
                    logs.append(f"Sector {sector}: {ffi_author} download failed ({e}).")

    return lc, used_source


def _normalize(lc: "LightCurve") -> "LightCurve":
    """Median-divide a light curve to a baseline flux of 1.0, dropping NaNs first.

    Also independently normalizes the raw sap_flux/pdcsap_flux columns (if
    present) to their own per-sector baseline of 1.0 — otherwise those raw
    columns stay at their original per-sector count scale (tens of thousands
    of electrons/s, different per sector), which makes them effectively
    unplottable once multiple sectors are stitched together.

    Parameters
    ----------
    lc : LightCurve
        Single-sector light curve, pre-normalization.

    Returns
    -------
    LightCurve
        The same light curve, NaN-cleaned and normalized in place (and
        returned for chaining).
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


def download_and_stitch(
    tic_id: str, sectors: list[int], prefer_source: str
) -> tuple["LightCurve | None", dict[int, "LightCurve"], list[str]]:
    """Download, normalize, and stitch the requested sectors.

    Prefers SPOC 2-minute cadence, falling back to FFI/QLP per sector when
    2-minute data isn't available. Each sector is independently normalized
    to a baseline flux of 1.0 (see `_normalize`) before stitching, with
    6-sigma outlier removal applied during the stitch.

    Not cached directly — see `download_and_stitch_cached` for the
    st.cache_resource-wrapped version app code should actually call.

    Parameters
    ----------
    tic_id : str
        TIC identifier, with or without a "TIC" prefix.
    sectors : list[int]
        Sector numbers to download.
    prefer_source : str
        Either SPOC_2MIN_LABEL or FFI_QLP_LABEL (see pht_app.config).

    Returns
    -------
    tuple[LightCurve | None, dict[int, LightCurve], list[str]]
        (stitched light curve or None if nothing downloaded successfully,
        per-sector light curves keyed by sector number, human-readable log
        messages describing what happened for each sector).
    """
    clean_id = clean_tic_id(tic_id)
    target = f"TIC {clean_id}"
    logs: list[str] = []
    per_sector_lcs: dict[int, "LightCurve"] = {}

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


@st.cache_resource(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def download_and_stitch_cached(
    tic_id: str, sectors: tuple[int, ...], prefer_source: str
) -> tuple["LightCurve | None", dict[int, "LightCurve"], list[str]]:
    """Cached wrapper around download_and_stitch.

    Uses st.cache_resource rather than st.cache_data: the return value
    contains LightCurve objects, which aren't cleanly picklable, and
    st.cache_resource caches by reference instead of serializing — the
    correct tool for heavyweight, non-serializable scientific objects
    (same category as a DB connection or an ML model in Streamlit's own
    docs). `sectors` must be passed as a tuple (hashable); a plain list
    would fail cache-key hashing since st.cache_resource still hashes its
    *arguments* even though it doesn't hash the *return value*.

    Note: st.cache_resource returns the SAME object to every caller,
    including across sessions on a shared deployment. Nothing downstream
    mutates the returned LightCurve in place (it's always reassigned, never
    e.g. `lc.flux[...] = ...`'d after this point) — safe today, but that
    invariant matters if this function's callers ever change.
    """
    return download_and_stitch(tic_id, list(sectors), prefer_source)
