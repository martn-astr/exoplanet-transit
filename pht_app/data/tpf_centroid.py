"""
Target Pixel File (TPF) centroid diagnostic.

Downloads the TPF covering a flagged transit window, builds an in-transit
minus out-of-transit difference image, and overlays Gaia DR3 source positions
so the user can visually confirm the photometric centroid does not shift onto
a background/blended star (a classic false-positive signature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
import lightkurve as lk
from astroquery.gaia import Gaia

from pht_app.config import CACHE_TTL_SECONDS
from pht_app.data.lightcurves import _search_safe

if TYPE_CHECKING:
    from astropy.wcs import WCS
    from lightkurve import TargetPixelFile

# Gaia archive queries have no timeout by default, which can hang the
# Streamlit run indefinitely on a slow/unresponsive connection.
Gaia.TIMEOUT = 30


class DifferenceImage(TypedDict):
    diff_image: npt.NDArray[np.float64]
    in_transit_image: npt.NDArray[np.float64]
    out_of_transit_image: npt.NDArray[np.float64]
    wcs: "WCS"
    ra: float
    dec: float


class CentroidEstimate(TypedDict):
    x: float
    y: float
    ra: Optional[float]
    dec: Optional[float]


class CentroidOffset(TypedDict):
    d_ra_arcsec: float
    d_dec_arcsec: float
    offset_arcsec: float


def download_tpf(tic_id: str, sector: int) -> Optional["TargetPixelFile"]:
    """Download the target pixel file for a given TIC + sector.

    Parameters
    ----------
    tic_id : str
        TIC identifier (numeric string, no "TIC" prefix expected here).
    sector : int
        TESS sector number.

    Returns
    -------
    TargetPixelFile or None
        None if no TPF product is found for this sector, or if the search
        itself fails (e.g. a MAST outage) — logged to Streamlit's UI via
        st.warning rather than raising.
    """
    target = f"TIC {tic_id}"
    logs: list[str] = []
    sr = _search_safe(lk.search_targetpixelfile, target, mission="TESS", sector=sector,
                       logs=logs, context=f"TIC {tic_id} sector {sector} TPF search")
    for msg in logs:
        st.warning(msg)
    if len(sr) == 0:
        return None
    return sr[0].download()


def build_difference_image(
    tpf: "TargetPixelFile", t0: float, duration_days: float, oot_buffer_days: float = 0.5
) -> Optional[DifferenceImage]:
    """Build an in-transit minus out-of-transit difference image from a TPF.

    Parameters
    ----------
    tpf : TargetPixelFile
        Target pixel file covering the transit window of interest.
    t0 : float
        Transit epoch (BTJD) to center the in-transit window on.
    duration_days : float
        Full transit duration in days.
    oot_buffer_days : float
        Width of the out-of-transit comparison window on each side, in days.

    Returns
    -------
    DifferenceImage or None
        None if fewer than 2 cadences fall in either the in-transit or
        out-of-transit window (not enough to form a reliable median image).
    """
    time_vals = tpf.time.value
    half_dur = duration_days / 2.0

    in_transit_mask = np.abs(time_vals - t0) <= half_dur
    out_of_transit_mask = (
        (np.abs(time_vals - t0) > half_dur) &
        (np.abs(time_vals - t0) <= half_dur + oot_buffer_days)
    )

    if in_transit_mask.sum() < 2 or out_of_transit_mask.sum() < 2:
        return None

    flux_cube = tpf.flux.value  # shape (n_time, ny, nx)
    in_transit_img = np.nanmedian(flux_cube[in_transit_mask], axis=0)
    out_of_transit_img = np.nanmedian(flux_cube[out_of_transit_mask], axis=0)
    diff_img = out_of_transit_img - in_transit_img  # positive = flux missing during transit

    return DifferenceImage(
        diff_image=diff_img,
        in_transit_image=in_transit_img,
        out_of_transit_image=out_of_transit_img,
        wcs=tpf.wcs,
        ra=tpf.ra,
        dec=tpf.dec,
    )


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def query_gaia_sources(
    ra: float, dec: float, radius_arcsec: float = 60.0, mag_limit: float = 18.0
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Query Gaia DR3 for sources near the target.

    Used to overlay on the TPF difference image and check whether the
    eclipse could originate from a blended neighbor rather than the target
    itself. Cached (plain float args are trivially hashable) so repeatedly
    inspecting the same field doesn't re-hit the Gaia archive.

    Parameters
    ----------
    ra, dec : float
        Target position in degrees (ICRS).
    radius_arcsec : float
        Search cone radius in arcseconds.
    mag_limit : float
        Only return sources brighter than this Gaia G magnitude.

    Returns
    -------
    tuple[pandas.DataFrame | None, str | None]
        (results, None) on success — results may be an empty DataFrame if
        the query succeeded but found nothing. (None, error_message) on
        failure, where error_message describes what went wrong (timeout,
        connection error, malformed query, etc.) so the caller can surface
        the actual reason instead of a generic "query failed" message.
    """
    radius_deg = radius_arcsec / 3600.0
    query = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND phot_g_mean_mag < {mag_limit}
    ORDER BY phot_g_mean_mag ASC
    """
    try:
        job = Gaia.launch_job(query)
        result = job.get_results()
        return result.to_pandas(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def centroid_shift_estimate(diff_image: npt.NDArray[np.float64], wcs: "WCS") -> Optional[CentroidEstimate]:
    """Flux-weighted centroid of a TPF difference image.

    Converts pixel coordinates to sky coordinates via the TPF's WCS, for
    comparison against the target's catalog position and any nearby Gaia
    sources.

    Parameters
    ----------
    diff_image : ndarray
        2D difference image (out-of-transit minus in-transit), as returned
        in DifferenceImage["diff_image"].
    wcs : astropy.wcs.WCS
        World coordinate system for the TPF the difference image came from.

    Returns
    -------
    CentroidEstimate or None
        None if the difference image has no positive (flux-missing) signal
        to center on.
    """
    img = np.nan_to_num(diff_image, nan=0.0)
    img = np.clip(img, 0, None)  # only positive (flux-missing) pixels are physically meaningful
    total = img.sum()
    if total <= 0:
        return None

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x_centroid = float((xx * img).sum() / total)
    y_centroid = float((yy * img).sum() / total)

    try:
        sky = wcs.pixel_to_world(x_centroid, y_centroid)
        return CentroidEstimate(x=x_centroid, y=y_centroid, ra=sky.ra.deg, dec=sky.dec.deg)
    except Exception:
        return CentroidEstimate(x=x_centroid, y=y_centroid, ra=None, dec=None)


def centroid_offset_arcsec(
    centroid: Optional[CentroidEstimate], target_ra_deg: Optional[float], target_dec_deg: Optional[float]
) -> Optional[CentroidOffset]:
    """Offset of the difference-image centroid from the target's catalog position.

    Matches the classic DV-report "TIC Position Centroid Offsets" plot. A
    large offset (well outside the photocenter uncertainty) suggests the
    eclipse originates from a nearby blended source rather than the target
    itself.

    Parameters
    ----------
    centroid : CentroidEstimate or None
        Result of centroid_shift_estimate().
    target_ra_deg, target_dec_deg : float or None
        Target's catalog position in degrees.

    Returns
    -------
    CentroidOffset or None
        None if the centroid or target position isn't available/valid.
    """
    if centroid is None or centroid.get("ra") is None or target_ra_deg is None or target_dec_deg is None:
        return None
    if np.isnan(target_ra_deg) or np.isnan(target_dec_deg):
        return None

    dec_rad = np.radians(target_dec_deg)
    d_ra_arcsec = (centroid["ra"] - target_ra_deg) * np.cos(dec_rad) * 3600.0
    d_dec_arcsec = (centroid["dec"] - target_dec_deg) * 3600.0
    offset_arcsec = float(np.hypot(d_ra_arcsec, d_dec_arcsec))

    return CentroidOffset(
        d_ra_arcsec=float(d_ra_arcsec),
        d_dec_arcsec=float(d_dec_arcsec),
        offset_arcsec=offset_arcsec,
    )
