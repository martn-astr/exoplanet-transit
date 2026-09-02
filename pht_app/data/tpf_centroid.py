"""
Target Pixel File (TPF) centroid diagnostic.

Downloads the TPF covering a flagged transit window, builds an in-transit
minus out-of-transit difference image, and overlays Gaia DR3 source positions
so the user can visually confirm the photometric centroid does not shift onto
a background/blended star (a classic false-positive signature).
"""

import numpy as np
import lightkurve as lk
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia


def download_tpf(tic_id: str, sector: int):
    """Download the target pixel file for a given TIC + sector."""
    target = f"TIC {tic_id}"
    sr = lk.search_targetpixelfile(target, mission="TESS", sector=sector)
    if len(sr) == 0:
        return None
    return sr[0].download()


def build_difference_image(tpf, t0: float, duration_days: float, oot_buffer_days: float = 0.5):
    """
    Build an in-transit minus out-of-transit difference image from a TPF.

    Returns dict with:
        diff_image: 2D array (out-of-transit median minus in-transit median)
        in_transit_image, out_of_transit_image: the two medians
        wcs: the TPF's WCS for coordinate overlay
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

    return {
        "diff_image": diff_img,
        "in_transit_image": in_transit_img,
        "out_of_transit_image": out_of_transit_img,
        "wcs": tpf.wcs,
        "ra": tpf.ra,
        "dec": tpf.dec,
    }


def query_gaia_sources(ra: float, dec: float, radius_arcsec: float = 60.0, mag_limit: float = 18.0):
    """
    Query Gaia DR3 for sources near the target, to overlay on the difference
    image and check whether the eclipse could originate from a blended
    neighbor rather than the target itself.
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
        return result.to_pandas()
    except Exception as e:
        return None


def centroid_shift_estimate(diff_image, wcs):
    """
    Flux-weighted centroid of the difference image, converted to sky
    coordinates via the TPF's WCS, for comparison against the target's
    catalog position and any nearby Gaia sources.
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
        return {"x": x_centroid, "y": y_centroid, "ra": sky.ra.deg, "dec": sky.dec.deg}
    except Exception:
        return {"x": x_centroid, "y": y_centroid, "ra": None, "dec": None}


def centroid_offset_arcsec(centroid, target_ra_deg, target_dec_deg):
    """
    Offset of the difference-image centroid from the target's catalog
    position, in RA/Dec arcsec — matching the classic DV-report "TIC
    Position Centroid Offsets" plot. A large offset (well outside the
    photocenter uncertainty) suggests the eclipse originates from a nearby
    blended source rather than the target itself.
    """
    if centroid is None or centroid.get("ra") is None or target_ra_deg is None or target_dec_deg is None:
        return None
    if np.isnan(target_ra_deg) or np.isnan(target_dec_deg):
        return None

    dec_rad = np.radians(target_dec_deg)
    d_ra_arcsec = (centroid["ra"] - target_ra_deg) * np.cos(dec_rad) * 3600.0
    d_dec_arcsec = (centroid["dec"] - target_dec_deg) * 3600.0
    offset_arcsec = float(np.hypot(d_ra_arcsec, d_dec_arcsec))

    return {
        "d_ra_arcsec": float(d_ra_arcsec),
        "d_dec_arcsec": float(d_dec_arcsec),
        "offset_arcsec": offset_arcsec,
    }
