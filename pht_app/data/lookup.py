"""TIC catalog resolution and ExoFOP cross-referencing."""

from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from astroquery.mast import Catalogs

from pht_app.config import CACHE_TTL_SECONDS


def clean_tic_id(tic_id: str) -> str:
    return tic_id.strip().upper().replace("TIC", "").replace(" ", "")


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def resolve_tic(tic_id: str):
    """Look up stellar parameters for a TIC ID via the MAST TIC catalog."""
    clean_id = clean_tic_id(tic_id)
    result = Catalogs.query_criteria(catalog="Tic", ID=clean_id)
    if len(result) == 0:
        return None

    row = result[0]

    def _f(col):
        val = row[col]
        return float(val) if val is not None and val is not np.ma.masked else np.nan

    return {
        "TIC_ID": clean_id,
        "ra": _f("ra"),
        "dec": _f("dec"),
        "Tmag": _f("Tmag"),
        "Teff": _f("Teff"),
        "R_star": _f("rad"),
        "M_star": _f("mass"),
        "logg": _f("logg"),
    }


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def query_exofop(tic_id: str):
    """
    Cross-reference a TIC ID against ExoFOP-TESS for existing dispositions
    (known planet, known false positive, cTOI, etc.) via the TOI CSV export.
    """
    clean_id = clean_tic_id(tic_id)
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php"
    params = {"target": clean_id, "output": "csv"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"status": "error", "message": str(e), "rows": []}

    if not resp.text.strip() or "TIC ID" not in resp.text:
        return {"status": "not_found", "message": "No ExoFOP TOI entries found.", "rows": []}

    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        return {"status": "error", "message": f"Parse error: {e}", "rows": []}

    if df.empty:
        return {"status": "not_found", "message": "No ExoFOP TOI entries found.", "rows": []}

    return {
        "status": "found",
        "message": f"{len(df)} ExoFOP entr(y/ies) found.",
        "rows": df.to_dict("records"),
    }
