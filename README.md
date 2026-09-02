# PHT Candidate Validator
# WIP - Work In Progress
A Streamlit application for validating exoplanet candidates and variable
stars from the **Planet Hunters TESS (PHT)** citizen-science project.
Pulls multi-sector TESS light curves, stitches them, and runs deterministic
diagnostic checks to help tell genuine transits apart from false positives
(eclipsing binaries, background blends, etc).

## Features

- **TIC search** — resolves stellar parameters (R★, M★, Teff, Tmag, logg)
  from the MAST TIC catalog.
- **ExoFOP cross-reference** — flags any existing TOI disposition for the
  target (known planet, known false positive, cTOI).
- **Multi-sector download & stitching** — defaults to SPOC 2-minute
  cadence, falls back automatically to FFI-derived data (TESS-SPOC / QLP)
  per sector when 2-min isn't available. Every sector is median-normalized
  to a baseline flux of 1.0 before stitching.
- **Synchronized 3-panel dashboard**
  1. **Timeline** — full stitched light curve with a SAP_FLUX / PDCSAP_FLUX
     toggle and box-select zoom.
  2. **Phase-folded view** — folds on a user-set (or BLS-derived) period and
     epoch, with an optional binned overlay.
  3. **Periodogram** — BLS or Lomb-Scargle, recomputed dynamically over
     whatever time window is currently zoomed/selected in Panel 1.
- **Deterministic single-transit estimator** — no AI/ML. Given a manually
  marked transit epoch (T0) and duration (T14), computes the maximum
  orbital period via Kepler's third law for a circular, central-transit
  orbit, and overlays predicted transit windows on the timeline. Also
  plots how the allowed period shrinks as impact parameter b → 1.
- **Signal masking** — remove known periodic signals from the search so
  residuals can be checked for additional, smaller planets.
- **False Positive diagnostics** — odd/even transit-depth comparison,
  V-shape vs U-shape transit classification, and a secondary-eclipse
  search at phase 0.5.
- **TPF spatial centroid check** — downloads the target pixel file for a
  flagged transit, builds an in-transit minus out-of-transit difference
  image, overlays Gaia DR3 sources, and estimates the flux-weighted
  centroid so you can confirm the signal isn't coming from a blended
  neighbor.

## Project structure

```
app.py                          Entry point (streamlit run app.py)
requirements.txt

pht_app/
  config.py                     Constants + session-state defaults
  data/
    lookup.py                   TIC catalog + ExoFOP queries
    lightcurves.py               Sector discovery + multi-sector download/stitch
    analysis.py                  Windowing, BLS/Lomb-Scargle, phase-folding
    single_transit.py            Deterministic Keplerian period estimator
    masking.py                   Known-signal masking
    fp_diagnostics.py            Odd/even, shape, secondary-eclipse tests
    tpf_centroid.py              TPF difference image + Gaia overlay + centroid
  ui/
    sidebar.py                   Search, sectors, data source, masking, export
    header.py                    Stellar params + ExoFOP status card
  panels/
    timeline.py                  Panel 1
    phasefold.py                 Panel 2
    periodogram.py                Panel 3
    single_transit.py            Single-transit estimator UI
    fp_and_tpf.py                 FP diagnostics + TPF centroid UI
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Enter a TIC ID in the sidebar and click **Search Target**.
2. Review the stellar parameters and any ExoFOP flags in the header.
3. Select sectors and click **Download & Stitch Selected Sectors**.
4. Use Panel 1's timeline to inspect the light curve; box-select a region
   to zoom Panel 3's periodogram into that window.
5. Set a period/epoch in Panel 2 (or click **Use this period/epoch for
   Panel 2 fold** from the periodogram) to see the folded transit.
6. For single-transit sectors, use the **Single-Transit Estimator** to
   compute a maximum period from a marked T0/T14, and optionally send it
   to Panel 2 or add it as a mask.
7. Run the **False Positive Diagnostics** once a period/epoch is set.
8. Use the **TPF Spatial Centroid Check** to rule out a background-star
   blend for any flagged transit.

## Notes / caveats

- The MAST, ExoFOP, and Gaia queries require outbound network access to
  their respective services; they were not exercised end-to-end against
  live servers in the environment this was built in (only package
  installation was network-accessible), so a smoke test against a known
  TIC ID (e.g. a confirmed TOI) is recommended before relying on it.
- The ExoFOP cross-reference scrapes their public TOI CSV export by TIC
  filter — if ExoFOP changes that endpoint's URL or schema, `lookup.py`
  will need a matching update.
- The single-transit period estimate is an **upper bound** (P_max at
  impact parameter b=0); always cross-check candidates against the BLS/LS
  periodogram and, where possible, against ExoFOP/TOI catalogs.
