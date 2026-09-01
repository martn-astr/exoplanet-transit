"""
Deterministic single-transit period estimator.

Pure Keplerian orbital physics — zero machine learning / black-box inference.
Given a user-clicked transit epoch (T0) and measured transit duration (T14),
plus stellar mass/radius, estimate the maximum possible orbital period assuming
a circular orbit (e=0) and central transit (b=0):

    P ≈ 93.25 * (M*/Msun) * (R*/Rsun)^-3 * (T14/hours)^3   [days]

This is the upper bound on P for b=0; true P shrinks as impact parameter b -> 1.
We also report the boundary curve for 0 <= b < 1 for context.
"""

import numpy as np

KEPLER_COEFFICIENT = 93.25  # days, per the practical-units formula


def estimate_max_period(m_star_msun: float, r_star_rsun: float, t14_hours: float) -> float:
    """P_max in days for a circular, central (b=0) transit."""
    if m_star_msun is None or r_star_rsun is None or t14_hours is None:
        return None
    if np.isnan(m_star_msun) or np.isnan(r_star_rsun) or r_star_rsun <= 0:
        return None
    return KEPLER_COEFFICIENT * m_star_msun * (r_star_rsun ** -3) * (t14_hours ** 3)


def period_vs_impact_parameter(m_star_msun: float, r_star_rsun: float, t14_hours: float, n_points: int = 50):
    """
    Return arrays (b, P) tracing how the allowed period shrinks as the impact
    parameter b goes from 0 (central, P=P_max) toward 1 (grazing, P->0),
    using T14 ∝ sqrt(1 - b^2) at fixed P, i.e. P(b) = P_max * (1 - b^2)^{1.5}.
    """
    p_max = estimate_max_period(m_star_msun, r_star_rsun, t14_hours)
    if p_max is None:
        return None, None
    b = np.linspace(0, 0.99, n_points)
    p = p_max * (1 - b ** 2) ** 1.5
    return b, p


def estimate_transit_duration_hours(t_start, t_end) -> float:
    """Convert two clicked timeline x-coordinates (in days, BTJD) to T14 in hours."""
    return abs(t_end - t_start) * 24.0


FORMULA_LATEX = r"P \approx 93.25 \times \left(\frac{M_*}{M_\odot}\right) " \
                 r"\left(\frac{R_*}{R_\odot}\right)^{-3} \left(\frac{T_{14}}{\text{hours}}\right)^3 \ \text{days}"
