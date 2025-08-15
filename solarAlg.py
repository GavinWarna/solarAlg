#!/usr/bin/env python3
# solar_tilt_gui.py
"""
Solar Tilt Optimizer — Streamlit GUI (pvlib-version-robust)

• Finds the best fixed tilt and a 2-position seasonal schedule (Apr–Sep, Oct–Mar by default).
• Weather modes:
    - clearsky (upper bound, fast)
    - tmy (PVGIS Typical Meteorological Year; realistic; no API key)
• Physics:
    - Perez/Hay-Davies transposition (with dni_extra + version fallbacks)
    - ASHRAE IAM for beam
    - SAPM (fallback to Faiman) module temperature
    - Simple DC power model with temperature coefficient
• Outputs:
    - Results panel with tilts and energies
    - Plot: energy vs tilt
    - Download CSV + PNG

Run:
    streamlit run solar_tilt_gui.py
"""

from __future__ import annotations

import io
import traceback
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pvlib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import streamlit as st


# ------------------------------
# Version-robust helpers
# ------------------------------
def pick_sapm_params():
    """Return a valid SAPM temperature parameter set across pvlib versions."""
    try:
        sapm_sets = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]
        for k in [
            "open_rack_glass_polymer",
            "open_rack_glass_glass",
            "close_mount_glass_polymer",
            "close_mount_glass_glass",
            "insulated_back_glass_polymer",
            "insulated_back_glass_glass",
        ]:
            if k in sapm_sets:
                return sapm_sets[k]
    except Exception:
        pass
    # Fallback approximate (Sandia-ish open-rack)
    return dict(a=-3.56, b=-0.075, deltaT=3.0)


def ensure_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Localize/convert a DatetimeIndex to tz; if naive, assume it's local and localize."""
    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)


def compute_irradiance_robust(
    surface_tilt: float,
    surface_azimuth: float,
    solpos: pd.DataFrame,
    irr_in: pd.DataFrame,   # expects columns: dni, ghi, dhi
    albedo: float,
    dni_extra: pd.Series,
    airmass: pd.Series,
) -> pd.DataFrame:
    """
    Try Perez with 'allsites', fall back to '1990', then to Hay-Davies.
    Handles pvlib versions that require dni_extra or lack 'allsites'.
    """
    kwargs = dict(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=irr_in["dni"], ghi=irr_in["ghi"], dhi=irr_in["dhi"],
        albedo=albedo,
        dni_extra=dni_extra,
        airmass=airmass,
    )

    try:
        return pvlib.irradiance.get_total_irradiance(
            **kwargs, model="perez", model_perez="allsites"
        )
    except Exception:
        pass

    try:
        return pvlib.irradiance.get_total_irradiance(
            **kwargs, model="perez", model_perez="1990"
        )
    except Exception:
        pass

    return pvlib.irradiance.get_total_irradiance(
        **kwargs, model="haydavies"
    )


def energy_for_tilt(
    tilt_deg: float,
    times: pd.DatetimeIndex,
    irr_in: pd.DataFrame,          # dni/ghi/dhi
    solpos: pd.DataFrame,
    surface_azimuth: float,
    albedo: float,
    ashr_b0: float,
    eta_stc: float,
    area_m2: float,
    gamma_p: float,
    temp_air: pd.Series,           # °C
    wind_mps: pd.Series,           # m/s
    dni_extra: Optional[pd.Series] = None,
    airmass: Optional[pd.Series] = None,
) -> float:
    """Energy in Wh for a given tilt over the provided times."""
    surf_tilt = float(tilt_deg)
    if dni_extra is None:
        dni_extra = pvlib.irradiance.get_extra_radiation(times)
    if airmass is None:
        airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

    irr = compute_irradiance_robust(
        surface_tilt=surf_tilt,
        surface_azimuth=surface_azimuth,
        solpos=solpos,
        irr_in=irr_in,
        albedo=albedo,
        dni_extra=dni_extra,
        airmass=airmass,
    )

    # IAM (ASHRAE) on beam only
    aoi = pvlib.irradiance.aoi(
        surf_tilt, surface_azimuth,
        solpos["apparent_zenith"], solpos["azimuth"]
    )
    iam = pvlib.iam.ashrae(aoi, b=ashr_b0)

    poa_direct_eff = (irr["poa_direct"] * np.clip(iam, 0, None)).fillna(0.0)
    poa = poa_direct_eff + (irr["poa_global"] - irr["poa_direct"]).fillna(0.0)

    # Temperature: try SAPM, fallback to Faiman
    try:
        tparams = pick_sapm_params()
        tc = pvlib.temperature.sapm_cell(poa, temp_air, wind_mps, **tparams)
    except Exception:
        tc = pvlib.temperature.faiman(poa, temp_air, wind_mps, u0=25.0, u1=6.84)

    # DC power with temp coefficient
    p_dc_w = eta_stc * area_m2 * poa * (1.0 + gamma_p * (tc - 25.0))
    p_dc_w = p_dc_w.clip(lower=0.0)

    # Integrate to Wh
    e_wh = p_dc_w.resample("1D").sum(min_count=1)
    return float(e_wh.sum())


def optimize_fixed(
    times: pd.DatetimeIndex,
    irr_in: pd.DataFrame,
    solpos: pd.DataFrame,
    tilt_grid: np.ndarray,
    **kwargs
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    scores: List[Tuple[float, float]] = []
    for tilt in tilt_grid:
        e = energy_for_tilt(tilt, times, irr_in, solpos, **kwargs)
        scores.append((float(tilt), float(e)))
    best = max(scores, key=lambda x: x[1])
    return best, scores


def subset_and_optimize(
    times: pd.DatetimeIndex,
    irr_in: pd.DataFrame,
    solpos: pd.DataFrame,
    months: List[int],
    tilt_grid: np.ndarray,
    **kwargs
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    mask = times.month.isin(months)
    tsub = times[mask]

    irr_sub = irr_in.loc[tsub]
    sp_sub = solpos.loc[tsub]

    sub_kwargs = dict(kwargs)
    for key in ["temp_air", "wind_mps", "dni_extra", "airmass"]:
        if key in sub_kwargs and isinstance(sub_kwargs[key], pd.Series):
            sub_kwargs[key] = sub_kwargs[key].loc[tsub]

    return optimize_fixed(tsub, irr_sub, sp_sub, tilt_grid, **sub_kwargs)


# ------------------------------
# Weather prep (clearsky / PVGIS TMY)
# ------------------------------
@st.cache_data(show_spinner=False)
def fetch_pvgis_tmy(lat: float, lon: float, tz: str):
    """Fetch PVGIS TMY and return (times, irr_in, solpos, temp_air, wind)."""
    tmy, meta, _ = pvlib.iotools.get_pvgis_tmy(lat, lon, map_variables=True)
    times = ensure_tz(tmy.index, tz)
    irr_in = tmy[["dni", "ghi", "dhi"]].clip(lower=0.0)
    temp_air = tmy["temp_air"].astype(float).rename("temp_air")
    wind = (tmy["wind_speed"] if "wind_speed" in tmy.columns else pd.Series(1.0, index=times)).astype(float)
    wind = wind.rename("wind_speed")

    solpos = pvlib.location.Location(lat, lon, tz=tz).get_solarposition(times)
    return times, irr_in, solpos, temp_air, wind


def prepare_weather(
    weather_mode: str,
    lat: float, lon: float, tz: str,
    year: int,
    const_temp: float, const_wind: float
):
    """Return (times, irr_in, solpos, temp_air, wind) for the chosen weather mode."""
    loc = pvlib.location.Location(lat, lon, tz=tz)
    if weather_mode == "clearsky":
        start = pd.Timestamp(f"{year}-01-01 00:00", tz=tz)
        end = pd.Timestamp(f"{year}-12-31 23:00", tz=tz)
        times = pd.date_range(start, end, freq="1h", tz=tz)

        irr_in = loc.get_clearsky(times, model="ineichen")[["dni", "ghi", "dhi"]]
        temp_air = pd.Series(const_temp, index=times, name="temp_air")
        wind = pd.Series(const_wind, index=times, name="wind_speed")
        solpos = loc.get_solarposition(times)
        return times, irr_in, solpos, temp_air, wind

    # TMY mode
    return fetch_pvgis_tmy(lat, lon, tz)


# ------------------------------
# Plot & downloads
# ------------------------------
def plot_energy_curve(
    tilt_vals: np.ndarray,
    energy_vals_kwh: np.ndarray,
    tilt_star: float,
    lat_abs: float,
    weather_label: str,
) -> bytes:
    """Return PNG bytes of the energy vs tilt plot."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(tilt_vals, energy_vals_kwh, linewidth=2)
    ax.axvline(tilt_star, linestyle="--", alpha=0.75, label=f"Best tilt = {tilt_star:.1f}°")
    ax.axvline(lat_abs, linestyle=":", alpha=0.75, label=f"Latitude tilt = {lat_abs:.1f}°")
    ax.set_title(f"Annual Energy vs Tilt ({weather_label})")
    ax.set_xlabel("Tilt (degrees)")
    ax.set_ylabel("Annual Energy (kWh) per module")
    ax.legend(loc="best")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_csv_bytes(scores: List[Tuple[float, float]]) -> bytes:
    df = pd.DataFrame(scores, columns=["tilt_deg", "energy_Wh"])
    df["energy_kWh"] = df["energy_Wh"] / 1000.0
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Solar Tilt Optimizer", page_icon="🔆", layout="wide")
st.title("🔆 Solar Tilt Optimizer (Physics-Based)")

with st.sidebar:
    st.header("Site & Weather")
    col_loc = st.columns(2)
    lat = col_loc[0].number_input("Latitude (°)", value=42.36, step=0.01, format="%.4f")
    lon = col_loc[1].number_input("Longitude (°)", value=-71.06, step=0.01, format="%.4f")
    tz = st.text_input("Timezone (IANA)", value="America/New_York")
    weather_mode = st.radio("Weather", options=["clearsky", "tmy"], index=0, horizontal=True)
    year = st.number_input("Design year (clearsky mode)", min_value=2000, max_value=2100, value=2024, step=1)

    if weather_mode == "clearsky":
        st.caption("Clear-sky is an optimistic upper bound (no clouds).")
        col_tw = st.columns(2)
        const_temp = col_tw[0].number_input("Ambient temp (°C)", value=25.0, step=1.0, format="%.1f")
        const_wind = col_tw[1].number_input("Wind speed (m/s)", value=1.0, step=0.5, format="%.1f")
    else:
        st.caption("TMY downloads PVGIS Typical Meteorological Year (needs internet).")
        const_temp, const_wind = 25.0, 1.0  # unused

    st.header("Module & Surface")
    az = st.slider("Surface azimuth (°) — 180° = South", min_value=0, max_value=360, value=180, step=1)
    albedo = st.slider("Ground albedo", min_value=0.0, max_value=0.9, value=0.20, step=0.01)
    eta_stc = st.number_input("Module efficiency at STC (η)", value=0.20, step=0.005, format="%.3f")
    area_m2 = st.number_input("Module area (m²)", value=1.70, step=0.1, format="%.2f")
    gamma_p = st.number_input("Power tempco γ (per °C)", value=-0.004, step=0.001, format="%.4f")
    b0 = st.number_input("ASHRAE IAM b₀", value=0.05, step=0.005, format="%.3f")

    st.header("Optimization")
    col_tilt = st.columns(3)
    tilt_min = col_tilt[0].number_input("Tilt min (°)", min_value=0, max_value=90, value=0, step=1)
    tilt_max = col_tilt[1].number_input("Tilt max (°)", min_value=0, max_value=90, value=70, step=1)
    tilt_step = col_tilt[2].number_input("Tilt step (°)", min_value=1, max_value=15, value=1, step=1)
    start_end = st.slider("Summer months range", min_value=1, max_value=12, value=(4, 9))
    summer_start, summer_end = start_end

    run_btn = st.button("Run Optimization", type="primary")

# Main panel
st.write(
    "This tool computes best **fixed** and **2-position seasonal** tilts using real irradiance transposition, "
    "incidence-angle losses, and temperature effects. "
    "Use **TMY** for realistic kWh; **clearsky** gives an optimistic upper bound."
)

if run_btn:
    try:
        # Weather & solar position
        times, irr_in, solpos, temp_air, wind = prepare_weather(
            weather_mode=weather_mode,
            lat=lat, lon=lon, tz=tz,
            year=int(year),
            const_temp=float(const_temp),
            const_wind=float(const_wind),
        )

        # Precompute
        dni_extra = pvlib.irradiance.get_extra_radiation(times)
        airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

        # Common kwargs
        common = dict(
            surface_azimuth=float(az),
            albedo=float(albedo),
            ashr_b0=float(b0),
            eta_stc=float(eta_stc),
            area_m2=float(area_m2),
            gamma_p=float(gamma_p),
            temp_air=temp_air,
            wind_mps=wind,
            dni_extra=dni_extra,
            airmass=airmass,
        )

        # Fixed-tilt optimization
        tilt_grid = np.arange(int(tilt_min), int(tilt_max) + 1e-9, int(tilt_step))
        (tilt_star, e_star), grid = optimize_fixed(times, irr_in, solpos, tilt_grid, **common)

        # Baseline: latitude tilt
        e_lat = energy_for_tilt(abs(lat), times, irr_in, solpos, **common)
        gain_pct = 100.0 * (e_star / e_lat - 1.0)

        # Seasonal optimization
        if summer_start <= summer_end:
            summer_months = list(range(summer_start, summer_end + 1))
        else:
            # wrap-around (e.g., Nov–Mar)
            summer_months = list(range(summer_start, 13)) + list(range(1, summer_end + 1))
        winter_months = [m for m in range(1, 13) if m not in summer_months]

        (tilt_summer, e_summer), scores_s = subset_and_optimize(times, irr_in, solpos, summer_months, tilt_grid, **common)
        (tilt_winter, e_winter), scores_w = subset_and_optimize(times, irr_in, solpos, winter_months, tilt_grid, **common)
        e_two_pos = e_summer + e_winter

        # Capacity factor (rough; DC-side upper bound)
        p_stc_kw = eta_stc * area_m2 * 1000.0 / 1000.0  # kW at 1000 W/m²
        hours = len(times)
        cf = (e_star / 1000.0) / (p_stc_kw * hours) if p_stc_kw > 0 else np.nan

        # Layout: results + plot
        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("Results")
            st.markdown(f"**Location:** {lat:.4f}, {lon:.4f}  |  **Weather:** `{weather_mode}`")
            st.metric("Best fixed tilt", f"{tilt_star:.1f}°")
            st.metric("Latitude-tilt energy", f"{e_lat/1000:.1f} kWh / module")
            st.metric("Best fixed-tilt energy", f"{e_star/1000:.1f} kWh / module", delta=f"{gain_pct:.2f}% vs latitude")
            st.metric("Two-position tilts", f"Summer {tilt_summer:.1f}°  |  Winter {tilt_winter:.1f}°")
            st.metric("Two-position energy", f"{e_two_pos/1000:.1f} kWh / module")
            st.caption(f"Approx. capacity factor (DC-side): {100*cf:.1f}%  — TMY is realistic; clearsky is optimistic.")

        with colB:
            st.subheader("Energy vs Tilt")
            tilt_vals = np.array([g for g, _ in grid], dtype=float)
            energy_vals_kwh = np.array([e for _, e in grid], dtype=float) / 1000.0

            png_bytes = plot_energy_curve(
                tilt_vals, energy_vals_kwh, tilt_star, abs(lat),
                weather_label=("TMY" if weather_mode == "tmy" else "Clear-sky")
            )
            st.image(png_bytes, caption="Annual energy per module vs tilt", use_container_width=True)

        # Downloads
        st.subheader("Download")
        csv_bytes = make_csv_bytes(grid)
        st.download_button("Download energy curve (CSV)", data=csv_bytes, file_name="tilt_energy_curve.csv", mime="text/csv")
        st.download_button("Download plot (PNG)", data=png_bytes, file_name="tilt_curve.png", mime="image/png")

        # Details / debug
        with st.expander("Details & Assumptions"):
            st.write({
                "azimuth_deg": az,
                "albedo": albedo,
                "eta_stc": eta_stc,
                "area_m2": area_m2,
                "gamma_p_per_C": gamma_p,
                "ashrae_b0": b0,
                "summer_months": summer_months,
                "winter_months": winter_months,
                "timeline_hours": hours,
            })
            if weather_mode == "clearsky":
                st.write({"clear_sky_temp_C": const_temp, "clear_sky_wind_mps": const_wind})
            else:
                st.write("PVGIS TMY fields available:", list(irr_in.columns) + ["temp_air", "wind_speed"])

    except Exception as e:
        st.error("Something went wrong. See details below.")
        st.code("".join(traceback.format_exc()))
