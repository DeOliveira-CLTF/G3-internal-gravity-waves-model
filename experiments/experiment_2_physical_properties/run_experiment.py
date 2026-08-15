"""Experiment 2: physical effects using the legacy plotting specification."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gravity_waves.common import (  # noqa: E402
    EARTH_ROTATION_RATE,
    N_BV,
    SCHEMES,
    add_panel_labels,
    modified_wavenumber,
    save_figure,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REFERENCE_DZ = 250.0
REFERENCE_LATITUDE = 45.0
REFERENCE_WIND = 15.0
CORIOLIS = 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(REFERENCE_LATITUDE))

def save_physical_figure(figure, stem):
    """Save at the exact 160-dpi raster resolution used by the legacy script."""
    save_figure(figure, OUTPUT_DIR, stem, dpi=160)


def write_csv(filename: str, rows: list[dict[str, object]]) -> None:
    """Write rows using the legacy diagnostic sampling."""
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rotating_frequency(k, m, coriolis=CORIOLIS, buoyancy_frequency=N_BV):
    """Continuous inertia-gravity frequency."""
    return np.sqrt(
        (buoyancy_frequency**2 * np.asarray(k) ** 2 + coriolis**2 * np.asarray(m) ** 2)
        / (np.asarray(k) ** 2 + np.asarray(m) ** 2)
    )


def numerical_rotating_frequency(k, m, dz, function, coriolis=CORIOLIS,
                                 buoyancy_frequency=N_BV):
    """Semi-discrete inertia-gravity frequency."""
    m_star = modified_wavenumber(m, dz, function)
    return np.sqrt(
        (buoyancy_frequency**2 * np.asarray(k) ** 2 + coriolis**2 * m_star**2)
        / (np.asarray(k) ** 2 + m_star**2)
    )


def intrinsic_frequency(k, m, buoyancy_frequency=N_BV):
    return buoyancy_frequency * np.abs(k) / np.sqrt(np.asarray(k) ** 2 + np.asarray(m) ** 2)


def observed_frequency(k, m, wind=REFERENCE_WIND):
    return wind * np.asarray(k) + intrinsic_frequency(k, m)


def numerical_observed_frequency(k, m, dz, function, wind=REFERENCE_WIND):
    m_star = modified_wavenumber(m, dz, function)
    return wind * np.asarray(k) + N_BV * np.abs(k) / np.sqrt(np.asarray(k) ** 2 + m_star**2)


def analytical_group_velocity(k, m):
    return -N_BV * k * m / (k**2 + m**2) ** 1.5


def numerical_group_velocity(k, m, dz, function, dm_fraction=1.0e-4):
    """Legacy centered differentiation of the discrete dispersion relation."""
    dm = dm_fraction * max(abs(m), 1.0 / dz)
    omega_plus = N_BV * abs(k) / np.sqrt(k**2 + modified_wavenumber(m + dm, dz, function) ** 2)
    omega_minus = N_BV * abs(k) / np.sqrt(k**2 + modified_wavenumber(m - dm, dz, function) ** 2)
    return (omega_plus - omega_minus) / (2.0 * dm)


def plot_separate_physical_effects() -> None:
    """Reproduce the three legacy single-panel physical-effect figures."""
    theta0 = np.pi / 2.0
    m = theta0 / REFERENCE_DZ
    horizontal_wavenumber = np.linspace(1.0e-3, 4.0, 400) * m

    figure, axis = plt.subplots(figsize=(8, 5.8))
    axis.plot(horizontal_wavenumber / m, rotating_frequency(horizontal_wavenumber, m) / N_BV,
              color="black", lw=2.4, label="Analytical", zorder=5)
    for name, scheme in SCHEMES.items():
        values = numerical_rotating_frequency(
            horizontal_wavenumber, m, REFERENCE_DZ, scheme["function"]
        ) / N_BV
        axis.plot(horizontal_wavenumber / m, values, scheme["linestyle"],
                  color=scheme["color"], lw=1.7, label=name)
    axis.axhline(CORIOLIS / N_BV, color="gray", lw=1, ls=":", label=r"Floor $\omega=f$")
    axis.set_xlabel(r"$k/m$", fontsize=14)
    axis.set_ylabel(r"$\omega/N$", fontsize=14)
    axis.set_title("Inertia-gravity waves\n(4-dz wave, f at 45 degrees N)", fontsize=16, fontweight="bold")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=10, loc="lower right")
    figure.tight_layout()
    save_physical_figure(figure, "rotation_effect")

    critical_wavenumber = N_BV / REFERENCE_WIND
    signed_wavenumber = np.linspace(-4.0, 4.0, 400) * critical_wavenumber
    figure, axis = plt.subplots(figsize=(8, 5.8))
    axis.plot(signed_wavenumber / critical_wavenumber, observed_frequency(signed_wavenumber, m) / N_BV,
              color="black", lw=2.4, label="Analytical", zorder=5)
    for name, scheme in SCHEMES.items():
        values = numerical_observed_frequency(
            signed_wavenumber, m, REFERENCE_DZ, scheme["function"]
        ) / N_BV
        axis.plot(signed_wavenumber / critical_wavenumber, values, scheme["linestyle"],
                  color=scheme["color"], lw=1.7, label=name)
    axis.set_xlabel(r"$k/k_c$,  $k_c \equiv N/U$  (negative = westward propagation)", fontsize=14)
    axis.set_ylabel(r"$\omega_{observed}/N$", fontsize=14)
    axis.set_title(f"Doppler shift\n(background wind U={REFERENCE_WIND:.0f} m/s)", fontsize=16, fontweight="bold")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=10, loc="upper left")
    figure.tight_layout()
    save_physical_figure(figure, "doppler_effect")

    theta = np.linspace(0.05, 0.98 * np.pi, 200)
    m_range = theta / REFERENCE_DZ
    fixed_k = m
    figure, axis = plt.subplots(figsize=(8, 5.8))
    analytical = [analytical_group_velocity(fixed_k, value) for value in m_range]
    axis.plot(theta / np.pi, analytical, color="black", lw=2.4,
              label="Analytical", zorder=5)
    for name, scheme in SCHEMES.items():
        numerical = [numerical_group_velocity(fixed_k, value, REFERENCE_DZ,
                                              scheme["function"]) for value in m_range]
        axis.plot(theta / np.pi, numerical, scheme["linestyle"],
                  color=scheme["color"], lw=1.7, label=name)
    axis.set_xlabel(r"$m\Delta z / \pi$")
    axis.set_ylabel(r"$c_{gz}$  [m/s]")
    axis.set_title("Vertical group velocity\n(fixed horizontal scale)",
                   fontsize=10, fontweight="bold")
    axis.legend(fontsize=6.5, loc="lower left")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    save_physical_figure(figure, "vertical_group_velocity")


def plot_doppler_critical_regime() -> None:
    """Reproduce the legacy 3-by-2 critical-regime figure and heatmaps."""
    ppwz_values = [4, 6, 8]
    chi_values = [0.20, 0.40, 0.60, 0.75, 0.85, 0.90, 0.95]
    intrinsic_errors = {name: np.full((3, 7), np.nan) for name in SCHEMES}
    stationarity_residuals = {name: np.full((3, 7), np.nan) for name in SCHEMES}
    rows: list[dict[str, object]] = []

    for chi_index, chi in enumerate(chi_values):
        k = chi * N_BV / abs(REFERENCE_WIND)
        m = (N_BV / abs(REFERENCE_WIND)) * np.sqrt(1.0 - chi**2)
        wavelength = 2.0 * np.pi / m
        analytical = intrinsic_frequency(k, m)
        for ppwz_index, ppwz in enumerate(ppwz_values):
            dz = wavelength / ppwz
            for name, scheme in SCHEMES.items():
                m_star = modified_wavenumber(m, dz, scheme["function"])
                numerical = N_BV * abs(k) / np.sqrt(k**2 + m_star**2)
                observed_numerical = REFERENCE_WIND * k - numerical
                relative_error = 100.0 * abs(numerical - analytical) / max(abs(analytical), 1.0e-15)
                residual = 100.0 * abs(observed_numerical) / N_BV
                intrinsic_errors[name][ppwz_index, chi_index] = relative_error
                stationarity_residuals[name][ppwz_index, chi_index] = residual
                rows.append({
                    "chi": chi, "wind_m_s": REFERENCE_WIND, "PPWz": ppwz,
                    "scheme": name, "k_m-1": k, "m_m-1": m,
                    "k_over_m": k / m, "vertical_wavelength_m": wavelength,
                    "dz_m": dz, "analytical_intrinsic_frequency_s-1": analytical,
                    "numerical_intrinsic_frequency_s-1": numerical,
                    "relative_intrinsic_frequency_error_pct": relative_error,
                    "stationarity_residual_over_N_pct": residual,
                })

    figure, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    for row_index, ppwz in enumerate(ppwz_values):
        left, right = axes[row_index]
        for name, scheme in SCHEMES.items():
            left.plot(chi_values, intrinsic_errors[name][row_index], scheme["linestyle"],
                      color=scheme["color"], lw=3.6, marker="o", ms=7, label=name)
            right.plot(chi_values, stationarity_residuals[name][row_index], scheme["linestyle"],
                       color=scheme["color"], lw=3.6, marker="o", ms=7, label=name)
        left.axvline(1.0, color="gray", ls=":", lw=1.1)
        right.axvline(1.0, color="gray", ls=":", lw=1.1)
        left.set_yscale("log")
        right.set_yscale("log")
        left.set_ylabel(f"PPWz={ppwz}\n$\omega_i$ error [%]", fontsize=16)
        right.set_ylabel(f"PPWz={ppwz}\n" + r"$|\omega_{obs,num}|/N$ [%]", fontsize=16)
        for axis in (left, right):
            axis.grid(alpha=0.3, which="both")
            axis.tick_params(axis="both", labelsize=14)
        if row_index == 0:
            left.set_title("Relative intrinsic-frequency error", fontsize=20, fontweight="bold")
            right.set_title("Stationarity residual", fontsize=20, fontweight="bold")
    axes[-1, 0].set_xlabel(r"$\chi=|Uk|/N$", fontsize=16)
    axes[-1, 1].set_xlabel(r"$\chi=|Uk|/N$", fontsize=16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=16,
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=[0.02, 0.05, 1.0, 0.94])
    add_panel_labels(figure, axes)
    save_physical_figure(figure, "doppler_critical_regime")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    for axis, (name, scheme) in zip(axes.flat, SCHEMES.items()):
        mesh = axis.pcolormesh(chi_values, ppwz_values, intrinsic_errors[name], shading="auto")
        figure.colorbar(mesh, ax=axis, label=r"$\omega_i$ error [%]")
        axis.set_title(name, fontsize=20, fontweight="bold")
        axis.set_xlabel(r"$\chi=|Uk|/N$", fontsize=16)
        axis.set_ylabel("PPWz", fontsize=16)
        axis.tick_params(axis="both", labelsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    add_panel_labels(figure, axes)
    save_physical_figure(figure, "doppler_critical_regime_heatmaps")
    write_csv("doppler_critical_regime_metrics.csv", rows)


def plot_rotation_resolution_interaction() -> None:
    """Reproduce the legacy 3-by-3 rotation-resolution comparison and summary."""
    ppwz_values = [4, 6, 8]
    latitudes = [0.0, 45.0, 90.0]
    wavelength = 2000.0
    m = 2.0 * np.pi / wavelength
    k_over_m = np.logspace(-3.0, np.log10(4.0), 400)
    k = k_over_m * m
    maximum_errors = {
        name: np.zeros((len(latitudes), len(ppwz_values))) for name in SCHEMES
    }

    figure, axes = plt.subplots(3, 3, figsize=(17, 16), sharex=True, sharey=True)
    for latitude_index, latitude in enumerate(latitudes):
        coriolis = 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(latitude))
        for ppwz_index, ppwz in enumerate(ppwz_values):
            axis = axes[latitude_index, ppwz_index]
            dz = wavelength / ppwz
            analytical = rotating_frequency(k, m, coriolis, N_BV) / N_BV
            axis.plot(k_over_m, analytical, color="black", lw=2.2,
                      label="Analytical", zorder=5)
            for name, scheme in SCHEMES.items():
                numerical = numerical_rotating_frequency(
                    k, m, dz, scheme["function"], coriolis, N_BV
                ) / N_BV
                axis.plot(k_over_m, numerical, scheme["linestyle"],
                          color=scheme["color"], lw=1.4, label=name)
                error = np.abs(numerical - analytical) / np.maximum(np.abs(analytical), 1.0e-12)
                maximum_errors[name][latitude_index, ppwz_index] = 100.0 * np.max(error)
            if coriolis > 0.0:
                axis.axhline(coriolis / N_BV, color="gray", lw=0.9, ls=":")
            axis.set_xscale("log")
            axis.set_ylim(0.0, 1.04)
            axis.grid(alpha=0.3, which="both")
            axis.tick_params(axis="both", labelsize=14)
            if latitude_index == 0:
                axis.set_title(f"PPWz={ppwz}\ndz={dz:.1f} m", fontsize=20, fontweight="bold")
            if ppwz_index == 0:
                axis.set_ylabel(rf"Lat={latitude:.0f}$^\circ$" + "\n" + r"$\omega/N$", fontsize=16)
            if latitude_index == len(latitudes) - 1:
                axis.set_xlabel(r"$k/m$", fontsize=16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, fontsize=16,
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=[0.02, 0.05, 1.0, 0.97])
    add_panel_labels(figure, axes)
    save_physical_figure(figure, "rotation_resolution_interaction")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    for axis, (name, scheme) in zip(axes.flat, SCHEMES.items()):
        for latitude_index, latitude in enumerate(latitudes):
            axis.plot(ppwz_values, maximum_errors[name][latitude_index],
                      marker="o", lw=1.5, label=rf"{latitude:.0f}$^\circ$")
        axis.set_yscale("log")
        axis.set_title(name, fontsize=20, fontweight="bold")
        axis.set_xlabel("PPWz")
        axis.set_ylabel(r"Maximum relative $\omega$ error [%]", fontsize=16)
        axis.grid(alpha=0.3, which="both")
        axis.tick_params(axis="both", labelsize=14)
        axis.legend(fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    add_panel_labels(figure, axes)
    save_physical_figure(figure, "rotation_resolution_error_summary")

def plot_rotation_physics() -> None:
    """Reproduce the two legacy latitude/stratification figures."""
    latitudes = [0.0, 30.0, 60.0, 90.0]
    buoyancy_frequencies = [0.003, 0.006, 0.012, 0.018]
    wavelength = 2000.0
    m = 2.0 * np.pi / wavelength
    k_over_m = np.logspace(-4.0, -0.3, 400)
    k = k_over_m * m

    figure, axis = plt.subplots(figsize=(10, 7))
    for latitude in latitudes:
        coriolis = 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(latitude))
        axis.plot(k_over_m, rotating_frequency(k, m, coriolis, 0.012) / 0.012,
                  lw=2.4, label=rf"{latitude:.0f}$^\circ$  ($f/N={coriolis / 0.012:.3f}$)")
    axis.set_xscale("log")
    axis.set_xlabel(r"$k/m$", fontsize=16)
    axis.set_ylabel(r"$\omega/N$", fontsize=16)
    axis.set_title("Rotation effect on the dispersion relation\n"
                   r"($N=0.012\ s^{-1}$; same vertical wave)", fontsize=20, fontweight="bold")
    axis.grid(alpha=0.3, which="both")
    axis.tick_params(axis="both", labelsize=14)
    axis.legend(fontsize=12)
    figure.tight_layout()
    save_physical_figure(figure, "rotation_physics_latitude")

    figure, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    for axis, buoyancy_frequency in zip(axes, buoyancy_frequencies):
        for latitude in latitudes:
            coriolis = 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(latitude))
            axis.plot(k_over_m, rotating_frequency(k, m, coriolis, buoyancy_frequency)
                      / buoyancy_frequency, lw=2.2, label=rf"{latitude:.0f}$^\circ$")
        axis.set_xscale("log")
        axis.set_ylabel(r"$\omega/N$", fontsize=14)
        axis.set_title(rf"$N={buoyancy_frequency:.3f}\ s^{{-1}}$", fontsize=16, fontweight="bold")
        axis.grid(alpha=0.3, which="both")
        axis.tick_params(axis="both", labelsize=12)
    axes[-1].set_xlabel(r"$k/m$", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=12,
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=[0, 0.05, 1, 0.95])
    add_panel_labels(figure, axes)
    save_physical_figure(figure, "rotation_physics_latitude_and_N")


def plot_rotation_error_maps() -> None:
    """Reproduce the legacy Coriolis maps with one global color scale."""
    ppwz_values = [4, 6, 8]
    wavelength = 2000.0
    m = 2.0 * np.pi / wavelength
    k_over_m = np.logspace(-4.0, 0.3, 220)
    f_over_n = np.logspace(-4.0, -0.2, 180)
    km_grid, fn_grid = np.meshgrid(k_over_m, f_over_n)
    k_grid = km_grid * m
    analytical = np.sqrt((k_grid**2 + fn_grid**2 * m**2) / (k_grid**2 + m**2))
    errors: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    global_maximum = 0.0
    for ppwz in ppwz_values:
        dz = wavelength / ppwz
        errors[ppwz] = {}
        for name, scheme in SCHEMES.items():
            m_star = modified_wavenumber(m, dz, scheme["function"])
            numerical = np.sqrt((k_grid**2 + fn_grid**2 * m_star**2) / (k_grid**2 + m_star**2))
            error = 100.0 * np.abs(numerical - analytical) / np.maximum(np.abs(analytical), 1.0e-14)
            errors[ppwz][name] = error
            global_maximum = max(global_maximum, float(np.nanmax(error)))
            rows.append({"scheme": name, "PPWz": ppwz, "dz_m": dz,
                         "maximum_relative_frequency_error_pct": float(np.nanmax(error))})

    for ppwz in ppwz_values:
        figure = plt.figure(figsize=(15, 11))
        grid = figure.add_gridspec(
            2, 3, width_ratios=[1.0, 1.0, 0.055], height_ratios=[1.0, 1.0],
            left=0.075, right=0.93, bottom=0.085, top=0.82, wspace=0.28, hspace=0.34,
        )
        axis00 = figure.add_subplot(grid[0, 0])
        axes = [
            axis00,
            figure.add_subplot(grid[0, 1], sharex=axis00, sharey=axis00),
            figure.add_subplot(grid[1, 0], sharex=axis00, sharey=axis00),
            figure.add_subplot(grid[1, 1], sharex=axis00, sharey=axis00),
        ]
        colorbar_axis = figure.add_subplot(grid[:, 2])
        mesh = None
        for axis, (name, scheme) in zip(axes, SCHEMES.items()):
            mesh = axis.pcolormesh(k_over_m, f_over_n, errors[ppwz][name],
                                   shading="auto", rasterized=True,
                                   vmin=0.0, vmax=global_maximum)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(name, fontsize=17, fontweight="bold", pad=10)
            axis.set_xlabel(r"$k/m$", fontsize=15)
            axis.set_ylabel(r"$f/N$", fontsize=15)
            axis.tick_params(axis="both", labelsize=12)
            axis.grid(alpha=0.15, which="both")
        assert mesh is not None
        colorbar = figure.colorbar(mesh, cax=colorbar_axis)
        colorbar.set_label(r"Relative $\omega$ error [%]", fontsize=15, labelpad=14)
        colorbar.ax.tick_params(labelsize=12)
        add_panel_labels(figure, axes)
        save_physical_figure(figure, f"rotation_error_km_fN_PPWz{ppwz}")
    write_csv("rotation_frequency_metrics.csv", rows)


def main() -> None:
    """Generate the physical figures using the legacy visual design."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "path"
    plot_separate_physical_effects()
    plot_doppler_critical_regime()
    plot_rotation_resolution_interaction()
    plot_rotation_physics()
    plot_rotation_error_maps()
    print(f"Experiment 2 complete. Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()