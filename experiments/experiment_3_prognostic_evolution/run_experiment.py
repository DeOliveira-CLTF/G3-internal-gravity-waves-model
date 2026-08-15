"""Experiment 3: prognostic dispersion tests with the legacy visual design."""

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
    N_BV, SCHEMES, analytical_frequency, analytical_vertical_group_velocity,
    numerical_frequency, numerical_vertical_group_velocity, save_figure,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LAMBDA_Z = 4_000.0
M0 = 2.0 * np.pi / LAMBDA_Z
K0 = M0
DOMAIN_LENGTH = 20.0 * LAMBDA_Z
PACKET_CENTER = 0.50 * DOMAIN_LENGTH
SIGMA_M = M0 / 20.0
STEPS_PER_PERIOD = 200
PPWZ_MODE = (2, 4, 6, 8, 12, 16, 24, 32)
PPWZ_MODE_FIGURE = (4, 8, 16, 32)
PPWZ_PACKET = (2.5, 3, 4, 8)
LEGEND_FONT = 11
PANEL_FONT = 16


def configure_legacy_typography() -> None:
    """Apply the publication typography used by the legacy prognostic script."""
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": LEGEND_FONT,
        # Paths retain the legacy appearance while preventing LaTeX text overlap.
        "svg.fonttype": "path",
    })


def label_legacy_panels(figure, axes) -> None:
    """Use the exact external panel-label placement of the legacy script."""
    figure.canvas.draw()
    for letter, axis in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        box = axis.get_position()
        figure.text(box.x0 - 0.035, box.y1 + 0.012, f"({letter})",
                    fontsize=PANEL_FONT, fontweight="bold", ha="right", va="bottom")


def write_csv(filename, rows) -> None:
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rk4_factor(omega, dt):
    argument = -1j * np.asarray(omega) * dt
    return 1.0 + argument + argument**2 / 2.0 + argument**3 / 6.0 + argument**4 / 24.0


def simulate_single_mode(ppwz, periods=50):
    dz = LAMBDA_Z / ppwz
    exact_omega = float(analytical_frequency(K0, M0))
    period = 2.0 * np.pi / exact_omega
    dt = period / STEPS_PER_PERIOD
    time = np.arange(periods * STEPS_PER_PERIOD + 1) * dt
    analytical = np.exp(-1j * exact_omega * time)
    result = {"time": time, "period": period, "schemes": {}}
    for name, scheme in SCHEMES.items():
        numerical_omega = float(numerical_frequency(K0, M0, dz, scheme["function"]))
        amplitude = rk4_factor(numerical_omega, dt) ** np.arange(time.size)
        result["schemes"][name] = {
            "omega": numerical_omega,
            "phase_error": np.unwrap(np.angle(amplitude)) + exact_omega * time,
            "predicted_phase_error": (exact_omega - numerical_omega) * time,
            "error_l2": np.abs(amplitude - analytical),
            "relative_energy": np.abs(amplitude) ** 2 - 1.0,
        }
    return result


def sample_indices(number_of_steps, number_of_samples):
    """Return the exact sampling indices used by the legacy experiment."""
    return np.unique(np.rint(np.linspace(0, number_of_steps, number_of_samples)).astype(int))


def modal_rk4_step(vorticity, buoyancy, dt, k, denominator):
    """Advance zeta_t=i k b and b_t=i k N^2 zeta/D by one RK4 step."""
    def tendency(vorticity_state, buoyancy_state):
        return (
            1j * k * buoyancy_state,
            1j * k * N_BV**2 * vorticity_state / denominator,
        )

    k1v, k1b = tendency(vorticity, buoyancy)
    k2v, k2b = tendency(vorticity + 0.5 * dt * k1v, buoyancy + 0.5 * dt * k1b)
    k3v, k3b = tendency(vorticity + 0.5 * dt * k2v, buoyancy + 0.5 * dt * k2b)
    k4v, k4b = tendency(vorticity + dt * k3v, buoyancy + dt * k3b)
    return (
        vorticity + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        buoyancy + dt * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0,
    )


def center_and_width(z, density):
    """Use the legacy non-periodic first and second energy moments."""
    weight = np.asarray(density, dtype=float)
    total = np.sum(weight)
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Invalid wave-packet energy.")
    center = float(np.sum(z * weight) / total)
    width = float(np.sqrt(np.sum((z - center) ** 2 * weight) / total))
    return center, width, float(total)


def exact_packet_state(amplitude, omega, time, k, vertical_operator, denominator):
    """Construct the positive-frequency modal eigenstate at a specified time."""
    streamfunction_hat = amplitude * np.exp(-1j * omega * time)
    vorticity_hat = -denominator * streamfunction_hat
    buoyancy_hat = N_BV**2 * k * streamfunction_hat / omega
    return vorticity_hat, buoyancy_hat


def diagnose_packet(vorticity_hat, buoyancy_hat, z, k, vertical_operator, denominator):
    """Diagnose the legacy kinetic-plus-potential energy density."""
    streamfunction_hat = -vorticity_hat / denominator
    streamfunction = np.fft.ifft(streamfunction_hat)
    buoyancy = np.fft.ifft(buoyancy_hat)
    horizontal_velocity = np.fft.ifft(-1j * vertical_operator * streamfunction_hat)
    vertical_velocity = np.fft.ifft(1j * k * streamfunction_hat)
    energy = 0.5 * (
        np.abs(horizontal_velocity) ** 2
        + np.abs(vertical_velocity) ** 2
        + np.abs(buoyancy) ** 2 / N_BV**2
    )
    center, width, total = center_and_width(z, energy)
    return center, width, total, streamfunction


def simulate_wave_packet(ppwz, periods=10):
    """Run the wave packet exactly as in the legacy implementation."""
    dz = LAMBDA_Z / ppwz
    points = int(round(DOMAIN_LENGTH / dz))
    z = np.arange(points) * dz
    m = 2.0 * np.pi * np.fft.fftfreq(points, d=dz)
    support = (m > 0.0) & (np.abs(m - M0) <= 4.0 * SIGMA_M)
    amplitude = np.zeros(points, dtype=complex)
    amplitude[support] = (
        np.exp(-0.5 * ((m[support] - M0) / SIGMA_M) ** 2)
        * np.exp(-1j * m[support] * PACKET_CENTER)
    )

    central_omega = float(analytical_frequency(K0, M0))
    period = 2.0 * np.pi / central_omega
    dt = period / STEPS_PER_PERIOD
    steps = int(round(periods * STEPS_PER_PERIOD))
    saved_steps = set(sample_indices(steps, 101).tolist())

    analytical_denominator = K0**2 + m**2
    analytical_omega = N_BV * abs(K0) / np.sqrt(analytical_denominator)
    safe_analytical_omega = np.where(analytical_omega > 0.0, analytical_omega, 1.0)
    result = {"z": z, "period": period, "schemes": {}}

    for name, scheme in SCHEMES.items():
        m_star = scheme["function"](m * dz) / dz
        numerical_denominator = K0**2 + m_star**2
        numerical_omega = N_BV * abs(K0) / np.sqrt(numerical_denominator)
        numerical_vorticity, numerical_buoyancy = exact_packet_state(
            amplitude, numerical_omega, 0.0, K0, m_star, numerical_denominator
        )

        times = []
        analytical_centers = []
        numerical_centers = []
        analytical_widths = []
        numerical_widths = []
        numerical_energies = []
        for step in range(steps + 1):
            if step in saved_steps:
                time = step * dt
                analytical_vorticity, analytical_buoyancy = exact_packet_state(
                    amplitude, safe_analytical_omega, time, K0, m,
                    analytical_denominator,
                )
                analytical_center, analytical_width, _, _ = diagnose_packet(
                    analytical_vorticity, analytical_buoyancy, z, K0, m,
                    analytical_denominator,
                )
                numerical_center, numerical_width, numerical_energy, _ = diagnose_packet(
                    numerical_vorticity, numerical_buoyancy, z, K0, m_star,
                    numerical_denominator,
                )
                times.append(time)
                analytical_centers.append(analytical_center)
                numerical_centers.append(numerical_center)
                analytical_widths.append(analytical_width)
                numerical_widths.append(numerical_width)
                numerical_energies.append(numerical_energy)
            if step < steps:
                numerical_vorticity, numerical_buoyancy = modal_rk4_step(
                    numerical_vorticity, numerical_buoyancy, dt, K0,
                    numerical_denominator,
                )

        times = np.asarray(times)
        analytical_centers = np.asarray(analytical_centers)
        numerical_centers = np.asarray(numerical_centers)
        analytical_widths = np.asarray(analytical_widths)
        numerical_widths = np.asarray(numerical_widths)
        numerical_energies = np.asarray(numerical_energies)
        predicted_cgz = float(numerical_vertical_group_velocity(
            K0, M0, dz, scheme["function"], scheme["derivative"]
        ))
        result["time"] = times
        result["analytical_center"] = analytical_centers
        result["analytical_width"] = analytical_widths
        result["schemes"][name] = {
            "center": numerical_centers,
            "width": numerical_widths,
            "predicted_cgz": predicted_cgz,
            "predicted_center_error": (
                predicted_cgz - analytical_vertical_group_velocity(K0, M0)
            ) * times,
            "relative_energy": numerical_energies / numerical_energies[0] - 1.0,
        }
    return result

def panel_title(ppwz):
    return (rf"PPW$_z$={ppwz:g}; $\Delta z$={LAMBDA_Z / ppwz:.0f} m; "
            rf"$\theta$={2.0 * np.pi / ppwz:.3f} rad")


def plot_single_mode(results):
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    for axis, ppwz in zip(axes.flat, PPWZ_MODE_FIGURE):
        result = results[ppwz]
        periods = result["time"] / result["period"]
        for name, scheme in SCHEMES.items():
            values = result["schemes"][name]
            axis.plot(periods, values["phase_error"], scheme["linestyle"],
                      color=scheme["color"], lw=1.8, label=name)
            axis.plot(periods, values["predicted_phase_error"],
                      color=scheme["color"], lw=0.8, alpha=0.45)
        axis.axhline(0.0, color="0.25", lw=0.8)
        axis.set_title(panel_title(ppwz))
        axis.set_xlabel(r"$t/T_{ana}$")
        axis.set_ylabel(r"$\Delta\phi=\phi_{num}-\phi_{ana}$ [rad]")
        axis.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=LEGEND_FONT)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.94,
                           wspace=0.34, hspace=0.38)
    label_legacy_panels(figure, axes)
    save_figure(figure, OUTPUT_DIR, "normal_mode_phase_error")

    result = results[8]
    periods = result["time"] / result["period"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for name, scheme in SCHEMES.items():
        values = result["schemes"][name]
        axes[0].plot(periods, values["phase_error"], scheme["linestyle"],
                     color=scheme["color"], lw=1.7, label=name)
        axes[1].plot(periods, values["error_l2"], scheme["linestyle"],
                     color=scheme["color"], lw=1.7)
        axes[2].semilogy(periods, np.maximum(np.abs(values["relative_energy"]), 1.0e-18),
                         scheme["linestyle"], color=scheme["color"], lw=1.7)
    axes[0].set_ylabel(r"$\Delta\phi=\phi_{num}-\phi_{ana}$ [rad]")
    axes[1].set_ylabel(r"$E_{L_2}(t)=\|\psi_{num}-\psi_{ana}\|_2/\|\psi_{ana}\|_2$")
    axes[2].set_ylabel(r"$|\Delta E/E_0|$")
    for axis in axes:
        axis.set_xlabel(r"$t/T_{ana}$")
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=LEGEND_FONT)
    figure.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.91, wspace=0.38)
    label_legacy_panels(figure, axes)
    save_figure(figure, OUTPUT_DIR, "normal_mode_time_integration_diagnostics")


def make_packet_axes():
    return plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)


def finish_packet_figure(figure, axes, stem):
    axes.flat[0].legend(fontsize=LEGEND_FONT)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.94,
                           wspace=0.34, hspace=0.38)
    label_legacy_panels(figure, axes)
    save_figure(figure, OUTPUT_DIR, stem)


def plot_wave_packet(results):
    figure, axes = make_packet_axes()
    for axis, ppwz in zip(axes.flat, PPWZ_PACKET):
        result = results[ppwz]
        periods = result["time"] / result["period"]
        axis.plot(periods, result["analytical_center"] / 1000.0,
                  color="black", lw=2.2, label="Analytical")
        for name, scheme in SCHEMES.items():
            axis.plot(periods, result["schemes"][name]["center"] / 1000.0,
                      scheme["linestyle"], color=scheme["color"], lw=1.8, label=name)
        axis.set_title(panel_title(ppwz))
        axis.set_xlabel(r"$t/T_{ana}$")
        axis.set_ylabel(r"energy centroid $z_E$ [km]")
        axis.grid(alpha=0.3)
    finish_packet_figure(figure, axes, "wave_packet_energy_centroid")

    figure, axes = make_packet_axes()
    for axis, ppwz in zip(axes.flat, PPWZ_PACKET):
        result = results[ppwz]
        periods = result["time"] / result["period"]
        for name, scheme in SCHEMES.items():
            values = result["schemes"][name]
            error = values["center"] - result["analytical_center"]
            axis.plot(periods, error / 1000.0, scheme["linestyle"],
                      color=scheme["color"], lw=1.8, label=name)
            axis.plot(periods, values["predicted_center_error"] / 1000.0,
                      color=scheme["color"], lw=0.8, alpha=0.45)
        axis.axhline(0.0, color="0.25", lw=0.8)
        axis.set_title(panel_title(ppwz))
        axis.set_xlabel(r"$t/T_{ana}$")
        axis.set_ylabel(r"$\Delta z_E=z_{E,num}-z_{E,ana}$ [km]")
        axis.grid(alpha=0.3)
    finish_packet_figure(figure, axes, "wave_packet_position_error")

    figure, axes = make_packet_axes()
    for axis, ppwz in zip(axes.flat, PPWZ_PACKET):
        result = results[ppwz]
        periods = result["time"] / result["period"]
        analytical_change = result["analytical_width"] - result["analytical_width"][0]
        axis.plot(periods, analytical_change / 1000.0,
                  color="black", lw=2.2, label="Analytical")
        for name, scheme in SCHEMES.items():
            width = result["schemes"][name]["width"]
            axis.plot(periods, (width - width[0]) / 1000.0, scheme["linestyle"],
                      color=scheme["color"], lw=1.8, label=name)
        axis.axhline(0.0, color="0.25", lw=0.8)
        axis.set_title(panel_title(ppwz))
        axis.set_xlabel(r"$t/T_{ana}$")
        axis.set_ylabel(r"$\Delta\sigma_z=\sigma_z(t)-\sigma_z(0)$ [km]")
        axis.grid(alpha=0.3)
    finish_packet_figure(figure, axes, "wave_packet_widening")

    figure, axes = make_packet_axes()
    for axis, ppwz in zip(axes.flat, PPWZ_PACKET):
        result = results[ppwz]
        periods = result["time"] / result["period"]
        analytical_change = result["analytical_width"] - result["analytical_width"][0]
        for name, scheme in SCHEMES.items():
            width = result["schemes"][name]["width"]
            excess = width - width[0] - analytical_change
            axis.plot(periods, excess / 1000.0, scheme["linestyle"],
                      color=scheme["color"], lw=1.8, label=name)
        axis.axhline(0.0, color="black", lw=1.0)
        axis.set_title(panel_title(ppwz))
        axis.set_xlabel(r"$t/T_{ana}$")
        axis.set_ylabel(r"$\delta\sigma_z=\Delta\sigma_{z,num}-\Delta\sigma_{z,ana}$ [km]")
        axis.grid(alpha=0.3)
    finish_packet_figure(figure, axes, "wave_packet_excess_widening")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_legacy_typography()
    mode_results = {ppwz: simulate_single_mode(ppwz) for ppwz in PPWZ_MODE}
    plot_single_mode(mode_results)
    mode_rows = []
    for ppwz, result in mode_results.items():
        for name, values in result["schemes"].items():
            mode_rows.append({
                "scheme": name, "PPWz": ppwz, "dz_m": LAMBDA_Z / ppwz,
                "frequency_rad_s": values["omega"],
                "final_phase_error_rad": values["phase_error"][-1],
                "predicted_final_phase_error_rad": values["predicted_phase_error"][-1],
                "final_L2_error": values["error_l2"][-1],
                "final_relative_energy_change": values["relative_energy"][-1],
            })
    write_csv("normal_mode_metrics.csv", mode_rows)

    packet_results = {ppwz: simulate_wave_packet(ppwz) for ppwz in PPWZ_PACKET}
    plot_wave_packet(packet_results)
    packet_rows = []
    for ppwz, result in packet_results.items():
        for name, values in result["schemes"].items():
            numerical_change = values["width"][-1] - values["width"][0]
            analytical_change = result["analytical_width"][-1] - result["analytical_width"][0]
            packet_rows.append({
                "scheme": name, "PPWz": ppwz,
                "dz_m": DOMAIN_LENGTH / len(result["z"]),
                "analytical_cgz_m_s": analytical_vertical_group_velocity(K0, M0),
                "predicted_numerical_cgz_m_s": values["predicted_cgz"],
                "final_energy_centroid_m": values["center"][-1],
                "final_centroid_error_m": values["center"][-1] - result["analytical_center"][-1],
                "initial_packet_width_m": values["width"][0],
                "packet_widening_m": numerical_change,
                "analytical_packet_widening_m": analytical_change,
                "excess_numerical_widening_m": numerical_change - analytical_change,
            })
    write_csv("wave_packet_metrics.csv", packet_rows)
    print(f"Experiment 3 complete. Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()