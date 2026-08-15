"""Supplementary linear Boussinesq model for buoyancy-bubble and mountain-wave tests.

This experiment is retained for teaching and model inspection. It is not part
of the manuscript's three controlled experiment groups. The prognostic system
in vorticity-streamfunction form is

    d zeta / dt = -U d zeta / dx + d b / dx,
    Laplacian(psi) = zeta,
    d b / dt = -U d b / dx - N^2 d psi / dx.

The horizontal direction is spectral, the vertical Poisson problem uses a
second-order centered operator, and time integration uses classical RK4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gravity_waves.common import N_BV, add_panel_labels, save_figure  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def configuration(mode: str) -> dict[str, object]:
    """Return physical and numerical parameters for a named scenario."""
    if mode == "bubble":
        return {
            "lx": 8_000.0, "lz": 8_000.0, "nx": 128, "nz": 129,
            "dt": 5.0, "total_time": 2_400.0,
            "saved_times": [0.0, 600.0, 1_200.0, 1_800.0, 2_400.0],
            "wind": 0.0, "ramp_time": 0.0, "sponge_fraction": None,
            "maximum_damping": 0.0,
        }
    if mode == "mountain":
        return {
            "lx": 5_000.0, "lz": 12_000.0, "nx": 64, "nz": 81,
            "dt": 8.0, "total_time": 14_400.0,
            "saved_times": [0.0, 3_600.0, 7_200.0, 10_800.0, 14_400.0],
            "wind": 5.0, "ramp_time": 3_600.0, "sponge_fraction": 0.7,
            "maximum_damping": 1.0 / 600.0,
        }
    raise ValueError("mode must be 'bubble' or 'mountain'")


def thomas_solve_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                       d: np.ndarray) -> np.ndarray:
    """Solve independent tridiagonal systems stored by row."""
    modes, interior_points = d.shape
    c_prime = np.zeros((modes, interior_points), dtype=complex)
    d_prime = np.zeros((modes, interior_points), dtype=complex)
    c_prime[:, 0] = c[:, 0] / b[:, 0]
    d_prime[:, 0] = d[:, 0] / b[:, 0]
    for index in range(1, interior_points):
        denominator = b[:, index] - a[:, index] * c_prime[:, index - 1]
        c_prime[:, index] = c[:, index] / denominator
        d_prime[:, index] = (
            d[:, index] - a[:, index] * d_prime[:, index - 1]
        ) / denominator
    solution = np.zeros_like(d_prime)
    solution[:, -1] = d_prime[:, -1]
    for index in range(interior_points - 2, -1, -1):
        solution[:, index] = d_prime[:, index] - c_prime[:, index] * solution[:, index + 1]
    return solution


def run(mode: str) -> None:
    """Integrate one supplementary scenario and save its snapshots."""
    parameters = configuration(mode)
    lx, lz = parameters["lx"], parameters["lz"]
    nx, nz = parameters["nx"], parameters["nz"]
    dt, total_time = parameters["dt"], parameters["total_time"]
    wind = parameters["wind"]
    dx, dz = lx / nx, lz / (nz - 1)
    x = np.arange(nx) * dx
    z = np.linspace(0.0, lz, nz)
    horizontal_wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    mode_count = horizontal_wavenumbers.size
    interior_count = nz - 2

    if mode == "mountain":
        mountain_height = 100.0
        topographic_wavenumber = 2.0 * np.pi / lx
        terrain = mountain_height * np.cos(topographic_wavenumber * x)
        lower_streamfunction_base = wind * np.fft.rfft(terrain)
        sponge_start = parameters["sponge_fraction"] * lz
        damping = np.where(
            z > sponge_start,
            parameters["maximum_damping"] * ((z - sponge_start) / (lz - sponge_start)) ** 2,
            0.0,
        )
    else:
        terrain = np.zeros_like(x)
        lower_streamfunction_base = np.zeros(mode_count, dtype=complex)
        sponge_start = lz
        damping = np.zeros(nz)

    def ramp(time: float) -> float:
        ramp_time = parameters["ramp_time"]
        if ramp_time <= 0.0 or time >= ramp_time:
            return 1.0
        return 0.5 * (1.0 - np.cos(np.pi * time / ramp_time))

    def solve_streamfunction(vorticity: np.ndarray, time: float) -> np.ndarray:
        lower_boundary = lower_streamfunction_base * ramp(time)
        lower = np.full((mode_count, interior_count), 1.0 / dz**2, dtype=complex)
        upper = lower.copy()
        diagonal = np.broadcast_to(
            -2.0 / dz**2 - horizontal_wavenumbers[:, None] ** 2,
            (mode_count, interior_count),
        ).copy()
        right_hand_side = vorticity[:, 1:-1].copy()
        right_hand_side[:, 0] -= lower_boundary / dz**2
        interior = thomas_solve_batch(lower, diagonal, upper, right_hand_side)
        streamfunction = np.zeros((mode_count, nz), dtype=complex)
        streamfunction[:, 0] = lower_boundary
        streamfunction[:, 1:-1] = interior
        return streamfunction

    def tendencies(vorticity: np.ndarray, buoyancy: np.ndarray,
                   time: float) -> tuple[np.ndarray, np.ndarray]:
        derivative_x = 1j * horizontal_wavenumbers[:, None]
        streamfunction = solve_streamfunction(vorticity, time)
        vorticity_tendency = -wind * derivative_x * vorticity + derivative_x * buoyancy
        buoyancy_tendency = -wind * derivative_x * buoyancy - N_BV**2 * derivative_x * streamfunction
        vorticity_tendency -= damping[None, :] * vorticity
        buoyancy_tendency -= damping[None, :] * buoyancy
        return vorticity_tendency, buoyancy_tendency

    def rk4_step(vorticity: np.ndarray, buoyancy: np.ndarray,
                 time: float) -> tuple[np.ndarray, np.ndarray]:
        k1v, k1b = tendencies(vorticity, buoyancy, time)
        k2v, k2b = tendencies(vorticity + 0.5 * dt * k1v, buoyancy + 0.5 * dt * k1b, time + 0.5 * dt)
        k3v, k3b = tendencies(vorticity + 0.5 * dt * k2v, buoyancy + 0.5 * dt * k2b, time + 0.5 * dt)
        k4v, k4b = tendencies(vorticity + dt * k3v, buoyancy + dt * k3b, time + dt)
        return (
            vorticity + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            buoyancy + dt * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0,
        )

    horizontal_grid, vertical_grid = np.meshgrid(x, z, indexing="ij")
    if mode == "bubble":
        initial_buoyancy = 0.05 * np.exp(-((horizontal_grid - lx / 2.0) / 300.0) ** 2)
        initial_buoyancy *= np.exp(-((vertical_grid - lz / 2.0) / 300.0) ** 2)
    else:
        initial_buoyancy = np.zeros_like(horizontal_grid)
    buoyancy = np.fft.rfft(initial_buoyancy, axis=0)
    vorticity = np.zeros_like(buoyancy)
    snapshots = {0.0: initial_buoyancy.copy()}
    saved_times = parameters["saved_times"]
    next_saved_index = 1
    for step in range(1, int(total_time / dt) + 1):
        previous_time = (step - 1) * dt
        vorticity, buoyancy = rk4_step(vorticity, buoyancy, previous_time)
        current_time = step * dt
        if next_saved_index < len(saved_times) and abs(current_time - saved_times[next_saved_index]) < dt / 2.0:
            snapshots[saved_times[next_saved_index]] = np.fft.irfft(buoyancy, n=nx, axis=0)
            next_saved_index += 1

    figure, axes = plt.subplots(1, len(snapshots), figsize=(4.2 * len(snapshots), 5.2), sharey=True)
    maximum = max(float(np.max(np.abs(field))) for field in snapshots.values()) or 1.0
    color_plot = None
    for axis, (time, field) in zip(np.ravel(axes), snapshots.items()):
        color_plot = axis.pcolormesh(x / 1000.0, z / 1000.0, field.T,
                                     shading="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        if mode == "mountain":
            axis.plot(x / 1000.0, 20.0 * terrain / 1000.0, color="black", linewidth=1.2)
            axis.axhline(sponge_start / 1000.0, color="0.4", linestyle=":")
        axis.set_title(f"t = {time:.0f} s", fontsize=11, fontweight="bold")
        axis.set_xlabel("x [km]")
    np.ravel(axes)[0].set_ylabel("z [km]")
    add_panel_labels(figure, axes)
    assert color_plot is not None
    figure.colorbar(color_plot, ax=np.ravel(axes).tolist(), label=r"$b'$ [m/s$^2$]")
    save_figure(figure, OUTPUT_DIR, f"{mode}_prognostic_model", dpi=160)
    print(f"Supplementary {mode} case complete. Outputs: {OUTPUT_DIR}")


def main() -> None:
    """Parse the scenario and run the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bubble", "mountain", "both"), default="both")
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "path"
    modes = ("bubble", "mountain") if arguments.mode == "both" else (arguments.mode,)
    for mode in modes:
        run(mode)


if __name__ == "__main__":
    main()
