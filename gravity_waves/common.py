# -*- coding: utf-8 -*-
"""Shared numerical operators and plotting utilities.

The project discretizes only the vertical direction. All experiments import the
same modified-wavenumber definitions from this module so that scheme names,
formulas, colors, and line styles remain identical throughout the repository.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_BV = 0.012
EARTH_ROTATION_RATE = 7.292e-5

SCHEME_NAMES = {
    "second_order": "Non-staggered, second order",
    "fourth_order": "Non-staggered, fourth order",
    "pade": "Compact Pade, fourth order",
    "lorenz": "Staggered Lorenz, second order",
}


def modified_wavenumber_dz_second_order(theta):
    """Return m* dz for the second-order centered non-staggered operator."""
    return np.sin(theta)


def modified_wavenumber_dz_fourth_order(theta):
    """Return m* dz for the fourth-order centered non-staggered operator."""
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def modified_wavenumber_dz_pade(theta):
    """Return m* dz for the fourth-order tridiagonal compact Pade operator."""
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def modified_wavenumber_dz_lorenz(theta):
    """Return m* dz for the second-order staggered Lorenz operator."""
    return 2.0 * np.sin(theta / 2.0)


def modified_wavenumber_derivative_second_order(theta):
    return np.cos(theta)


def modified_wavenumber_derivative_fourth_order(theta):
    return (8.0 * np.cos(theta) - 2.0 * np.cos(2.0 * theta)) / 6.0


def modified_wavenumber_derivative_pade(theta):
    return 3.0 * (2.0 * np.cos(theta) + 1.0) / (2.0 + np.cos(theta)) ** 2


def modified_wavenumber_derivative_lorenz(theta):
    return np.cos(theta / 2.0)


SCHEMES = {
    SCHEME_NAMES["second_order"]: {
        "function": modified_wavenumber_dz_second_order,
        "derivative": modified_wavenumber_derivative_second_order,
        "color": "#c0392b",
        "linestyle": "--",
        "formal_order": 2,
        "flops": 2,
    },
    SCHEME_NAMES["fourth_order"]: {
        "function": modified_wavenumber_dz_fourth_order,
        "derivative": modified_wavenumber_derivative_fourth_order,
        "color": "#2980b9",
        "linestyle": "-.",
        "formal_order": 4,
        "flops": 6,
    },
    SCHEME_NAMES["pade"]: {
        "function": modified_wavenumber_dz_pade,
        "derivative": modified_wavenumber_derivative_pade,
        "color": "#27ae60",
        "linestyle": ":",
        "formal_order": 4,
        "flops": 10,
    },
    SCHEME_NAMES["lorenz"]: {
        "function": modified_wavenumber_dz_lorenz,
        "derivative": modified_wavenumber_derivative_lorenz,
        "color": "#8e44ad",
        "linestyle": "-",
        "formal_order": 2,
        "flops": 2,
    },
}


def modified_wavenumber(m, dz, function):
    """Return the effective vertical wavenumber m*."""
    return function(np.asarray(m) * dz) / dz


def analytical_frequency(k, m, buoyancy_frequency=N_BV):
    """Continuous internal-gravity-wave frequency."""
    return (
        buoyancy_frequency
        * np.abs(k)
        / np.sqrt(np.asarray(k) ** 2 + np.asarray(m) ** 2)
    )


def numerical_frequency(k, m, dz, function, buoyancy_frequency=N_BV):
    """Semi-discrete frequency obtained by replacing m with m*."""
    m_star = modified_wavenumber(m, dz, function)
    return (
        buoyancy_frequency
        * np.abs(k)
        / np.sqrt(np.asarray(k) ** 2 + m_star**2)
    )


def analytical_rotating_frequency(k, m, coriolis, buoyancy_frequency=N_BV):
    """Continuous inertia-gravity-wave frequency."""
    return np.sqrt(
        (
            buoyancy_frequency**2 * np.asarray(k) ** 2
            + coriolis**2 * np.asarray(m) ** 2
        )
        / (np.asarray(k) ** 2 + np.asarray(m) ** 2)
    )


def numerical_rotating_frequency(
    k, m, dz, function, coriolis, buoyancy_frequency=N_BV
):
    """Semi-discrete inertia-gravity-wave frequency."""
    m_star = modified_wavenumber(m, dz, function)
    return np.sqrt(
        (
            buoyancy_frequency**2 * np.asarray(k) ** 2
            + coriolis**2 * m_star**2
        )
        / (np.asarray(k) ** 2 + m_star**2)
    )


def analytical_vertical_group_velocity(k, m, buoyancy_frequency=N_BV):
    """Continuous vertical group velocity d omega / d m."""
    return (
        -buoyancy_frequency
        * np.abs(k)
        * np.asarray(m)
        / (np.asarray(k) ** 2 + np.asarray(m) ** 2) ** 1.5
    )


def numerical_vertical_group_velocity(
    k, m, dz, function, derivative, buoyancy_frequency=N_BV
):
    """Semi-discrete vertical group velocity from the chain rule."""
    m_star = modified_wavenumber(m, dz, function)
    dm_star_dm = derivative(np.asarray(m) * dz)
    return (
        -buoyancy_frequency
        * np.abs(k)
        * m_star
        * dm_star_dm
        / (np.asarray(k) ** 2 + m_star**2) ** 1.5
    )


def configure_plots():
    """Apply publication-scale typography suitable for an A4 page."""
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            # Converting text to paths avoids LaTeX/InkScape text-overlay issues.
            "svg.fonttype": "path",
        }
    )


def add_panel_labels(figure, axes, fontsize=14):
    """Place panel labels exactly as in the legacy plotting helper."""
    for index, axis in enumerate(np.asarray(axes).flat):
        axis.text(
            -0.10,
            1.04,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )

def save_figure(figure, output_directory, stem, dpi=180):
    """Save matching PNG and SVG versions and close the figure."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_directory / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


configure_plots()
