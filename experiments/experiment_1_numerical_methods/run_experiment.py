# -*- coding: utf-8 -*-
"""Experiment 1 - numerical-method characterization.

This script reproduces the numerical-method experiments described in Sections
2.6 and 3.1 of the manuscript:

1. continuous PPWz spectral sweep;
2. controlled contrasts of formal order, staggering, and compactness;
3. accuracy-versus-arithmetic-cost comparison at PPWz = 4;
4. grid-convergence test for a fixed physical vertical wavelength.

Only the vertical derivative is discretized. The horizontal direction remains
analytical so that vertical-resolution effects are isolated.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gravity_waves.common import (  # noqa: E402
    N_BV,
    SCHEMES,
    SCHEME_NAMES,
    add_panel_labels,
    analytical_frequency,
    numerical_frequency,
    save_figure,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIXED_DZ = 250.0
FIXED_VERTICAL_WAVELENGTH = 2000.0
PPWZ_SWEEP = np.arange(3.0, 13.0)
PPWZ_CONTRASTS = np.array([4.0, 6.0, 8.0, 12.0])
PPWZ_CONVERGENCE = np.array([4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0])
ASPECT_RATIOS = [1.0, 0.5, 2.0]


def write_csv(filename, rows):
    """Write a list of dictionaries to the experiment output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_modified_wavenumber_error(ppwz, scheme):
    """Return |m*/m - 1| for a specified PPWz."""
    theta = 2.0 * np.pi / np.asarray(ppwz)
    return np.abs(scheme["function"](theta) / theta - 1.0)


def run_baseline_dispersion():
    """Reproduce the legacy three-panel dispersion figure and convergence panel."""
    resolutions = [
        (np.pi, "2-dz wave (worst resolved case: Nyquist)"),
        (np.pi / 2.0, "4-dz wave"),
        (np.pi / 4.0, "8-dz wave"),
    ]
    k_over_m = np.linspace(1.0e-3, 4.0, 400)
    figure = plt.figure(figsize=(13, 9.5))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.38, wspace=0.32)
    top_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    for axis, (theta, label) in zip(top_axes, resolutions):
        m = theta / FIXED_DZ
        k = k_over_m * m
        axis.plot(k_over_m, analytical_frequency(k, m) / N_BV,
                  color="black", lw=2.4, label="Analytical (exact)", zorder=5)
        for name, scheme in SCHEMES.items():
            axis.plot(k_over_m,
                      numerical_frequency(k, m, FIXED_DZ, scheme["function"]) / N_BV,
                      scheme["linestyle"], color=scheme["color"], lw=1.8, label=name)
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_xlabel(r"$k/m$  (dimensionless horizontal wavenumber)")
        axis.set_ylabel(r"$\omega/N$")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)
    top_axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

    error_axis = figure.add_subplot(grid[1, :])
    theta = np.linspace(1.0e-3, np.pi, 400)
    for name, scheme in SCHEMES.items():
        error = np.abs(scheme["function"](theta) / theta - 1.0)
        error_axis.semilogy(theta / np.pi, error, scheme["linestyle"],
                            color=scheme["color"], lw=2.0, label=name)
    error_axis.set_xlabel(
        r"$m\Delta z\,/\,\pi$   (0 = long/well-resolved wave; 1 = 2-dz/Nyquist wave)"
    )
    error_axis.set_ylabel(r"Relative error in $m^*$ (log scale)")
    error_axis.set_title("Convergence: vertical phase error versus resolution, by scheme",
                         fontsize=11, fontweight="bold")
    error_axis.grid(alpha=0.3, which="both")
    error_axis.legend(loc="upper left", fontsize=9)
    all_axes = top_axes + [error_axis]
    add_panel_labels(figure, all_axes)
    save_figure(figure, OUTPUT_DIR, "baseline_dispersion", dpi=160)

def run_ppwz_sweep():
    """Evaluate dispersion and modified-wavenumber error from PPWz 3 to 12."""
    k_over_m = np.linspace(1.0e-3, 4.0, 400)
    rows = []
    n_cols, n_rows = 5, 2
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.7 * n_rows), sharey=True)

    for axis, ppwz in zip(axes.flat, PPWZ_SWEEP):
        theta = 2.0 * np.pi / ppwz
        m = theta / FIXED_DZ
        k = k_over_m * m
        vertical_wavelength = ppwz * FIXED_DZ
        analytical = analytical_frequency(k, m) / N_BV
        axis.plot(k_over_m, analytical, color="black", lw=2.2, label="Analytical (exact)", zorder=5)

        row = {
            "PPWz": ppwz,
            "theta_over_pi": theta / np.pi,
            "vertical_wavelength_m": vertical_wavelength,
        }
        for name, scheme in SCHEMES.items():
            numerical = numerical_frequency(
                k, m, FIXED_DZ, scheme["function"]
            ) / N_BV
            axis.plot(
                k_over_m,
                numerical,
                scheme["linestyle"],
                color=scheme["color"],
                lw=1.5,
                label=name,
            )
            row[f"modified_wavenumber_error_pct__{name}"] = (
                100.0 * relative_modified_wavenumber_error(ppwz, scheme)
            )

        rows.append(row)
        axis.set_title(
            rf"PPWz = {ppwz:.0f}  ($\lambda_z$={vertical_wavelength:.0f} m)"
        )
        axis.set_title(axis.get_title(), fontsize=10, fontweight="bold")
        axis.set_xlabel(r"$k/m$", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.3)

    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\omega/N$")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.015),
    )
    figure.subplots_adjust(
        left=0.07, right=0.99, bottom=0.13, top=0.95,
        wspace=0.18, hspace=0.34
    )
    add_panel_labels(figure, axes, fontsize=12)
    save_figure(figure, OUTPUT_DIR, "ppwz_spectral_sweep")
    write_csv("ppwz_spectral_sweep.csv", rows)
    return rows


def run_scheme_contrasts():
    """Isolate formal-order, staggering, and compactness effects."""
    errors = {
        name: relative_modified_wavenumber_error(PPWZ_CONTRASTS, scheme)
        for name, scheme in SCHEMES.items()
    }
    contrasts = [
        {
            "title": "Formal-order effect",
            "subtitle": "non-staggered: second vs fourth order",
            "schemes": [
                SCHEME_NAMES["second_order"],
                SCHEME_NAMES["fourth_order"],
            ],
        },
        {
            "title": "Staggering effect",
            "subtitle": "second order: non-staggered vs Lorenz",
            "schemes": [
                SCHEME_NAMES["second_order"],
                SCHEME_NAMES["lorenz"],
            ],
        },
        {
            "title": "Compactness effect",
            "subtitle": "fourth order: explicit vs compact",
            "schemes": [
                SCHEME_NAMES["fourth_order"],
                SCHEME_NAMES["pade"],
            ],
        },
    ]

    rows = []
    for ppwz_index, ppwz in enumerate(PPWZ_CONTRASTS):
        for contrast in contrasts:
            baseline, improved = contrast["schemes"]
            baseline_error = errors[baseline][ppwz_index]
            improved_error = errors[improved][ppwz_index]
            rows.append(
                {
                    "contrast": contrast["title"],
                    "PPWz": ppwz,
                    "baseline_scheme": baseline,
                    "improved_scheme": improved,
                    "baseline_error_pct": 100.0 * baseline_error,
                    "improved_error_pct": 100.0 * improved_error,
                    "error_reduction_factor": baseline_error / improved_error,
                }
            )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for axis, contrast in zip(axes, contrasts):
        for name in contrast["schemes"]:
            scheme = SCHEMES[name]
            axis.plot(
                PPWZ_CONTRASTS,
                100.0 * errors[name],
                scheme["linestyle"],
                color=scheme["color"],
                marker="o",
                ms=7,
                lw=2.0,
                label=name,
            )
        axis.set_yscale("log")
        axis.set_xticks(PPWZ_CONTRASTS)
        axis.set_xlabel(r"PPW$_z$")
        axis.set_title(f"{contrast['title']}\n{contrast['subtitle']}", fontsize=10.5, fontweight="bold")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8.5, loc="upper right")

    axes[0].set_ylabel(r"Relative error in $m^*$ [\%]")
    figure.subplots_adjust(
        left=0.08, right=0.99, bottom=0.14, top=0.90, wspace=0.22
    )
    add_panel_labels(figure, axes)
    save_figure(figure, OUTPUT_DIR, "scheme_contrasts")
    write_csv("scheme_contrasts.csv", rows)


def run_contrast_reduction_ratios():
    """Plot the error-reduction ratios used in Figure 1 of the manuscript."""
    ppwz_values = np.array([4.0, 6.0, 8.0, 12.0, 16.0])
    scheme_errors = {
        name: relative_modified_wavenumber_error(ppwz_values, scheme)
        for name, scheme in SCHEMES.items()
    }
    contrasts = [
        {
            "label": "(a) Formal-order effect",
            "baseline": SCHEME_NAMES["second_order"],
            "improved": SCHEME_NAMES["fourth_order"],
            "color": "#4c72b0",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "label": "(b) Staggering effect",
            "baseline": SCHEME_NAMES["second_order"],
            "improved": SCHEME_NAMES["lorenz"],
            "color": "#dd8452",
            "linestyle": "--",
            "marker": "s",
        },
        {
            "label": "(c) Compactness effect",
            "baseline": SCHEME_NAMES["fourth_order"],
            "improved": SCHEME_NAMES["pade"],
            "color": "#55a868",
            "linestyle": ":",
            "marker": "^",
        },
    ]

    rows = []
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for contrast in contrasts:
        ratios = (
            scheme_errors[contrast["baseline"]]
            / scheme_errors[contrast["improved"]]
        )
        axis.plot(
            ppwz_values,
            ratios,
            color=contrast["color"],
            linestyle=contrast["linestyle"],
            marker=contrast["marker"],
            markersize=7,
            linewidth=2.0,
            label=contrast["label"],
        )
        for ppwz, ratio in zip(ppwz_values, ratios):
            axis.annotate(
                f"{ratio:.1f}x",
                (ppwz, ratio),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=contrast["color"],
            )
            rows.append(
                {
                    "contrast": contrast["label"],
                    "PPWz": ppwz,
                    "baseline_scheme": contrast["baseline"],
                    "improved_scheme": contrast["improved"],
                    "baseline_modified_wavenumber_error": float(
                        scheme_errors[contrast["baseline"]][
                            np.where(ppwz_values == ppwz)[0][0]
                        ]
                    ),
                    "improved_modified_wavenumber_error": float(
                        scheme_errors[contrast["improved"]][
                            np.where(ppwz_values == ppwz)[0][0]
                        ]
                    ),
                    "error_reduction_ratio": float(ratio),
                }
            )

    axis.set_yscale("log")
    axis.set_xticks(ppwz_values)
    axis.set_xlabel("PPWz (points per vertical wavelength)")
    axis.set_ylabel("Error reduction (times smaller; log scale)")
    axis.grid(alpha=0.30, which="major")
    axis.grid(alpha=0.16, which="minor")
    axis.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    save_figure(figure, OUTPUT_DIR, "contrast_error_reduction_ratio", dpi=150)
    write_csv("contrast_error_reduction_ratio.csv", rows)
    return rows

def run_accuracy_cost_analysis():
    """Reproduce the manuscript cost-benefit diagram and ideal operating zone."""
    from matplotlib.patches import Rectangle

    ppwz = 4.0
    marker_by_scheme = {
        SCHEME_NAMES["second_order"]: "o",
        SCHEME_NAMES["fourth_order"]: "s",
        SCHEME_NAMES["pade"]: "D",
        SCHEME_NAMES["lorenz"]: "^",
    }
    short_label_by_scheme = {
        SCHEME_NAMES["second_order"]: "Non-staggered\nsecond order",
        SCHEME_NAMES["fourth_order"]: "Non-staggered\nfourth order",
        SCHEME_NAMES["pade"]: "Compact Pade\nfourth order",
        SCHEME_NAMES["lorenz"]: "Staggered Lorenz\nsecond order",
    }
    annotation_offset = {
        SCHEME_NAMES["second_order"]: (6, 0),
        SCHEME_NAMES["fourth_order"]: (6, 0),
        SCHEME_NAMES["pade"]: (6, -2),
        SCHEME_NAMES["lorenz"]: (6, -2),
    }

    rows = []
    for name, scheme in SCHEMES.items():
        error = float(100.0 * relative_modified_wavenumber_error(ppwz, scheme))
        cost = scheme["flops"]
        rows.append(
            {
                "scheme": name,
                "PPWz": ppwz,
                "nominal_flops_per_grid_point": cost,
                "modified_wavenumber_error_pct": error,
                "inside_ideal_operating_region": bool(cost <= 3.0 and error <= 15.5),
            }
        )

    # Retain the formal non-dominance classification in the machine-readable table.
    for candidate in rows:
        candidate["pareto_optimal"] = not any(
            other["nominal_flops_per_grid_point"]
            <= candidate["nominal_flops_per_grid_point"]
            and other["modified_wavenumber_error_pct"]
            <= candidate["modified_wavenumber_error_pct"]
            and (
                other["nominal_flops_per_grid_point"]
                < candidate["nominal_flops_per_grid_point"]
                or other["modified_wavenumber_error_pct"]
                < candidate["modified_wavenumber_error_pct"]
            )
            for other in rows
        )

    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    ideal_zone = Rectangle(
        (0.0, 0.0),
        3.0,
        15.5,
        facecolor="none",
        fill=False,
        edgecolor="#d62728",
        linewidth=1.8,
        zorder=4,
        label="Ideal operating region",
    )
    axis.add_patch(ideal_zone)

    for row in rows:
        scheme = SCHEMES[row["scheme"]]
        axis.scatter(
            row["nominal_flops_per_grid_point"],
            row["modified_wavenumber_error_pct"],
            s=115,
            color=scheme["color"],
            marker=marker_by_scheme[row["scheme"]],
            edgecolor="none",
            zorder=5,
        )
        axis.annotate(
            short_label_by_scheme[row["scheme"]],
            (
                row["nominal_flops_per_grid_point"],
                row["modified_wavenumber_error_pct"],
            ),
            xytext=annotation_offset[row["scheme"]],
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.55",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
        )

    axis.set_xlim(0.0, 12.0)
    axis.set_ylim(0.0, 45.0)
    axis.set_xticks(np.arange(0.0, 12.1, 2.0))
    axis.set_yticks(np.arange(0.0, 45.1, 5.0))
    axis.set_xlabel("Estimated computational cost (FLOPs per grid point)")
    axis.set_ylabel(r"Spectral error [\%] evaluated at PPW$_z=4$")
    axis.grid(alpha=0.30, linestyle="--")
    axis.legend(handles=[ideal_zone], loc="upper right", fontsize=9)
    figure.tight_layout()
    save_figure(figure, OUTPUT_DIR, "accuracy_cost_pareto", dpi=150)
    write_csv("accuracy_cost_pareto.csv", rows)

def run_grid_convergence():
    """Refine dz while keeping the physical vertical wave fixed."""
    m = 2.0 * np.pi / FIXED_VERTICAL_WAVELENGTH
    dz_values = FIXED_VERTICAL_WAVELENGTH / PPWZ_CONVERGENCE
    rows = []
    results = {}

    for aspect_ratio in ASPECT_RATIOS:
        k = aspect_ratio * m
        analytical = analytical_frequency(k, m)
        results[aspect_ratio] = {}
        for name, scheme in SCHEMES.items():
            numerical = np.array(
                [
                    numerical_frequency(k, m, dz, scheme["function"])
                    for dz in dz_values
                ]
            )
            error = np.abs(numerical - analytical) / analytical
            results[aspect_ratio][name] = error
            slope = float(np.polyfit(np.log(dz_values[-4:]), np.log(error[-4:]), 1)[0])
            for ppwz, dz, value in zip(PPWZ_CONVERGENCE, dz_values, error):
                rows.append(
                    {
                        "scheme": name,
                        "k_over_m": aspect_ratio,
                        "vertical_wavelength_m": FIXED_VERTICAL_WAVELENGTH,
                        "PPWz": ppwz,
                        "dz_m": dz,
                        "relative_frequency_error_pct": 100.0 * value,
                        "observed_order_last_four_resolutions": slope,
                    }
                )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for axis, aspect_ratio in zip(axes, ASPECT_RATIOS):
        for name, scheme in SCHEMES.items():
            axis.plot(
                PPWZ_CONVERGENCE,
                100.0 * results[aspect_ratio][name],
                scheme["linestyle"],
                color=scheme["color"],
                marker="o",
                ms=6,
                lw=1.9,
                label=name,
            )
        axis.set_yscale("log")
        axis.set_xlabel("PPWz  (increases = finer grid)")
        axis.set_title(rf"$k/m = {aspect_ratio:g}$", fontsize=11, fontweight="bold")
        axis.grid(alpha=0.3, which="both")

    axes[0].set_ylabel(r"Relative frequency error [\%]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.015),
    )
    figure.subplots_adjust(
        left=0.08, right=0.99, bottom=0.18, top=0.92, wspace=0.22
    )
    add_panel_labels(figure, axes)
    save_figure(figure, OUTPUT_DIR, "fixed_wavelength_grid_convergence")
    write_csv("fixed_wavelength_grid_convergence.csv", rows)
    return rows


def main():
    """Run every experiment in the numerical-method group."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Experiment 1 - numerical-method characterization")
    run_baseline_dispersion()
    run_ppwz_sweep()
    run_scheme_contrasts()
    run_contrast_reduction_ratios()
    run_accuracy_cost_analysis()
    convergence_rows = run_grid_convergence()

    print("\nObserved orders from the four finest grids:")
    printed = set()
    for row in convergence_rows:
        key = (row["scheme"], row["k_over_m"])
        if key not in printed:
            print(
                f"  {row['scheme']}, k/m={row['k_over_m']:g}: "
                f"p={row['observed_order_last_four_resolutions']:.3f}"
            )
            printed.add(key)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
