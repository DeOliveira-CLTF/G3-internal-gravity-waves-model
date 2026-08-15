# Vertical discretization of atmospheric internal gravity waves

This repository is the reproducible code companion to the manuscript *Impact of vertical resolution and finite-difference schemes on the simulation of atmospheric internal gravity waves*.

The experiments isolate vertical-discretization error. The physical vertical wavelength is held fixed in every convergence test; refinement changes `dz = lambda_z / PPWz`, not the wave being represented. All scripts use Matplotlib's non-interactive `Agg` backend and write both PNG and SVG figures. SVG text is converted to vector paths to prevent duplicated or overlaid labels when figures are included in LaTeX through Inkscape.

## Repository map

| Paper component | Reproducible module | Main outputs |
|---|---|---|
| Numerical-method characterization (Sections 2.6 and 3.1; Figures 1-3) | `experiments/experiment_1_numerical_methods/run_experiment.py` | PPWz sweep, controlled scheme contrasts, contrast error-reduction ratio, accuracy-cost Pareto diagram, fixed-wavelength convergence |
| Physical consequences (Sections 2.7 and 3.2; Figures 4-6) | `experiments/experiment_2_physical_properties/run_experiment.py` | rotational frequency error, Doppler/stationarity residual, vertical group velocity |
| Prognostic error accumulation (Sections 2.8 and 3.3; Figures 7-9) | `experiments/experiment_3_prognostic_evolution/run_experiment.py` | normal-mode phase drift, wave-packet energy centroid, position error, packet widening |
| Supplementary model | `experiments/supplementary_bubble_mountain/run_experiment.py` | buoyancy-bubble and mountain-wave snapshots |
| Shared equations and visual style | `gravity_waves/common.py` | modified-wavenumber operators, dispersion relations, group velocities, plotting helpers |

## Figures included in the final manuscript

The table below lists only the figures that are actually numbered in the final manuscript. The captions are faithful English translations of the captions in the final Portuguese PDF. Other plots available in the experiment output directories are supplementary diagnostics and are not numbered figures in the article.

| Article figure | Caption in the final manuscript | Reproducible files | Generating script |
|---|---|---|---|
| **Figure 1** | Error-reduction ratio by scheme contrast. | [PNG](experiments/experiment_1_numerical_methods/outputs/contrast_error_reduction_ratio.png) · [SVG](experiments/experiment_1_numerical_methods/outputs/contrast_error_reduction_ratio.svg) | [run_experiment.py](experiments/experiment_1_numerical_methods/run_experiment.py) |
| **Figure 2** | Computational efficiency and cost-benefit analysis: trajectory of spectral-error convergence projected against the fixed arithmetic cost in FLOPs per grid point. | [PNG](experiments/experiment_1_numerical_methods/outputs/accuracy_cost_pareto.png) · [SVG](experiments/experiment_1_numerical_methods/outputs/accuracy_cost_pareto.svg) | [run_experiment.py](experiments/experiment_1_numerical_methods/run_experiment.py) |
| **Figure 3** | The plotted content is the physical-wavelength grid-convergence experiment for `k/m = 1, 0.5, 2`. | [PNG](experiments/experiment_1_numerical_methods/outputs/fixed_wavelength_grid_convergence.png) · [SVG](experiments/experiment_1_numerical_methods/outputs/fixed_wavelength_grid_convergence.svg) | [run_experiment.py](experiments/experiment_1_numerical_methods/run_experiment.py) |
| **Figure 4** | Numerical frequency error as a function of `k/m` and `f/N`, using a grid with `PPWz = 4` and a fixed vertical wavelength `lambda_z = 2000 m`. | [PNG](experiments/experiment_2_physical_properties/outputs/rotation_error_km_fN_PPWz4.png) · [SVG](experiments/experiment_2_physical_properties/outputs/rotation_error_km_fN_PPWz4.svg) | [run_experiment.py](experiments/experiment_2_physical_properties/run_experiment.py) |
| **Figure 5** | Sensitivity of intrinsic frequency and stationarity to vertical discretization. | [PNG](experiments/experiment_2_physical_properties/outputs/doppler_critical_regime.png) · [SVG](experiments/experiment_2_physical_properties/outputs/doppler_critical_regime.svg) | [run_experiment.py](experiments/experiment_2_physical_properties/run_experiment.py) |
| **Figure 6** | Analytical and numerical vertical group velocity as a function of the normalized vertical wavenumber `m dz / pi`, keeping the horizontal wave scale fixed. | [PNG](experiments/experiment_2_physical_properties/outputs/vertical_group_velocity.png) · [SVG](experiments/experiment_2_physical_properties/outputs/vertical_group_velocity.svg) | [run_experiment.py](experiments/experiment_2_physical_properties/run_experiment.py) |
| **Figure 7** | Temporal evolution of the phase difference between the numerical normal mode and the analytical solution for different vertical resolutions. | [PNG](experiments/experiment_3_prognostic_evolution/outputs/normal_mode_phase_error.png) · [SVG](experiments/experiment_3_prognostic_evolution/outputs/normal_mode_phase_error.svg) | [run_experiment.py](experiments/experiment_3_prognostic_evolution/run_experiment.py) |
| **Figure 8** | Temporal evolution of the vertical energy centroid of wave packets for different vertical resolutions. The black line represents the analytical solution, and the colored lines represent the different vertical-discretization schemes. | [PNG](experiments/experiment_3_prognostic_evolution/outputs/wave_packet_energy_centroid.png) · [SVG](experiments/experiment_3_prognostic_evolution/outputs/wave_packet_energy_centroid.svg) | [run_experiment.py](experiments/experiment_3_prognostic_evolution/run_experiment.py) |
| **Figure 9** | Temporal evolution of the vertical widening of wave packets for different vertical resolutions. Widening is defined as the change in energetic width relative to the initial state. | [PNG](experiments/experiment_3_prognostic_evolution/outputs/wave_packet_widening.png) · [SVG](experiments/experiment_3_prognostic_evolution/outputs/wave_packet_widening.svg) | [run_experiment.py](experiments/experiment_3_prognostic_evolution/run_experiment.py) |
Each experiment stores figures and machine-readable CSV tables in its own `outputs/` directory. Panel letters are placed outside the plotting axes, and the figures have no overall title so that the manuscript caption remains the single source of description.


## Numerical formulation

For a two-dimensional, non-rotating internal gravity wave, the continuous dispersion relation is

```math
\omega = N\frac{|k|}{\sqrt{k^2+m^2}},
```

where `N` is the Brunt-Vaisala frequency and `k` and `m` are the horizontal and vertical wavenumbers. Vertical finite differences replace `m` with an effective wavenumber `m*`. With `theta = m dz`, the four tested operators are:

| Scheme label used in figures | Modified wavenumber |
|---|---|
| Non-staggered, second order | `m* dz = sin(theta)` |
| Non-staggered, fourth order | `m* dz = [8 sin(theta) - sin(2 theta)] / 6` |
| Compact Pade, fourth order | `m* dz = 3 sin(theta) / [2 + cos(theta)]` |
| Staggered Lorenz, second order | `m* dz = 2 sin(theta/2)` |

The semi-discrete frequency is

```math
\omega_n = N\frac{|k|}{\sqrt{k^2+(m^*)^2}}.
```

With rotation, the continuous and semi-discrete inertia-gravity frequencies follow

```math
\omega^2 = \frac{N^2k^2+f^2m^2}{k^2+m^2},
```

using `m` and `m*`, respectively. The analytical vertical group velocity is

```math
c_{gz}=\frac{\partial\omega}{\partial m}=-N\frac{|k|m}{(k^2+m^2)^{3/2}}.
```

Experiment 3 tests the time-domain consequences predicted by those spectral relations:

```math
\Delta\phi(t) \simeq [\omega_n-\omega]t,
\qquad
\Delta z_E(t) \simeq [c_{gz,n}-c_{gz}]t.
```

## Reproduce the results

Python 3.10 or newer is recommended. Only NumPy and Matplotlib are required; `csv`, `pathlib`, `argparse`, and `subprocess` come from the Python standard library.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_all.py
```

Run one paper group independently:

```bash
python experiments/experiment_1_numerical_methods/run_experiment.py
python experiments/experiment_2_physical_properties/run_experiment.py
python experiments/experiment_3_prognostic_evolution/run_experiment.py
```

The supplementary demonstrations are excluded from the default workflow because they are slower and are not used for the controlled paper comparisons:

```bash
python experiments/supplementary_bubble_mountain/run_experiment.py --mode both
# or
python scripts/run_all.py --include-supplementary
```

## Experiment parameters

- Spectral sweep: `dz = 250 m`, `PPWz = 3,...,12`, and `k/m = 0.001,...,4`.
- Grid convergence: `lambda_z = 2000 m` fixed, `PPWz = 4,5,6,8,10,12,16`, and `k/m = 0.5,1,2`.
- Rotation: `N = 0.012 s^-1`, `lambda_z = 2000 m`, and separate error maps for `PPWz = 4,6,8`.
- Doppler/stationarity: uniform wind `U = 15 m s^-1` and `PPWz = 4,6,8`.
- Normal mode: `lambda_z = 4000 m`, `k/m = 1`, 50 wave periods, and 200 RK4 steps per analytical period.
- Wave packet: `PPWz = 2.5,3,4,8`, `z0 = 0.5 Lz`, `Lz = 20 lambda_z`, and spectral width `sigma_m = m0/20`. The positive-wavenumber packet is truncated at four standard deviations, keeping it below Nyquist even at `PPWz = 2.5`.


## Reproducibility notes and limitations

- The experiments are linear and two-dimensional; they isolate numerical dispersion rather than representing a complete atmospheric model.
- The horizontal direction remains analytical in Experiments 1 and 2.
- `PPWz = 2` is the vertical Nyquist scale. Some non-staggered operators have `m* = 0` there; this is a property of the stencil, not a missing value.
- The compact Pade operator is compared through its exact Fourier symbol. Its arithmetic count is a nominal local-work estimate and excludes implementation-dependent communication or solver overhead.
- Wave-packet centroids and widths use the same linear energy moments as the legacy experiment, with $E=\tfrac12(|u|^2+|w|^2+|b|^2/N^2)$. The packet starts at the domain center to avoid boundary crossing during the diagnosed interval.
