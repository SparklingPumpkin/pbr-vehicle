# Delivery Report

## Delivered Scope

`pbr-vehicle-auto 1.4.1` implements the promoted `ALM-v5` auto-light-material contract and portable Torch CUDA evaluator:

- fixed and independent direct sun RGB;
- a luminance-normalized `3500-7200 K` CCT environment path at fixed energy;
- full Viser template preservation for vehicle pose, projection, fixed material and all non-fitted state;
- fitting limited to sun intensity, fill light and environment CCT; vehicle saturation is manual only;
- one rear-view PBR-to-ordinary-Gaussian DC bake and opacity-weighted luminance CDF-L1 evaluation.
- a projection template whose extension is the continuous parallel-light sweep between the vehicle footprint and its ground-projected footprint.
- fixed absolute parameter walls in Viewer slider coordinates: both `sun_intensity` and `ambient_fill` use `[-0.3, 0.3]`, independent of the input template values and enforced in both coarse and final searches.
- `--device auto` keeps static PBR, mapping and histogram tensors on the first process-visible CUDA GPU for the full grid; NumPy CPU remains the automatic fallback.

The promoted P-v4 asset path is `pbr-vehicle-single-ply-v1`: it reads `files.pbr`, its `normal_0..2`, and global `material.albedo_rgb`. No proxy, mapping, or compatibility field family is required.

The grid optimum is always emitted and baked. Original vehicle DC remains a diagnostic baseline only and is not a second acceptance gate. `metrics.json` records `selection_policy=always_apply_grid_best`; consequently `improvement_percent` can be zero or negative while the result status remains `candidate_only` for Panel/Viser compatibility.

The package contains source, tests, a generic full Viser template, and a real-input validation result. It does not include vehicle or scene Gaussian assets.

## Default Profile

The delivered template is a portable normalization of the requested full Viser state. It retains the saved vehicle material, vehicle-lighting state and projection, and sets every `projection.extension.brightness_distance_to_white` value to `0.0`. Asset references use the deployment-relative `Assets/pbr_assets` convention. Result configs explicitly persist active `vehicle_lighting.sun_color_rgb`; this prevents fresh Viewer imports from depending on unrelated GUI slider values.

## Validation

Real inputs: ordinary Argoverse003 background Gaussian PLY and PBR asset `10010`. The second run used the first run's emitted config as its input template.

| Check | Result |
| --- | --- |
| Unit tests | 10 passed |
| First CUDA fit runtime | 1.537 s |
| Repeated CUDA fit runtime | 1.804 s |
| Candidate evaluations per run | 42 |
| First/repeated selected sun intensity | `0.7 / 0.7` |
| First/repeated selected ambient fill | `0.545 / 0.545` |
| First/repeated selected environment CCT | `6900 K / 6900 K` |
| Raw luminance CDF-L1 | 0.0726236 |
| Baked luminance CDF-L1 | 0.0746070 |
| Diagnostic change | -2.731% |
| Baked vehicle points | 191,427 |
| Nonzero higher-order SH coefficients | 0 |

Both runs recorded `bounds_policy=absolute_viser_slider` and `ui_min/ui_max=-0.3/0.3` for sun intensity and ambient fill. Although the second template contained the first result, both runs selected identical parameters and metrics.

Artifact: `dist/pbr_vehicle_auto-1.4.1-py3-none-any.whl` (20,831 bytes). SHA-256: `e25df32dabe6c554dfef3babc97a53325469679ecf7d83e631aed2b368600f7a`; it is also recorded in `dist/SHA256SUMS`. Runtime dependencies constrain NumPy to `>=1.24,<2` to avoid the known Torch 2.1/2.2 NumPy-2 ABI incompatibility.

The previously rejected scene-003 case also confirms the always-grid-best policy: the selected candidate is published despite its negative diagnostic change.

## Boundary

This package adapts a single rear-view, DC-only appearance proxy. It does not establish recovered physical illumination, material truth, scene correspondence, or multi-view realism.
