# Delivery Report

## Delivered Scope

`pbr-vehicle-auto 1.2.0` implements the promoted `ALM-v5` auto-light-material contract:

- fixed and independent direct sun RGB;
- a luminance-normalized `3500-7200 K` CCT environment path at fixed energy;
- full Viser template preservation for vehicle pose, projection, fixed material and all non-fitted state;
- fitting limited to sun intensity, fill light and environment CCT; vehicle saturation is manual only;
- one rear-view PBR-to-ordinary-Gaussian DC bake and opacity-weighted luminance CDF-L1 evaluation.

The promoted P-v4 asset path is `pbr-vehicle-single-ply-v1`: it reads `files.pbr`, its `normal_0..2`, and global `material.albedo_rgb`. No proxy, mapping, or compatibility field family is required.

The result is reject-safe: if the constrained candidate does not improve the opacity-weighted DC objective, the emitted configuration and baked DC retain the base appearance and report `rejected_no_improvement` rather than overwriting it with a worse fit.

The package contains source, tests, a generic full Viser template, and a real-input validation result. It does not include vehicle or scene Gaussian assets.

## Default Profile

The delivered template is a portable normalization of the requested full Viser state. It retains the saved vehicle material, vehicle-lighting state and projection, and sets every `projection.extension.brightness_distance_to_white` value to `0.0`. Asset references use the deployment-relative `Assets/pbr_assets` convention. Result configs explicitly persist active `vehicle_lighting.sun_color_rgb`; this prevents fresh Viewer imports from depending on unrelated GUI slider values.

## Validation

Real inputs: ordinary Argoverse004 background Gaussian PLY and PBR asset `10010`.

| Check | Result |
| --- | --- |
| Unit tests | 4 passed |
| DC fit runtime | 17.889 s |
| Candidate evaluations | 48 |
| Raw luminance CDF-L1 | 0.0527102 |
| Baked luminance CDF-L1 | 0.0492155 |
| Improvement | 6.630% |
| Baked vehicle points | 191,427 |
| Nonzero higher-order SH coefficients | 0 |

The delivery implementation uses the requested template's active `56/45` direction. Its independent CPU validation is contract-compatible but does not claim numerical identity with the project-native renderer.

## Boundary

This package adapts a single rear-view, DC-only appearance proxy. It does not establish recovered physical illumination, material truth, scene correspondence, or multi-view realism.
