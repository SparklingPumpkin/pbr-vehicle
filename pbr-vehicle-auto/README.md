# PBR Vehicle Auto

`pbr-vehicle-auto` is the delivered `ALM-v5` implementation for adapting a PBR vehicle asset to an ordinary Gaussian scene without 2D RGB observations. The scene input is only Gaussian DC and opacity; the vehicle input is a PBR asset folder plus a full Viser state template. Chinese setup and command reference: [使用指南.md](使用指南.md).

## Contract

- The full Viser template is the baseline. Vehicle pose, projection, material, scene state, sun direction and sun RGB are manual inputs and are preserved.
- Direct `sun_color_rgb` is fixed and independent from environment colour.
- Environment color is constrained to a luminance-normalized `3500-7200 K` CCT path at fixed energy.
- Vehicle material is not fitted. `material.saturation` is manual only, with `1.0` meaning unchanged.
- Only `sun_intensity`, `ambient_fill`, and `environment_cct_kelvin` are searched.
- The objective is opacity-weighted vehicle/scene DC luminance CDF-L1, using one vehicle rear view. The optional bake writes those PBR colours as ordinary Gaussian DC and clears higher-order SH.
- A candidate that does not beat the original vehicle DC objective is rejected; its output preserves the base configuration and reports `rejected_no_improvement`.

This is an appearance-adaptation proxy. It does not claim physical illumination recovery, material truth, spatial correspondence, or multi-view realism.

## Install

```bash
cd pbr-vehicle/pbr-vehicle-auto
python -m pip install -e '.[test]'
```

## Default Template

The packaged `templates/default_vehicle_pbr_viser_template.json` is a portable full Viser state template. It retains the entire state, including vehicle projection. Asset references use the `Assets/pbr_assets` deployment-relative convention; pass the actual asset directory through `--asset-dir`. The fitter explicitly writes the active `vehicle_lighting.sun_color_rgb` in each result so importing into a fresh Viser process does not depend on existing GUI slider values.

Regenerate it from another Viser saved state:

```bash
pbr-vehicle-auto-default --saved-state <state.json> --output <template.json>
```

## Fit

```bash
pbr-vehicle-auto-fit \
  --scene-ply <ordinary_scene_background.ply> \
  --asset-dir <pbr_asset_folder> \
  --template-config templates/default_vehicle_pbr_viser_template.json \
  --output-dir <run_dir> \
  --final-config <asset_folder>/configs/config_auto_<vehicle>_<scene>.json
```

The package does not include vehicle or scene Gaussian assets. The template's `pbr_asset` uses `pbr-vehicle-single-ply-v1`: `files.pbr` is relative to `--asset-dir`, `normal_0..2` are read from that PLY, and `material.albedo_rgb` is broadcast as the global base color. No proxy, mapping, or `r3gw_*` field is required.

## Tests

```bash
python -m pytest -q
```
