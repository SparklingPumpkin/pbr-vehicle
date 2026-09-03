# PBR Vehicle SDK

## PBR Asset Contract (1.5.0)

Formal assets use `pbr-vehicle-single-ply-v1`. Each vehicle delivers one visible PBR Gaussian PLY and one canonical configuration:

```text
<asset-id>/
  pbr_<asset-id>.ply
  configs/
    config_<asset-id>.json
    config_*.json
```

`pbr_<asset-id>.ply` retains the source standard 3DGS fields and adds only `normal_0..2`. It contains no proxy, mapping, per-point albedo, or `r3gw_*` compatibility fields. Global `albedo_rgb`, roughness, metallic, reflectance, and clearcoat are stored in `material` of the canonical configuration.

```bash
pbr-vehicle complete-asset vehicle.ply vehicle_asset \
  --asset-id vehicle001 --config examples/configs/config_default.json
pbr-vehicle inspect-complete vehicle_asset
```

`complete-asset` is the formal delivery command. The retained `convert`, `inspect`, `render`, and `bake` APIs are historical generic KNN SDK interfaces; they do not produce or consume this single-PLY delivery contract.

Projection is generated at runtime by `receiver_space_mask_v1`, with no `projection_*.ply`. The contact layer is a vehicle-local rounded rectangle. The extension is calculated from the current PBR Gaussian along the current parallel sun direction. Both layers are anchored to the Gaussian bottom support surface, so changing sun direction, vehicle pose, or projection parameters immediately recomputes the masks.

`contact` independently controls dimensions, XY offset, corner radius, softness, opacity, and grayscale target. `extension` independently controls opacity, grayscale target, distance decay, and distance-dependent softness. `brightness` is the grayscale target (`0` black, `1` white); opacity is an independent coverage value.

Input PLY must be binary little-endian standard 3DGS with `x/y/z`, `f_dc_0..2`, `scale_0..2`, `rot_0..3`, and `opacity`.

## Installation

```bash
python -m pip install .
```

Install optional dependencies only for the historical image-rendering interfaces:

```bash
python -m pip install '.[render]'
```

Chinese command and parameter documentation is in [使用指南.md](../使用指南.md).
