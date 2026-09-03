# PBR Vehicle Convert 1.5.0 Delivery Report

## Delivered Contract

`pbr-vehicle-sdk 1.5.0` delivers the P-v4 `pbr-vehicle-single-ply-v1` contract:

- `pbr_<asset-id>.ply` preserves the visible source 3DGS fields and adds only `normal_0..2`.
- `configs/config_<asset-id>.json` stores global material, lighting, projection, and integrity metadata.
- No proxy PLY, mapping NPZ, projection PLY, or `r3gw_*` compatibility fields are delivered.

The formal conversion command is `pbr-vehicle complete-asset`. Projection is runtime-only `receiver_space_mask_v1`: its contact and parallel-light PBR-Gaussian-outline masks share the Gaussian-bottom-surface anchor.

## Compatibility

The canonical config is accepted by `pbr-vehicle-panel` and `pbr-vehicle-auto`. Their single-PLY loaders use PLY normals and broadcast `material.albedo_rgb`, roughness, and metallic across the visible Gaussian layer. Historical generic KNN SDK APIs remain separate legacy interfaces and are not the delivery conversion path.

## Verification

- SDK tests: 29 passed.
- Real simulation2 asset `10010` passes `inspect-complete` loading.
- Panel real-asset tests: 3 passed.
- Auto tests: 4 passed.
