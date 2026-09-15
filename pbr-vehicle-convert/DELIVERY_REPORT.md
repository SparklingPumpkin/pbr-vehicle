# PBR Vehicle Convert 1.6.0 Delivery Report

## Delivered Contract

`pbr-vehicle-sdk 1.6.0` preserves the P-v4 `pbr-vehicle-single-ply-v1` contract and adds portable local-GPU execution:

- `pbr_<asset-id>.ply` preserves the visible source 3DGS fields and adds only `normal_0..2`.
- `configs/config_<asset-id>.json` stores global material, lighting, projection, and integrity metadata.
- No proxy PLY, mapping NPZ, projection PLY, or `r3gw_*` compatibility fields are delivered.
- The runtime projection descriptor defines cast extension as the swept convex hull from the vehicle footprint to the parallel-light endpoint footprint, so low-elevation shadows remain connected to the vehicle.
- `RenderConfig.device="auto"` selects the first process-visible CUDA GPU; GGX relight, mapping transfer, bake and gsplat rendering share that result, with a NumPy CPU fallback for non-raster operations.

The formal conversion command is `pbr-vehicle complete-asset`. Projection is runtime-only `receiver_space_mask_v1`: its contact and parallel-light PBR-Gaussian-outline masks share the Gaussian-bottom-surface anchor.

## Compatibility

The canonical config is accepted by `pbr-vehicle-panel` and `pbr-vehicle-auto`. Their single-PLY loaders use PLY normals and broadcast `material.albedo_rgb`, roughness, and metallic across the visible Gaussian layer. Historical generic KNN SDK APIs remain separate legacy interfaces and are not the delivery conversion path.

## Verification

- SDK tests: 29 passed.
- Real simulation2 asset `10010` passes `inspect-complete` loading.
- Panel real-asset tests: 3 passed.
- Auto tests: 4 passed.

Artifact: `package/dist/pbr_vehicle_sdk-1.6.0-py3-none-any.whl` (33,992 bytes). SHA-256: `d75dca0cc38870adb22a999f450f904368e6a18479b4e4ca852aeef0691ab878`; it is also recorded in `package/dist/SHA256SUMS`. Runtime dependencies constrain NumPy to `>=1.24,<2` to avoid the known Torch 2.1/2.2 NumPy-2 ABI incompatibility.
