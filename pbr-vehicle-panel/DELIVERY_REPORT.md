# PBR Vehicle Panel 1.3.1 Delivery Report

`pbr-vehicle-panel 1.3.1` implements the promoted `VSP-v2` panel contract. It opens each vehicle with six direct controls: 太阳光强度、太阳方位角、太阳高度角、车辆色温、饱和度和亮度，加上 Center orbit 命令。色温使用 `-0.5~0.5`，饱和度使用直接倍率 `0~2`，亮度使用 `-1~1`；内部仍使用兼容的 Kelvin、saturation 和 fill-light 物理量。Config 操作隔离在 `Config`；完整控件按 `Transform / Material / R3GW Lighting / Projection` 分类，子面板不再嵌套“高级”或“次要参数”。

Saturation is applied to the original vehicle color in `Relight Original`, so direct-white proxy assets respond correctly. Exact environment RGB remains in saved configs for compatibility but is derived from color temperature and no longer exposed in the panel.

The panel loads ordinary Gaussian PLY or a DriveStudio PTH background, then provides vehicle placement, material, independent sun/environment light, runtime receiver masks, multi-vehicle state, and configuration save/load.

It accepts the P-v4 `pbr-vehicle-single-ply-v1` config-folder contract: `files.pbr` identifies the visible Gaussian, `normal_0..2` provide surface normals, and global material values are broadcast from config. No proxy or mapping is required.

The project and distribution are now both named `pbr-vehicle-panel`. The `pbr-vehicle-panel` CLI is primary; `pbr-vehicle-viewer` remains an alias. The Python module and prior saved config kind retain their original names for backward compatibility.

The panel is an interactive editor and preview, not a claim of pixel parity with the native DriveStudio rasterizer. Its packaged wheel checksum is recorded in `dist/SHA256SUMS`.

The runtime cast extension uses the convex hull of both the vehicle XY footprint and the parallel-light endpoint footprint. This keeps the shadow connected to the vehicle at low sun elevations while preserving the physical direction opposite the sun azimuth. Contact core geometry and controls remain unchanged.

Version 1.3.1 assigns each exposed control its intended user range instead of applying one common range: temperature `-0.5~0.5`, saturation `0~2`, and brightness `-1~1`. Saved configs remain backward compatible. The Chinese usage guide covers installation, a complete panel launch command, every CLI parameter, and common launch combinations. All delivery documentation, manifests, tests, and wheel metadata use portable paths rather than machine-specific absolute paths.
