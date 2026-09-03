# PBR Vehicle Panel 1.3.0 Delivery Report

`pbr-vehicle-panel 1.3.0` implements the promoted `VSP-v2` panel contract. It opens each vehicle with six direct controls: 太阳光强度、太阳方位角、太阳高度角、车辆色温、饱和度和亮度，加上 Center orbit 命令。四个非角度参数使用 `-1~1` 归一化滑条，`0` 对应原有默认值；内部仍使用兼容的物理量。Config 操作隔离在 `Config`；完整控件按 `Transform / Material / R3GW Lighting / Projection` 分类，子面板不再嵌套“高级”或“次要参数”。

Saturation is applied to the original vehicle color in `Relight Original`, so direct-white proxy assets respond correctly. Exact environment RGB remains in saved configs for compatibility but is derived from color temperature and no longer exposed in the panel.

The panel loads ordinary Gaussian PLY or a DriveStudio PTH background, then provides vehicle placement, material, independent sun/environment light, runtime receiver masks, multi-vehicle state, and configuration save/load.

It accepts the P-v4 `pbr-vehicle-single-ply-v1` config-folder contract: `files.pbr` identifies the visible Gaussian, `normal_0..2` provide surface normals, and global material values are broadcast from config. No proxy or mapping is required.

The project and distribution are now both named `pbr-vehicle-panel`. The `pbr-vehicle-panel` CLI is primary; `pbr-vehicle-viewer` remains an alias. The Python module and prior saved config kind retain their original names for backward compatibility.

The panel is an interactive editor and preview, not a claim of pixel parity with the native DriveStudio rasterizer. Its packaged wheel checksum is recorded in `dist/SHA256SUMS`.

Version 1.3.0 adds Chinese labels and normalized `-1~1` exposed controls while retaining backward-compatible config and rendering ranges. The Chinese usage guide covers installation, a complete panel launch command, every CLI parameter, and common launch combinations. All delivery documentation, manifests, tests, and wheel metadata use portable paths rather than machine-specific absolute paths.
