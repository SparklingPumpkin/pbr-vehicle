# PBR Vehicle Panel 1.7.1 Delivery Report

Version 1.7.1 synchronizes the standalone delivery with the current integrated Viewer: adjacent `config.yaml` scene binding, two-round SSE progress and published-round parsing, sun-angle propagation to every active vehicle, and lock-protected NumPy 2 checkpoint loading on NumPy 1.x. The optional Auto integration targets `pbr-vehicle-auto 1.2.1`, including the template-relative intensity and fill parameter walls. The Panel SciPy range is compatible with `pbr-vehicle-sun 1.2.0`, and the wheel contains no DriveStudio import or workspace absolute path.

## 1.7.0

Version 1.7.0 adds a per-vehicle textured environment-sphere preview. The captured cubemap is resampled to an equirectangular texture and exported as an emissive UV sphere. Independent mode uses editable XYZ coordinates as both capture and display position. Follow-vehicle mode keeps capture fixed at the vehicle center and applies editable XYZ only as a visualization offset, so moving the sphere does not alter its environment content. Capture excludes every inserted vehicle, projection layer, label, and existing preview sphere. Preview visibility, follow mode, independent position, and relative offset round-trip through vehicle configs. The visualization remains independent from the opt-in IBL shading switch.

Validation: `35 passed, 2 skipped`. The packaged wheel is `dist/pbr_vehicle_panel-1.7.0-py3-none-any.whl`; its SHA-256 is recorded in `dist/SHA256SUMS`.

## 1.6.0

Version 1.6.0 added the opt-in, per-vehicle scene environment-map IBL preview. The implementation is retained unchanged in this release; 1.7.0 adds the separate textured environment-sphere visualization.

Validation: `32 passed, 2 skipped` in the 1.6.0 release. The packaged wheel and checksum remain recorded in `dist/SHA256SUMS`.

Version 1.5.0 integrates `pbr-vehicle-auto 1.2.0 / ALM-v5` as an asynchronous per-vehicle action. The button snapshots the active vehicle and uses the full ordinary-Gaussian scene PLY; a PTH source is materialized to an unsampled cached PLY first. Only an accepted `candidate_only` result is applied. It switches that vehicle to independent lighting, enables its sun, and fills exactly `sun_intensity`, `ambient_fill` (亮度), and environment CCT (车辆色温). Sun direction, sun RGB, saturation, material, transform, and projection are preserved. A rejected, failed, invalid, or timed-out run leaves all controls untouched, and run artifacts remain in a system temporary directory.

The integration uses the public `pbr-vehicle-auto-fit` process boundary, with an optional portable JSON command configuration. The panel package does not copy the Auto implementation or embed machine-specific paths.

Version 1.4.0 integrates the delivered `pbr-vehicle-sun 1.1.0 / SSE-v8` scene estimator through its public `pbr-vehicle-sun-scene` command. The Scene panel starts estimation asynchronously, writes all generated artifacts to a retained system temporary directory, parses the fail-closed scene result, converts its road-plane direction into panel coordinates, enables directional sunlight, and fills the shared sun azimuth/elevation controls. A failed or confidence-rejected estimate leaves existing controls unchanged. External model locations remain caller-supplied through a portable JSON configuration; no machine-specific path is embedded in this package.

`pbr-vehicle-panel 1.3.1` implements the promoted `VSP-v2` panel contract. It opens each vehicle with six direct controls: 太阳光强度、太阳方位角、太阳高度角、车辆色温、饱和度和亮度，加上 Center orbit 命令。色温使用 `-0.5~0.5`，饱和度使用直接倍率 `0~2`，亮度使用 `-1~1`；内部仍使用兼容的 Kelvin、saturation 和 fill-light 物理量。Config 操作隔离在 `Config`；完整控件按 `Transform / Material / R3GW Lighting / Projection` 分类，子面板不再嵌套“高级”或“次要参数”。

Saturation is applied to the original vehicle color in `Relight Original`, so direct-white proxy assets respond correctly. Exact environment RGB remains in saved configs for compatibility but is derived from color temperature and no longer exposed in the panel.

The panel loads ordinary Gaussian PLY or a DriveStudio PTH background, then provides vehicle placement, material, independent sun/environment light, runtime receiver masks, multi-vehicle state, and configuration save/load.

It accepts the P-v4 `pbr-vehicle-single-ply-v1` config-folder contract: `files.pbr` identifies the visible Gaussian, `normal_0..2` provide surface normals, and global material values are broadcast from config. No proxy or mapping is required.

The project and distribution are now both named `pbr-vehicle-panel`. The `pbr-vehicle-panel` CLI is primary; `pbr-vehicle-viewer` remains an alias. The Python module and prior saved config kind retain their original names for backward compatibility.

The panel is an interactive editor and preview, not a claim of pixel parity with the native DriveStudio rasterizer. Its packaged wheel checksum is recorded in `dist/SHA256SUMS`.

The runtime cast extension uses the convex hull of both the vehicle XY footprint and the parallel-light endpoint footprint. This keeps the shadow connected to the vehicle at low sun elevations while preserving the physical direction opposite the sun azimuth. Contact core geometry and controls remain unchanged.

Version 1.3.1 assigns each exposed control its intended user range instead of applying one common range: temperature `-0.5~0.5`, saturation `0~2`, and brightness `-1~1`. Saved configs remain backward compatible. The Chinese usage guide covers installation, a complete panel launch command, every CLI parameter, and common launch combinations. All delivery documentation, manifests, tests, and wheel metadata use portable paths rather than machine-specific absolute paths.
