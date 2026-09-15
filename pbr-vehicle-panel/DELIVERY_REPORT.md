# PBR Vehicle Panel 1.9.1 Delivery Report

Version 1.9.1 integrates `pbr-vehicle-auto 1.4.1`. Sun intensity and brightness now use the fixed absolute `[-0.3, 0.3]` interval in Viser slider coordinates. Repeated Auto clicks no longer recenter the search domain on the previously applied result.

Validation: `48 passed, 2 skipped`. Artifact: `dist/pbr_vehicle_panel-1.9.1-py3-none-any.whl` (54,516 bytes). SHA-256: `7cfdae6460734b4d7c5a8219385c495dafbec617afad7aba67372c148678ab41`.

## 1.9.0

Version 1.9.0 adds a portable CUDA compute backend. `--device auto` selects the first GPU visible to the process and therefore respects `CUDA_VISIBLE_DEVICES`; it does not encode an A100 model or physical index. Proxy/direct PBR shading, environment cubemap lookup and mapping transfer run in Torch CUDA with per-vehicle static Tensor caches. NumPy remains the no-CUDA/PyTorch fallback, and Gaussian counts remain unlimited by default.

Real `normal_unit` validation used 224,201 Gaussians on an NVIDIA A100-SXM4-80GB. Warm cached shading measured 0.0045 s versus 0.163 s for NumPy CPU (about 36x), with maximum color difference `5.96e-7`.

Artifact: `dist/pbr_vehicle_panel-1.9.0-py3-none-any.whl` (54,383 bytes). SHA-256: `bf8022cb519b17b623ec6350583e0b2908b7b1ecefa0acb5fb7becc51b925ede`; it is also recorded in `dist/SHA256SUMS`.

## 1.8.2

Version 1.8.2 integrates `pbr-vehicle-auto 1.3.0`. A successful search always applies the best candidate inside the configured grid; raw vehicle DC is diagnostic only and is no longer a second acceptance gate. The Panel still validates the `candidate_only` result contract and fails closed for process errors, timeouts, malformed files, or unsupported legacy statuses. Zero and negative `improvement_percent` values are displayed rather than rejected.

## 1.8.1

Version 1.8.1 closes the standalone distribution boundary. Runtime modules contain no training-repository imports, workspace path injection, or machine-specific default paths. User documentation now treats scenes and vehicle assets as caller-provided inputs, points only to files within this delivery directory, and uses the native-resolution `?fixedDpr=1` URL. The default scene Gaussian limit is zero, so the complete display layer is loaded unless the caller explicitly opts into a positive limit.

Validation completed on 2026-09-15:

- `48 passed` with the real `normal_unit` asset and canonical config enabled; no tests skipped.
- The wheel was installed into a new Python 3.12 virtual environment with no workspace `PYTHONPATH`; `pip check` reported no broken requirements.
- Both `pbr-vehicle-panel` and `pbr-vehicle-viewer` console scripts resolved to `pbr_vehicle_standalone.cli:main`.
- The installed wheel loaded the real `normal_unit` asset (`224,201` Gaussian points), started Viser 1.1.0, and returned HTTP 200.
- The installed wheel loaded the real scene cache with `3,412,690/3,412,690` Gaussian points.
- A separate Python 3.9 venv with the optional PyTorch dependency loaded the real 017 PTH checkpoint as `3,412,690/3,412,690` Gaussian points; the imported module came from that venv's `site-packages`.
- Wheel inspection found all 14 runtime Python modules and no references to workspace paths or training-repository source.

Artifact: `dist/pbr_vehicle_panel-1.8.1-py3-none-any.whl` (49,562 bytes). SHA-256: `f37dc1825c7aef0f3569c2cd22f40c21963d7db42a02ca0dc05c7547d0ce9bae`.

Validation includes a real on-disk miniature Gaussian PLY and single-Ply asset path, CLI metadata, a clean virtual-environment wheel install, and a real `normal_unit` asset smoke test. The packaged wheel and final test counts are recorded below after release validation.

## 1.8.0

Version 1.8.0 adds a persisted, per-vehicle `use_proxy_relighting` switch under `Advanced / Material`. Enabled is backward-compatible: PBR is evaluated on the proxy and its illumination ratio modulates the visible Gaussian appearance. Disabled bypasses ratio transfer and renders the PBR Gaussian layer directly from its albedo, normals, roughness, metallic and active lighting. Single-Ply assets share one Gaussian buffer between their logical visible and proxy roles; legacy three-file assets retain the physical proxy plus mapping. Older configs default to enabled.

Validation: `43 passed, 2 skipped`. The packaged wheel is `dist/pbr_vehicle_panel-1.8.0-py3-none-any.whl`; its SHA-256 is recorded in `dist/SHA256SUMS`.

## 1.7.1

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
