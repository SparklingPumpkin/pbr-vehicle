# PBR 车辆资产格式说明

本目录保存供 `pbr-vehicle-convert`、`pbr-vehicle-panel`、`pbr-vehicle-auto` 和渲染端共同使用的 PBR 车辆资产。当前资产合同为：

```text
asset_contract = pbr-vehicle-single-ply-v1
schema_version = 10
```

该合同使用一个可独立渲染的 PBR Gaussian PLY，不再保存实体 proxy、original-to-proxy 映射或投影 PLY。逐点几何、原始外观和法线位于 PLY；全局材质、默认光照与运行时投影参数位于 JSON 配置。

## 1. 目录结构

```text
Assets/
  PBR资产格式说明.md
  pbr_assets/
    <asset-id>/
      pbr_<asset-id>.ply
      configs/
        config_<asset-id>.json
        config_*.json
```

当前目录中的预置资产为 `10010`：

```text
pbr_assets/10010/
  pbr_10010.ply
  configs/
    config_10010.json
    config_*.json
```

其中 `configs/config_<asset-id>.json` 是 canonical 资产配置。`configs/` 还可以保存 Panel、Auto 或特定场景生成的其他 `config_*.json`，但每个资产目录只能有一份符合命名规则的 canonical 配置。

资产目录根部只能有一个二进制 payload，即 `pbr_<asset-id>.ply`；所有 JSON 配置统一放在 `configs/`，且文件名必须以 `config_` 开头。

## 2. 单 PLY 设计

`pbr_<asset-id>.ply` 同时承担最终可见车辆和 PBR 几何输入两种职责：

- 保留输入 3DGS PLY 的点数、Gaussian 中心、形状、旋转、opacity 和可见颜色。
- 保留输入中已有的其他字段，例如 `f_rest_*` 高阶 SH。
- 只追加单位法线 `normal_0..2`。
- 不写入逐点 albedo、roughness、metallic 等当前全局材质参数。
- 不写入 `r3gw_*`、`pbr_*` 等旧兼容前缀字段。

因此，新格式不会生成或依赖以下旧格式文件：

```text
original_<asset-id>.ply
proxy_<asset-id>.ply
map_<asset-id>.npz
projection_<asset-id>.ply
```

### 2.1 必需 PLY 字段

输入和输出都必须是标准 3DGS Gaussian PLY，至少包含：

| 字段 | 含义 |
| --- | --- |
| `x`, `y`, `z` | Gaussian 中心坐标。 |
| `f_dc_0`, `f_dc_1`, `f_dc_2` | 原始外观的零阶 SH/DC 颜色。 |
| `scale_0`, `scale_1`, `scale_2` | Gaussian 三轴 log scale。 |
| `rot_0`, `rot_1`, `rot_2`, `rot_3` | Gaussian 四元数旋转。 |
| `opacity` | Gaussian 的 logit opacity。 |
| `normal_0`, `normal_1`, `normal_2` | 转换后追加的单位表面法线。 |

`f_rest_*` 不是此合同的必需字段，但若输入 PLY 已包含它们，转换会原样保留。`scale_*` 和 `rot_*` 仍是可见 3DGS 渲染所需的形状字段，同时也是计算法线和估计车辆底面的依据，不能从单 PLY 中删除。

### 2.2 法线生成

每个 Gaussian 的法线取其最小协方差轴：

1. 用 `rot_0..3` 恢复 Gaussian 的旋转矩阵。
2. 从 `scale_0..2` 中选取最小尺度对应的旋转轴。
3. 以全车点中心的中位数为参考，将法线方向翻转为大致朝向车体外侧。
4. 归一化后写入 `normal_0..2`。

这是基于单个 Gaussian 形状的法线估计，不进行邻域平均或实验性全车法线平滑。中心朝外只能统一大致方向，并不保证所有局部区域的法线都连续或符合真实表面拓扑。

## 3. Canonical 配置

canonical 配置固定为：

```text
<asset-dir>/configs/config_<asset-id>.json
```

核心结构如下：

```json
{
  "schema_version": 10,
  "asset_contract": "pbr-vehicle-single-ply-v1",
  "asset_id": "10010",
  "files": {
    "pbr": "pbr_10010.ply",
    "config": "configs/config_10010.json"
  },
  "material": {
    "albedo_rgb": [0.82, 0.82, 0.82],
    "roughness": 0.32,
    "metallic": 0.02,
    "reflectance": 0.04,
    "clearcoat": 0.35,
    "exposure": 1.0,
    "relight_strength": 1.0
  },
  "light": {
    "sun_enabled": true,
    "sun_azimuth_degrees": 126.0,
    "sun_elevation_degrees": 45.0,
    "intensity": 1.0,
    "color_rgb": [1.0, 0.98, 0.92],
    "environment_color_rgb": [0.55, 0.62, 0.72]
  },
  "surface": {
    "geometry": "visible_original_gaussians",
    "normal_fields": ["normal_0", "normal_1", "normal_2"],
    "normal_source": "minimum_covariance_axis",
    "normal_orientation": "center_outward",
    "normal_version": 1
  },
  "relighting": {
    "control_layer": "logical",
    "geometry_source": "pbr_gaussian",
    "material_source": "config.material",
    "transfer": "same_point_multiplicative_ratio",
    "mapping_required": false
  },
  "projection": {
    "runtime": "receiver_space_mask_v1",
    "shape_model": "procedural-mask-contact-silhouette-v4",
    "composition": "receiver_alpha_blend_to_grayscale_v1"
  },
  "sha256": {
    "pbr": "..."
  }
}
```

规则如下：

- `files.*` 均相对于当前资产根目录解析，不允许依赖开发机器上的绝对路径。
- `files.pbr` 指向唯一的 PLY payload。
- `files.config` 指向 canonical 配置自身。
- `sha256.pbr` 校验 PLY 是否与交付时一致；修改 PLY 后必须重新生成配置和哈希。
- `surface` 和 `relighting` 描述固定的数据合同，不是逐帧调节参数。
- `material`、`light` 和 `projection` 是可编辑的默认状态。场景专用结果应保存成新的 `config_*.json`，不要用其替换 canonical 配置。

## 4. 材质与逻辑代理层

当前材质是全车共享的全局参数：

| 字段 | 含义 |
| --- | --- |
| `albedo_rgb` | BRDF 计算使用的全局基础反照率，不等于 PLY 的原始可见颜色。 |
| `roughness` | 微表面粗糙度，控制高光宽度。 |
| `metallic` | 金属工作流混合比例。 |
| `reflectance` | 非金属部分的基础镜面反射率 F0。 |
| `clearcoat` | 叠加在基础材质上的清漆层强度。 |
| `exposure` | 最终车辆亮度倍率。 |
| `relight_strength` | 原始外观与完整重光照响应之间的混合强度。 |

新格式取消了磁盘上的实体 proxy，但保留运行时的“逻辑代理层”。渲染器直接使用同一 PLY 的 `xyz + normal` 和配置中的全局材质计算 BRDF 响应，再按同一点将光照倍率乘回原始 Gaussian 颜色：

```text
logical_lit = BRDF(albedo_rgb, normal, material, light, view)
ratio       = logical_lit / albedo_rgb
final_rgb   = original_rgb * mix(1, ratio, relight_strength) * exposure
```

上式是数据流说明，实际实现还会进行数值稳定处理和颜色范围约束。由于可见点和逻辑 PBR 点是一一对应的同一批 Gaussian，不再需要 KNN、索引映射或 `map_*.npz`。

## 5. 运行时投影

车辆投影不存储为 PLY，也不通过固定 2DGS 拉伸实现，而是由渲染器在接收面上实时生成连续 mask：

```text
projection.runtime      = receiver_space_mask_v1
projection.shape_model  = procedural-mask-contact-silhouette-v4
projection.composition  = receiver_alpha_blend_to_grayscale_v1
```

投影由两层组成：

| 层 | 形状与作用 |
| --- | --- |
| `contact` | 车辆正下方固定的车辆局部 XY 圆角矩形，用于接地阴影。可独立调节长宽、XY 偏移、圆角、边缘柔化、opacity 和目标灰度。 |
| `extension` | 对当前 PBR Gaussian 的车辆 XY footprint 与太阳投影终点 footprint 的并集求凸包，形成从车体到远端轮廓的连续平行光扫掠区域。可独立调节 opacity、目标灰度、距离衰减和随距离增长的边缘柔化。 |

接收面的高度由 `projection.anchor` 决定：

```text
mode = gaussian_bottom_surface
```

运行时会结合 Gaussian 中心和协方差估计车辆底面，并应用 `z_offset_m`。当太阳方位、太阳仰角、车辆姿态或投影参数变化时，`extension` 会重新计算，因此一个固定投影文件无法表达完整行为。

`brightness` 表示覆盖区域的目标灰度：`0` 为黑，`1` 为白且不压暗；`opacity` 表示向该目标灰度混合的覆盖强度。两者语义不同，可以分别控制颜色与混合程度。

完整投影参数及默认值见 Convert 交付包中的 `使用指南.md`。

## 6. 配置类型

| 类型 | 识别方式 | 用途 |
| --- | --- | --- |
| canonical 资产配置 | 文件名为 `config_<asset-id>.json`，合同为 `pbr-vehicle-single-ply-v1`，且 `files.config` 指向自身 | 定义唯一 PLY、哈希和默认 PBR 状态。 |
| Panel 场景配置 | `config_*.json`，通常包含场景、车辆姿态和手调状态 | 保存特定场景的渲染状态。 |
| Auto 结果配置 | `config_*.json`，通常包含 `auto_fit` | 保存自动拟合输入、结果和诊断信息。 |

下游程序应先通过 canonical 配置发现资产 PLY，再按需加载场景或 Auto 配置作为参数覆盖。不要根据文件名前缀推断旧的 original、proxy 或 mapping 文件。

## 7. 生成与校验

从标准 3DGS PLY 生成新资产：

```bash
pbr-vehicle complete-asset \
  path/to/vehicle.ply \
  Assets/pbr_assets/vehicle_new \
  --asset-id vehicle_new \
  --config pbr-vehicle-convert/package/examples/configs/config_default.json
```

输出目录必须不存在或为空。输入 PLY 若已包含 `normal_0..2`，转换器会拒绝再次转换，以避免法线来源不明确。

校验已有资产：

```bash
pbr-vehicle inspect-complete Assets/pbr_assets/10010
```

校验内容包括：

- `configs/` 是否存在。
- 是否恰好存在一份合法 canonical 配置。
- 资产根目录是否只包含配置声明的单个 PLY payload。
- `configs/` 中的文件是否全部符合 `config_*.json` 命名。
- `files.config` 是否正确指向 canonical 配置。
- `sha256.pbr` 是否与实际 PLY 一致。

调整材质、光照或投影参数时，可另存新的 `config_*.json`。不要直接编辑已交付的 PLY 后继续使用旧的 canonical 哈希；需要改变几何或法线时，应重新执行转换并生成新的资产目录。
