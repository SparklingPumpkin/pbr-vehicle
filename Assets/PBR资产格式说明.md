# PBR 资产格式说明

本目录存放可由 `pbr-vehicle-convert`、`pbr-vehicle-panel` 与 `pbr-vehicle-auto` 共同使用的 PBR 车辆资产。共同合同为 `pbr-vehicle-config-folder-v4`。

## 资产库

```text
Assets/
  pbr_assets/
    manifest.json                 # 资产库清单、哈希和生成记录
    validation.json               # 资产库验收结果
    <asset-id>/
      original_<asset-id>.ply     # 最终可见的原始车辆 Gaussian
      proxy_<asset-id>.ply        # PBR proxy Gaussian
      map_<asset-id>.npz          # original -> proxy 映射
      configs/
        config_<asset-id>.json    # canonical 资产配置
        config_*.json             # Panel 或 Auto 的场景配置
```

当前预置资产为 `10010`、`10011`、`10012`、`10013`、`10014` 和 `10015`。每辆车都经过 identity 中心/opacity、mapping 与运行时投影合同验收；总览见 `pbr_assets/manifest.json` 与 `pbr_assets/validation.json`。

## 核心原则

- `original_<asset-id>.ply` 是最终可见车辆层。PBR 不替换车辆几何，而是将 proxy 计算的重光照响应映射回 original。
- `proxy_<asset-id>.ply` 是隐藏控制层，承载 PBR albedo、normal 和材质字段。
- `map_<asset-id>.npz` 规定 original 到 proxy 的绑定。消费者不得假设点顺序相同，必须使用映射。
- canonical 配置是发现以上三个文件的唯一入口。消费者应读取 `files` 字段，不应拼接固定文件名。
- 场景配置和自动配置不得替代或修改 canonical 配置中定义的资产文件与哈希。

## Canonical 配置

每辆车必须有且仅有一个 canonical 配置：

```text
<asset-dir>/configs/config_<asset-id>.json
```

最小必要结构如下：

```json
{
  "schema_version": 9,
  "asset_contract": "pbr-vehicle-config-folder-v4",
  "asset_id": "10010",
  "files": {
    "original": "original_10010.ply",
    "proxy": "proxy_10010.ply",
    "mapping": "map_10010.npz",
    "config": "configs/config_10010.json"
  },
  "material": { "roughness": 0.32, "metallic": 0.02 },
  "light": { "sun_enabled": true, "sun_azimuth_degrees": 126.0 },
  "projection": { "runtime": "receiver_space_mask_v1" },
  "sha256": { "original": "...", "proxy": "...", "mapping": "..." }
}
```

`files.*` 路径相对于资产根目录。`files.config` 必须指向当前 canonical 文件本身。`sha256` 用于校验三个 payload 是否仍与交付时一致。

配置还保存默认材质、光照和投影参数。它们是可编辑的默认状态；原始、proxy 与 mapping 文件在同一资产内应保持不变。场景状态、车辆姿态或 Auto 结果应另存为新的 `config_*.json`。

## 三个二进制 payload

### Original PLY

`original_<asset-id>.ply` 必须是标准 3DGS PLY，至少含有：

| 字段 | 含义 |
| --- | --- |
| `x`, `y`, `z` | Gaussian 中心。 |
| `f_dc_0`, `f_dc_1`, `f_dc_2` | DC 颜色系数。 |
| `scale_0..2` | Gaussian 的 log scale。 |
| `rot_0..3` | 四元数旋转。 |
| `opacity` | logit opacity。 |

可包含 `f_rest_*` 等高阶 SH。Auto 产出的诊断 bake 会将其清零，因为该 bake 只写 DC；这不改变原始资产文件。

### Proxy PLY

`proxy_<asset-id>.ply` 也是 Gaussian PLY，除几何与 opacity 字段外，需要一套 PBR 字段。消费者支持以下两种命名之一：

| 语义 | P-v3 direct-white 命名 | 一般 KNN proxy 命名 |
| --- | --- | --- |
| 反照率 RGB | `r3gw_albedo_0..2` | `pbr_albedo_0..2` |
| 法线 XYZ | `r3gw_normal_0..2` | `pbr_normal_0..2` |
| 粗糙度 | `r3gw_roughness`，或 `roughness` logit | `pbr_roughness` |
| 金属度 | `r3gw_metallic`，或 `metalness` logit | `pbr_metallic` |

P-v3 direct-white 资产的 proxy 保留输入 Gaussian 中心和 opacity，把最小协方差轴压薄为白色 2DGS 表面，并以该表面的外向法线作为 PBR normal。当前预置 `10010-10015` 均属于此类型。

### Mapping NPZ

映射支持两种合法形式：

| 类型 | 必需数组 | 含义 |
| --- | --- | --- |
| identity | `clean_to_proxy_idx` | 每个 original 点对应一个 proxy 点。P-v3 资产同时带有单邻居 KNN 数组以便兼容。 |
| KNN | `raw_to_proxy_knn_idx`、`raw_to_proxy_knn_weight` | 每个 original 点对应一个或多个 proxy 点；权重必须归一化。 |

可选数组 `raw_to_proxy_knn_distance` 只用于诊断。Panel 和 Auto 会以权重混合 proxy 的光照比值，再乘回 original 颜色。映射下标必须位于 proxy 点数范围内，且行数必须等于 original 点数。

## 配置类型与使用方式

| 类型 | 识别方式 | 用途 |
| --- | --- | --- |
| canonical 资产配置 | `asset_contract: pbr-vehicle-config-folder-v4` 且 `files.config` 指向自身 | 定义资产与默认 PBR 状态。 |
| Panel 保存状态 | 包含 `scene`、`pbr_asset` 与 `vehicle` | 保存场景路径、车辆姿态、手调材质、光照和投影。 |
| Auto 结果 | 含 `auto_fit` | 记录 DC 适配的输入、指标、候选搜索和最终/拒绝状态。 |

Panel 可直接加载 canonical、场景配置或 Auto 配置。Auto 可接收 canonical 配置或完整 Panel 保存状态作为 `--template-config`。当 Auto 状态为 `rejected_no_improvement` 时，配置保留输入模板外观；此文件只记录一次被拒绝的搜索，不代表新的光照参数。

## 投影合同

投影不是资产 PLY，而是运行时生成的接收面 mask：

```text
projection.runtime      = receiver_space_mask_v1
projection.shape_model  = procedural-mask-contact-silhouette-v4
projection.composition  = receiver_alpha_blend_to_grayscale_v1
```

它包含车辆底部的圆角矩形接触层和按平行太阳光投射的车辆外轮廓延伸层。投影会随太阳方向、车辆姿态和投影参数重算；资产目录中不应存在或依赖 `projection_<asset-id>.ply`。

## 完整性校验

对由 `complete-asset` 生成或本库提供的 P-v3 资产，可运行：

```bash
pbr-vehicle inspect-complete Assets/pbr_assets/10010
```

该命令校验配置布局、payload 文件、canonical 路径和 SHA-256。库级的 `manifest.json` 记录各预置资产的点数与哈希，`validation.json` 记录六辆车通过 identity、opacity、mapping 和投影合同验证的结果。

不要编辑 `original_*.ply`、`proxy_*.ply` 或 `map_*.npz` 后继续沿用旧哈希。若需要生成新车辆，请通过 Convert 创建新的资产目录；若只需要调整外观，请创建新的 `config_*.json`。
