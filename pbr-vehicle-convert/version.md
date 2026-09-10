# pbr-vehicle-convert 版本说明

## 当前版本

- 当前包版本：`1.5.0`，以本目录 `package/pyproject.toml` 中声明的版本为准。
- 当前转换器面向 `pbr-vehicle-single-ply-v1` 单 PLY 资产合同。

## 功能演进

### 1.5.0

- 支持从普通 Gaussian PLY 生成单 PLY PBR 车辆资产。
- 追加 `normal_0..2`，保留原始 Gaussian 几何和已有 SH 字段。
- 生成 canonical config、相对路径和 PLY SHA256。
- 支持资产检查、配置校验和投影参数合同。
- 投影描述器使用车辆 footprint 与平行光投影终点 footprint 的连续扫掠凸包，避免低太阳高度下 extension 与车辆脱离。

### 1.3.x

- 维护 original/proxy/mapping 多文件资产转换合同。
- 支持 direct-white identity 映射和通用 KNN 映射。
- 增加配置目录校验、文件命名约束和转换后资产报告。

### 1.0.x - 1.2.x

- 建立普通 Gaussian PLY 到 PBR 车辆资产的 Python API、CLI、测试和 wheel 交付流程。
- 支持基础材质、光照、车辆投影配置以及相对路径 manifest。

## 兼容边界

- 转换器版本不等同于资产合同版本；输出合同由生成的 config 中 `asset_contract` 和 `schema_version` 决定。
- 资产几何或法线发生变化时必须重新转换并更新哈希。
