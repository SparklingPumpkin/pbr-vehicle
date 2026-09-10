# PBR 车辆资产版本说明

本文件只记录 `Assets/` 资产合同的变化，不记录面板、转换器或自动拟合工具的版本。

## P-v4 / `pbr-vehicle-single-ply-v1` / schema 10

- 当前资产采用单个 `pbr_<asset-id>.ply` 加 `configs/config_*.json` 的自包含布局。
- PLY 保留可见 Gaussian 的原始中心、协方差、旋转、opacity、DC/高阶 SH 字段，并追加 `normal_0..2` 单位法线。
- 全局材质、默认光照、投影和哈希写入 JSON；不再需要实体 proxy PLY、original-to-proxy 映射 NPZ 或 projection PLY。
- 运行时逻辑代理在同一批 Gaussian 上计算 PBR 光照倍率，再乘回原始可见颜色。
- 资产根目录只允许一个二进制 PLY payload；所有配置集中在 `configs/`，路径相对于资产根目录。

## P-v3 / direct-white 2DGS

- original PLY、proxy PLY 和 mapping NPZ 构成完整车辆资产。
- proxy 用于法线、Albedo、粗糙度、金属度和 PBR 控制；mapping 将 proxy 的光照倍率传回 original。
- direct-white identity 映射保持 original Gaussian 中心和数量不变，不做位置迁移、补洞或高斯数量限制。

## P-v2 / 双层车辆资产

- 初始合同将可见 original 与 PBR proxy 分开保存，并支持 KNN/identity 映射。
- 后续加入固定车底 contact core 与随太阳方向变化的 cast extension 接收面投影。

## 兼容约定

- 新消费者应优先识别 `pbr-vehicle-single-ply-v1`；旧的 original/proxy/mapping 目录仅作为历史兼容输入。
- canonical 配置为 `configs/config_<asset-id>.json`，`files.*` 不得依赖机器绝对路径。
- 修改 PLY 后必须重新计算 `sha256.pbr`，不得继续使用旧 canonical 哈希。
- 资产版本与面板版本、转换器版本、自动拟合版本分别维护。
