# pbr-vehicle-panel 版本说明

## 当前版本：1.3.1

- 顶层车辆面板使用中文参数：太阳光强度、太阳方位角、太阳高度角、车辆色温、饱和度、亮度和 Center orbit。
- 色温滑条范围为 `-0.5~0.5`，映射 `2000~12000K`，`0=6500K`。
- 饱和度为直接倍率 `0~2`，`1` 为中性；亮度为 `-1~1`，`0` 对应内部 fill light `0.35`。
- 太阳光强度继续使用 `-1~1`，角度继续使用度数；配置文件保存内部兼容物理量。
- 主线 DriveStudio Viewer 与独立 Viser 交付包保持同一参数映射。
- 修复低太阳高度下 cast extension 只绘制投影终点、与车辆脱离的问题；extension 现在覆盖车辆 footprint 到投影终点 footprint 的完整平行光扫掠区域，方向保持为太阳方位角加 `180` 度。

## 1.3.0

- 暴露参数改为中文，增加亮度命名和统一归一化控制。
- 独立交付包支持单 PLY `pbr-vehicle-single-ply-v1` 资产合同。
- 增加渲染原理、公式、Viser buffer 和普通 Gaussian 差异说明。

## 1.2.x

- 增加中文使用指南、可移植相对路径示例和 PTH 场景转换说明。
- 清理交付包中的机器绝对路径，真实资产测试改为环境变量可选启用。
- 完成六项快捷参数、单层高级分类、Config 隔离、双向镜像和多车辆控制。

## 1.1.x

- 从 DriveStudio 链路拆出独立 Viser 面板。
- 支持场景 PLY/PTH、车辆添加/删除、逐车或共享光照、配置保存/加载。
- 使用固定车底 contact core 与动态 cast extension 两层程序化投影。

## 1.0.x

- 建立 Gaussian PBR 车辆面板的首个独立交付版本。
- 支持车辆变换、材质、R3GW 光照、投影调节和全量 Gaussian 加载。

## 维护规则

每次发布新版本时同步更新 `pyproject.toml`、`MAINLINE_MANIFEST.json`、`DELIVERY_REPORT.md`、本文件、测试和 `dist/SHA256SUMS`。资产合同变化记录在 `../Assets/version.md`，转换器和自动拟合工具分别记录在各自子项目的 `version.md`。
