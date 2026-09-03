# PBR Vehicle Panel

这是从 DriveStudio 训练/数据加载链路中拆出的最小运行包。它直接加载场景 Gaussian PLY（或从 PTH 提取 Background）、完整车辆 PBR 资产文件夹，并通过独立 Viser 面板完成车辆变换、材质、逐车/共享光照和两层投影调节。

当前交付实现对应 PBR-Inserts 项目控制面的 `VSP-v2` 主线；机器可读身份和证据指针见 [MAINLINE_MANIFEST.json](MAINLINE_MANIFEST.json)。

## 资产约定

车辆目录必须包含 canonical `configs/config_<车辆名>.json`。主线合同为 `pbr-vehicle-single-ply-v1`，其中 `files.pbr` 是相对路径。

```text
车辆目录/
├── pbr_<车辆名>.ply
└── configs/
    ├── config_<车辆名>.json
    └── config_*.json
```

程序以 `configs/config_<车辆名>.json` 中的 `files.pbr` 解析可见 PBR Gaussian。材质由 `material.albedo_rgb` 和全局材质字段广播到该层。未指定 `--config` 时，会从同目录的其他 `config_*.json` 中按 `--seed` 随机选择一个；若没有其他配置，则使用资产主配置。

## 安装

推荐沿用已有 DriveStudio Python 环境，但运行时不需要 DriveStudio 仓库：

```bash
cd pbr-vehicle-panel
python -m pip install -e .
```

仅加载 PLY 时不要求 PyTorch。加载 PTH 需要环境中已有 PyTorch，或安装 `.[pth]`。

## 启动

加载车辆和 PTH 场景：

```bash
python -m pbr_vehicle_standalone \
  --scene \
    scenes/checkpoint.pth \
  --vehicle-asset-folder \
    assets/vehicle_example \
  --port 18091 \
  --scene-cache-dir \
    .cache/pbr_vehicle_scenes
```

仅加载车辆（启动更快）：

```bash
python -m pbr_vehicle_standalone \
  --vehicle-asset-folder \
    assets/vehicle_example \
  --port 18091
```

浏览器访问 `http://localhost:18091`。面板可在运行时替换场景、添加/删除多辆车，并为每辆车选择共享场景光照或独立光照。保存按钮会在车辆目录的 `configs/` 下创建 `config_<场景>_<车辆>_<序号>_<时间>.json`。

已安装时也可运行 `pbr-vehicle-panel`；`pbr-vehicle-viewer` 保留为兼容别名。

完整启动命令、全部参数含义和常用组合见 [使用指南.md](使用指南.md)。文档中的资产与场景路径均为相对路径示例。

## 范围与边界

- 场景保持普通 Gaussian SH 颜色，不执行 R3GW PBR。
- 场景与车辆始终加载全部 Gaussian，不提供数量限制或抽样选项。
- 车辆面板直接展示太阳强度、方位角、高度角、车辆色温、饱和度、补光强度和 Center orbit；配置操作收入 `Config`，完整参数按 `Transform / Material / R3GW Lighting / Projection` 四类直接平铺在外层“高级”下，子面板不再嵌套“高级”或“次要参数”。
- 顶部六项参数会与“高级”中 Material / R3GW Lighting 分类下的同名参数双向同步。
- 环境光仅展示色温，RGB 由色温派生并写入兼容配置；太阳 RGB 仍独立保存在“次要参数”中。
- 车辆 Albedo 饱和度是 P0 参数；`Relight Original` 中直接作用于可见 PBR Gaussian 颜色，重光照比率按同一点计算，不依赖 mapping。
- contact core 固定在车辆底部；cast extension 根据太阳方向重新生成，两层直接 alpha 叠加。
- 投影以 glTF `alphaMode=BLEND` 平面发送给 Viser，不依赖修改 Viser 前端安装文件。
- 这是交互预览实现，不包含 DriveStudio 相机批渲染、训练器、数据集或 CUDA rasterizer；它不宣称与 DriveStudio 原生 GGX rasterizer 像素一致。

## 测试

```bash
python -m pytest -q
```
