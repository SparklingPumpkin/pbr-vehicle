# PBR Vehicle Panel

这是一个可独立安装的 Gaussian PBR 车辆交互面板。它直接加载调用方提供的场景 Gaussian PLY（或从兼容 PTH 提取 Background）和完整车辆 PBR 资产文件夹，并通过 Viser 完成车辆变换、材质、逐车/共享光照和两层投影调节。运行时不导入或读取任何训练仓库源码。

当前交付实现为 `VSP-v2`，机器可读身份见 [MAINLINE_MANIFEST.json](MAINLINE_MANIFEST.json)。版本 `1.9.1` 对接 `pbr-vehicle-auto 1.4.1` 的固定绝对搜索范围；默认通过 `--device auto` 使用本机可见 CUDA GPU，并继续对接 `pbr-vehicle-sun 1.3.0`。无 CUDA 时保留 CPU 回退，完整 Gaussian 和原生分辨率合同不变。

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

建议在任意 Python 3.9+ 虚拟环境中安装；不需要训练仓库或工作区 `PYTHONPATH`：

```bash
cd pbr-vehicle-panel
python -m pip install '.[gpu,pth]'
```

`gpu` 与 `pth` 都安装 PyTorch；已有与本机 CUDA 匹配的 PyTorch 时也可直接 `python -m pip install .`。仅加载 PLY 且强制 `--device cpu` 时不要求 PyTorch。

## 启动

加载车辆和 PTH 场景：

```bash
python -m pbr_vehicle_standalone \
  --scene \
    scenes/checkpoint.pth \
  --vehicle-asset-folder \
    assets/vehicle_example \
  --port 18091 \
  --device auto \
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

浏览器访问 `http://localhost:18091/?fixedDpr=1`。`fixedDpr=1` 固定原生画布分辨率，避免 Viser 在完整场景压力下自动降到低分辨率。面板可在运行时替换场景、添加/删除多辆车，并为每辆车选择共享场景光照或独立光照。保存按钮会在车辆目录的 `configs/` 下创建 `config_<场景>_<车辆>_<序号>_<时间>.json`。

每辆车的“高级 / R3GW Lighting”中可选“使用场景环境贴图”。启用后，Panel 在车辆中心对不含插入车辆和投影的场景渲染六个 `90°` cubemap 面：漫反射使用二阶 irradiance SH，镜面项使用当前观察相机方向和反射方向采样 cubemap，并按 roughness 选择 mip。车辆位置变化会重新捕获；相机位置变化只重算车辆颜色。该开关默认关闭，关闭或捕获失败时继续使用原有 SH 环境光预览。

同一区域的“Environment-map 预览”可显示贴有当前环境图的球体。独立模式下，环境球 XYZ 同时是采样和显示位置，移动后重新捕获；“跟随车辆”模式下，采样中心固定为车辆中心，显示偏移 XYZ 只把球移到便于观察的位置，不会改变贴图内容。捕获期间会隐藏所有车辆、两层投影和已有环境球，避免自反射。

Scene 区域新增“太阳估计”。安装相邻 `pbr-vehicle-sun` wheel、准备其外部模型环境，并在“太阳估计配置”填写可移植 JSON 后，点击按钮会异步执行场景级 SSE-v8。中间文件保留在系统临时目录；成功后自动启用太阳并回填共享光照的方位角和高度角。示例配置见 [examples/sun-estimator-config.example.json](examples/sun-estimator-config.example.json)。

加载 PTH 时，Panel 会读取同目录 `config.yaml` 的 `data_root` 与 `scene_idx`，将太阳估计绑定到当前场景的 RGB、标定和道路 mask；换场景后不会沿用旧数据。两轮估计会显示阶段进度，只采用 `two_round_result.json` 指定的发布轮次。成功后同时更新共享场景和所有现有车辆的太阳角，但保持每辆车原有的共享/独立光照模式。

每辆车顶部新增“Auto 识别车辆参数”。安装相邻 `pbr-vehicle-auto` wheel 后，按钮会在后台用完整普通 Gaussian 场景执行 ALM-v5；PTH 会先转换成不抽样的缓存 PLY。Auto 1.4.1 在 CUDA 可用时把整轮候选计算留在 GPU，并将太阳光强度、亮度固定在 Viser 滑条绝对区间 `[-0.3, 0.3]` 搜索，不跟随点击前参数移动。搜索成功后始终应用网格最优候选并回填太阳光强度、亮度和车辆色温。失败、超时或无效输出仍不覆盖面板。可移植配置示例见 [examples/auto-fit-config.example.json](examples/auto-fit-config.example.json)。

已安装时也可运行 `pbr-vehicle-panel`；`pbr-vehicle-viewer` 保留为兼容别名。

完整启动命令、全部参数含义和常用组合见 [使用指南.md](使用指南.md)。文档中的资产与场景路径均为相对路径示例。

## 范围与边界

- 场景保持普通 Gaussian DC 颜色，不执行 R3GW PBR；PTH 仅是兼容输入格式，不需要对应训练仓库源码。
- Environment-map 是可选的交互预览分支，默认关闭；使用当前浏览器对场景做 LDR cubemap 捕获，不声称恢复物理 HDR 辐射值。
- 太阳估计只依赖数据集 RGB/标定与独立 `pbr-vehicle-sun` 模型链，不改变场景 Gaussian；没有通过置信度门时不会覆盖当前太阳参数。
- Auto 识别使用普通 Gaussian 场景的 DC/opacity 与当前车辆状态，只拟合太阳光强度、亮度和车辆色温；它是外观匹配代理，不代表物理光照或材质真值。
- 场景与车辆默认加载全部 Gaussian，不设置数量上限；只有调用方显式传入非零 `--scene-max-splats` 时才对场景显示层抽样。
- 车辆面板直接展示太阳光强度、太阳方位角、太阳高度角、车辆色温、饱和度、亮度和 Center orbit；配置操作收入 `Config`，完整参数按 `Transform / Material / R3GW Lighting / Projection` 四类直接平铺在外层“高级”下，子面板不再嵌套“高级”或“次要参数”。
- 车辆/环境色温范围为 `-0.5~0.5`（`0=6500K`），饱和度为直接倍率 `0~2`（`1` 为中性），亮度为 `-1~1`（`0` 对应 fill light `0.35`）。
- 顶部六项参数会与“高级”中 Material / R3GW Lighting 分类下的同名参数双向同步。
- 环境光仅展示色温，RGB 由色温派生并写入兼容配置；太阳 RGB 仍独立保存在“次要参数”中。
- 车辆 Albedo 饱和度是 P0 参数；`Relight Original` 中直接作用于可见 PBR Gaussian 颜色，重光照比率按同一点计算，不依赖 mapping。
- `高级 / Material / 使用代理重光照` 默认开启：开启时保持代理光照比例回传方法；关闭时直接显示 PBR Gaussian 的朴素 PBR 着色。该选项逐车保存，旧配置默认开启。
- contact core 固定在车辆底部；cast extension 根据太阳方向重新生成，两层直接 alpha 叠加。
- 投影以 glTF `alphaMode=BLEND` 平面发送给 Viser，不依赖修改 Viser 前端安装文件。
- 服务端 PBR/GGX、mapping 和 cubemap 查询可由 Torch CUDA 加速；Viser Gaussian 光栅化仍在浏览器客户端执行。roughness 使用 cubemap mip 近似预过滤，不宣称与原生 GGX importance-sampled prefilter 或 DriveStudio rasterizer 像素一致。

## 测试

```bash
python -m pytest -q
```
