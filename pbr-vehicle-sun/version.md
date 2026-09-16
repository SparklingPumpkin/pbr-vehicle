# pbr-vehicle-sun 版本说明

## 当前版本：1.4.0

- 对应加速主线 `SSE-v9`；SSISv2、P95、相机可见轮廓、搜索网格和硬门语义不变。
- YOLO batch 16；SAM2.1 批量图像编码并将同图全部 box 一次送入 decoder，仍穷举全场景候选。
- SSISv2 前置门控；有效车辆按视图分组，并在调用方指定的最多三张 GPU 上并行准备。
- InfiniDepth 单次共享前向直接导出深度和源视图逐像素 Gaussian，默认跳过重复深度推理、2M 点采样及 PLY I/O；保留 `legacy_ply` 兼容模式。
- 三车独立拟合并行，每车候选角默认 8 线程；搜索仅保留标量，证据按需重算，联合拟合复用数学等价的独立粗网格。
- 新增完整调用内容缓存；输入、参数、模型元数据或包源码变化会使缓存失效。

## 1.3.0

- 所有公开推理入口默认使用 `--device auto`；有 CUDA 时选择当前进程可见的第一张本地 GPU，无 CUDA时回退 CPU。
- 自动把统一设备转换为 Torch/Detectron2 的 `cuda:0` 与 Ultralytics 的 `0`，修复 `auto`/`cpu` 被错误拼成 CUDA 字符串的问题。
- 遵循 `CUDA_VISIBLE_DEVICES`，不写死 GPU 型号或物理卡号；子进程会打印实际解析设备。
- 运行依赖约束 NumPy 为 `>=1.24,<2`，避免旧版 Torch 与 NumPy 2 的 ABI 不兼容组合。

## 1.2.0

- 默认 `pbr-vehicle-sun-scene` 已加入两轮换车：首轮全局排名 1–3，无有效角时复用检测和排名并更换为排名 4–6。
- 第二轮仍无有效角才返回 `no_valid_sun_information`；进程失败立即停止并返回非零码，不触发换车。
- 新增轮级重试、完成结果恢复和 `two_round_result.json`；保留 `pbr-vehicle-sun-scene-round` 作为高级单轮入口。
- Argoverse 000–049 审计中 26/50 发布、24/50 无有效角、0 个进程失败；007–049 中 9 个场景由第二轮挽救。

## 1.1.0

- 对应研究主线 `SSE-v8`，其阴影协议严格采用实验 `SSE-v6-b`，并继承 `SSE-v7` 的通用多车门控。
- 以 SSISv2 object-shadow association 替换已弃用的旧阴影检测链。
- 官方 SSISv2 shadow mask 直接参与 InfiniDepth 提升，不执行差集、连通域清洗或形态学后处理。
- 默认全场景 Top 3 物理车辆，通用支持 Top 1–5。
- 每车执行 P95 相机可见轮廓拟合和双硬门；保留通过车辆的加权角与共享角联合拟合，全部拒绝时不发布太阳角。
- wheel 不包含模型、权重、数据资产、缓存结果或机器绝对路径。
