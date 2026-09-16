# PBR Vehicle Sun 1.4.0 Delivery Report

Version 1.4.0 promotes the accelerated `SSE-v9` runtime. It preserves the SSISv2 official association, P95 camera-visible contour objective, search grid and confidence gates from SSE-v8 while parallelizing independent work.

车辆选择仍穷举全部帧、相机和合格 YOLO11 观测。SAM2.1 批量编码图像，并一次解码同图全部 box；未执行候选预筛。Argoverse 002 上 SAM2 排名由 69.09 秒降至 14.28 秒，Top-6 选择和输出 mask 与同输入逐张版本一致。

SSISv2 在 InfiniDepth 前并行执行，拒绝车辆不再计算三维几何。通过车辆按源视图分组，在 `--worker-devices` 上并行；一次 InfiniDepth 前向同时导出深度和逐像素源视图可见 Gaussian，跳过重复 dense-depth 前向、2M 点随机采样及约 128 MB PLY 写入/重读。与旧 PLY 路径的双向最近点中位差为 2.2/2.7 mm，固定角分数由 -0.15761 变为 -0.15608；因此该几何采样变更以 SSE-v9 单独标识，并保留 `legacy_ply`。

拟合侧同时并行最多三辆车，每车默认 8 个候选角线程；小轮廓 KDTree 禁止创建全机线程池，搜索阶段只保存标量，最终证据按需重算。三车 scene-002 同输入拟合由约 99 秒降至 24.61 秒，角度和门控结果逐值一致。联合粗网格仅在数学等价时复用单车分数；存在随机降采样风险时自动回退完整联合粗搜。

完整首次运行实测和缓存命中计时记录在 `delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/`。缓存键覆盖完整参数、路径资产元数据和包内 Python 源码；二进制结果硬链接恢复，文本 manifest 使用独立副本并重写输出路径。

默认场景入口仍执行两轮换车合同：首轮排名 1–3，仅在正常完成但无有效角时复用检测/排名并改用排名 4–6。进程失败不触发换车。活动链不含已弃用阴影检测器，也不对 SSISv2 官方 shadow mask 做差集、连通域或形态学后处理。

交付物：`dist/pbr_vehicle_sun-1.4.0-py3-none-any.whl`（88,519 bytes，SHA-256 `92126fabbdfe60f3f4740036fe8c3736e9c197a42cc47038fb4d1838f3e2c220`）。wheel 不包含权重、模型仓库、场景资产、缓存或机器路径；完整验收见 `delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/verification_summary.json`。
