# SSE-v9 运行时加速实验报告

本轮已把完整三车冷启动从旧版 scene-002 的 1093.5 秒降到双 GPU 实测 129.3 秒，约 8.46 倍加速。由于基准时第三张本机 GPU 被无关任务持续占满，无法取得无污染三卡实测；按本轮实际调度，三卡会消除 19.224 秒的第三视图串行 InfiniDepth 段，预计约 110.0 秒，达到 2 分钟目标。完全相同输入的缓存命中由 0.61 秒数据指纹和约 0.05 秒产物恢复组成。

主要改动：YOLO batch 16；SAM2.1 batch image encoder 与同图多 box decoder；SSISv2 前置并行门控；一次 InfiniDepth 前向共享深度和源视图可见 Gaussian；最多三卡车辆准备；三车并行拟合、每车 8 路候选角；KDTree/BLAS/OpenCV 防止线程超额订阅；搜索标量化和证据按需重算；联合粗网格在数学等价时复用；完整内容缓存。

科学协议仍使用 SSISv2 官方 object-shadow pair、未经后处理的 shadow mask、P95 相机可见轮廓、原两阶段角度网格和原硬门。车辆选择继续穷举全部合格观测，没有候选预筛。InfiniDepth 的 direct-dense 几何采样替代旧 2M PLY 随机采样，属于轻微方法变更，因此新主线明确编号 `SSE-v9`；旧路径可用 `--geometry-mode legacy_ply` 恢复。

主要证据：

- `full-cold-final/two_round_result.json`：双 GPU 完整冷启动与发布角。
- `full-cold-final/round-1/scene_result.json`：逐阶段真实计时。
- `selection_benchmark.json`：SAM2 批处理性能和选择等价性。
- `sam2_batch4_vs_batch16_audit.json`：同输入 batch 4/16 的 Top-6 与像素 mask 等价性。
- `benchmark_summary.json`：机器可读的汇总、限制和三卡推算方法。
- `verification_summary.json`：测试、wheel、路径扫描和安装验收。
