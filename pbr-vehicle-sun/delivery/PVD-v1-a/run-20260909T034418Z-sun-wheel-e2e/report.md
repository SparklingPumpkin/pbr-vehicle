# PVD-v1-a 交付核验

## 结论

`pbr-vehicle-sun 1.0.0` wheel 已通过独立安装与冻结输入复现，交付合同与上游 `SSE-v6`（由 `SSE-v5-f` 晋级）一致。

## 核验项

- 19 个 Python 模块完成编译；`pytest` 为 `2 passed`。
- wheel 在隔离 target 目录安装成功，6 个 console entry point 均可加载并输出帮助。
- wheel 内容扫描未发现 `/mnt/fjn`、`/mnt/data/fjn`、`/mnt/data/why`、`/root/` 或 `PBR-Inserts` 运行时硬编码。
- 默认 dry-run 为单帧 `feedforward_infinidepth`，启用 `camera_cone_middle` 和源图边缘段剔除；观测 floor 与候选 footprint 二次差集均关闭。
- 冻结样本 `scene-017/t025/cam2/vehicle-instance-19` 完整两阶段拟合输出 `181°/15°`，耗时 `137.91 s`。
- 该结果的最优解、目标函数、相机轮廓合同及扣除开关与上游 `SSE-v5-f` 对应结果逐字段相同。

## 核验中修复

首次回放发现交付调度器没有传递上游 `--no-observed-floor-subtraction`，虽然角度仍为 `181°/15°`，但评分目标和边界点数发生变化。交付默认已修正为观测阴影不再扣 floor，并新增显式旧行为开关；修正后严格复现上游分数 `-0.1527805903612691` 和 534 个观测边界点。

## 产物

- `pbr_vehicle_sun-1.0.0-py3-none-any.whl`：最终 wheel。
- `SHA256SUMS`：wheel 完整性校验。
- `verification_summary.json`：结构化核验摘要。
- `dry-run-final/`：最终默认分派 manifest。
- `real-single-frame-v2/`：冻结样本拟合结果、最佳轮廓图与同目标 Top-25。
