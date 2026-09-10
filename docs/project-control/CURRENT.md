# 当前项目控制面

> 由机器状态生成，只展示当前版本与仍活跃的候选；完整历史见 `VERSION_LEDGER.md`。

项目：PBR Vehicle Delivery（`pbr-vehicle-delivery`）  
更新时间：2026-09-09T04:28:26+00:00

## 项目主线

当前版本：`PVD-v2`（four-package-sun-delivery）

交付根现包含Convert、Panel、Auto与Sun四个独立可安装子包；Sun 1.0.0冻结SSE-v6双几何分支、单/多帧路由、全场景Top-3车辆选择和边缘安全相机锥形轮廓拟合。

当前证据：
- `pbr-vehicle-sun/delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e/verification_summary.json`：证明测试、隔离安装、CLI、wheel路径审计和181度/15度严格复现全部通过
- `pbr-vehicle-sun/DELIVERY_REPORT.md`：定义Sun交付范围并记录wheel与冻结样本核验结果

活跃候选：
- 无。
