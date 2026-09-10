# PVD-v2：four-package-sun-delivery

> 该文件是版本建立时的不可变快照。状态变化和后续候选见 `../VERSION_LEDGER.md`。

作用域：项目主线（`project`）  
建立时间：2026-09-09T04:28:26+00:00  
父版本：`PVD-v1`  
来源候选：`PVD-v1-a`  
旧标识：`无`

## 当前真相摘要

交付根现包含Convert、Panel、Auto与Sun四个独立可安装子包；Sun 1.0.0冻结SSE-v6双几何分支、单/多帧路由、全场景Top-3车辆选择和边缘安全相机锥形轮廓拟合。

## 改变类型

- `delivery_contract`

## 证据

- `pbr-vehicle-sun/delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e/verification_summary.json`：证明测试、隔离安装、CLI、wheel路径审计和181度/15度严格复现全部通过
- `pbr-vehicle-sun/DELIVERY_REPORT.md`：定义Sun交付范围并记录wheel与冻结样本核验结果

## 晋级依据

- 候选：`PVD-v1-a`
- 假设：将SSE-v6核心脚本封装进无PBR-Inserts运行时路径依赖的pbr-vehicle-sun wheel，并显式配置外部模型仓库/权重，可复现核心单帧拟合且完整场景编排能通过CLI和输入合同核验。
- 结论：新增第四个独立太阳识别交付包，wheel无上游源码路径依赖，默认合同与SSE-v6一致并严格复现SSE-v5-f冻结样本。
