# 项目版本总账

> 本文件由 `fjn-project-workflow` 根据不可追加覆盖的事件账本生成；请勿手改。

更新时间：2026-09-15T10:16:51+00:00

## 项目主线（project）

作用域类型：`project`  
当前版本：`PVD-v3`

### 主线版本

| 版本 | 日期 | 状态 | 父版本 | 来源候选 | 变更类型 | 摘要 | 证据 |
|---|---|---|---|---|---|---|---|
| PVD-v1 | 2026-09-09T03:40:40+00:00 | superseded | — | — | — | 当前交付根包含Convert、Panel、Auto三个独立子包和车辆资产库，尚未包含太阳角识别子包。 | 使用说明.md：记录现有三个可独立安装子包及其职责 |
| PVD-v2 | 2026-09-09T04:28:26+00:00 | superseded | PVD-v1 | PVD-v1-a | delivery_contract | 交付根现包含Convert、Panel、Auto与Sun四个独立可安装子包；Sun 1.0.0冻结SSE-v6双几何分支、单/多帧路由、全场景Top-3车辆选择和边缘安全相机锥形轮廓拟合。 | pbr-vehicle-sun/delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e/verification_summary.json：证明测试、隔离安装、CLI、wheel路径审计和181度/15度严格复现全部通过; pbr-vehicle-sun/DELIVERY_REPORT.md：定义Sun交付范围并记录wheel与冻结样本核验结果 |
| PVD-v3 | 2026-09-15T10:16:51+00:00 | current | PVD-v2 | PVD-v2-a | code_architecture | Sun 1.4.0/SSE-v9与Viser适配器完成端到端加速、三卡调度和缓存交付。 | pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/verification_summary.json：证明测试、wheel隔离安装、通用性扫描和运行时证据均通过; pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/report.md：汇总SSE-v9加速设计、真实性能、限制和交付证据 |

### 子实验候选

| 候选 | 基线 | 状态 | 假设 | 结论 | 运行数 | 证据 |
|---|---|---|---|---|---:|---|
| PVD-v1-a | PVD-v1 | promoted | 将SSE-v6核心脚本封装进无PBR-Inserts运行时路径依赖的pbr-vehicle-sun wheel，并显式配置外部模型仓库/权重，可复现核心单帧拟合且完整场景编排能通过CLI和输入合同核验。 | 新增第四个独立太阳识别交付包，wheel无上游源码路径依赖，默认合同与SSE-v6一致并严格复现SSE-v5-f冻结样本。 | 1 | pbr-vehicle-sun/delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e/verification_summary.json：证明测试、隔离安装、CLI、wheel路径审计和181度/15度严格复现全部通过 |
| PVD-v2-a | PVD-v2 | promoted | 通过消除InfiniDepth重复前向、前移SSISv2门控、并行独立车辆拟合、修正小轮廓KDTree线程开销、批量候选分割与持久缓存，可以保持SSE-v8输出语义并显著缩短Viser端到端时间，目标三车首轮低于120秒。 | 将SSISv2前置门控、批量SAM2、共享InfiniDepth前向、多GPU准备、并行拟合和内容缓存合并为SSE-v9；双GPU已接近两分钟，三GPU配置预计低于两分钟。 | 1 | pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/verification_summary.json：证明测试、wheel隔离安装、通用性扫描和运行时证据均通过 |

## 血缘解释

仅当事件账本中同时保存父版本、晋级候选和证据时，才生成承接关系。没有证据的旧关系应登记为未知，不得推断连线。
