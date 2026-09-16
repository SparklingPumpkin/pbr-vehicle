# PVD-v3：sse-v9-runtime-accelerated-delivery

> 该文件是版本建立时的不可变快照。状态变化和后续候选见 `../VERSION_LEDGER.md`。

作用域：项目主线（`project`）  
建立时间：2026-09-15T10:16:51+00:00  
父版本：`PVD-v2`  
来源候选：`PVD-v2-a`  
旧标识：`无`

## 当前真相摘要

Sun 1.4.0/SSE-v9与Viser适配器完成端到端加速、三卡调度和缓存交付。

## 改变类型

- `code_architecture`

## 证据

- `pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/verification_summary.json`：证明测试、wheel隔离安装、通用性扫描和运行时证据均通过
- `pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark/report.md`：汇总SSE-v9加速设计、真实性能、限制和交付证据

## 晋级依据

- 候选：`PVD-v2-a`
- 假设：通过消除InfiniDepth重复前向、前移SSISv2门控、并行独立车辆拟合、修正小轮廓KDTree线程开销、批量候选分割与持久缓存，可以保持SSE-v8输出语义并显著缩短Viser端到端时间，目标三车首轮低于120秒。
- 结论：将SSISv2前置门控、批量SAM2、共享InfiniDepth前向、多GPU准备、并行拟合和内容缓存合并为SSE-v9；双GPU已接近两分钟，三GPU配置预计低于两分钟。
