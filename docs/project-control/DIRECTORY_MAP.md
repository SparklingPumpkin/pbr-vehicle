# 文件夹职责总表

> 每个受管文件夹必须有唯一职责、层级和父级。新增实验目录前先记录布局决策。

活跃平级目录预算：6  
最大受管深度：8

| ID | 路径 | 层级 | 父级 | 作用域 | 状态 | 管理直接子目录 | 职责 |
|---|---|---|---|---|---|---|---|
| dir-0001 | . | project | — | project | active | 否 | 项目边界；其子目录必须按业务职责继续登记。 |
| dir-0002 | pbr-vehicle-sun | workstream | dir-0001 | project | active | 是 | 第四个独立可安装太阳角识别交付包，承载SSE-v6核心拟合、完整场景编排、文档与wheel |
| dir-0003 | pbr-vehicle-sun/delivery | experiment_family | dir-0002 | project | active | 是 | 承载太阳识别交付包的候选构建与验证记录 |
| dir-0004 | pbr-vehicle-sun/delivery/PVD-v1-a | candidate | dir-0003 | project | active | 是 | 将PBR-Inserts SSE-v6双分支、单多帧、全场景Top3与证据可视化打包为独立wheel |
| dir-0006 | pbr-vehicle-sun/delivery/PVD-v2-a | candidate | dir-0003 | project | active | 否 | 在保持SSE-v8科学与发布合同不变的前提下，将Viser三车场景太阳估计压缩到两分钟目标并同步独立wheel交付 |
| dir-0005 | pbr-vehicle-sun/delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e | run | dir-0004 | project | active | 是 | 构建wheel并在隔离安装目录验证CLI、核心太阳角单帧结果、场景编排dry-run和路径独立性 |
| dir-0007 | pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark | run | dir-0006 | project | active | 否 | 对SSE-v8旧版与加速版进行单场景三车端到端计时、结果等价性和wheel安装验证 |

## 布局决策历史

| 决策 | 状态 | 动作 | 目标路径 | 理由 | 备选方案 | 豁免理由 |
|---|---|---|---|---|---|---|
| layout-0001 | applied | new_group | pbr-vehicle-sun | 太阳识别与Convert、Panel、Auto职责和依赖独立，应作为平级子交付包而非混入现有包 | 把脚本散落复制进pbr-vehicle-auto会混淆外观适配和太阳角识别边界 | — |
| layout-0002 | applied | new_group | pbr-vehicle-sun/delivery | 项目工作流要求候选位于实验族下，交付源码仍位于pbr-vehicle-sun根目录 | 跳过实验族直接登记候选不符合目录合同 | — |
| layout-0003 | applied | new_child | pbr-vehicle-sun/delivery/PVD-v1-a | 这是从三包交付基线扩展为四包的可验证交付合同变化 | 只复制说明文档而不提供可安装CLI无法构成交付包 | — |
| layout-0004 | applied | new_child | pbr-vehicle-sun/delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e | 交付包必须证明安装后不依赖PBR-Inserts源码目录 | 仅运行源目录pytest不能证明wheel交付完整 | — |
| layout-0005 | applied | new_child | pbr-vehicle-sun/delivery/PVD-v2-a | 运行时调度、缓存与等价数值实现属于交付性能合同变化，需要独立候选审计 | 直接覆盖PVD-v2会失去速度和结果等价性的回归证据 | — |
| layout-0006 | applied | new_child | pbr-vehicle-sun/delivery/PVD-v2-a/run-20260915T091753Z-three-vehicle-runtime-benchmark | 需要持久记录两分钟目标是否达到及每项优化的真实收益 | 只报告微基准无法证明Viser完整流程性能 | — |
