# PBR Vehicle Sun 1.2.0 Delivery Report

本交付包冻结 `SSE-v8`：上游严格采用 `SSE-v6-b` 的 SSISv2 官方关联阴影 mask 协议，下游继承 `SSE-v7` 的 P95 相机可见轮廓及通用 1–5 车置信度门控/联合拟合。

默认场景入口已纳入两轮换车合同：首轮使用全局排名 1–3；仅在首轮正常完成但无有效角时，复用 YOLO 检测和 SAM2 排名，更换为排名 4–6。第二轮仍无有效角才返回 `no_valid_sun_information`。进程、文件或资源失败保持独立失败类型，不触发换车。

活动执行链中已移除旧阴影检测器、阴影减车辆、连通域清洗、形态学修改及其概率图证据门。SSISv2 object member 与 YOLO/SAM2 目标车辆通过 IoU 绑定，关联 shadow member 直接进入同帧 InfiniDepth 几何提升。

Argoverse 000–049 的冻结审计结果为 26/50 场景发布太阳角、24/50 场景返回无有效太阳信息、0 个进程失败。007–049 中有 9 个场景由第二轮排名 4–6 的替换车辆挽救。

交付 wheel 仅包含通用代码。外部仓库、权重、解释器、数据和输出路径均由调用参数提供，不捆绑场景资产。wheel SHA256 记录于本目录 `dist/SHA256SUMS`。

交付验收：完整测试 `9 passed`；Python 3.8 源码编译通过；Python 3.12 隔离安装通过；默认与高级 console script 映射通过；wheel 机器路径和实验产物扫描通过。
