# PBR Vehicle Sun 1.0.0 Delivery Report

本交付包将 `PBR-Inserts` 的场景太阳识别 `SSE-v6` 主线冻结为独立 Python wheel。核心拟合器只消费对齐后的 NPZ/mask，不依赖上游源码路径；完整场景入口将外部 YOLO、SAM2、MTMT 与 InfiniDepth 仓库/权重作为显式参数。

默认无 Gaussian 合同是纯 RGB InfiniDepth、全场景跨帧物理车辆 Top-3、阴影概率/面积 fail-closed 门、源图边缘段剔除，以及观测/预测统一的最大相机锥形近侧完整轮廓。观测阴影默认不再扣除车辆 floor，候选车辆投影也不执行会制造贴车 footprint 红线的布尔差集；两项旧行为均为显式兼容开关。

交付验证包含源码测试、wheel 内容审计、隔离目录安装、CLI 帮助、默认分派 manifest，以及使用冻结真实几何输入完成一次核心角度拟合。最终结果为 `2 passed`，6 个 CLI 入口均可加载，wheel 未包含机器绝对路径；冻结 `scene-017/t025/cam2` 样本严格复现上游 `181°/15°`、分数 `-0.1527805903612691` 和相机轮廓合同。验证记录位于 `delivery/PVD-v1-a/run-20260909T034418Z-sun-wheel-e2e/`。
