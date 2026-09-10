# PBR Vehicle Sun

`pbr-vehicle-sun 1.0.0` 是场景太阳方位角/高度角识别的独立交付包，对应研究主线 `SSE-v6`。它从车辆投影几何与原视图车辆阴影恢复太阳角，不修改车辆 PBR 材质，也不拟合太阳 RGB 或强度。默认严格复现已接受的 `SSE-v5-f` 合同：观测阴影不再扣除车辆 floor，候选投影也不做 footprint 差集；两项旧行为只能显式开启。

## 两个入口

- `pbr-vehicle-sun-fit`：核心拟合入口。输入已经对齐的车辆几何 NPZ、阴影深度提升 NPZ 和后处理阴影 mask；支持单帧、多帧以及自带 Gaussian / InfiniDepth 前馈两种几何分支。
- `pbr-vehicle-sun-scene`：Argoverse 完整场景入口。遍历全部帧和相机，以 YOLO + SAM2 对物理车辆去重，按最大 mask 面积选前三辆；每车执行 MTMT、后处理、阴影证据门、纯 RGB InfiniDepth 几何和独立角度拟合，再按轮廓吻合度选 winner。

完整中文说明见 [使用指南.md](使用指南.md)。

## 安装与测试

```bash
python -m pip install -e '.[scene,test]'
python -m pytest -q
```

核心 wheel 不捆绑模型权重。完整场景入口要求调用方显式传入 YOLO、SAM2、MTMT 和 InfiniDepth 的仓库或权重路径。
