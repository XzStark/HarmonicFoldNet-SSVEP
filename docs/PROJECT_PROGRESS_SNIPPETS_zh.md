# 对外材料统一进展表述

以下文本统一来源于 `runs/kim2025_full_v1/evaluation.json`。正式发布时补充 GitHub、Hugging Face、技术报告和 DOI 链接。

## 宇树材料

项目近期完成了轻量化 SSVEP 脑电识别模型的首版训练与验证。模型基于 40 名参与者、9,600 个公开同步脑电试次进行被试隔离训练和测试，使用 8 个枕区通道完成 40 类刺激识别。当前固定测试集 Top-1 为 88.65%，Top-5 为 95.73%；相较同条件固定双谐波功率基线，Top-1 提高 9.59 个百分点。训练结构经等价重参数化后，CPU 单线程和 GPU 的单次模型计算延迟分别由 8.37 ms 降至 6.01 ms、由 8.01 ms 降至 5.20 ms。现阶段正在补充短窗口、完整跨被试和真实端侧验证，该方向作为眼镜多模态交互研究的一部分推进，不作为常驻频闪交互或医疗功能。

## 教授材料

脑电交互方向已完成第一阶段可复现实验。当前方法将刺激候选频率及谐波形成的物理证据，与时域、频域局部特征和后期注意力融合，并通过一维训练-部署结构重参数化降低推理开销。在 40 名参与者、9,600 个试次的公开数据上，采用 32/4/4 名参与者完全隔离的训练、验证和测试划分，完整模型测试 Top-1 为 88.65%，固定双谐波功率基线为 79.06%。部署结构通过数值等价验证，并将 CPU/GPU 模型计算延迟降低 28.2%/35.0%。该结果仍属于阶段性证据，尚需通过短窗口、LOSO、传统与深度学习基线、统计显著性和真实设备实验验证其普适性。

## a16z 申请网站

### Efficient Cross-Subject SSVEP Decoding

I built a compact 40-class EEG decoding model that combines candidate-frequency evidence, time/frequency representations, late attention, and an equivalent train-to-deploy reparameterization path. On a fixed subject-disjoint split of a 40-participant public dataset, the first checkpoint reached 88.65% Top-1 and 95.73% Top-5 accuracy. The deployment graph reduced batch-one model latency by 28.2% on a single CPU thread and 35.0% on a laptop GPU, with numerical equivalence checks passing. The code, weights, configurations, and evaluation records are being released as a noncommercial research artifact. Short-window, full leave-one-subject-out, and real-device evaluation remain in progress.

## 朋友圈正文

最近把脑电交互方向往前推进了一步。

我做了一个不到 90 万参数的 40 类 SSVEP 识别模型，结合候选频率物理证据、时频特征和面向部署的结构重参数化。首版使用 40 名参与者、9,600 个公开同步试次，并把训练、验证和测试参与者完全隔离。

当前固定测试集 Top-1 为 88.65%，Top-5 为 95.73%；相较同条件固定双谐波功率基线，Top-1 提高了 9.59 个百分点。部署结构在输出数值等价的情况下，CPU 和 GPU 模型计算延迟分别降低约 28% 和 35%。

这还是阶段性结果，目前继续补短窗口、完整跨被试和真实端侧验证。代码、模型权重、配置和评估记录会在复现实验材料整理完成后，以非商用研究许可开源。

项目链接：开源时补充
