## Action-Conditioned OpenSora

### 项目概述

本项目基于 [Open-Sora](https://github.com/hpcaitech/Open-Sora) 框架，并结合 [WMPO](https://github.com/WM-PO/WMPO) 中的相关实现进行二次开发，旨在将原本面向通用视频生成的模型，扩展为一个 **action-conditioned world model（动作条件世界模型）**。

不同于原始 Open-Sora 主要依赖文本或视觉条件进行视频生成，本项目的模型**显式建模环境动力学（environment dynamics）**，以更好地服务于决策、规划与控制等任务场景。模型学习如下映射关系：

- **输入**：历史观测（observations）与动作序列（actions）  
- **输出**：对应的未来观测（future observations）

在该设定下，模型可被视为一个**视觉世界模型（visual world model）**，能够在给定当前观测及未来动作序列的情况下，预测环境在时间维度上的演化过程及其未来视觉观测轨迹。

当前实现中，本项目作为一个**独立的世界模型组件**，用于支持 [RLinf](https://github.com/Rlinf/RLinf) 中对世界模型的集成与调用。

---

### 版权与致谢（Acknowledgement & License Notice）

本项目基于 **Open-Sora** 开源项目进行修改与扩展，模型整体架构与核心实现来源于 Open-Sora 社区。在此对 Open-Sora 的作者及所有贡献者表示诚挚感谢。

此外，本项目中的部分修改代码来源于 **WMPO** 项目。我们在 WMPO 的实现基础上，将其中依赖的 Open-Sora 相关代码进行了**单独抽离**，形成了一个可独立使用的 **Open-Sora-based module**，并在此基础上进一步扩展为 **action-conditioned world model**。

本项目仅对原有代码进行了工程结构层面的调整与功能扩展，**不改变原始项目的版权归属与开源协议**。  
Open-Sora 与 WMPO 的原始版权分别归其作者所有，具体请参见对应项目中的 LICENSE 文件。

---
