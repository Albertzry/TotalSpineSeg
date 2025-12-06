# TotalSpineSeg Step1/Step2/Step5 技术报告（中英文双语）

> 说明：本报告总结了 TotalSpineSeg 项目中 Step1、Step2 以及 Step5（LDH 专项优化）的设计思路、网络结构、训练配置与技术亮点，供论文撰写与答辩准备使用。
>
> Note: This report summarizes the design, network architecture, training configuration, and technical highlights of Step1, Step2, and Step5 (LDH-specific optimization) in the TotalSpineSeg project, for paper writing and defense preparation.

---

## 1. 总体框架概述 / Overall Framework

**中文**  
TotalSpineSeg 采用“两阶段分割 + 微小病灶专项优化”的整体框架：

1. **Step1（Dataset101）**：对全脊柱进行多类别 3D 语义分割，获取椎体、椎间盘、椎管、脊髓等结构，并精确识别四个关键椎间盘解剖标志点（C2–C3、C7–T1、T12–L1、L5–S）及 C1 椎体，为后续编号提供“坐标系”和锚点。
2. **Step2（Dataset102）**：在 Step1 的结果基础上，结合从 Step1 中提取的“奇数椎间盘”作为第二输入通道，对所有椎骨、椎间盘、骶骨和椎管/脊髓进行更加精细、带奇偶信息的编号与分割。
3. **Step5（Dataset105, LDH）**：在完成整体解剖标注后，针对腰椎间盘突出（Lumbar Disc Herniation, LDH）这样体积极小、类别极度不平衡的病灶，单独构建数据集和训练器，通过解剖先验注意力和多损失函数组合进行专项优化。

从整体上看，Step1 更偏向“建立坐标系和粗分割”，Step2 负责“精细编号和结构一致性”，Step5 则是在该坐标系中对极小病灶进行“放大镜式”的精细检测与分割。

**English**  
TotalSpineSeg adopts a global framework of **two-stage segmentation plus a dedicated small-lesion optimization module**:

1. **Step1 (Dataset101)**: Performs multi-class 3D semantic segmentation of the entire spine, segmenting vertebrae, intervertebral discs, spinal canal, and spinal cord, while accurately detecting four key disc landmarks (C2–C3, C7–T1, T12–L1, L5–S) and the C1 vertebra. This step builds a reliable anatomical coordinate system for subsequent labeling.
2. **Step2 (Dataset102)**: Based on Step1 outputs, Step2 uses the odd-numbered intervertebral discs extracted from Step1 as a second input channel. It then performs finer segmentation and labeling of all vertebrae, discs, sacrum, canal, and cord, explicitly encoding odd/even information.
3. **Step5 (Dataset105, LDH)**: After the global anatomical labeling is completed, Step5 specifically targets **lumbar disc herniation (LDH)**, a very small and highly imbalanced lesion. A dedicated dataset and trainer are built, combining anatomical prior attention and multiple loss terms for specialized optimization.

In summary, Step1 focuses on **building the coordinate system and coarse segmentation**, Step2 ensures **fine-grained labeling and structural consistency**, and Step5 acts as a **magnifying glass** for precise detection and segmentation of tiny LDH lesions.

---

## 2. Step1：Dataset101 – 粗分割与关键标志点识别 / Coarse Segmentation and Landmark Detection

### 2.1 任务定义与数据预处理 / Task Definition and Pre-processing

**中文**  
- **任务类型**：3D 全分辨率多类别语义分割。
- **输入**：单通道脊柱 MRI 体数据，预处理包括：
  - 重采样到 $1\times1\times1$ mm 等方分辨率；
  - 重定向到 LPI(-) 方向；
  - Z-score 强度归一化。
- **输出类别（共 9 类）**：
  0. 背景（background）  
  1. 椎间盘（disc，未指明水平的普通椎间盘）  
  2. C2–C3 椎间盘（disc_C2_C3）  
  3. C7–T1 椎间盘（disc_C7_T1）  
  4. T12–L1 椎间盘（disc_T12_L1）  
  5. L5–S 椎间盘（disc_L5_S）  
  6. 椎骨（vertebrae，一般椎体，不区分具体水平）  
  7. C1 椎骨（vertebrae_C1）  
  8. 椎管（canal）  
  9. 脊髓（cord）  

**设计目标**：
1. 在全脊柱范围内获得稳定的椎骨 / 椎间盘 / 椎管 / 脊髓的粗分割结果；
2. 准确识别 4 个关键椎间盘 landmark 与 C1 椎体，为后续迭代编号提供锚点；
3. 输出结果将作为 Step2 和 Step5 的重要先验信息来源。

**English**  
- **Task type**: 3D full-resolution multi-class semantic segmentation.
- **Input**: Single-channel spinal MRI volume, with the following pre-processing:
  - Resampled to isotropic $1\times1\times1$ mm resolution;
  - Reoriented to LPI(-) convention;
  - Z-score intensity normalization.
- **Output classes (9 in total)**:
  0. background  
  1. disc (general, level-unspecified)  
  2. disc_C2_C3  
  3. disc_C7_T1  
  4. disc_T12_L1  
  5. disc_L5_S  
  6. vertebrae (general vertebrae)  
  7. vertebrae_C1  
  8. canal  
  9. cord  

**Design goals**:
1. Obtain robust coarse segmentation of vertebrae/discs/canal/cord over the entire spine;
2. Accurately detect the four key disc landmarks and the C1 vertebra to serve as anchors for iterative labeling;
3. Provide strong anatomical priors for Step2 and Step5.

### 2.2 网络结构与训练配置 / Network Architecture and Training Configuration

**中文**  
- **基础框架**：nnU-Net v2，3D 全分辨率配置（`3d_fullres`），使用 `nnUNetPlans_small` 以适配 GPU 内存。
- **主干结构**：典型 3D U-Net：
  - 编码器（Encoder）：5 层，下采样 4 次；每层包含 Conv3D → InstanceNorm3D → LeakyReLU；
  - 瓶颈层（Bottleneck）：最高特征通道数约为 320；
  - 解码器（Decoder）：与编码器对称的上采样结构，采用转置卷积 + skip connection；
  - 输出头（Head）：`Conv3D(32 → 9)`，随后接 Softmax 或 logits。
- **损失函数**：Dice Loss + Cross Entropy 的加权组合，用于缓解类别不平衡：
  $$ L_{total} = \alpha L_{Dice} + (1-\alpha)L_{CE} $$
- **优化与训练策略**：
  - 优化器：Adam；学习率由 nnU-Net 自动设定（通常约为 $1e^{-3}$）；
  - 5 折交叉验证（fold=0–4），最终推理时支持模型集成；
  - 数据增强：旋转（±15°）、缩放（0.85–1.25）、弹性形变、亮度/对比度/Gamma 调整、高斯噪声与模糊等；
  - 重要约束：使用 `NoMirroring` 训练器，不进行左右/前后镜像翻转，以保持脊柱在头脚和左右方向上的真实解剖结构。

**English**  
- **Base framework**: nnU-Net v2, 3D full resolution (`3d_fullres`), using `nnUNetPlans_small` to adapt to GPU memory.
- **Backbone**: a standard 3D U-Net:
  - Encoder: 5 levels with 4 down-sampling steps; each level contains Conv3D → InstanceNorm3D → LeakyReLU;
  - Bottleneck: deepest layer with ~320 feature channels;
  - Decoder: symmetric up-sampling path with transposed convolutions and skip connections;
  - Output head: `Conv3D(32 → 9)` followed by Softmax or logits.
- **Loss function**: a weighted combination of Dice Loss and Cross-Entropy to address class imbalance:
  $$ L_{total} = \alpha L_{Dice} + (1-\alpha)L_{CE} $$
- **Optimization and training strategy**:
  - Optimizer: Adam, with learning rate automatically determined by nnU-Net (typically around $1e^{-3}$);
  - 5-fold cross-validation (folds 0–4) with ensemble at inference;
  - Data augmentation: rotations (±15°), scaling (0.85–1.25), elastic deformations, brightness/contrast/Gamma adjustments, Gaussian noise, and Gaussian blur;
  - Important constraint: we use a `NoMirroring` trainer and thus **no flipping augmentation** is applied, in order to preserve the anatomical directions of the spine.

### 2.3 迭代标注与奇数椎间盘提取 / Iterative Labeling and Odd-disc Extraction

**中文**  
Step1 的输出不仅是一个 9 类的分割图，更重要的是为后续提供了解剖“刻度尺”与迭代标注的基础：

1. **迭代标注算法（`iterative_label.py`）**：
   - 利用 C2–C3、C7–T1、T12–L1、L5–S 四个关键椎间盘 landmark 以及 C1 椎体，作为序列两端和中间的锚点；
   - 沿着脊柱纵向，从这些锚点向上或向下逐一推断其余椎间盘与椎骨的相对顺序和编号；
   - 该算法将 Step1 的“语义类别”转换成具有解剖顺序含义的标签，为 Step2 和最终输出打下基础。
2. **奇数椎间盘提取（`extract_alternate.py`）**：
   - 在已编号的椎间盘中提取奇数序号的椎间盘（例如 C3–C4, C5–C6, T1–T2, T3–T4 等）；
   - 这些奇数椎间盘的分割结果会被编码为一个二值或多值 mask，作为 Step2 的第二输入通道，向网络显式提供脊柱纵向尺度信息。

**English**  
The output of Step1 is more than just a 9-class segmentation map; it serves as the basis for a longitudinal anatomical scale and iterative labeling:

1. **Iterative labeling algorithm (`iterative_label.py`)**:
   - Uses the four key disc landmarks (C2–C3, C7–T1, T12–L1, L5–S) and C1 vertebra as anchors at the ends and middle of the spinal sequence;
   - Iteratively propagates labels upward and downward along the spine to infer the order and indices of all remaining discs and vertebrae;
   - This algorithm converts Step1 semantic classes into anatomically ordered labels, which are crucial for Step2 and the final outputs.
2. **Odd-disc extraction (`extract_alternate.py`)**:
   - From the labeled intervertebral discs, it extracts those with odd indices (e.g., C3–C4, C5–C6, T1–T2, T3–T4, ...);
   - The resulting masks are encoded as a binary/multi-value map and used as the second input channel of Step2, explicitly injecting longitudinal scale information into the network.

---

## 3. Step2：Dataset102 – 精细编号与多结构联合分割 / Fine-grained Labeling and Multi-structure Segmentation

### 3.1 任务与输入输出 / Task and I/O Definition

**中文**  
- **任务类型**：基于 Step1 结果的精细化 3D 多类别分割与编号。
- **输入**：
  - 通道 0：与 Step1 相同预处理的 MRI 体数据；
  - 通道 1：由 Step1 迭代标注和奇数椎间盘提取算法生成的奇数椎间盘 mask。
- **输出类别（共 11 类）**：
  0. 背景（background）  
  1. 椎间盘（disc，一般）  
  2. disc_C2_C3  
  3. disc_C7_T1  
  4. disc_T12_L1  
  5. disc_L5_S  
  6. 通用椎骨（vertebrae，一般）  
  7. 奇数椎骨（vertebrae_O，例如 C1, C3, C5, T2, T4 ...）  
  8. 偶数椎骨（vertebrae_E，例如 C2, C4, C6, T1, T3 ...）  
  9. 骶骨（sacrum）  
  10. 椎管（canal）  
  11. 脊髓（cord）  

**English**  
- **Task type**: fine-grained 3D multi-class segmentation and labeling conditioned on Step1 results.
- **Inputs**:
  - Channel 0: MRI volume with the same pre-processing as Step1;
  - Channel 1: odd-disc mask generated by the iterative labeling and odd-disc extraction in Step1.
- **Output classes (11 in total)**:
  0. background  
  1. disc (general)  
  2. disc_C2_C3  
  3. disc_C7_T1  
  4. disc_T12_L1  
  5. disc_L5_S  
  6. vertebrae (general)  
  7. vertebrae_O (odd vertebrae, e.g., C1, C3, C5, T2, T4, ...)  
  8. vertebrae_E (even vertebrae, e.g., C2, C4, C6, T1, T3, ...)  
  9. sacrum  
  10. canal  
  11. cord  

### 3.2 双通道设计与网络结构 / Dual-channel Design and Architecture

**中文**  
- Step2 的网络主体仍然基于 nnU-Net 3D U-Net 结构，与 Step1 保持高度一致，以便共享经验与配置；
- 关键结构差异在于：
  - 第一层卷积的输入通道数从 1 变为 2：`Conv3D(2 → 32, kernel=3)`；
  - 输出头通道数从 9 变为 11：`Conv3D(32 → 11)`；
  - 其余编码器/解码器层数、卷积核、归一化和激活函数基本一致。

**双通道设计动机**：
1. 奇数椎间盘 mask 提供了“纵向刻度线”，网络不必从零学习序列关系；
2. 显式注入 Step1 的结构信息，能显著减少相邻椎体和椎间盘之间的混淆；
3. 通过 vertebrae_O / vertebrae_E + landmark，可以在后处理阶段稳定地恢复 C1–L5 和骶骨的完整编号。

**English**  
- The core architecture of Step2 remains a 3D U-Net based on nnU-Net, largely identical to that of Step1 for better knowledge transfer and configuration sharing;
- The key structural differences are:
  - The first convolutional layer now takes 2-channel input: `Conv3D(2 → 32, kernel=3)`;
  - The output head predicts 11 channels instead of 9: `Conv3D(32 → 11)`;
  - All other encoder/decoder layers, kernel sizes, normalization, and activations remain the same.

**Motivation of the dual-channel design**:
1. The odd-disc mask provides explicit longitudinal scale marks so that the network does not need to learn the entire sequence relation from scratch;
2. Injecting Step1 structural information explicitly reduces confusion between adjacent vertebrae and discs;
3. Combined with vertebrae_O / vertebrae_E and key landmarks, the post-processing stage can reliably reconstruct complete labeling from C1 to L5 and the sacrum.

### 3.3 后处理与最终编号 / Post-processing and Final Labeling

**中文**  
Step2 的预测结果将经过以下后处理步骤：

1. **最大连通域提取**：对椎体、椎管等结构保留最大连通区域，去除噪声小块；
2. **迭代标注算法再次应用**：
   - 综合利用 Step2 的 vertebrae_O / vertebrae_E、sacrum 以及四个 landmark 椎间盘；
   - 在整个脊柱范围内一致地恢复 C1–C7、T1–T12、L1–L5 与骶骨的标签；
3. **椎管填充（`fill_canal.py`）**：对椎管进行拓扑修复，填补小空洞，保证其连通性和解剖合理性。

**English**  
The predictions of Step2 are further refined by the following post-processing operations:

1. **Largest connected component extraction**: keep only the largest connected component for structures like vertebrae and canal to remove noisy fragments;
2. **Iterative labeling applied again**:
   - Combine vertebrae_O / vertebrae_E, sacrum, and the four disc landmarks from Step2;
   - Reconstruct consistent labels from C1–C7, T1–T12, L1–L5, and sacrum along the whole spine;
3. **Canal filling (`fill_canal.py`)**: perform topological correction on the canal to fill small holes and ensure connectivity and anatomical plausibility.

### 3.4 Step2 — Design Goal, Contract, Edge Cases and Metrics / Step2 — 设计目标、合同、边界情况与评价指标

**中文要点**
- Design goal：在 Step1 提供的解剖先验（分割、landmark、odd-disc mask）基础上，完成对全脊柱的细粒度、一致编号的多结构分割（C1–L5 + sacrum），并在局部边界与小结构上明显优于 Step1。
- Contract（输入/输出/失败模式/成功标准）：
  - Inputs: preprocessed MRI, odd-disc mask
  - Outputs: full anatomical segmentation (11 classes) and stable vertebra indices C1→L5
  - Failure modes: missing/misaligned odd-disc mask, Step1 landmark errors, severe anatomical deformity
  - Success criteria: per-class Dice improved over Step1, high vertebra-level ID accuracy, low sequence-consistency error rate
- Edge cases & mitigations：
  - Missing odd-disc mask → fallback to landmark-only iterative labeling;  
  - Severe deformity/surgery → include augmented or pathological samples in training;  
  - Cross-protocol images → strict preprocessing + intensity augmentation
- Suggested metrics：voxel-wise Dice per class, vertebra-level ID accuracy / sequence-consistency rate, Hausdorff distance, % volumes with full-sequence correct labeling

**English bullets**
- Design goal: using Step1 anatomical priors (segmentation, landmarks, odd-disc mask), produce fine-grained, consistently indexed full-spine segmentation (C1–L5 and sacrum) with improved local boundary accuracy versus Step1.
- Contract (I/O / failure modes / success criteria):
  - Inputs: preprocessed MRI, odd-disc mask
  - Outputs: full anatomical segmentation (11 classes) and stable vertebra indices C1→L5
  - Failure modes: missing/misaligned odd-disc mask, Step1 landmark errors, severe anatomical deformity
  - Success criteria: per-class Dice outperforming Step1, high vertebra-level ID accuracy, low sequence-consistency error rate
- Edge cases & mitigations:
  - Missing odd-disc mask → fallback to landmark-based iterative labeling;  
  - Severe deformity/surgical cases → include such samples and augmentations in training;  
  - Cross-protocol variability → rigorous preprocessing and augmentation
- Suggested metrics: voxel-wise Dice per class, vertebra-level ID accuracy / sequence-consistency rate, Hausdorff distance, % volumes with correct full-sequence labeling

---

## 4. Step5：Dataset105 – LDH 微小病灶分割专项优化 / Dedicated Optimization for LDH Small-lesion Segmentation

### 4.1 背景与挑战 / Background and Challenges

**中文**  
腰椎间盘突出（Lumbar Disc Herniation, LDH）具有以下特点：

1. **体积极小**：在整幅 3D MRI 体数据中所占体积比例极低，前景/背景不平衡极其严重；
2. **位置高度受解剖约束**：LDH 只会发生在椎间盘、椎骨、椎管/脊髓交界处，不可能出现在肌肉或椎体内部任意位置；
3. **标注可能不完全**：在大规模 Dataset102 中，一些 LDH 区域可能没有被明确标注，直接将 LDH 作为 Step2 的一个新增类别会引入大量“假负样本”。

如果在 Step2 中直接加入 LDH 类别，将面临：
- 正负样本极端不平衡，损失主导被背景牵制；
- 未标注 LDH 被当作背景，造成系统性偏差；
- 很难在保证整体解剖分割的同时，把如此微小的病灶学好。

**English**  
Lumbar disc herniation (LDH) presents several unique challenges:

1. **Very small volume**: LDH occupies only a tiny fraction of the 3D MRI volume, leading to extremely severe foreground–background imbalance;
2. **Strong anatomical constraints**: LDH can only appear at the interface between disc, vertebrae, and canal/cord, and should never occur in arbitrary locations such as muscle or deep inside vertebral bodies;
3. **Potentially incomplete annotation**: In the large-scale Dataset102, some LDH regions might not be explicitly annotated, making LDH as an additional class in Step2 prone to many false negatives.

If LDH were directly added as another output class in Step2, we would face:
- Extreme class imbalance causing the loss to be dominated by background;
- Systematic bias due to unannotated LDH being treated as background;
- Difficulty in jointly optimizing for both global anatomical segmentation and very small lesions.

### 4.2 专门数据集 Dataset105 的构建 / Construction of Dedicated Dataset105

**中文**  
为了解决上述问题，我们单独构建了 **Dataset105** 用于 LDH 分割：

- **数据来源**：从增强后的 Dataset100/102 中筛选出具有 LDH 标注的子集（sub-LDH*）；
- **输入通道**：
  - 通道 0：原始或预处理后的 MRI；
  - 通道 1：来自 Step1 的全解剖预测结果，用于构造解剖先验注意力（而非直接 one-hot 编码所有结构）；
- **输出类别（2 类）**：
  - 0：背景（background）；
  - 1：LDH 病灶（来自原 label101 中对 LDH 的精细标注）。

通过这种方式：
1. 仅在“确定有 LDH 的样本”上训练，避免把未标注的 LDH 当作负样本；
2. 任务被简化为二分类小目标检测，使网络可以更专注于病灶自身形态；
3. Step1 的输出作为附加通道，为后续的解剖先验注意力提供基础。

**English**  
To address the above problems, we construct a **dedicated Dataset105** for LDH segmentation:

- **Data source**: a subset (sub-LDH*) filtered from the augmented Dataset100/102 that contains explicit LDH annotations;
- **Input channels**:
  - Channel 0: original or pre-processed MRI volumes;
  - Channel 1: full anatomical predictions from Step1, which are used to build anatomical prior attention maps (rather than simply feeding all labels as one-hot channels);
- **Output classes (binary)**:
  - 0: background;
  - 1: LDH lesion (fine annotations originating from label101).

This design:
1. Trains only on samples with confirmed LDH, preventing unannotated LDH from being incorrectly treated as negatives;
2. Reduces the task to binary small-lesion detection, allowing the network to focus more on lesion morphology;
3. Uses Step1 outputs as an auxiliary channel to support anatomical prior attention.

### 4.3 多层解剖注意力机制 / Multi-level Anatomical Attention Mechanism

**中文**  
Step1 的预测标签（disc、vertebrae、canal、cord 等）蕴含了丰富的解剖先验信息。我们在 Step5 中显式构造了三层解剖注意力图，将“LDH 可能出现的区域”与“解剖上不合理的区域”区分开来。

1. **椎间盘边界注意力（Disc Boundary Attention，权重 0.3）**：
   - 利用椎间盘类（1–5）进行形态学膨胀和腐蚀处理，取二者差集构造近似的椎间盘边界；
   - 物理意义：LDH 通常起源于椎间盘纤维环破裂，因此最早发生变化的是椎间盘边界区域。

2. **椎间盘–椎管/脊髓接触注意力（Disc–Cord/Canal Interface Attention，权重 0.5）**：
   - 对椎间盘与椎管（标签 8）/脊髓（标签 9）分别膨胀后求交集，得到“突出的椎间盘与神经结构接触区域”；
   - 物理意义：临床上最关心的是“椎间盘是否压迫到脊髓或神经根”，这一接触区域往往与 LDH 的功能影响直接相关，因此赋予最高权重 0.5。

3. **椎体间隙注意力（Inter-vertebral Attention，权重 0.2）**：
   - 对椎骨标签（6–7）进行膨胀，并减去原椎体区域，近似获得“椎体之间的间隙”；
   - 物理意义：LDH 只可能发生在相邻椎体之间的椎间盘及其周边软组织，而不可能出现在椎体内部或远离椎柱的区域。

**综合注意力图**：

```python
attention = 0.5 * disc_cord_interface \
          + 0.3 * disc_boundary \
          + 0.2 * inter_vertebral
```

在训练过程中，该注意力图将：
- 用于放大注意力高区域（解剖上高度可疑区域）内的损失贡献；
- 并对注意力低区域中错误预测的 LDH 给出更强惩罚，从而引导网络“遵守解剖学常识”。

**English**  
The predictions from Step1 (disc, vertebrae, canal, cord, etc.) encode rich anatomical priors. In Step5, we explicitly construct three anatomical attention maps to distinguish between regions where LDH is anatomically plausible and those where it is not.

1. **Disc boundary attention (weight 0.3)**:
   - Perform morphological dilation and erosion on disc labels (1–5), and take their difference to approximate disc boundaries;
   - Intuition: LDH originates from annulus fibrosus rupture, and the earliest changes occur around the disc boundary.

2. **Disc–cord/canal interface attention (weight 0.5, most important)**:
   - Dilate disc labels and canal (label 8) / cord (label 9) respectively, then take their intersection to form the contact region between bulging discs and neural structures;
   - Intuition: Clinically, the key question is whether the disc compresses the spinal cord or nerve roots. Therefore this region is given the highest weight (0.5).

3. **Inter-vertebral attention (weight 0.2)**:
   - Dilate vertebrae labels (6–7) and subtract the original vertebral regions, approximating the inter-vertebral spaces;
   - Intuition: LDH can only occur in the inter-vertebral disc region between adjacent vertebrae, not inside vertebral bodies or far from the spine.

**Combined attention map**:

```python
attention = 0.5 * disc_cord_interface \
          + 0.3 * disc_boundary \
          + 0.2 * inter_vertebral
```

During training, this attention map is used to:
- Emphasize loss contributions in high-attention regions (anatomically suspicious zones);
- Apply stronger penalties to LDH predictions in low-attention regions, enforcing anatomical plausibility.

### 4.4 多损失函数组合设计 / Combination of Multiple Loss Functions

**中文**  
在 Step5 中，我们将解剖注意力与多种适合小目标分割的损失函数结合，形成如下总损失：

$$
L_{total} = L_{AttDice} + L_{CE} + 0.5 L_{FocalTversky} + 0.5 L_{Boundary} + 2.0 L_{AnatomicalPenalty}
$$

其中：

1. **Attention-weighted Dice Loss（解剖注意力加权 Dice）**
   - 在注意力高的区域（可能存在 LDH）中赋予更大权重计算 Dice；
   - 目的：专注于小体积病灶周围的分割质量，减轻大面积背景对梯度的稀释。

2. **Weighted Cross-Entropy（加权交叉熵损失）**
  - 对正类（LDH）赋予远大于背景的损失权重（例如 `pos_weight=10`）；
  - 目的：缓解前景样本稀少导致的梯度不足问题，避免网络倾向于全部预测为背景。

3. **Focal Tversky Loss（焦点 Tversky 损失）**
  - 基于 Tversky 指数：

$$
TI = \frac{TP}{TP + \alpha FP + \beta FN}
$$

  - Focal Tversky 损失：

$$
L_{FocalTversky} = (1 - TI)^{\gamma}
$$

  - 在我们的设置中，$\alpha=0.3, \beta=0.7, \gamma=0.75$，更偏向惩罚漏检（FN），从而提高 LDH 的召回率，并通过 $\gamma>0$ 将优化重点放在“难样本”上。

4. **Boundary Loss（边界损失）**
  - 基于 GT 掩膜的 signed distance map $dist\_map$：

$$
L_{Boundary} = \operatorname{mean}(\hat{y} \cdot dist\_map)
$$

  - 其中 $\hat{y}$ 为预测概率。该损失鼓励预测轮廓贴近真实边界，特别适用于窄带结构和细长病灶。

5. **Anatomical Penalty（解剖一致性惩罚项）**
  - 当模型在解剖注意力很低的区域预测 LDH 时（例如椎体内部、远离椎柱的软组织中），该项给出额外惩罚；
  - 从优化角度看，该项相当于对“解剖不合理的前景预测”添加加权 L2 / CE 罚项，进一步约束模型输出空间。

**English**  
In Step5, we combine anatomical attention with several loss functions tailored for small lesion segmentation, leading to the following total loss:

$$
L_{total} = L_{AttDice} + L_{CE} + 0.5 L_{FocalTversky} + 0.5 L_{Boundary} + 2.0 L_{AnatomicalPenalty}
$$

where:

1. **Attention-weighted Dice Loss**
  - Computes Dice loss with higher weights in high-attention regions where LDH is likely to appear;
  - Goal: focus on segmentation quality around small lesions and reduce the dilution of gradients by large background regions.

2. **Weighted Cross-Entropy**
  - Assigns a much larger weight (e.g., `pos_weight=10`) to the positive class (LDH) than to the background;
  - Goal: alleviate the shortage of positive gradients caused by rare foreground voxels and prevent the trivial all-background solution.

3. **Focal Tversky Loss**
  - Based on the Tversky index:

$$
TI = \frac{TP}{TP + \alpha FP + \beta FN}
$$

  - and the Focal Tversky loss:

$$
L_{FocalTversky} = (1 - TI)^{\gamma}
$$

  - With $\alpha=0.3, \beta=0.7, \gamma=0.75$, the loss penalizes false negatives more heavily, improving LDH recall and emphasizing hard examples.

4. **Boundary Loss**
  - Uses the signed distance map $dist\_map$ of the ground-truth mask:

$$
L_{Boundary} = \operatorname{mean}(\hat{y} \cdot dist\_map)
$$

  - This encourages the predicted contour $\hat{y}$ to align with the true boundary, which is particularly useful for thin and elongated lesions.

5. **Anatomical Penalty**
  - Adds extra penalties when LDH is predicted in low-attention regions (e.g., inside vertebral bodies or far from the spine);
  - From an optimization perspective, it acts as a weighted penalty term on anatomically implausible foreground predictions, further constraining the output space。

### 4.5 Step5 — Task Type, Design Goal, Contract, Edge Cases and Metrics / Step5 — 任务类型、设计目标、合同、边界情况与评价指标

**中文要点**
- Task type：binary small-lesion segmentation / detection（0=background, 1=LDH），专注于极小体积且受解剖约束的病灶；
- Design goal：在高召回（sensitivity）前提下实现临床可用的 LDH 检出与精确分割，同时通过解剖先验严格控制假阳性，确保输出边界精确且解剖一致；
- Contract（输入/输出/失败模式/成功标准）：
  - Inputs: preprocessed MRI, Step1 anatomical predictions (segmentation + landmarks)
  - Outputs: binary LDH mask (same resolution), optional lesion centroids/volumes
  - Failure modes: incomplete training labels (unlabeled LDH), lesions below voxel-level detectability, false positives in anatomically impossible areas
  - Success criteria: high lesion-wise recall, acceptable precision / low FP per volume, voxel-level Dice & boundary accuracy, anatomical plausibility (predictions concentrated in attention regions)
- Edge cases & mitigations：
  - Unlabeled LDH in training data → train only on LDH-positive subset (Dataset105) + consider weak supervision or manual relabeling;  
  - Multiple adjacent small lesions → ensure multi-lesion samples and connected-component based separation in post-processing;  
  - Very tiny lesions (< a few voxels) → consider patch zoom-in / super-resolution or increase sampling resolution when available
- Suggested metrics：lesion-level sensitivity/recall, precision, F1, false positives per volume, voxel-wise Dice, 95% HD or ASD, clinical-impact metrics

**English bullets**
- Task type: binary small-lesion segmentation / detection (0=background, 1=LDH), specialized for tiny, anatomically constrained lesions;
- Design goal: achieve clinically useful LDH detection and accurate segmentation under a high-recall constraint, while strictly controlling anatomically implausible false positives via anatomical priors;
- Contract (I/O / failure modes / success criteria):
  - Inputs: preprocessed MRI, Step1 anatomical predictions (segmentation + landmarks)
  - Outputs: binary LDH mask, optional lesion-level centroids/volumes
  - Failure modes: incomplete training labeling, lesions below voxel detectability, anatomically impossible false positives
  - Success criteria: high lesion-wise recall, acceptable precision / low FP per volume, voxel-level Dice & boundary accuracy, anatomical plausibility
- Edge cases & mitigations:
  - Unlabeled LDH → train on LDH-positive subset (Dataset105) and consider weak supervision / relabeling;  
  - Multiple adjacent lesions → include such cases in training and separate via connected-component analysis;  
  - Tiny lesions → use patch zoom-in or super-resolution approaches, or higher-resolution sampling if available
- Suggested metrics: lesion-level sensitivity/recall, precision, F1, FP/vol, voxel-wise Dice, 95% HD / ASD, clinical impact metrics

---

## 5. 当前进展与阶段成果提纲 / Outline of Current Progress and Outcome

> 本节以“Current progress and outcome”为主题，给出答辩时约 4 页 PPT 的推荐提纲，可直接用于展示当前工作进展与阶段性成果。
>
> This section provides a suggested 4-page PPT outline for the "Current progress and outcome" part of the defense, summarizing the present progress and intermediate achievements.

### 第 1 页：总体框架与研究定位 / Page 1: Overall Framework and Positioning

**中文要点**  
- 简要回顾研究背景：全脊柱自动分割与标注在临床与科研中的需求；
- 给出 TotalSpineSeg 的整体 pipeline 图：MRI → Step1 → 迭代标注与奇数椎间盘 → Step2 → 完整解剖标注 → Step5（LDH 微小病灶）；
- 强调本工作的核心思想：将“全局解剖建模”和“局部病灶检测”解耦，通过分阶段 + 解剖先验方式提升鲁棒性和可解释性。

**English bullets**  
- Briefly review the motivation: the need for fully automatic spine segmentation and labeling in clinical practice and research;
- Present the overall TotalSpineSeg pipeline: MRI → Step1 → iterative labeling & odd-disc extraction → Step2 → full anatomical labeling → Step5 (LDH small-lesion module);
- Highlight the core idea: decoupling global anatomical modeling from local lesion detection via staged design and anatomical priors, improving robustness and interpretability.

### 第 2 页：Step1 & Step2 的技术实现与效果 / Page 2: Implementation and Results of Step1 & Step2

**中文要点**  
- 概述 Step1 的任务（9 类分割 + landmark 识别）和 Step2 的任务（11 类精细分割 + 奇偶椎体 + 骶骨）；
- 展示 3D nnU-Net 的网络结构示意图，说明 Step1/2 共用主干，仅在输入/输出通道上有差异；
- 用一张表或若干截图展示典型切片上的分割结果，对比 Step1 粗分割与 Step2 精细编号的差异；
- 简要给出 Dice 等指标或 Qualitative 评价，说明在全脊柱分割和编号上的准确性与鲁棒性。

**English bullets**  
- Summarize the tasks of Step1 (9-class segmentation + landmark detection) and Step2 (11-class refined segmentation with odd/even vertebrae and sacrum);
- Show a schematic of the 3D nnU-Net architecture, emphasizing that Step1 and Step2 share the same backbone with only input/output channels differing;
- Present example slices to compare coarse Step1 outputs and fine-grained Step2 labels;
- Provide quantitative metrics (e.g., Dice scores) or qualitative assessments to demonstrate accuracy and robustness in full-spine segmentation and labeling.

### 第 3 页：Step5 LDH 专项优化方法 / Page 3: Step5 LDH-specific Optimization Method

**中文要点**  
- 说明 LDH 的临床重要性及其“小体积 + 极度不平衡 + 解剖约束强”的特点；
- 重点阐述 Dataset105 的构建思路：仅在有 LDH 标注的子集上进行二分类训练，输入为 MRI + Step1 解剖预测；
- 用示意图解释三层解剖注意力（椎间盘边界、盘–椎管/脊髓接触、椎体间隙）以及它们在空间上的关系；
- 列出多损失函数组合及其设计动机，突出 Focal Tversky、Boundary Loss 和 Anatomical Penalty 在提升小病灶召回率与边界精度方面的作用。

**English bullets**  
- Explain the clinical importance of LDH and its characteristics: very small volume, extreme class imbalance, and strong anatomical constraints;
- Describe the construction of Dataset105: binary training only on LDH-positive cases, with MRI and Step1 anatomical predictions as inputs;
- Use diagrams to illustrate the three-level anatomical attention (disc boundary, disc–cord/canal interface, inter-vertebral space) and their spatial relationships;
- List the combined loss functions and their motivations, highlighting how Focal Tversky, Boundary Loss, and Anatomical Penalty improve small-lesion recall and boundary accuracy.

### 第 4 页：当前实验结果与后续计划 / Page 4: Current Experimental Results and Future Work

**中文要点**  
- 汇总当前在公共数据集或内部数据上的定量结果（全脊柱分割的 Dice、LDH 检出率/假阳性率等）；
- 展示若干 LDH 典型病例的可视化结果，对比是否开启 Step5 模块的差异；
- 总结当前阶段的结论：证明 Step1+Step2 能提供稳定的全脊柱解剖标注，Step5 明显提升了微小 LDH 病灶的检测与分割效果；
- 简要规划后续工作，如：更多中心/扫描协议的泛化测试、与临床评分或预后指标的关联分析等。

**English bullets**  
- Summarize current quantitative results on public or internal datasets (e.g., Dice for full-spine segmentation, LDH detection rate and false positive rate);
- Show visual examples of LDH cases, comparing results with and without the Step5 module;
- Conclude the current stage: Step1+Step2 provide robust full-spine anatomical labeling, while Step5 significantly improves LDH detection and segmentation;
- Outline future work, such as generalization to more centers/scanning protocols and correlation with clinical scores or prognosis.

---

*本双语技术报告可直接用作论文方法部分与答辩 PPT 的基础文本，后续如需根据具体篇幅或学校模板做进一步精简与排版，可在此基础上裁剪与调整。*

*This bilingual technical report can serve as the basis for the Methods section of a manuscript and for defense slides. It can be further condensed or reformatted according to specific length limits or institutional templates.*
