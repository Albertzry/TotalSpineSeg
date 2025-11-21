# TotalSpineSeg 项目修改说明 - 腰椎间盘突出 (LDH) 模块集成

本文档详细记录了为集成腰椎间盘突出 (LDH) 分割功能而对 `TotalSpineSeg` 项目进行的修改。

## 1. 数据准备与转换

### 1.1 原始数据处理 (`scripts/convert_data_ori.py`)
*   **目的**: 将非标准格式的原始数据集 (`data_ori`) 转换为符合 BIDS 标准的格式 (`data-ldh`)，以便纳入后续的统一处理流程。
*   **改动内容**:
    *   创建了 python 脚本 `scripts/convert_data_ori.py`。
    *   **重命名规则**: 删除了文件名中的中文字符，统一采用 `sub-编号_日期` 格式 (例如 `sub-100_20130610`)。
    *   **标签映射**: 将原始数据中的标签 `1` (LDH) 临时映射为 `101`，以避免与 `TotalSpineSeg` 现有的标签系统（如椎间盘标签）发生冲突。
    *   **后缀标准化**: 为标签文件添加 `_T2w` 标识，并统一存放至 `derivatives/labels_iso` 目录。

### 1.2 数据集生成流程 (`scripts/prepare_datasets.sh`)
*   **目的**: 在生成 nnUNet 训练数据时，自动识别并融合新的 LDH 数据集。
*   **改动内容**:
    *   **忽略原始目录**: 修改遍历逻辑，自动跳过非标准的 `data_ori` 目录。
    *   **LDH 数据集集成**: 增加了对 `LDH` 数据集 (`data-ldh`) 的识别逻辑。
        *   对于 LDH 数据，直接进行文件复制，跳过了针对普通脊柱数据的复杂标签映射（如椎管、脊髓合并等），因为该数据集仅包含单一的 LDH 标签。
    *   **数据增强**: 在 `totalspineseg_augment` 命令的 `--seg-classes` 参数中**追加了 `101`**。
        *   **关键点**: 确保数据增强脚本不会忽略 LDH 标签，保证了该结构也能获得旋转、缩放等增强效果。

## 2. 配置文件更新

### 2.1 标签映射 (`resources/labels_maps/`)
*   **目的**: 定义训练时的标签 ID。
*   **修改文件**: `nnunet_step1.json`, `nnunet_step2.json`
*   **改动内容**:
    *   保留了原有标签的 ID 映射（尽量减少变动）。
    *   新增了 `"101": 10` (Step 1) 和 `"101": 12` (Step 2) 的映射。
    *   **结果**: 在训练中，LDH 结构被分配为最后一个类别的 ID。

### 2.2 数据集定义 (`resources/datasets/`)
*   **目的**: 更新 nnUNet 的数据集配置，注册新类别。
*   **修改文件**: `dataset_step1.json`, `dataset_step2.json`
*   **改动内容**:
    *   **Labels**: 在 `labels` 字典中新增了 `"LDH": 10` (Step 1) 和 `"LDH": 12` (Step 2)。
    *   **Regions Order**: 更新了 `regions_class_order` 列表。
        *   **Step 1**: `[1, 2, ..., 9, 10]`
        *   **Step 2**: `[1, 2, ..., 11, 12]`
    *   **说明**: 这决定了训练日志中 Dice 指标的打印顺序。目前的配置使得 **LDH 的 Dice 指标显示在打印数组的最后一位**。

## 3. 训练与推理

*   **训练**: 使用修改后的配置运行 `scripts/train.sh` 时，模型将学习新的 LDH 类别。
    *   验证集 Dice 输出：`[Disc, ..., LDH]` (LDH 为最后一项)。
*   **推理**: 推理流程保持不变，输出结果中将包含新的标签 ID (10 或 12)，代表分割出的腰椎间盘突出区域。

## 4. 执行步骤总结

1.  **数据转换**:
    ```bash
    python scripts/convert_data_ori.py
    ```
2.  **生成数据集**:
    ```bash
    bash scripts/prepare_datasets.sh
    ```
3.  **开始训练**:
    ```bash
    bash scripts/train.sh 101  # 或 102
    ```

