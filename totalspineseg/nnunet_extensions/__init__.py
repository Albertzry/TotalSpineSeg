"""
nnU-Net Extensions for TotalSpineSeg

历史说明：
本目录曾用于放置 Dataset105 的 legacy nnUNet trainer（nnUNetTrainer_LDH，基于 Step1 prior 的 whole-spine 分割）。
按照当前项目重构，LDH 训练已迁移为两阶段（disc-level detection -> ROI segmentation）并使用 Step2-derived disc index prior。
因此旧 trainer 与注册逻辑已删除，避免误用。
"""
__all__ = []

