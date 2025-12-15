"""
LDH two-stage pipeline (disc-level detection -> ROI fine segmentation) with disc index prior.

This module intentionally does NOT depend on nnUNet trainers because the refactor
requires disc-level samples and multi-task targets (mask + SDM).
"""


