#!/bin/bash

# 修复损坏的文件脚本
# 这个脚本会：
# 1. 检查并安装 git-annex
# 2. 下载 git-annex 指针文件的实际内容
# 3. 删除或重新下载损坏的文件

set -e

TOTALSPINESEG_DATA="${TOTALSPINESEG_DATA:-data}"
bids="$TOTALSPINESEG_DATA/bids"

echo "=== 修复损坏的文件 ==="
echo ""

# 检查 git-annex 是否安装
if ! command -v git-annex >/dev/null 2>&1 && ! git annex version >/dev/null 2>&1; then
    echo "错误: git-annex 未安装"
    echo "请先安装 git-annex:"
    echo "  conda install -c conda-forge git-annex -y"
    echo "  或: apt-get install git-annex -y"
    exit 1
fi

echo "✓ git-annex 已安装"
echo ""

# 处理 data-multi-subject
if [ -d "$bids/data-multi-subject" ]; then
    echo "处理 data-multi-subject..."
    cd "$bids/data-multi-subject"
    
    # 检查是否是 git-annex 仓库
    if [ -d ".git/annex" ]; then
        echo "  下载 git-annex 文件..."
        git annex get --all || git annex get || echo "  警告: 部分文件下载失败"
        
        # 删除损坏的文件（在 derivatives/labels_iso 中）
        echo "  检查并删除损坏的文件..."
        corrupted_files=(
            "derivatives/labels_iso/sub-mniPilot1/anat/sub-mniPilot1_T1w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-cmrrb05/anat/sub-cmrrb05_T1w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-ucdavis02/anat/sub-ucdavis02_T1w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-cardiff04/anat/sub-cardiff04_T1w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-unf01/anat/sub-unf01_T1w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-beijingGE04/anat/sub-beijingGE04_T1w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-mniS08/anat/sub-mniS08_flip-2_mt-off_MTS_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-vallHebron01/anat/sub-vallHebron01_flip-1_mt-off_MTS_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-tehranS06/anat/sub-tehranS06_T2w_space-resampled_label-spine_dseg.nii.gz"
            "derivatives/labels_iso/sub-mgh01/anat/sub-mgh01_T2w_space-resampled_label-spine_dseg.nii.gz"
        )
        
        for file in "${corrupted_files[@]}"; do
            if [ -f "$file" ]; then
                echo "    删除损坏的文件: $file"
                rm -f "$file"
                # 尝试重新下载
                git annex get "$file" 2>/dev/null || echo "    无法重新下载: $file"
            fi
        done
    else
        echo "  警告: 不是 git-annex 仓库"
    fi
    cd - > /dev/null
fi

# 处理 data-single-subject
if [ -d "$bids/data-single-subject" ]; then
    echo "处理 data-single-subject..."
    cd "$bids/data-single-subject"
    
    if [ -d ".git/annex" ]; then
        echo "  下载 git-annex 文件..."
        git annex get --all || git annex get || echo "  警告: 部分文件下载失败"
    else
        echo "  警告: 不是 git-annex 仓库"
    fi
    cd - > /dev/null
fi

echo ""
echo "=== 完成 ==="
echo "请运行检查脚本验证文件是否已修复:"
echo "  python3 check_files.py"


