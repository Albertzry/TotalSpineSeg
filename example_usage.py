"""
示例：在其他程序中显式指定 TOTALSPINESEG_DATA 的几种方式
"""

import os
import subprocess
from pathlib import Path

# ============================================
# 方法 1: 使用 --data-dir 参数（推荐）
# ============================================
def run_with_data_dir_param():
    """使用命令行参数显式指定数据目录"""
    cmd = [
        "totalspineseg",
        "--input-dir", "/path/to/input",
        "--output-dir", "/path/to/output",
        "--data-dir", "/path/to/TotalSpineSegData"  # 显式指定
    ]
    subprocess.run(cmd)


# ============================================
# 方法 2: 在 Python 中设置环境变量
# ============================================
def run_with_env_var():
    """在 Python 代码中设置环境变量"""
    # 设置环境变量（仅对当前进程有效）
    os.environ["TOTALSPINESEG_DATA"] = "/path/to/TotalSpineSegData"
    
    # 然后运行命令
    cmd = [
        "totalspineseg",
        "--input-dir", "/path/to/input",
        "--output-dir", "/path/to/output"
    ]
    subprocess.run(cmd)


# ============================================
# 方法 3: 使用 subprocess 传递环境变量（推荐用于子进程）
# ============================================
def run_with_subprocess_env():
    """在 subprocess 中传递环境变量"""
    env = os.environ.copy()
    env["TOTALSPINESEG_DATA"] = "/path/to/TotalSpineSegData"
    
    cmd = [
        "totalspineseg",
        "--input-dir", "/path/to/input",
        "--output-dir", "/path/to/output"
    ]
    subprocess.run(cmd, env=env)


# ============================================
# 方法 4: 直接调用 Python 函数（最灵活）
# ============================================
def run_direct_function_call():
    """直接调用 inference 函数，完全控制参数"""
    from totalspineseg.inference import inference
    from totalspineseg.utils.utils import ZIP_URLS
    from totalspineseg.init_inference import init_inference
    
    # 显式指定数据路径
    data_path = Path("/path/to/TotalSpineSegData")
    input_path = Path("/path/to/input")
    output_path = Path("/path/to/output")
    
    # 初始化（下载权重等）
    init_inference(
        data_path=data_path,
        dict_urls=ZIP_URLS,
        quiet=False
    )
    
    # 运行推理
    inference(
        input_path=input_path,
        output_path=output_path,
        data_path=data_path,
        default_release=list(ZIP_URLS.values())[0].split('/')[-2],
        output_iso=False,
        loc_path=None,
        suffix=[''],
        loc_suffix='',
        step1_only=False,
        keep_only=[''],
        max_workers=os.cpu_count(),
        max_workers_nnunet=1,
        device='cuda',
        quiet=False
    )


# ============================================
# 方法 5: 使用 context manager 临时设置环境变量
# ============================================
class TotalSpineSegContext:
    """上下文管理器，临时设置 TOTALSPINESEG_DATA"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.original_value = None
    
    def __enter__(self):
        self.original_value = os.environ.get("TOTALSPINESEG_DATA")
        os.environ["TOTALSPINESEG_DATA"] = self.data_path
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_value is None:
            os.environ.pop("TOTALSPINESEG_DATA", None)
        else:
            os.environ["TOTALSPINESEG_DATA"] = self.original_value


def run_with_context_manager():
    """使用上下文管理器"""
    with TotalSpineSegContext("/path/to/TotalSpineSegData"):
        cmd = [
            "totalspineseg",
            "--input-dir", "/path/to/input",
            "--output-dir", "/path/to/output"
        ]
        subprocess.run(cmd)
    # 退出上下文后，环境变量自动恢复


if __name__ == "__main__":
    # 示例：使用上下文管理器
    run_with_context_manager()
