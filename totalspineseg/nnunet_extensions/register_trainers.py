"""
Register custom trainers with nnUNet

This module registers the custom trainers defined in this package with nnUNet's
trainer discovery system. Import this module before running nnUNet training
to make custom trainers available.

Usage:
    # In Python
    from totalspineseg.nnunet_extensions import register_trainers
    
    # Or from command line (before nnUNet commands)
    python -c "from totalspineseg.nnunet_extensions import register_trainers"
"""

import os
import sys

def register_with_nnunet():
    """
    Register custom trainers with nnUNet's trainer discovery system.
    
    nnUNet discovers trainers by looking for classes that inherit from nnUNetTrainer
    in modules that are importable. By importing our custom trainers here, they
    become discoverable when this module is imported before training.
    """
    # Add the parent directory to path if not already there
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import trainers to register them
    try:
        from totalspineseg.nnunet_extensions.nnUNetTrainer_LDH import nnUNetTrainer_LDH
        print(f"Registered trainer: nnUNetTrainer_LDH")
        return True
    except ImportError as e:
        print(f"Warning: Could not register custom trainers: {e}")
        return False


# Auto-register when this module is imported
_registered = register_with_nnunet()

