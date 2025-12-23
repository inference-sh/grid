"""
Torch compatibility patch for missing torch.library.register_fake function.

This patch adds a dummy implementation of torch.library.register_fake for
compatibility with older PyTorch versions that don't have this function.

Usage:
    Import this module before any torch/torchvision imports:
    
    import torch_compatibility_patch
    import torch
    import torchvision
"""

import torch

def apply_register_fake_patch():
    """Apply the register_fake compatibility patch."""
    if not hasattr(torch.library, 'register_fake'):
        def register_fake(op_name):
            """
            Dummy implementation of register_fake for compatibility.
            
            The register_fake decorator is used for PyTorch's fake tensor system.
            This dummy implementation just returns the original function unchanged,
            which is sufficient for basic compatibility.
            """
            def decorator(func):
                return func
            return decorator
        
        torch.library.register_fake = register_fake
        print("Applied torch.library.register_fake compatibility patch")
        return True
    return False

# Apply the patch automatically when this module is imported
apply_register_fake_patch()