"""显存管理与设备工具。"""
import torch
import gc


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_vram_usage() -> dict:
    """返回 GPU 显存使用量 (MB)。仅 CUDA 有效，非 CUDA 环境返回全零。"""
    if not torch.cuda.is_available():
        return {'allocated_mb': 0, 'reserved_mb': 0, 'total_mb': 0}
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }
