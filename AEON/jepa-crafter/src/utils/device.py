"""Device selection and hardware utilities."""
import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device.
    
    Accepts:
        'auto': picks 'cuda' if available, else 'mps' if available on mac, else 'cpu'.
        'cuda': requires CUDA.
        'mps': requires Apple Silicon MPS.
        'cpu': forced CPU.
    """
    device_str = device_str.lower().strip()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    elif device_str == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        return torch.device(device_str)


def get_device_info(device: torch.device) -> dict:
    """Return hardware information for logging and manifest."""
    info = {
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_capability"] = torch.cuda.get_device_capability(device)
        info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return info
