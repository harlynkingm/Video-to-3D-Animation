from __future__ import annotations

import torch

def sys_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def empty_cuda_cache(selected_device: torch.device | None = None) -> None:
    if selected_device is not None and selected_device.type != "cuda":
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()