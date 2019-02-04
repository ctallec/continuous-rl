"""Define conversion facilities."""
import numpy as np
import torch
from abstract import Arrayable, Tensorable

def check_array(arr: Arrayable) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr)

def check_tensor(tens: Tensorable, device='cpu') -> torch.Tensor:
    if isinstance(tens, torch.Tensor):
        return tens
    return torch.Tensor(tens).to(device)

def arr_to_th(arr: Arrayable, device) -> torch.Tensor:
    return torch.from_numpy(check_array(arr)).float().to(device)

def th_to_arr(tens: torch.Tensor) -> np.ndarray:
    return tens.cpu().detach().numpy()
