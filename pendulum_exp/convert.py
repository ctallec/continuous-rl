"""Define conversion facilities."""
import numpy as np
import torch
from abstract import Arrayable

def check_array(arr: Arrayable) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr)

def arr_to_th(arr: Arrayable, device) -> torch.Tensor:
    return torch.from_numpy(check_array(arr)).float().to(device)

def th_to_arr(tens: torch.Tensor) -> Arrayable:
    return tens.cpu().detach().numpy()
