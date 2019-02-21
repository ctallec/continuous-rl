"""Define conversion facilities."""
import numpy as np
import torch
from abstract import Arrayable, Tensorable


def check_array(arr: Arrayable) -> np.ndarray:
    """Arrayable to numpy array."""
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr)


def check_tensor(tens: Tensorable, device='cpu') -> torch.Tensor:
    """Tensorable to torch tensor."""
    if isinstance(tens, torch.Tensor):
        return tens
    return torch.Tensor(tens).to(device)


def arr_to_th(arr: Arrayable, device) -> torch.Tensor:
    """Arrayable to tensor."""
    return torch.from_numpy(check_array(arr)).float().to(device)


def th_to_arr(tens: torch.Tensor) -> np.ndarray:
    """Tensorable to numpy array."""
    return check_tensor(tens).cpu().detach().numpy()
