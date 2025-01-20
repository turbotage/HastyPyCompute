from typing import TypeAlias
import threading
import numpy as np

import sys
print(sys._is_gil_enabled())
import torch
print(sys._is_gil_enabled())
import finufft
print(sys._is_gil_enabled())
import cufinufft
print(sys._is_gil_enabled())

_torch_to_numpy_dtype = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_
}

_numpy_to_torch_dtype = {
    np.float32, torch.float32,
    np.float64, torch.float64,
    np.int32,   torch.int32,
    np.int64,   torch.int64,
    np.uint8,   torch.uint8,
    np.bool_,   torch.bool
}

def dtype_conv(dtype):
    if isinstance(dtype, torch.dtype):
        return _torch_to_numpy_dtype[dtype]
    elif isinstance(dtype, np.dtype):
        return _numpy_to_torch_dtype[dtype]
    else:
        raise ValueError("Invalid dtype")

class CTensor:
    lock: threading.RLock
    tensordict: dict[torch.device, torch.Tensor]
    cached: bool

    def __init__(self, tensor: torch.Tensor, name: str):
        self.name = name
        self.lock = threading.RLock()
        self.tensordict = {}
        self.cached = False

        self.shape = tensor.shape
        self.dtype = tensor.dtype

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self.tensordict[tensor.device] = tensor

    def get_tensor(self, device: torch.device) -> torch.Tensor:
        # We are probably holding the lock unnecessarily long here
        with self.lock:
            if device in self.tensordict:
                return self.tensordict[device]
            
            # There are no devices which we can transfer from, fallback to cache
            if len(self.tensordict) == 0:
                # 3. From Cache
                return_tensor = self.unload_cached_tensor().to(device)
                self.tensordict[device] = return_tensor
                return return_tensor

            # We try to get the tensor in the following order:
            # 1. From Host memory
            if torch.device('cpu') in self.tensordict:
                return_tensor = self.tensordict['cpu'].to(device)
                self.tensordict[device] = return_tensor
                return return_tensor

            # 2. From Device memory
            return_tensor = self.tensordict[self.tensordict.keys()[0]].to(device)
            self.tensordict[device] = return_tensor
            return return_tensor

                
    def free_tensor(self, device: torch.device):
        # We are probably holding the lock unnecessarily long here
        with self.lock:
            # If no device tensor for specified device, return
            if device not in self.tensordict:
                return

            # If this device is the only memory backed version and the
            # tensor isn't cached, we must cache it.
            if (len(self.tensordict) == 1) and (not self.cached):
                self.cache_tensor(device_hint=device)

            # Finally we can free the device tensor
            del self.tensordict[device]

    def unload_cached_tensor(self) -> torch.Tensor:
        # We are probably holding the lock unnecessarily long here
        return torch.tensor(np.fromfile(
            self.name + '.rtbf', dtype=dtype_conv(self.dtype))).view(self.shape)

    def cache_tensor(self, device_hint: torch.device | None = None):
        if not self.cached:
            if device_hint is None:
                if 'cpu' in self.tensordict:
                    device_hint = 'cpu'
                else:
                    device_hint = self.tensordict[self.tensordict.keys()[0]].device

            self.tensordict[device_hint].numpy().tofile(self.name + '.rtbf')


class CTensorCache:
    def __init__(self):
        self.lock = threading.RLock()
        self.cache: dict[str, CTensor] = {}

    def get(self, name: str) -> CTensor:
        with self.lock:
            self.cache[name]
    
    def add(self, tensor: CTensor):
        with self.lock:
            self.cache[tensor.name] = tensor

    def add(self, name: str, tensor: torch.Tensor):
        with self.lock:
            self.cache[name] = CTensor(tensor, name)

    def free_name(self, name: str, device: torch.device | None = None):
        with self.lock:
            if device is None:
                del self.cache[name]
            else:
                self.cache[name].free_tensor(device)

    def replace_name(self, name: str, tensor: torch.Tensor):
        with self.lock:
            self.cache[name] = CTensor(name, tensor)

    def free(self, device: torch.device | None = None):
        with self.lock:
            if device is None:
                self.cache.clear()
            else:
                for cten in self.cache.items():
                    cten.free_tensor(device)



