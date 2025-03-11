import torch
import threading
from typing import Dict

class GPUMemoryManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUMemoryManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.pools: Dict[str, Dict[int, torch.Tensor]] = {}
        self.reserved_memory = 0
        
    def allocate_buffer(self, key: str, size: int, dtype=torch.float16) -> torch.Tensor:
        """Allocate or reuse a GPU tensor from the pool"""
        if key not in self.pools:
            self.pools[key] = {}
        
        if size in self.pools[key]:
            # Return existing buffer if available
            buffer = self.pools[key][size]
            # Zero out the buffer for safety
            buffer.zero_()
            return buffer
        
        # Create new buffer
        buffer = torch.zeros(size, dtype=dtype, device='cuda')
        self.pools[key][size] = buffer
        self.reserved_memory += buffer.element_size() * buffer.nelement()
        return buffer
    
    def get_stats(self) -> Dict[str, int]:
        """Return memory statistics for monitoring"""
        return {
            "reserved_mb": self.reserved_memory / (1024 * 1024),
            "total_buffers": sum(len(sizes) for sizes in self.pools.values())
        }