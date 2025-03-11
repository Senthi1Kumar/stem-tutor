from prometheus_client import start_http_server, Gauge, Counter, Histogram
import torch
import time
from functools import wraps

class Telemetry:
    def __init__(self):
        # Resource Metrics
        self.vram_usage = Gauge('vram_usage_gb', 'Current VRAM usage in GB', ['component'])
        self.peak_vram = Gauge('peak_vram_gb', 'Peak VRAM usage in GB')
        self.total_vram = Gauge('total_vram_gb', 'Total available VRAM in GB')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU utilization')
        
        # Performance Metrics
        self.llm_tokens = Counter('llm_tokens_total', 'Total tokens processed', ['direction'])
        self.llm_latency = Histogram('llm_latency_seconds', 'LLM processing time')
        self.stt_latency = Histogram('stt_latency_seconds', 'STT processing time')
        self.tts_latency = Histogram('tts_latency_seconds', 'TTS processing time')

        if torch.cuda.is_available():
            self.total_vram.set(round(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024, 3))
            self.base_vram = round(torch.cuda.memory_allocated()/1024/1024/1024, 3)
        else:
            self.base_vram = 0

        
    def start_server(self, port=9090):
        start_http_server(port)

    def track_vram(self, component):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_allocated = torch.cuda.memory_allocated()
                
                result = func(*args, **kwargs)
                
                delta = round((torch.cuda.memory_allocated() - start_allocated)/1024/1024/1024, 3)
                self.vram_usage.labels(component=component).set(abs(delta))
                
                with torch.cuda.device(0):
                    self.peak_vram.set(
                        round(torch.cuda.max_memory_allocated()/1024/1024/1024, 3)
                    )
                
                return result
            return wrapper
        return decorator


    def track_latency(self, metric):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                metric.observe(time.time() - start_time)
                return result
            return wrapper
        return decorator

telemetry = Telemetry()