from functools import cached_property
from typing import cast

import GPUtil
import psutil
import torch
from pydantic import BaseModel, Field, computed_field


class ResourceManager(BaseModel):
    memory_usage: float = Field(
        default=0.0, title="Memory Usage", description="Memory usage in GB"
    )
    cpu_usage: float = Field(
        default=0.0, title="CPU Usage", description="CPU usage in %"
    )
    gpu_usage: float = Field(
        default=0.0, title="GPU Usage", description="GPU usage in %"
    )
    network_usage: float = Field(
        default=0.0, title="Network Usage", description="Network usage in MB/s"
    )

    def __call__(self):
        self.update()
        return self

    def update(self):
        self.memory_usage = psutil.virtual_memory().used / self.memory
        self.cpu_usage = psutil.cpu_percent()
        self.network_usage = psutil.net_io_counters().bytes_sent / 1024 / 1024

    @computed_field(return_type=str)
    @cached_property
    @torch.no_grad()  # type: ignore
    def device(self) -> str:
        return str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @computed_field(return_type=float)
    def cpu_count(self) -> float:
        return psutil.cpu_count()

    @computed_field(return_type=float)
    @property
    def memory(self) -> float:
        return cast(float, psutil.virtual_memory().total / 1024 / 1024 / 1024)  # type: ignore

    @computed_field(return_type=float)
    @property
    def get_gpu_usage(self) -> float:
        try:
            return GPUtil.getGPUs()[0].load * 100  # type: ignore
        except Exception:
            return 0.0  # type: ignore
