"""Private stock-vLLM runtime owner used by ModelFabric acceptance."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, signal, socket, subprocess, time, urllib.request, json

@dataclass(frozen=True)
class NativeRuntimeReceipt:
    model: str
    port: int
    pid: int
    status: str

class NativeRuntimeOwner:
    def __init__(self, *, vllm_bin: Path, model_dir: Path, include_dir: Path, port: int, log: Path):
        self.vllm_bin=Path(vllm_bin); self.model_dir=Path(model_dir); self.include_dir=Path(include_dir)
        self.port=int(port); self.log=Path(log); self.proc: subprocess.Popen[str] | None=None
    def start(self) -> NativeRuntimeReceipt:
        if self.proc and self.proc.poll() is None: raise RuntimeError("NATIVE_ALREADY_RUNNING")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        inner=(f"mount --bind {self.include_dir} /usr/include; "
               "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nv_dispi.inf_amd64_e980fd2c7c4fce8; "
               f"exec {self.vllm_bin} serve {self.model_dir} --host 127.0.0.1 --port {self.port} "
               "--served-model-name native-fixture --dtype float16 --max-model-len 128 "
               "--max-num-seqs 1 --max-num-batched-tokens 128 --gpu-memory-utilization 0.50 --enforce-eager")
        with self.log.open("a", encoding="utf-8") as out:
            self.proc=subprocess.Popen(["unshare","--user","--map-root-user","--mount","--propagation","private","bash","-c",inner], stdout=out, stderr=subprocess.STDOUT, text=True)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2) as r:
                    if r.status==200: return NativeRuntimeReceipt(str(self.model_dir),self.port,self.proc.pid,"READY")
            except Exception: pass
            if self.proc.poll() is not None: raise RuntimeError("NATIVE_START_FAILED")
            time.sleep(1)
        raise TimeoutError("NATIVE_READY_TIMEOUT")
    def stop(self) -> None:
        if not self.proc: return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try: self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait(timeout=5)
        self.proc=None
    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()
