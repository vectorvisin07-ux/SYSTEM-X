"""Toolkit-only CUDA 13.3 policy and future source installation."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .command import Runner, require_success
from .errors import BootstrapError, ErrorCode


def forbidden_package(name: str, patterns: Sequence[str]) -> bool:
    base = name.split(":", 1)[0]
    return any(fnmatch.fnmatchcase(base, pattern) for pattern in patterns)


def assert_toolkit_only_packages(names: Sequence[str], cuda_lock: Mapping[str, Any]) -> None:
    patterns = cuda_lock["forbidden_package_patterns"]
    rejected = sorted({name for name in names if forbidden_package(name, patterns)})
    if rejected:
        raise BootstrapError(
            ErrorCode.HOST_UNSUPPORTED,
            "Linux NVIDIA display-driver or CUDA meta-package is prohibited in WSL",
            context={"packages": rejected},
        )
    toolkit = cuda_lock["toolkit_package"]["name"]
    if names and toolkit not in names and not all(name.startswith("cuda-") and "-13-3" in name for name in names):
        raise BootstrapError(ErrorCode.HOST_UNSUPPORTED, "CUDA plan is not constrained to the 13.3 toolkit family")


def download_pinned_keyring(
    cuda_lock: Mapping[str, Any],
    destination: Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    record = cuda_lock["keyring_package"]
    request = urllib.request.Request(record["url"], headers={"User-Agent": "system-x-bootstrap/1"})
    context = ssl.create_default_context()
    with opener(request, context=context, timeout=30) as response:
        if getattr(response, "status", 200) != 200:
            raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "official CUDA keyring download failed")
        payload = response.read(record["bytes"] + 1)
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "official CUDA keyring hash or size mismatch")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def configure_official_cuda_source(cuda_lock: Mapping[str, Any], runner: Runner) -> None:
    """Install only the pinned public keyring package; apt verifies InRelease."""

    with tempfile.TemporaryDirectory(prefix="system-x-cuda-keyring-") as temporary:
        package_path = Path(temporary) / "cuda-keyring.deb"
        download_pinned_keyring(cuda_lock, package_path)
        require_success(
            runner(("dpkg", "--install", str(package_path)), timeout=120),
            purpose="pinned CUDA repository keyring installation failed",
        )
    keyring_path = Path("/usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg")
    if not keyring_path.is_file():
        raise BootstrapError(ErrorCode.INTEGRITY_FAILURE, "CUDA repository keyring was not installed")
