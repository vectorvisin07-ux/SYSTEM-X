# Dependencies

`SYSTEM_X_DEPENDENCY_LOCK.json` is the machine-readable evidence-linked specification. Component-specific locks remain authoritative for operations.

## Target host

- Ubuntu 26.04 LTS
- WSL2, x86_64
- systemd PID 1 and a functioning systemd user manager
- CPython 3.14

## Direct Ubuntu packages

| Package | Accepted observation | Purpose |
| --- | --- | --- |
| `build-essential` | `12.12ubuntu2.26.04.2` | C/C++ toolchain |
| `ca-certificates` | `20260601~26.04.1` | TLS trust |
| `cmake` | `4.2.3-2ubuntu2` | llama.cpp configure |
| `curl` | `8.18.0-1ubuntu2.3` | bounded HTTPS diagnostics |
| `dbus-user-session` | `1.16.2-2ubuntu4` | user DBus/session |
| `git` | `1:2.53.0-1ubuntu1` | source operations |
| `ninja-build` | `1.13.2-1` | locked generator |
| `pkg-config` | `2.5.1-4` | dependency discovery |
| `python3.14` | `3.14.4-1ubuntu0.1` | runtime interpreter |
| `python3.14-venv` | `3.14.4-1ubuntu0.1` | private environments |
| `systemd` | `259.5-0ubuntu3` | PID 1/user services |
| `cuda-toolkit-13-3` | `13.3.1-1` | toolkit-only CUDA |

These are accepted observations, not an instruction to silently accept arbitrary substitutions. Names, major/minor identities, source boundaries, and forbidden packages are hard requirements. A compatible patch-level difference requires the committed regex/policy, complete physical validation, and explicit authority. Optional operator tools are not runtime dependencies unless listed above.

## Accepted tool and device identities

- Python `3.14.4`
- CUDA Toolkit package `13.3.1-1`
- nvcc `13.3.73`
- CMake `4.2.3`
- Ninja `1.13.2`
- GCC/G++ `15.2.0`
- Git `2.53.0`
- NVIDIA GeForce RTX 3060, compute capability `8.6`
- Windows driver observation `595.97`

## CUDA/WSL law

Windows owns the NVIDIA display driver. Ubuntu consumes `/dev/dxg` and the WSL bridge libraries and installs CUDA Toolkit 13.3 only. Linux `nvidia-driver-*`, `cuda-drivers*`, and broad `cuda` meta-packages are prohibited.

## llama.cpp

- origin: `https://github.com/ggml-org/llama.cpp`
- tag: `b10092`
- commit: `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`
- tree: `cfbb48d88cfd7f0530f27e08b0112dd66c001816`
- generator/build: Ninja / Release
- CUDA: enabled; target architecture `86`
- target: `llama-server`

## Python environments

The Inspector environment is private CPython 3.14 and uses the committed standard-library/internal dependency policy where its lock proves that closure. The API service environment is private CPython 3.14 reconstructed from `requirements.in`, `requirements.lock`, `environment.lock.json`, and the bootstrap environment lock. Neither environment is repository content.
