# Installation

## Scope and safety boundary

Installation reconstructs portable source into local generated state. It does not authorize model download/admission, service handoff, or copying an old credential. Keep the source clone, runtime state, models, private environments, and external products separate.

## Prerequisites

- Ubuntu 26.04 LTS under WSL2 on x86_64
- systemd as PID 1
- a working per-user manager, runtime directory, DBus, and `systemctl --user`
- the direct packages in `SYSTEM_X_DEPENDENCY_LOCK.json`
- Windows NVIDIA driver support exposed through WSL
- CUDA Toolkit 13.3 in Ubuntu, without a Linux NVIDIA display driver

## Clone

```bash
git clone https://github.com/vectorvisin07-ux/SYSTEM-X.git system-x
cd system-x
```

The completed R1 tree vendors llama.cpp as ordinary files. Do not use `--recursive`; no submodule initialization is required.

## Read-only discovery and frozen plan

Run as the target user:

```bash
python3.14 -I -S -B bootstrap/run_bootstrap.py identify
python3.14 -I -S -B bootstrap/run_bootstrap.py inspect-host
python3.14 -I -S -B bootstrap/run_bootstrap.py plan
```

Preserve the plan output and verify its source identity before privileged work. `apply-host` is the only host-package stage and must be run as root only for that exact frozen plan. Never broaden the package list interactively.

## Host stage

The direct package set is:

```text
build-essential ca-certificates cmake curl dbus-user-session git ninja-build
pkg-config python3.14 python3.14-venv systemd cuda-toolkit-13-3
```

The CUDA package is toolkit-only. Reject `cuda`, `cuda-13-3`, `cuda-drivers*`, `nvidia-driver-*`, and other packages that pull a Linux display-driver stack.

## Generated stages

After host acceptance, use the committed bootstrap operations in this order:

1. `build-environments` — creates the two private CPython 3.14 environments from locks.
2. `build-llama-server` — verifies vendored source identity and builds only the pinned CUDA target.
3. `initialize-runtime` — creates empty current-schema runtime and database state.
4. `initialize-credentials` — generates a new local API credential without printing it.
5. `register-platform-service` — writes/registers the user unit only after the user-manager gate passes.

Use `status` and `verify` between stages. The high-level `reconstruct` operation is appropriate only when its plan and all gates are understood and accepted.

## Service and model boundary

Enable/start/stop through `model-api-gguf/service_control/platform_adapters`; do not kill processes directly. Model admission is a later explicit Inspector operation. A healthy empty runtime should remain `WAITING_FOR_MODEL`.

## Verification

Verify package identities, forbidden driver absence, environment imports, the vendored source manifest, CUDA device listing, resolved dynamic libraries, empty runtime schemas, owner-only credential state, platform-adapter receipts, authenticated health, and a clean source tree. Never print the API key as evidence.
