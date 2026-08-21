# Installation

## Scope and safety boundary

Installation reconstructs portable source into local generated state. It does not authorize model download/admission or copying an old credential. The authorized reconstruction registers the local user service; model handoff remains a separate Inspector operation. Keep the source clone, runtime state, models, private environments, and external products separate.

## Prerequisites

- Ubuntu 26.04 LTS under WSL2 on x86_64
- systemd as PID 1
- a working per-user manager, runtime directory, DBus, and `systemctl --user`
- the direct packages in `SYSTEM_X_DEPENDENCY_LOCK.json`
- Windows NVIDIA driver support exposed through WSL
- CUDA Toolkit 13.3 in Ubuntu, without a Linux NVIDIA display driver

## Clone

```bash
git clone --branch rebuild/b8bd-system-x --single-branch https://github.com/vectorvisin07-ux/SYSTEM-X.git system-x
cd system-x
```

The `rebuild/b8bd-system-x` branch is based on exact b8bd39f and vendors llama.cpp as ordinary files at b10092 / 3ce7da2c / cfbb48d. Do not use `--recursive`; no submodule initialization is required. Installation, build, runtime initialization, service activation, and model admission are explicit generated-state operations.

The portable tree manifest is a `FULL_TREE` source contract. It covers every regular candidate Git-tree member except its explicit self-exclusion and records exact Git mode, portable mode, bytes, and SHA-256. The tracked producer/validator is `bootstrap/system_x_bootstrap/portable_manifest.py`; do not copy a source tree, environment, build, runtime, credential, model, service unit, or packet evidence into a clone.

## Read-only discovery and frozen plan

Run as the target user:

```bash
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py identify
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py inspect-host
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py plan
```

Preserve the plan output and verify its source identity before authorized reconstruction. Never broaden the package list interactively.

## Authorized reconstruction

After the read-only plan is reviewed, run this product command exactly once from the clean clone as the target user:

```bash
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py reconstruct --authorize
```

The product performs its own read-only host plan. If locked host packages are missing, the product validates the platform-owned elevation route and preserves the validated installation user; on WSL it may re-enter the exact `Ubuntu-26.04` distribution through `/init` when noninteractive sudo is unavailable, then performs one product-owned privilege handoff before starting its single host mutation transaction. The packet or operator wrapper must not run `sudo`, `apt`, `systemctl`, or any replacement command. The one reconstruction command then verifies the vendored source, builds the two private environments and only `llama-server`, initializes empty current-schema runtime state, generates a new local credential, registers the accepted user service adapter, and verifies model-free authenticated `WAITING_FOR_MODEL`. Do not invoke the lower-level operations individually for this proof. On an already host-ready Ubuntu 26.04 target, no elevation is performed.

The host package contract remains bounded to:

```text
build-essential ca-certificates cmake curl dbus-user-session git ninja-build
pkg-config python3.14 python3.14-venv systemd cuda-toolkit-13-3
```

The CUDA package is toolkit-only. Reject `cuda`, `cuda-13-3`, `cuda-drivers*`, `nvidia-driver-*`, and other packages that pull a Linux display-driver stack.

After reconstruction, `status` and `verify --level waiting-for-model` are read-only checks. Model admission is a separate explicit Inspector operation.

## Service and model boundary

Enable/start/stop through `model-api-gguf/service_control/platform_adapters`; do not kill processes directly. Model admission is a later explicit Inspector operation. A healthy empty runtime should remain `WAITING_FOR_MODEL`.

## Verification

Verify package identities, forbidden driver absence, environment imports, the vendored source manifest, CUDA device listing, resolved dynamic libraries, empty runtime schemas, owner-only credential state, platform-adapter receipts, authenticated health, and a clean source tree. Never print the API key as evidence.
