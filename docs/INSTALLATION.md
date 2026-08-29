# Installation

## Scope and safety boundary

Installation reconstructs portable source into private local state. It does not download a model or copy an old credential. The product-owned reconstruction registers the local user service through the selected platform adapter; automatic GGUF placement is the next product workflow.

## Prerequisites

- Ubuntu 26.04 LTS under WSL2 on x86_64
- systemd as PID 1
- a working per-user manager, runtime directory, DBus, and user-service adapter
- the direct packages in `SYSTEM_X_DEPENDENCY_LOCK.json`
- Windows NVIDIA driver support exposed through WSL
- CUDA Toolkit 13.3 in Ubuntu, without a Linux NVIDIA display driver

## Clone and inspect

```bash
git clone --branch rebuild/b8bd-system-x --single-branch https://github.com/vectorvisin07-ux/SYSTEM-X.git system-x
cd system-x
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py identify
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py inspect-host
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py plan
```

These commands are read-only. Review the emitted plan and source identity before authorizing reconstruction. Do not broaden the package list or substitute a lower-level command.

## Product-owned reconstruction

Run this once from the clean clone as the target user:

```bash
/usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py reconstruct --authorize
```

The product validates the frozen host contract, builds the private environments and pinned `llama-server`, initializes empty current-schema runtime state, generates a new local credential, registers the user-service adapter, and verifies authenticated `WAITING_FOR_MODEL`. Never use a manual service-manager command as a replacement for this product operation.

## Automatic first-model placement

After reconstruction:

1. Wait for authenticated `WAITING_FOR_MODEL`.
2. Copy one supported `.gguf` file completely into the direct `INSPECTOR/MODEL-TEST` intake directory.
3. Do not provide a model name or call the normal `deploy-gguf` operation for this path.
4. Let the product observe copy completion and reconcile the single stable candidate.
5. Wait for `READY`, then read the connection receipt if connection details are needed.

The automatic branch is GGUF-only and uses the pinned `llama.cpp` source with `llama-server`. If there is no visible candidate, the service remains `WAITING_FOR_MODEL`. If there are multiple candidates, the copy is still changing, the path is unsafe, or qualification fails, the product records a safe non-ready result. A ready model fences automatic replacement.

## Read-only CLI contract

The bootstrap read-only operations are `identify`, `inspect-host`, `plan`, `status`, and `verify --level waiting-for-model`. The Inspector machine interface exposes `show-connection` as a zero-argument read-only receipt check:

```bash
cd INSPECTOR
/usr/bin/python3.14 -I -S -B -m system_x_inspector.machine show-connection
```

It does not accept a model name, deployment selector, generation request, or receipt mutation. It reports the stored receipt identity and never prints the raw key.

## API contract

Use the base URLs and resolved model reference `default` from the receipt.

- Native: `/system/v1/health`, `/system/v1/models`, `/system/v1/models/{model_id}`, and `/system/v1/generate`.
- OpenAI-compatible: base URL `<origin>/v1`, with `/models`, `/completions`, `/chat/completions`, and `/responses`.
- Messages-compatible: base URL `<origin>`, with `/v1/models`, `/v1/messages`, and `/v1/messages/count_tokens`.

Use `x-api-key` or `Authorization: Bearer` only as documented by the receipt. Store credentials only in owner-only local state.

## Verification

Verify package identities, forbidden-driver absence, private-environment imports, the vendored source manifest, CUDA device visibility, dynamic libraries, empty runtime schemas, owner-only credential state, adapter receipts, authenticated health, and a clean source tree. Do not print the API key. The source tree does not contain generated environments, builds, runtime state, credentials, or model bytes.


## Public installation path

The repository root `system-x` command is the supported user entry point. Run
`system-x install` once after the clone; it supplies authorization to the
existing bootstrap reconstruction and does not take a model name or model
policy. The service remains authenticated and running while the model state is
`WAITING_FOR_MODEL`.

To admit a model, ensure there is exactly one complete stable `.gguf` file in
`INSPECTOR/MODEL-TEST/`. For a large artifact, use a hidden temporary file in
that same directory, flush and close it, then atomically rename it to the final
visible `.gguf` name. The automatic intake path performs validation and
lifecycle work. Do not use a manual deployment command, edit the registry, or
operate the service manager directly.

Poll `system-x status` until readiness is `READY`, then run `system-x
connection`. Use the receipt's `default` model reference with the native,
OpenAI-compatible, or Messages-compatible API family. The receipt exposes only
sanitized connection data and a non-secret key identifier; the raw key is never
printed. OpenClaw is not required.
