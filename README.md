# System X

System X is a private, local-first model-serving platform with a stable API and explicit lifecycle control. It separates model inspection, model-format routing, inference engines, API compatibility, credentials, runtime state, and platform service control so that each boundary can be audited independently.

## Current architecture

- **Inspector** is model-format neutral. It identifies and validates artifacts before any admission.
- **GGUF** uses the pinned `llama.cpp` source and `llama-server`.
- **Native models** are assigned to vLLM. The native branch is not currently built or physically accepted by this repository publication.
- **System X API** provides the stable local interface and compatibility adapters.
- **Service control** owns desired state and delegates activation to the selected platform adapter.

With no admitted model, healthy System X reports `WAITING_FOR_MODEL`. That is an intentional zero-model state, not a claim that inference is ready.

## Repository boundary

At the completed SYSMIN.02.PUSH.R1 publication tip, `model-api-gguf/llama.cpp` is ordinary vendored source pinned to tag `b10092`, commit `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`. A normal clone is sufficient; recursive submodule initialization is not required. Intermediate commits in the ordered publication retain the prior gitlink until all vendor batches have been remotely verified.

The repository contains portable first-party source, configuration, schemas, tests, scripts, documentation, deterministic fixtures, dependency locks, and the exact pinned third-party source. It deliberately does not contain generated Python environments, CUDA/toolchain binaries, compiled builds, model weights, mutable runtime state, databases, local credentials, caches, bytecode, or temporary files.

HEYCHAT / `system-x-chat` is a separate product. `UNCENSORED-ENV` and model-lab artifacts are also separate products. None is a System X dependency or part of this tree.

## Supported host

The accepted host profile is Ubuntu 26.04 LTS on WSL2, x86_64, with systemd PID 1 and CPython 3.14. Windows owns the NVIDIA display driver. Ubuntu installs CUDA Toolkit 13.3 only; Linux NVIDIA display-driver packages are prohibited. See [dependencies](docs/DEPENDENCIES.md) for exact identities.

## Clone and inspect

```bash
git clone https://github.com/vectorvisin07-ux/SYSTEM-X.git system-x
cd system-x
python3.14 -I -S -B bootstrap/run_bootstrap.py identify
python3.14 -I -S -B bootstrap/run_bootstrap.py inspect-host
python3.14 -I -S -B bootstrap/run_bootstrap.py plan
```

`identify`, `inspect-host`, and `plan` are read-only and run as the target user. Review and freeze the emitted plan before any privileged host action. Do not treat this README as authorization to mutate a host.

## Reconstruction sequence

1. Prove the user manager, `/run/user/<uid>`, user DBus, and `systemctl --user` gate.
2. Run the three read-only bootstrap commands above as the target user and freeze the plan.
3. Apply only the frozen host plan as root. This installs the direct Ubuntu packages and CUDA Toolkit 13.3 without a Linux display driver.
4. Rebuild the Inspector and API private Python environments from committed locks.
5. Verify the vendored llama source and build only `llama-server` with the committed CUDA/Ninja profile.
6. Initialize only empty current-schema runtime directories and databases.
7. Generate a new local credential; never import a raw credential into Git.
8. Register and control the user unit only through the accepted platform adapter.
9. Admit a model only through a separate explicit Inspector workflow.

Detailed commands and gates are in [Installation](docs/INSTALLATION.md) and [Build and runtime](docs/BUILD_AND_RUNTIME.md).

## State origins

- Models come from an operator-selected external source and require explicit Inspector admission.
- Credentials are freshly generated or explicitly imported into owner-only local state; raw values never come from Git.
- Runtime databases and records are initialized from tracked schemas and then remain mutable local state.
- Environments and builds are reconstructed from tracked locks and source.

## Health states

`WAITING_FOR_MODEL` means the service is authenticated and healthy but no model is admitted. `READY` requires separate physical model admission and runtime acceptance. Failure and transition states are documented with the lifecycle schemas; a repository clone alone proves neither service health nor model readiness.

## Acceptance boundary

This source tree records accepted historical identities and the procedure for a fresh-clone verification. Documentation committed before the final verification cell intentionally does not claim that a clean clone has already passed that cell. See [History and acceptance](docs/HISTORY_AND_ACCEPTANCE.md) for accepted, superseded, separate-product, and still-unproved boundaries.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Dependencies](docs/DEPENDENCIES.md)
- [Repository layout](docs/REPOSITORY_LAYOUT.md)
- [Generated, private, and model state](docs/GENERATED_PRIVATE_AND_MODEL_STATE.md)
- [Build and runtime](docs/BUILD_AND_RUNTIME.md)
- [History and acceptance](docs/HISTORY_AND_ACCEPTANCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
