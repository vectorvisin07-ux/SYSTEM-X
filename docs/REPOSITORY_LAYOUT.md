# Repository layout

This layout describes the portable source tree. It contains no generated build, initialized runtime, credential, or model, and it does not by itself prove service health or model readiness.

```text
.
├── INSPECTOR/                    portable Inspector source, schemas and tests
├── bootstrap/                    clean-host discovery, plan and reconstruction
├── model-api-gguf/
│   ├── api_service/              stable API source, locks, schema and tests
│   ├── api_service_controller/   lifecycle controller source
│   ├── branch_controller/        GGUF branch controller source
│   ├── llama.cpp/                exact ordinary vendored upstream source
│   └── service_control/          supervisor, recovery and platform adapters
├── model-api-native/             native/vLLM layout contract; not accepted for automatic placement
├── docs/                         installation, dependency and operating guidance
├── SYSTEM_X_*                    machine-readable manifests and contracts
└── THIRD_PARTY_NOTICES.md        third-party source identity and notices
```

The final `model-api-gguf/llama.cpp` directory is not a gitlink and has no nested `.git`. It contains only files tracked by the exact upstream source identity. The System X identity record is outside that upstream tree; upstream `build/` output is absent.

Generated roots such as `.venv`, `RUNTIME`, `MODEL`, capability records, caches, and compiler output are described in `SYSTEM_X_EXCLUDED_STATE_MANIFEST.json` and `SYSTEM_X_DIRECTORY_CONTRACTS.json`; they are not source payload. Required empty directories are represented by contracts because Git does not store directories.

## Excluded-root reconstruction contract

| Exact path | Owner | Why excluded | Created or reconstructed by | Required before runtime | Cleanup owner | Secrets | Model bytes |
|---|---|---|---|---|---|---|---|
| `.system-x-bootstrap-state` | portable bootstrap | local plans, locks, transactions, receipts, and status | bootstrap operations | only for authorized reconstruction | bootstrap transaction/recovery | no raw values by contract | no |
| `INSPECTOR/.venv` | Inspector | private generated Python environment | `build-environments` from the committed CPython lock | yes, before Inspector execution | Inspector environment builder/operator | no | no |
| `INSPECTOR/MODEL-TEST` | Inspector automatic intake | one operator-supplied supported `.gguf` candidate | product-owned automatic reconciliation after copy completion | no for `WAITING_FOR_MODEL`; required for automatic first-model placement | intake owner/operator | no | yes |
| `INSPECTOR/RUNTIME` | Inspector | mutable logs, locks, status, transactions, and results | runtime initialization and product operations | yes, before Inspector operations | Inspector runtime/transaction owner | no raw credential by contract | no |
| `INSPECTOR/capabilities` | Inspector | generated capability records and bindings | Inspector capability/decision operations | before admission and handoff, not clone inspection | Inspector capability owner | no | no |
| `model-api-gguf/MODEL` | GGUF model lifecycle | accepted production model artifacts | automatic first-model acceptance or an explicitly authorized separate workflow | no for `WAITING_FOR_MODEL`; yes for `READY` | model lifecycle owner | no | yes |
| `model-api-gguf/RUNTIME` | GGUF API and service control | mutable databases, credentials, logs, PIDs, locks, status, and staging | runtime initialization, fresh credential generation, and product transactions | yes, except model state may remain empty | API/platform owners | yes | yes only in controlled replacement staging |
| `model-api-gguf/api_service/.venv` | GGUF API | private generated dependency environment | `build-environments` from committed hashed wheel locks | yes, before API runtime | API environment builder/operator | no | no |
| `model-api-gguf/llama.cpp/build` | GGUF engine | generated CMake/Ninja/CUDA output | `build-llama-server` from vendored source and build lock | yes, before inference startup | build operation/operator | no | no |
| `model-api-native/MODEL` | native model lifecycle | external native model artifacts | future explicit native-model workflow | not required for automatic GGUF placement | native model owner | no | yes |
| `model-api-native/RUNTIME` | native service | mutable native runtime state | future native-runtime workflow | not currently required or accepted | native runtime owner | no raw value by contract | no |
| `model-api-native/vLLM` | native environment builder | generated native environment | future native dependency workflow | not currently required or accepted | native environment builder/operator | no | no production weights |

External generated user-service state is also excluded. The selected platform adapter owns the service unit and enablement link; neither contains secrets or model bytes.

Caches and temporary files are incidental products of interpreters, tests, compilers, and runtime operations. Their owning producer cleans them; they are never portable payload.

Outer and nested Git object databases are never mirrored as repository content. External workflow evidence remains in its owning external workspace.

HEYCHAT / `system-x-chat` and `UNCENSORED-ENV` are outside this layout.
## Client reference

The accepted GGUF branch is served by `llama-server`. Read the zero-argument `show-connection` result for local origins and use `default` as the recommended client model reference.
