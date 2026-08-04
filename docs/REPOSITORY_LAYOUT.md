# Repository layout

This layout describes `rebuild/b8bd-system-x`, normalized independently from the exact b8bd39f active submodule and vendor-staging trees. The source snapshot contains no generated build, initialized runtime, credential, or model, and it does not prove System X health.

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
├── model-api-native/             native/vLLM layout contract; not yet accepted
├── docs/                         installation, dependency and operating guidance
├── SYSTEM_X_*                    machine-readable manifests and contracts
└── THIRD_PARTY_NOTICES.md        third-party source identity and notices
```

The final `model-api-gguf/llama.cpp` directory is not a gitlink and has no nested `.git`. It contains only files tracked by the exact upstream commit. The System X identity record is outside that upstream tree at `model-api-gguf/LLAMA_CPP_SOURCE_IDENTITY.json`; upstream `build/` output is absent.

Generated roots such as `.venv`, `RUNTIME`, `MODEL`, capability records, caches, and compiler output are described in `SYSTEM_X_EXCLUDED_STATE_MANIFEST.json` and `SYSTEM_X_DIRECTORY_CONTRACTS.json`; they are not source payload. Required empty directories are represented by contracts because Git does not store directories.

## Excluded-root reconstruction contract

| Exact path | Owner | Why excluded | Created or reconstructed by | Required before runtime | Cleanup owner | Secrets | Model bytes |
|---|---|---|---|---|---|---|---|
| `.system-x-bootstrap-state` | portable bootstrap | local plans, locks, transactions, receipts, and status | bootstrap operations | only for authorized reconstruction | bootstrap transaction/recovery | no raw values by contract | no |
| `INSPECTOR/.venv` | Inspector | private generated Python environment | `build-environments` from the committed CPython lock | yes, before Inspector execution | Inspector environment builder/operator | no | no |
| `INSPECTOR/MODEL-TEST` | Inspector intake | operator-supplied inspection candidates | explicit operator intake | no; only before inspection | Inspector intake transaction/operator | no | yes |
| `INSPECTOR/RUNTIME` | Inspector | mutable logs, locks, status, transactions, and results | `initialize-runtime` and Inspector operations | yes, before Inspector operations | Inspector runtime/transaction owner | no raw credential by contract | no |
| `INSPECTOR/capabilities` | Inspector | generated capability records and bindings | Inspector capability/decision operations | before admission and handoff, not clone inspection | Inspector capability owner | no | no |
| `INSPECTOR/environment.lock.json` | Inspector environment builder | host-specific environment observation | successful private-environment build | before accepted Inspector runtime | Inspector environment builder | no | no |
| `model-api-gguf/MODEL` | GGUF model lifecycle | admitted production model artifacts | explicit Inspector admission from an external operator-selected source | no for `WAITING_FOR_MODEL`; yes for `READY` | Inspector retirement/model owner | no | yes |
| `model-api-gguf/RUNTIME` | GGUF API and service control | mutable databases, credentials, logs, PIDs, locks, status, and staging | `initialize-runtime`, fresh credential generation, and runtime transactions | yes, except model state may remain empty | GGUF runtime/platform adapter owners | yes | yes, only in controlled replacement staging |
| `model-api-gguf/api_service/.venv` | GGUF API | private generated dependency environment | `build-environments` from committed hashed wheel locks | yes, before API runtime | API environment builder/operator | no | no |
| `model-api-gguf/api_service/environment.lock.json` | GGUF API environment builder | host-specific environment observation | successful private-environment build | before accepted API runtime | API environment builder | no | no |
| `model-api-gguf/llama.cpp/build` | GGUF engine | generated CMake/Ninja/CUDA output | `build-llama-server` from vendored source and build lock | yes, before GGUF inference process startup | llama build operation/operator | no | no |
| `model-api-native/MODEL` | native model lifecycle | external native model artifacts | future explicit native-model admission | not for the currently accepted GGUF path | native model lifecycle owner | no | yes |
| `model-api-native/RUNTIME` | native service | mutable native runtime state | future authorized native-runtime initialization | not currently required or accepted | native runtime owner | no raw value by contract | no |
| `model-api-native/vLLM` | native environment builder | generated vLLM/toolchain environment | future committed native dependency workflow | not currently required or accepted | native environment builder/operator | no | no production weights |

External generated systemd-user state is also excluded: `.config/systemd/user/system-x.service` is owned and rendered by the Linux systemd-user platform adapter, and its `default.target.wants` link is owned by adapter enablement. Both are required only for registered service operation, may be removed only by that adapter/operator workflow, and contain neither secrets nor model bytes.

Caches and temporary files are incidental products of interpreters, tests, compilers, and runtime operations. Their owning producer cleans them; they are never prerequisites or portable payload.

Outer and nested Git object databases are never mirrored as repository content. External workflow evidence is absent from the product tree and remains in its owning external workspace.

HEYCHAT / `system-x-chat` and `UNCENSORED-ENV` are outside this layout.
