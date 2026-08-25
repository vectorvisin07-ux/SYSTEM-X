# System X

System X is a private, local-first model-serving platform with a stable local API, explicit ownership boundaries, and a product-owned path from a clean clone to the first accepted GGUF model.

## Product flow

Use this sequence for a new local installation:

1. Clone the single repository branch and enter the clone.

   ```bash
   git clone --branch rebuild/b8bd-system-x --single-branch https://github.com/vectorvisin07-ux/SYSTEM-X.git system-x
   cd system-x
   ```

2. Run the product-owned reconstruction command as the target user.

   ```bash
   /usr/bin/python3.14 -I -S -B bootstrap/run_bootstrap.py reconstruct --authorize
   ```

   The product performs the locked host, environment, pinned `llama.cpp`, empty-runtime, credential, and user-service work through its own adapters. Do not run lower-level construction commands as a substitute.

3. Wait for authenticated `WAITING_FOR_MODEL`. This means the service is available and protected, but inference is intentionally not ready.

4. Place exactly one supported `.gguf` file in:

   `INSPECTOR/MODEL-TEST`

   The file must be fully copied before it becomes visible.

5. Do not type the model name into a command. The automatic path derives its candidate identity from the stable file.

6. Do not run `deploy-gguf` in the normal first-model workflow. Automatic placement uses the pinned `llama.cpp` and `llama-server` GGUF branch.

7. Wait for System X to reach authenticated `READY`. A ready model is not automatically replaced by a later candidate.

8. Read the connection receipt at `INSPECTOR/RUNTIME/status/api-connection.json`, or run the zero-argument read-only connection command:

   ```bash
   cd INSPECTOR
   /usr/bin/python3.14 -I -S -B -m system_x_inspector.machine show-connection
   ```

   The command returns the stored receipt identity and never prints the raw API key.

9. Use model `default` as the recommended client reference.
The automatic path is GGUF-only. Native model support and its vLLM branch are separate layout contracts and are not part of automatic GGUF placement.

## API connections

The receipt provides the local origin and the resolved immutable model while keeping the recommended client reference `default`.

| Client family | Base URL from the receipt | Authentication |
| --- | --- | --- |
| System X native | local origin | `x-api-key` or `Authorization: Bearer` |
| OpenAI-compatible | local origin followed by `/v1` | `Authorization: Bearer` |
| Messages-compatible | local origin | `x-api-key` and `anthropic-version: 2023-06-01` |

Native health and model discovery use `/system/v1/health`, `/system/v1/models`, and `/system/v1/models/{model_id}`. The native generation route is `/system/v1/generate`. OpenAI-compatible clients use `/v1/models`, `/v1/completions`, `/v1/chat/completions`, and `/v1/responses`. Messages-compatible clients use `/v1/models`, `/v1/messages`, and `/v1/messages/count_tokens`.

Example native request:

```json
{"model":"default","prompt":"Write one short sentence.","stream":false}
```

Example OpenAI-compatible request:

```json
{"model":"default","messages":[{"role":"user","content":"Write one short sentence."}],"stream":false}
```

Example Messages-compatible request headers:

```text
x-api-key: <local key held in owner-only state>
anthropic-version: 2023-06-01
```

Never place a raw key in Git, documentation, shell history, logs, or evidence.

## State and safety boundaries

- `WAITING_FOR_MODEL` is a healthy authenticated zero-model state.
- `READY` requires the automatic candidate to pass the complete Inspector and GGUF acceptance chain.
- The automatic path never uses vLLM, never downloads a model, and never replaces a ready model.
- Generated environments, builds, runtime state, credentials, service-manager state, and model bytes remain outside the portable source tree.
- Service lifecycle changes belong to the selected platform adapter; direct process control is not a product operation.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Dependencies](docs/DEPENDENCIES.md)
- [Repository layout](docs/REPOSITORY_LAYOUT.md)
- [Generated, private, and model state](docs/GENERATED_PRIVATE_AND_MODEL_STATE.md)
- [Build and runtime](docs/BUILD_AND_RUNTIME.md)
- [History and acceptance](docs/HISTORY_AND_ACCEPTANCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
