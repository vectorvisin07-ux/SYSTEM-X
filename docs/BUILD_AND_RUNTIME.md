# Build and runtime

The committed tree contains no generated build, initialized runtime, credential, or model. The operations below reconstruct private state under explicit product authorization.

## Environment build

Create the private CPython 3.14 environments through the committed bootstrap operation. The Inspector lock defines its standard-library/internal closure. The API locks define direct pins, resolved lock lines, and artifact identities. Post-build checks import the committed modules with system-site-packages disabled.

## Vendored source verification

Before configuration, bootstrap verifies ordinary vendored mode: no gitlink, no nested `.git`, no `build/`, no upstream identity mismatch, and every upstream path, mode, byte count, and SHA-256 equal to `model-api-gguf/LLAMA_CPP_SOURCE_IDENTITY.json`. Verification does not contact the network in vendored mode. The compatibility command `initialize-submodules` reports `VENDORED_SOURCE_VERIFIED`; it never initializes a submodule.

## llama-server build

The committed build profile uses:

```text
generator: Ninja
build type: Release
CUDAToolkit_ROOT: /usr/local/cuda-13.3
CMAKE_CUDA_COMPILER: /usr/local/cuda-13.3/bin/nvcc
GGML_CUDA: ON
LLAMA_BUILD_SERVER: ON
target compute capability: 86
target: llama-server
```

Generated files go only beneath `model-api-gguf/llama.cpp/build`. A no-model verification may run `llama-server --version`, list devices, prove CUDA0, and inspect dynamic-library resolution. It must leave no listener or process and must not load a model.

## Runtime initialization

`initialize-runtime` validates tracked schemas and creates only empty current-schema directories and databases. `initialize-credentials` creates a new local secret without printing it. `register-platform-service` is gated by the user manager, user DBus, runtime directory, and the selected platform adapter.

## Automatic GGUF placement

After the service reaches authenticated `WAITING_FOR_MODEL`, the product-owned automatic coordinator watches the direct `INSPECTOR/MODEL-TEST` inbox. One fully copied supported `.gguf` file is the complete input; no model name is supplied. The coordinator waits for stable metadata and bytes, then dispatches the pinned GGUF branch through `llama.cpp` and `llama-server`.

The automatic coordinator uses first-model placement only. It does not replace a ready model, does not treat multiple visible candidates as an instruction to choose one, and records safe rejection or copy-in-progress states. A successful path publishes the connection receipt without exposing the raw key.

## Lifecycle control

Desired-state changes and activation are accepted only through the selected platform adapter. Do not kill processes directly. `WAITING_FOR_MODEL` is the stable authenticated state before automatic placement; `READY` requires physical candidate acceptance and runtime verification. Read-only receipt inspection does not mutate service, model, registry, or credential state.
## Receipt and client contract

The zero-argument `show-connection` operation is read-only. It checks the stored connection receipt, reports its byte identity, and never prints the raw key. The receipt supplies the local origins and the recommended client model reference `default` for native, OpenAI-compatible, and Messages-compatible requests.
