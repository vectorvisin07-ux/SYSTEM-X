# Build and runtime

## Environment build

Create two private CPython 3.14 environments through the committed bootstrap operation. The Inspector lock defines its standard-library/internal closure. The API locks define direct pins, resolved lock lines, and artifact identities. Post-build checks must import the committed modules with system-site-packages disabled.

## Vendored source verification

Before configuration, bootstrap verifies ordinary vendored mode: no gitlink, no nested `.git`, no `build/`, origin/tag/commit/tree identity metadata present, and every upstream file/mode/hash equal to `model-api-gguf/LLAMA_CPP_SOURCE_IDENTITY.json`. Verification must not contact the network in vendored mode. The compatibility command `initialize-submodules` reports `VENDORED_SOURCE_VERIFIED`; it never initializes a submodule or contacts GitHub.

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

Generated files go only beneath `model-api-gguf/llama.cpp/build`. A no-model verification may run `llama-server --version`, list devices, prove CUDA0, and inspect dynamic-library resolution. It must leave no listener/process and must not load a model.

## Runtime initialization

`initialize-runtime` validates tracked schemas and creates only empty current-schema directories/databases. It must not migrate private state from another host. `initialize-credentials` creates a new local secret without printing it. `register-platform-service` is gated by the user manager, user DBus, runtime directory, and `systemctl --user`.

## Lifecycle control

Desired state changes and activation are accepted only through the selected platform adapter. Do not kill processes directly. `WAITING_FOR_MODEL` is the stable authenticated health state before explicit model admission. Model admission and `READY` acceptance are separate workflows.
