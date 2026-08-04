# Third-party notices

## llama.cpp

System X vendors the complete clean Git tree of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at tag `b10092`, commit `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`, tree `cfbb48d88cfd7f0530f27e08b0112dd66c001816`.

The upstream `LICENSE`, notices, vendored dependencies, fixtures, documentation, and source-controlled assets are preserved at `model-api-gguf/llama.cpp`. The ordinary tree was independently normalized from the exact b8bd39f active submodule and vendor-staging trees with zero source patches. Generated build output and the upstream `.git` database are not vendored.

## Host and Python dependencies

Ubuntu, NVIDIA CUDA Toolkit, and Python packages remain externally installed dependencies. Their binaries are not copied into this repository. Package names, accepted versions, sources, and reconstruction policy are recorded in `SYSTEM_X_DEPENDENCY_LOCK.json`, committed component locks, and `docs/DEPENDENCIES.md`. Each dependency remains subject to its own license.
