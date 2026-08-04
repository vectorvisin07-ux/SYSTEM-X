# History and acceptance

## Current authority

The active source-control branch is `rebuild/b8bd-system-x`, based on `b8bd39fc6fd86f5bd3a8392e8422bb6b7e3af4e7`. Its ordinary llama.cpp source was independently normalized from matching authenticated upstream trees at tag `b10092`, commit `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`, tree `cfbb48d88cfd7f0530f27e08b0112dd66c001816`, with zero source patches. An ordinary clone is sufficient and recursive submodule initialization is not required.

Conflicts are resolved in this order: current explicit system requirements, current locked architecture, current live filesystem, current physical execution evidence, accepted architecture maps, then older Git history.

## Accepted product boundary

System X source, schemas, bootstrap configuration, tests, and exact vendored llama.cpp are tracked. Private environments, generated builds, runtime databases, credentials, service-manager state, model artifacts, caches, and external workflow evidence are not tracked. Historical source states remain available in Git.

## Separate products

HEYCHAT / `system-x-chat`, `UNCENSORED-ENV`, and model-lab acquisition or evaluation artifacts are separate products. They are not executed by this repository.

## Claim ceiling

A source commit establishes only its recorded source and repository properties. Host, build, service, model, and inference claims require current physical verification against generated local state.
