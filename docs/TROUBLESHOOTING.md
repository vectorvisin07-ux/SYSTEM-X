# Troubleshooting

## Service is `WAITING_FOR_MODEL`

This is the correct healthy state when no model is admitted. Confirm that the service is authenticated, then place exactly one fully copied supported `.gguf` file in the direct `INSPECTOR/MODEL-TEST` directory. Do not provide a model name. The product-owned coordinator will observe stability and perform first-model placement.

## The service stays waiting after a copy

Check that the candidate is a direct visible child, has a `.gguf` suffix, is a regular file, and is no longer changing. A partial copy is intentionally reported as copy-in-progress. A hidden file is not a visible candidate. Do not create a second candidate to force progress.

## Multiple candidates are visible

The safe result is no automatic selection. Remove or relocate the extra source through the operator-owned intake workflow, then leave one complete candidate visible. A ready model is never automatically replaced.

## A candidate is rejected

The Inspector may reject a directory, symlink, hard link, special file, unreadable file, path escape, mount substitution, unsupported GGUF, failed qualification, or changed source. Preserve the recorded reason and repair the intake source or candidate through the product workflow. Do not bypass the Inspector with a direct deployment or model-server command.

## Connection details are unavailable

Use the zero-argument read-only receipt operation:

```bash
cd INSPECTOR
/usr/bin/python3.14 -I -S -B -m system_x_inspector.machine show-connection
```

It reports whether the stored receipt is absent, stale, unavailable, or ready. It returns the stored receipt identity and never prints the raw API key. Do not read credential files or put keys in logs.

## API compatibility checks

Use only the base URLs and model reference `default` shown by the receipt.

- Native health: `/system/v1/health`
- Native models: `/system/v1/models`
- Native generation: `/system/v1/generate`
- OpenAI-compatible models and inference: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/responses`
- Messages-compatible models and inference: `/v1/models`, `/v1/messages`, `/v1/messages/count_tokens`

## User manager or DBus fails

Do not register or activate System X manually. Prove the target user has a distinct user manager, runtime directory, user bus, and a functioning selected platform adapter. Repair that boundary through the product-owned bootstrap or adapter operation before service work.

## Build or CUDA verification fails

Compare `llama-build.lock.json`, `cuda-wsl.lock.json`, the vendored source identity, the configured CMake values, and the toolkit-only package policy. Do not install a Linux display-driver stack or rebuild from an unpinned source.

## Runtime or credential state is absent

Initialize empty current-schema runtime and generate a new local credential through bootstrap. Do not copy an old database, pepper, or API key. Never print secrets into logs or evidence.

## Receipt or registry state is inconsistent

Stop automatic placement and preserve the existing records. Use read-only status and receipt inspection, then repair through the owning Inspector, API, or platform adapter operation. Do not write a registry, edit a receipt, or kill a service by hand.
## Model becomes `READY`

A successful automatic path reaches `READY` only after the candidate is accepted through the pinned `llama.cpp` and `llama-server` GGUF branch. Use the receipt for the resolved client reference; do not change service or model state by hand.


## Public front-door troubleshooting

- `SOURCE_ONLY`: the clone is present but the authenticated installation has
  not run. Use `system-x install`.
- `WAITING_FOR_MODEL`: the service is running and waiting for one complete,
  stable `.gguf` file in `INSPECTOR/MODEL-TEST/`. Use one hidden temporary copy
  and a same-directory atomic rename so intake never sees a partial artifact.
- `READY`: run `system-x connection` and use model reference `default`.
- `CONNECTION_STALE` or `DEGRADED`: keep the receipt private and rerun the
  read-only status/connection commands; do not edit generated state or resend
  a generation request.
- `REPAIR_REQUIRED`: preserve the installation and its receipts, then inspect
  the read-only `system-x doctor` result. Do not manually start a backend,
  change the registry, or manipulate the service manager.

The public commands do not require OpenClaw and never display the raw API key,
private model-child endpoint, or physical model path.
