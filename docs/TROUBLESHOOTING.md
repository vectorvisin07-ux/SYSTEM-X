# Troubleshooting

## User manager or DBus fails

Do not register/start System X. Prove the target user has a distinct `user@<uid>.service`, `/run/user/<uid>`, user bus, and successful `systemctl --user`. Repair the user-session boundary before service work; do not add an artificial WSL keepalive.

## GitHub authentication fails

Verify the signed credential-manager binary and authenticated account without printing the credential. Confirm repository `vectorvisin07-ux/SYSTEM-X`, private visibility, write permission, branch `main`, expected parent, and unchanged tags. Never force-push or rewrite history.

## Vendored-source verification fails

Compare every path, mode, byte count, SHA-256, and upstream Git blob identity against tag `b10092` / commit `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`. Reject missing/extra/changed files, nested `.git`, `build/`, or a remaining gitlink. Vendored mode must not fetch the network.

## CUDA is missing or wrong

Check `/usr/local/cuda-13.3/bin/nvcc`, the WSL bridge paths, `nvidia-smi`, Toolkit package `cuda-toolkit-13-3`, and forbidden Linux display-driver packages. Do not install `cuda`, `cuda-13-3`, `cuda-drivers*`, or `nvidia-driver-*` as a shortcut.

## Environment reconstruction fails

Use CPython 3.14, private venvs, system-site-packages disabled, and the exact committed component/bootstrap locks. Do not copy an old environment or relax hashes silently.

## Service reports `WAITING_FOR_MODEL`

That is the correct healthy state when no model is admitted. Do not download/load a model merely to change the state. Use the separate Inspector admission workflow when explicitly authorized.

## Runtime or credential state is absent

Initialize empty current-schema runtime and generate a new local credential through bootstrap. Do not copy an old database, pepper, or API key; never print secrets into logs/evidence.

## Publication race

Stop before push if remote `main` is not the exact expected parent. Re-authenticate and reconcile under a new authorized plan. Never force, rebase published history, push tags, or push another branch.
