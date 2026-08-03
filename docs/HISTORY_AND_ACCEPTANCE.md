# History and acceptance

The publication reviewed every stable physical packet, final output, failed-run record, incomplete output, X.5/XER5 record, and Macro/Mini/SYSMIN evidence file through EOF. Duplicate bytes were analyzed once by SHA-256 while every physical filename remains represented in `SYSTEM_X_EVIDENCE_INDEX.json`. Raw OpenClaw output is not copied into Git.

## Current authority

Conflicts are resolved in this order: current explicit SIR, current locked architecture, current live filesystem, current physical execution evidence, accepted architecture/maps, then older history.

Current architectural facts are:

- GGUF: llama.cpp / llama-server
- native: vLLM
- Inspector: model-format neutral
- zero-model healthy state: `WAITING_FOR_MODEL`
- host: Ubuntu 26.04 LTS / WSL2 / x86_64 / systemd
- CUDA: Toolkit 13.3 only; Windows display driver boundary

## Accepted evidence

Earlier accepted records establish the stable API/runtime lifecycle, Inspector and branch decision architecture, no-model build/smoke behavior, platform-adapter contract, and previous sanitized Git publication parent `08f144e8c2e702eccfa923ca8a600f9b7367c244`.

## Superseded directions

- routing GGUF through vLLM is historical, not current guidance;
- representing llama.cpp only by a gitlink is superseded at the completed R1 vendor switch;
- an encrypted full-machine snapshot in Git/LFS is superseded by the portable-source policy.

## Separate products and failed work

HEYCHAT / `system-x-chat`, `UNCENSORED-ENV`, and model-lab acquisition/evaluation artifacts are separate. Incomplete/failed SYSMIN.03 handoff evidence is retained as repair history, not installation acceptance. It is not executed by this publication.

## Claim ceiling

The repository may claim exact source/dependency identities and source-only verification when those gates pass. Until the final fresh-clone cell is physically complete, it does not claim that a clean clone has already passed that cell. Repository publication alone does not prove a running service, admitted model, `READY`, performance, or native/vLLM acceptance.
