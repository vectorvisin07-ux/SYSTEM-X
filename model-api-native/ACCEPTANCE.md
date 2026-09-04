# Native branch acceptance contract

The native branch is a private stock-vLLM environment and remains installed
idle when the original GGUF model is current. It has no vLLM plugin and sets
`VLLM_PLUGINS` to the empty string. Native activation never installs packages,
builds engines, downloads model files, or edits the registry directly.

The bounded acceptance fixture used during V7 was
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, loaded with
`trust_remote_code=false` from local safetensors. The fixture is disposable and
must be retired after acceptance; it is not part of the release tree.
