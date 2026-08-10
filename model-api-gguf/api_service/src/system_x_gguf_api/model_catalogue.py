"""Public model catalogue and immutable registry-backed request resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from .backend import BackendError
from .errors import SystemXError
from .model_registry import ModelRegistry, ModelRegistryError
from .operation_records import OperationRecorder
from .schemas import (
    ModelCapabilities,
    ModelDetailResponse,
    ModelListResponse,
    PlannerDefaultGeneration,
    PlannerModelCapabilities,
    PublicModelDetail,
    PublicModelRecord,
    TokenCountOperations,
)


Operation = Literal["generate", "chat", "responses", "token_count"]
CapabilityName = Literal["tool_calling", "structured_output", "streaming"]
CapabilityState = Literal["available", "not_tested", "unavailable"]


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    requested_reference: str
    registry_generation: int
    public_model_id: str
    bundle_id: str
    router_model_id: str
    registration_state: str
    created_utc: str
    aliases: tuple[str, ...]
    capability_manifest_identity: str
    context_bound: int | None
    chat_template_present: bool
    tool_calling_state: CapabilityState
    structured_output_state: CapabilityState
    parallel_tool_calling_state: CapabilityState
    streaming_state: CapabilityState

    def verification_fields(self) -> dict[str, Any]:
        return {
            "model_version_id": self.public_model_id,
            "bundle_id": self.bundle_id,
            "router_model_id": self.router_model_id,
            "state": self.registration_state,
            "capability_manifest_sha256": self.capability_manifest_identity,
            "created_utc": self.created_utc,
            "aliases": list(self.aliases),
            "artifact_present": True,
        }


class ModelCatalogue:
    """Separate public presentation from private registry/router identities."""

    def __init__(
        self,
        registry: ModelRegistry,
        backend: Any,
        operations: OperationRecorder,
    ) -> None:
        self.registry = registry
        self.backend = backend
        self.operations = operations
        self._proven_operations: dict[str, set[Operation]] = {}

    @staticmethod
    def _manifest_facts(
        row: dict[str, Any],
    ) -> tuple[
        int | None,
        bool,
        CapabilityState,
        CapabilityState,
        CapabilityState,
        CapabilityState,
    ]:
        try:
            manifest = json.loads(str(row["capability_manifest_json"]))
            context = manifest.get("context")
            chat_template = manifest.get("chat_template")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model metadata is unavailable",
            ) from exc
        if not isinstance(context, dict) or not isinstance(chat_template, dict):
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model metadata is unavailable",
            )
        context_bound = context.get("default_n_ctx")
        if type(context_bound) is not int or context_bound <= 0:
            context_bound = None
        template_present = chat_template.get("present")
        if type(template_present) is not bool:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model metadata is unavailable",
            )
        derived = manifest.get("derived_template_capabilities")
        runtime_tests = manifest.get("runtime_generation_tests")
        if not isinstance(derived, dict) or not isinstance(runtime_tests, dict):
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model metadata is unavailable",
            )
        template_tools = derived.get("tool_calling")
        template_parallel = derived.get("parallel_tool_calling")
        if template_tools not in {True, False, "unknown"} or template_parallel not in {
            True,
            False,
            "unknown",
        }:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model metadata is unavailable",
            )
        if not template_present or template_tools is False:
            tool_calling: CapabilityState = "unavailable"
        elif runtime_tests.get("tool_calling") == "AVAILABLE":
            tool_calling = "available"
        else:
            tool_calling = "not_tested"
        structured_output: CapabilityState = (
            "available"
            if runtime_tests.get("structured_output") == "AVAILABLE"
            else "not_tested"
        )
        streaming: CapabilityState = (
            "available"
            if runtime_tests.get("streaming") == "AVAILABLE"
            else "not_tested"
        )
        if template_parallel is False or tool_calling == "unavailable":
            parallel_tool_calling: CapabilityState = "unavailable"
        elif runtime_tests.get("parallel_tool_calling") == "AVAILABLE":
            parallel_tool_calling = "available"
        else:
            parallel_tool_calling = "not_tested"
        return (
            context_bound,
            template_present,
            tool_calling,
            structured_output,
            parallel_tool_calling,
            streaming,
        )

    @classmethod
    def _snapshot(
        cls,
        reference: str,
        generation: int,
        row: dict[str, Any],
    ) -> ModelSnapshot:
        (
            context_bound,
            chat_template_present,
            tool_calling_state,
            structured_output_state,
            parallel_tool_calling_state,
            streaming_state,
        ) = cls._manifest_facts(row)
        aliases = row.get("aliases")
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) for alias in aliases
        ):
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model metadata is unavailable",
            )
        return ModelSnapshot(
            requested_reference=reference,
            registry_generation=generation,
            public_model_id=str(row["model_version_id"]),
            bundle_id=str(row["bundle_id"]),
            router_model_id=str(row["router_model_id"]),
            registration_state=str(row["state"]),
            created_utc=str(row["created_utc"]),
            aliases=tuple(sorted(aliases)),
            capability_manifest_identity=str(
                row["capability_manifest_sha256"]
            ),
            context_bound=context_bound,
            chat_template_present=chat_template_present,
            tool_calling_state=tool_calling_state,
            structured_output_state=structured_output_state,
            parallel_tool_calling_state=parallel_tool_calling_state,
            streaming_state=streaming_state,
        )

    async def resolve(self, reference: str) -> ModelSnapshot:
        try:
            result = await self.registry.resolve_public_model(reference)
        except ModelRegistryError as exc:
            raise SystemXError(
                503,
                "system_x_backend_unavailable",
                "Model registry is unavailable",
                retryable=True,
            ) from exc
        resolution = result.get("resolution")
        if resolution == "not_found":
            configured_default = str(
                getattr(
                    getattr(self.registry, "settings", None),
                    "registry_default_alias",
                    "default",
                )
            )
            if reference == configured_default:
                raise SystemXError(
                    503,
                    "system_x_no_ready_model",
                    "No READY model is currently available",
                    retryable=True,
                )
            raise SystemXError(
                404,
                "system_x_model_not_found",
                "Requested model was not found",
            )
        if resolution == "unavailable":
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Requested model is unavailable",
                retryable=True,
            )
        row = result.get("model")
        generation = result.get("registry_generation")
        if resolution != "ready" or not isinstance(row, dict) or type(generation) is not int:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Requested model is unavailable",
                retryable=True,
            )
        return self._snapshot(reference, generation, row)

    async def verify(self, snapshot: ModelSnapshot) -> bool:
        try:
            return await self.registry.public_model_snapshot_matches(
                snapshot.requested_reference,
                snapshot.verification_fields(),
            )
        except ModelRegistryError:
            return False

    async def _runtime_states(self) -> dict[str, str]:
        try:
            inventory = await self.backend.current_router_inventory()
        except (BackendError, RuntimeError):
            return {}
        result: dict[str, str] = {}
        for model in inventory.models:
            if model.status in {"loaded", "sleeping"}:
                state = "loaded"
            elif model.status == "loading":
                state = "loading"
            elif model.status == "unloaded":
                state = "unloaded"
            else:
                state = "unavailable"
            result[model.model_id] = state
        return result

    def mark_operation_proven(
        self, public_model_id: str, operation: Operation
    ) -> None:
        self._proven_operations.setdefault(public_model_id, set()).add(operation)

    async def mark_capability_proven(
        self,
        public_model_id: str,
        capability: CapabilityName,
        request_id: str,
        service_transaction_id: str,
        router_transaction_id: str,
        observed_protocol_surfaces: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        try:
            return await self.registry.record_runtime_capability(
                public_model_id,
                capability,
                request_id,
                service_transaction_id,
                router_transaction_id,
                observed_protocol_surfaces,
            )
        except ModelRegistryError as exc:
            raise SystemXError(
                503,
                "system_x_backend_unavailable",
                "Capability evidence could not be persisted",
                retryable=True,
            ) from exc

    @staticmethod
    def require_capability(
        snapshot: ModelSnapshot,
        capability: CapabilityName,
    ) -> None:
        state = (
            snapshot.tool_calling_state
            if capability == "tool_calling"
            else snapshot.structured_output_state
        )
        if state == "unavailable":
            code = (
                "system_x_tool_capability_unavailable"
                if capability == "tool_calling"
                else "system_x_capability_unavailable"
            )
            raise SystemXError(
                503,
                code,
                "Requested model capability is unavailable",
            )

    def _capabilities(self, snapshot: ModelSnapshot) -> ModelCapabilities:
        proven = self._proven_operations.get(snapshot.public_model_id, set())
        return ModelCapabilities(
            generate="available" if "generate" in proven else "not_tested",
            chat=(
                "unavailable"
                if not snapshot.chat_template_present
                else "available"
                if "chat" in proven
                else "not_tested"
            ),
            responses="available" if "responses" in proven else "not_tested",
            token_count="available" if "token_count" in proven else "not_tested",
            tool_calling=snapshot.tool_calling_state,
            structured_output=snapshot.structured_output_state,
            parallel_tool_calling=snapshot.parallel_tool_calling_state,
            streaming=snapshot.streaming_state,
        )

    def _public_record(
        self,
        snapshot: ModelSnapshot,
        runtime_states: dict[str, str],
        *,
        public_id: str | None = None,
        aliases: list[str] | None = None,
    ) -> PublicModelRecord:
        runtime_state = runtime_states.get(snapshot.router_model_id, "unavailable")
        return PublicModelRecord(
            id=public_id or snapshot.public_model_id,
            aliases=list(snapshot.aliases) if aliases is None else aliases,
            branch="gguf",
            registration_state="ready",
            runtime_state=runtime_state,
            capabilities=self._capabilities(snapshot),
        )

    @staticmethod
    def _positive_integer(value: object) -> int | None:
        return value if type(value) is int and value > 0 else None

    async def request_limits(
        self, snapshot: ModelSnapshot
    ) -> tuple[int | None, int | None]:
        """Return active context and maximum-output limits without content."""
        properties = await self.backend.active_model_properties(
            snapshot.router_model_id
        )
        defaults = (
            properties.get("default_generation_settings")
            if isinstance(properties, dict)
            else None
        )
        if not isinstance(defaults, dict):
            defaults = {}
        active_context = self._positive_integer(defaults.get("n_ctx"))
        maximum_output = self._positive_integer(defaults.get("max_tokens"))
        if maximum_output is None:
            maximum_output = self._positive_integer(defaults.get("n_predict"))
        return active_context or snapshot.context_bound, maximum_output
    async def _planner_detail(
        self,
        snapshot: ModelSnapshot,
        runtime_states: dict[str, str],
    ) -> PublicModelDetail:
        runtime_state = runtime_states.get(
            snapshot.router_model_id, "unavailable"
        )
        props = await self.backend.active_model_properties(
            snapshot.router_model_id
        )
        defaults: dict[str, Any] = {}
        if isinstance(props, dict):
            candidate = props.get("default_generation_settings")
            if isinstance(candidate, dict):
                defaults = candidate
        active_context = self._positive_integer(defaults.get("n_ctx"))
        maximum_output = self._positive_integer(
            defaults.get("max_tokens")
        ) or self._positive_integer(defaults.get("n_predict"))
        temperature_value = defaults.get("temperature")
        temperature = (
            float(temperature_value)
            if type(temperature_value) in {int, float}
            and 0.0 <= float(temperature_value) <= 2.0
            else None
        )
        chat_template = (
            props.get("chat_template") if isinstance(props, dict) else None
        )
        if not isinstance(chat_template, str):
            chat_template = ""
        supports_reasoning_output = (
            snapshot.chat_template_present
            and any(
                marker in chat_template
                for marker in (
                    "enable_thinking",
                    "<think>",
                    "reasoning_content",
                )
            )
        )
        supports_template_control = (
            supports_reasoning_output and "enable_thinking" in chat_template
        )
        basic = self._capabilities(snapshot)
        requested_alias = (
            snapshot.requested_reference
            if snapshot.requested_reference in snapshot.aliases
            else None
        )
        effective_context = active_context or snapshot.context_bound
        return PublicModelDetail(
            public_model_id=snapshot.public_model_id,
            requested_alias=requested_alias,
            resolved_model_id=snapshot.public_model_id,
            artifact_version_id=snapshot.bundle_id,
            state="ready",
            runtime_state=runtime_state,
            capabilities=PlannerModelCapabilities(
                chat=basic.chat,
                completion=basic.generate,
                responses=basic.responses,
                streaming=basic.streaming,
                tool_calling=basic.tool_calling,
                structured_output=basic.structured_output,
                reasoning_output=(
                    "available"
                    if supports_reasoning_output
                    else "not_tested"
                    if snapshot.chat_template_present
                    else "unavailable"
                ),
                reasoning_control=(
                    "available"
                    if supports_template_control
                    else "unavailable"
                ),
            ),
            reasoning_formats=(
                ["reasoning_content", "canonical_reasoning_delta"]
                if supports_reasoning_output
                else []
            ),
            reasoning_control_modes=(
                ["template_toggle"] if supports_template_control else []
            ),
            context_window_tokens=effective_context,
            model_declared_context_window_tokens=snapshot.context_bound,
            active_context_window_tokens=active_context,
            maximum_output_tokens=maximum_output,
            default_generation=PlannerDefaultGeneration(
                temperature=temperature
            ),
            token_count_operations=TokenCountOperations(
                system_x_native="available",
                messages_compatible="available",
                openai_compatible="available",
            ),
        )

    async def reasoning_template_control_available(
        self, snapshot: ModelSnapshot
    ) -> bool:
        props = await self.backend.active_model_properties(
            snapshot.router_model_id
        )
        chat_template = (
            props.get("chat_template") if isinstance(props, dict) else None
        )
        return (
            isinstance(chat_template, str)
            and "enable_thinking" in chat_template
        )

    async def _ready_snapshots(self) -> tuple[int, list[ModelSnapshot]]:
        try:
            result = await self.registry.public_model_rows()
        except ModelRegistryError as exc:
            raise SystemXError(
                503,
                "system_x_backend_unavailable",
                "Model registry is unavailable",
                retryable=True,
            ) from exc
        generation = result.get("registry_generation")
        rows = result.get("models")
        if type(generation) is not int or not isinstance(rows, list):
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Model catalogue is unavailable",
            )
        snapshots = [
            self._snapshot(str(row["model_version_id"]), generation, row)
            for row in rows
            if isinstance(row, dict)
        ]
        if len(snapshots) != len(rows):
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Model catalogue is unavailable",
            )
        return generation, sorted(
            snapshots,
            key=lambda item: item.public_model_id,
        )

    async def compatibility_snapshots(self) -> list[ModelSnapshot]:
        """Expose only immutable public catalogue evidence to local adapters."""

        _, snapshots = await self._ready_snapshots()
        return snapshots

    async def list_models(self, request_id: str) -> ModelListResponse:
        generation, snapshots = await self._ready_snapshots()
        runtime_states = await self._runtime_states()
        response = ModelListResponse(
            request_id=request_id,
            registry_generation=generation,
            models=[
                record
                for snapshot in snapshots
                for record in (
                    (
                        self._public_record(
                            snapshot,
                            runtime_states,
                            public_id="default",
                            aliases=["default"],
                        ),
                    )
                    if "default" in snapshot.aliases
                    else ()
                )
                + (self._public_record(snapshot, runtime_states),)
            ],
        )
        self.operations.note_terminal(
            request_id,
            state="completed",
            finish_reason=None,
        )
        return response

    async def model_detail(
        self, request_id: str, reference: str
    ) -> ModelDetailResponse:
        snapshot = await self.resolve(reference)
        self.operations.note_model(
            request_id,
            snapshot.public_model_id,
            snapshot.bundle_id,
        )
        runtime_states = await self._runtime_states()
        response = ModelDetailResponse(
            request_id=request_id,
            registry_generation=snapshot.registry_generation,
            model=await self._planner_detail(snapshot, runtime_states),
        )
        self.operations.note_terminal(
            request_id,
            state="completed",
            finish_reason=None,
        )
        return response
