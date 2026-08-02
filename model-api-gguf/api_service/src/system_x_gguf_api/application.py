"""Side-effect-free construction and lifespan-managed private backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
import platform
from importlib.metadata import version as distribution_version
import sqlite3

from fastapi import FastAPI, Query, Request, Response

from .authentication import AuthenticationManager
from .authentication_openapi import apply_authentication_openapi
from .backend import BackendCoordinator
from .anthropic_adapter import AnthropicCompatibilityAdapter
from .anthropic_contract import (
    COMPATIBILITY_CONTRACT as ANTHROPIC_COMPATIBILITY_CONTRACT,
    COMPATIBILITY_VERSION as ANTHROPIC_COMPATIBILITY_VERSION,
)
from .anthropic_routes import build_anthropic_router
from .anthropic_stream import ANTHROPIC_STREAMING_CONTRACT
from .credential_store import CredentialStore
from .credential_types import AUTHENTICATION_CONTRACT
from .compatibility_models import build_compatibility_models_router
from .errors import SYSTEM_X_ERROR_RESPONSES, install_system_error_handling
from .external_static import configured_external_static
from .inference_service import InferenceService
from .model_catalogue import ModelCatalogue
from .model_registry import ModelRegistry
from .openai_adapter import OpenAICompatibilityAdapter
from .openai_contract import (
    COMPATIBILITY_CONTRACT,
    COMPATIBILITY_VERSION,
)
from .openai_routes import build_openai_router
from .openai_stream import OPENAI_STREAMING_CONTRACT
from .operation_records import OPERATION_RECORD_SCHEMA, OperationRecorder
from .registry_types import (
    MODEL_ADAPTATION_CONTRACT,
    REGISTRY_SCHEMA_IDENTITY,
    REGISTRY_SCHEMA_VERSION,
)
from .request_context import request_id_for
from .schemas import HealthDetail, HealthResponse, VersionResponse
from .settings import ServiceSettings
from .stream_control import ActiveStreamRegistry
from .streaming_inference import StreamingInferenceService
from .system_routes import build_system_router
from .system_stream import SYSTEM_X_STREAMING_CONTRACT
from .tool_contract import (
    AGENT_CLIENT_CONTRACT,
    ANTHROPIC_TOOL_EXTENSION,
    OPENAI_TOOL_EXTENSION,
    STRUCTURED_OUTPUT_CONTRACT,
)
from .warm_model import WarmModelCoordinator, full_readiness
from .runtime_recovery import RuntimeRecoveryCoordinator


def create_application(
    settings: ServiceSettings | None = None,
    *,
    credential_store: CredentialStore | None = None,
) -> FastAPI:
    """Construct the app; private lifecycle work begins only in ASGI lifespan."""

    active_settings = settings or ServiceSettings.from_environment()
    external_static = configured_external_static(active_settings)
    backend = BackendCoordinator(active_settings)
    registry = ModelRegistry(active_settings, backend)
    operations = OperationRecorder()
    catalogue = ModelCatalogue(registry, backend, operations)
    warm_model = WarmModelCoordinator(
        active_settings, catalogue, registry, backend
    )
    runtime_recovery = RuntimeRecoveryCoordinator(
        active_settings, backend, warm_model
    )
    inference = InferenceService(catalogue, backend, operations)
    compatibility = OpenAICompatibilityAdapter(catalogue, inference)
    anthropic_compatibility = AnthropicCompatibilityAdapter(catalogue, inference)
    active_streams = ActiveStreamRegistry(operations)
    streaming = StreamingInferenceService(inference, active_streams)
    credentials = credential_store or CredentialStore()
    authentication = AuthenticationManager(
        credentials,
        enabled=active_settings.authentication_enabled,
    )
    runtime_readiness = {"authentication_ready": False}

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        authentication.validate_startup()
        runtime_readiness["authentication_ready"] = True
        operations.startup()
        await backend.startup()
        try:
            try:
                await registry.startup()
                try:
                    await warm_model.startup(start_observer=False)
                    await runtime_recovery.startup()
                    try:
                        yield
                    finally:
                        await runtime_recovery.shutdown()
                finally:
                    await warm_model.shutdown()
            finally:
                try:
                    await active_streams.shutdown()
                finally:
                    await registry.shutdown()
        finally:
            runtime_readiness["authentication_ready"] = False
            try:
                await backend.shutdown()
            finally:
                operations.shutdown()

    application = FastAPI(
        title=active_settings.service_name,
        version=active_settings.service_version,
        description=(
            "System X GGUF API service with native and bounded local "
            "System X, OpenAI-compatible, and Messages-compatible streaming "
            "and non-streaming contracts over one private router adapter, "
            "protected by branch-local private API-key authentication."
        ),
        openapi_url="/openapi.json",
        docs_url="/docs" if active_settings.docs_enabled else None,
        redoc_url="/redoc" if active_settings.docs_enabled else None,
        swagger_ui_oauth2_redirect_url=None,
        lifespan=lifespan,
    )
    application.state.settings = active_settings
    application.state.backend = backend
    application.state.registry = registry
    application.state.catalogue = catalogue
    application.state.warm_model = warm_model
    application.state.runtime_recovery = runtime_recovery
    application.state.inference = inference
    application.state.compatibility = compatibility
    application.state.anthropic_compatibility = anthropic_compatibility
    application.state.active_streams = active_streams
    application.state.streaming = streaming
    application.state.credentials = credentials
    application.state.authentication = authentication
    application.state.operations = operations
    install_system_error_handling(application, authentication)

    @application.get(
        "/system/v1/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        responses=SYSTEM_X_ERROR_RESPONSES,
        tags=["system"],
    )
    async def system_health(
        request: Request,
        response: Response,
        detail: bool = Query(default=False),
    ) -> HealthResponse:
        warm_status = await warm_model.observe_once()
        backend_state = await backend.public_state()
        registry_summary = await registry.public_summary()
        readiness = full_readiness(
            warm_status,
            backend_state,
            registry_summary,
            authentication_ready=runtime_readiness[
                "authentication_ready"
            ],
            recovery_status=runtime_recovery.public_status,
        )
        response.status_code = (
            200 if readiness["service_available"] else 503
        )
        detail_value = None
        if detail:
            detail_value = HealthDetail(
                public_listener_configured=active_settings.public_port is not None,
            )
        return HealthResponse(
            request_id=request_id_for(request),
            service_name=active_settings.service_name,
            service_readiness_state=readiness[
                "service_readiness_state"
            ],
            model_service_state=readiness["model_service_state"],
            service_available=readiness["service_available"],
            inference_ready=readiness["inference_ready"],
            ready=readiness["ready"],
            service_status=readiness["service_status"],
            contract_version=active_settings.contract_version,
            backend_status=backend_state.status,
            backend_process_running=backend_state.process_running,
            backend_control_plane_ready=backend_state.control_plane_ready,
            loaded_model_count=backend_state.loaded_model_count,
            maximum_loaded_models=backend_state.maximum_loaded_models,
            model_ready=backend_state.model_ready,
            environment_name=active_settings.environment_name,
            registry_status=registry_summary.registry_status,
            registered_model_count=registry_summary.registered_model_count,
            ready_model_count=registry_summary.ready_model_count,
            candidate_model_count=registry_summary.candidate_model_count,
            rejected_artifact_count=registry_summary.rejected_artifact_count,
            registry_generation=registry_summary.registry_generation,
            last_reconcile_utc=registry_summary.last_reconcile_utc,
            default_alias=readiness["default_alias"],
            configured_default_alias=readiness[
                "configured_default_alias"
            ],
            resolved_default_alias=readiness[
                "resolved_default_alias"
            ],
            resolved_public_model_id=readiness[
                "resolved_public_model_id"
            ],
            artifact_version_id=readiness["artifact_version_id"],
            reason_code=readiness["reason_code"],
            warm_identity=readiness["warm_identity"],
            recovery_state=runtime_recovery.public_status[
                "recovery_state"
            ],
            recovery_reason_code=runtime_recovery.public_status[
                "primary_reason_code"
            ],
            recovery_attempt=runtime_recovery.public_status[
                "current_attempt"
            ],
            detail=detail_value,
        )

    @application.get(
        "/system/v1/version",
        response_model=VersionResponse,
        responses=SYSTEM_X_ERROR_RESPONSES,
        tags=["system"],
    )
    async def system_version(request: Request) -> VersionResponse:
        return VersionResponse(
            request_id=request_id_for(request),
            service_name=active_settings.service_name,
            service_version=active_settings.service_version,
            contract_version=active_settings.contract_version,
            authentication_contract=AUTHENTICATION_CONTRACT,
            authentication_enabled=active_settings.authentication_enabled,
            operation_record_contract=OPERATION_RECORD_SCHEMA,
            model_adaptation_contract=MODEL_ADAPTATION_CONTRACT,
            compatibility_version=COMPATIBILITY_VERSION,
            anthropic_compatibility_version=ANTHROPIC_COMPATIBILITY_VERSION,
            agent_client_contract=AGENT_CLIENT_CONTRACT,
            structured_output_contract=STRUCTURED_OUTPUT_CONTRACT,
            streaming_contract=SYSTEM_X_STREAMING_CONTRACT,
            openai_streaming_contract=OPENAI_STREAMING_CONTRACT,
            anthropic_streaming_contract=ANTHROPIC_STREAMING_CONTRACT,
            openai_tool_extension=OPENAI_TOOL_EXTENSION,
            anthropic_tool_extension=ANTHROPIC_TOOL_EXTENSION,
            python_version=platform.python_version(),
            fastapi_version=distribution_version("fastapi"),
            uvicorn_version=distribution_version("uvicorn"),
            pydantic_version=distribution_version("pydantic"),
            private_backend_adapter="llama-server-router",
            registry_schema_identity=REGISTRY_SCHEMA_IDENTITY,
            registry_schema_version=REGISTRY_SCHEMA_VERSION,
            watchfiles_version=distribution_version("watchfiles"),
            sqlite_library_version=sqlite3.sqlite_version,
        )

    application.include_router(
        build_system_router(catalogue, inference, streaming)
    )
    application.include_router(
        build_compatibility_models_router(
            compatibility, anthropic_compatibility
        )
    )
    application.include_router(build_openai_router(compatibility, streaming))
    application.include_router(
        build_anthropic_router(anthropic_compatibility, streaming)
    )
    if external_static is not None:
        external_static.install(application)
    original_openapi = application.openapi

    def compatibility_openapi():
        schema = original_openapi()
        compatibility_operations = (
            ("/v1/models", "get"),
            ("/v1/completions", "post"),
            ("/v1/chat/completions", "post"),
            ("/v1/responses", "post"),
        )
        anthropic_operations = (
            ("/v1/messages", "post"),
            ("/v1/messages/count_tokens", "post"),
        )
        header_contract = {
            "x-request-id": {
                "description": "System X request identity",
                "schema": {"type": "string"},
            },
            "X-System-X-Request-ID": {
                "description": "System X request identity",
                "schema": {"type": "string"},
            },
            "X-System-X-Compatibility-Version": {
                "description": "Compatibility contract identity",
                "schema": {
                    "type": "string",
                    "const": COMPATIBILITY_VERSION,
                },
            },
        }
        for path, method in compatibility_operations:
            responses = schema["paths"][path][method]["responses"]
            responses.pop("422", None)
            for response in responses.values():
                response.setdefault("headers", {}).update(header_contract)
        anthropic_header_contract = {
            "request-id": {
                "description": "Messages compatibility request identity",
                "schema": {"type": "string"},
            },
            "X-System-X-Request-ID": {
                "description": "System X request identity",
                "schema": {"type": "string"},
            },
            "X-System-X-Anthropic-Compatibility": {
                "description": "Messages compatibility contract identity",
                "schema": {
                    "type": "string",
                    "const": ANTHROPIC_COMPATIBILITY_VERSION,
                },
            },
        }
        for path, method in anthropic_operations:
            responses = schema["paths"][path][method]["responses"]
            responses.pop("422", None)
            for response in responses.values():
                response.setdefault("headers", {}).update(
                    anthropic_header_contract
                )
        streaming_operations = {
            "/system/v1/generate": SYSTEM_X_STREAMING_CONTRACT,
            "/system/v1/chat": SYSTEM_X_STREAMING_CONTRACT,
            "/system/v1/responses": SYSTEM_X_STREAMING_CONTRACT,
            "/v1/completions": OPENAI_STREAMING_CONTRACT,
            "/v1/chat/completions": OPENAI_STREAMING_CONTRACT,
            "/v1/responses": OPENAI_STREAMING_CONTRACT,
            "/v1/messages": ANTHROPIC_STREAMING_CONTRACT,
        }
        for path, contract in streaming_operations.items():
            successful = schema["paths"][path]["post"]["responses"]["200"]
            successful.setdefault("content", {})["text/event-stream"] = {
                "schema": {"type": "string"},
                "x-system-x-streaming-contract": contract,
            }
        schema["x-system-x-native-contract-version"] = (
            active_settings.contract_version
        )
        schema["x-system-x-compatibility-version"] = COMPATIBILITY_VERSION
        schema["x-system-x-openai-compatibility"] = COMPATIBILITY_CONTRACT
        schema["x-system-x-anthropic-compatibility-version"] = (
            ANTHROPIC_COMPATIBILITY_VERSION
        )
        schema["x-system-x-anthropic-compatibility"] = (
            ANTHROPIC_COMPATIBILITY_CONTRACT
        )
        schema["x-system-x-agent-client-contract"] = AGENT_CLIENT_CONTRACT
        schema["x-system-x-structured-output-contract"] = (
            STRUCTURED_OUTPUT_CONTRACT
        )
        schema["x-system-x-streaming-contract"] = SYSTEM_X_STREAMING_CONTRACT
        schema["x-system-x-openai-streaming-contract"] = (
            OPENAI_STREAMING_CONTRACT
        )
        schema["x-system-x-anthropic-streaming-contract"] = (
            ANTHROPIC_STREAMING_CONTRACT
        )
        schema["x-system-x-openai-tool-extension"] = OPENAI_TOOL_EXTENSION
        schema["x-system-x-anthropic-tool-extension"] = (
            ANTHROPIC_TOOL_EXTENSION
        )
        return apply_authentication_openapi(
            schema,
            enabled=active_settings.authentication_enabled,
        )

    application.openapi = compatibility_openapi
    return application


app = create_application()
