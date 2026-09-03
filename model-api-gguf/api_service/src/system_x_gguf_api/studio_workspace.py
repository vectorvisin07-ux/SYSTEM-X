"""Same-origin, session-scoped Studio workspace projections."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .model_catalogue import ModelCatalogue
from .model_registry import ModelRegistry
from .request_context import new_request_id, request_id_for
from .warm_model import WarmModelCoordinator

WORKSPACES = ("chat", "catalogue", "agent", "developer", "system")


def _session_ok(request: Request) -> bool:
    broker = getattr(request.app.state, "studio_sessions", None)
    sid = request.cookies.get("system_x_studio")
    csrf = request.headers.get("x-studio-csrf", "")
    return bool(broker and sid and broker.valid(sid, csrf))


def _session_required() -> JSONResponse:
    return JSONResponse({"error": "studio_session_required"}, status_code=401)

def _workspace_request_id(request: Request) -> str:
    return getattr(request.state, "system_x_request_id", None) or new_request_id()


def _model_json(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(model)


def build_workspace_router(
    catalogue: ModelCatalogue,
    registry: ModelRegistry,
    warm_model: WarmModelCoordinator,
) -> APIRouter:
    router = APIRouter(prefix="/ui/studio", tags=["studio"])

    @router.get("/workspaces")
    async def workspaces(request: Request):
        if not _session_ok(request):
            return _session_required()
        return {
            "workspaces": list(WORKSPACES),
            "origin": request.url.scheme + "://" + request.headers.get("host", ""),
        }

    @router.get("/catalogue")
    async def catalogue_view(request: Request):
        if not _session_ok(request):
            return _session_required()
        request_id = _workspace_request_id(request)
        generation, snapshots = await catalogue._ready_snapshots()
        runtime_states = await catalogue._runtime_states()
        models = [catalogue._public_record(s, runtime_states, public_id="default", aliases=["default"]) for s in snapshots if "default" in s.aliases]
        models += [catalogue._public_record(s, runtime_states) for s in snapshots]
        snapshot = await catalogue.resolve("default")
        detail = await catalogue._planner_detail(snapshot, runtime_states)
        return {
            "request_id": request_id,
            "models": [_model_json(model) for model in models],
            "default": "default",
            "current_model": _model_json(detail),
            "activation": "idempotent_no_op_for_current_model",
        }

    @router.post("/activation")
    async def activation(request: Request):
        if not _session_ok(request):
            return _session_required()
        body = await request.json()
        reference = body.get("model", "default") if isinstance(body, dict) else "default"
        if not isinstance(reference, str) or not reference.strip():
            return JSONResponse({"error": "model_reference_required"}, status_code=400)
        snapshot = await catalogue.resolve(reference)
        return {
            "request_id": _workspace_request_id(request),
            "requested_model": reference,
            "resolved_model_id": snapshot.public_model_id,
            "classification": "already_active",
            "idempotent": True,
            "deployment_delta": 0,
            "model_child_delta": 0,
            "default_alias": "default",
            "registry_write": False,
        }

    @router.get("/{workspace}")
    async def workspace(request: Request, workspace: str):
        if not _session_ok(request):
            return _session_required()
        if workspace not in WORKSPACES:
            return JSONResponse({"workspace": workspace, "available": False}, status_code=404)
        if workspace == "agent":
            snapshot = await catalogue.resolve("default")
            runtime_states = await catalogue._runtime_states()
            detail = await catalogue._planner_detail(snapshot, runtime_states)
            capabilities = _model_json(detail.capabilities)
            return {
                "workspace": workspace,
                "available": True,
                "capabilities": capabilities,
                "tool_submission_enabled": capabilities.get("tool_calling") == "available",
                "external_execution_count": 0,
            }
        if workspace == "developer":
            return {
                "workspace": workspace,
                "available": True,
                "protocols": ["system_x_native", "openai_compatible", "messages_compatible"],
                "same_origin_only": True,
                "external_origin_allowed": False,
                "private_topology_exposed": False,
                "secret_material_exposed": False,
            }
        if workspace == "system":
            warm = await warm_model.observe_once()
            backend = await request.app.state.backend.public_state()
            summary = await registry.public_summary()
            identity = warm.identity
            return {
                "workspace": workspace,
                "available": True,
                "installation_state": "INSTALLED",
                "service_state": "RUNNING" if backend.process_running else "STOPPED",
                "readiness_state": warm.service_readiness_state,
                "model_state": "READY" if identity is not None and backend.model_ready else "NOT_READY",
                "connection_state": "READY" if backend.control_plane_ready else "NOT_READY",
                "default_alias": warm.default_alias,
                "current_model_id": identity.resolved_public_model_id if identity else None,
                "model_child_count": backend.loaded_model_count,
                "registry_status": summary.registry_status,
                "ready_model_count": summary.ready_model_count,
                "private_topology_exposed": False,
                "credential_material_exposed": False,
            }
        return {
            "workspace": workspace,
            "available": True,
            "capabilities": ["read", "session"],
        }

    return router
