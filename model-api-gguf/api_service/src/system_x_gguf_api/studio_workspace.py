"""Same-origin, session-scoped Studio workspace metadata."""
from fastapi import APIRouter, Request

WORKSPACES = ("chat", "catalogue", "agent", "developer", "system")

def build_workspace_router() -> APIRouter:
    router = APIRouter(prefix="/ui/studio", tags=["studio"])

    @router.get("/workspaces")
    async def workspaces(request: Request):
        return {"workspaces": list(WORKSPACES), "origin": request.url.scheme + "://" + request.headers.get("host", "")}

    @router.get("/catalogue")
    async def catalogue():
        return {"models": [], "default": "default", "activation": "session-scoped"}

    @router.post("/activation")
    async def activation(request: Request):
        body = await request.json()
        model = body.get("model", "default") if isinstance(body, dict) else "default"
        return {"model": model, "active": True, "scope": "session"}

    @router.get("/{workspace}")
    async def workspace(workspace: str):
        if workspace not in WORKSPACES:
            return {"workspace": workspace, "available": False}
        return {"workspace": workspace, "available": True, "capabilities": ["read", "session"]}
    return router
