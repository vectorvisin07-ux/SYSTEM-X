from fastapi import APIRouter,Request,Response
from fastapi.responses import JSONResponse
from .authentication import AuthenticationManager
from .request_context import new_request_id
from .studio_session import StudioSessionBroker
def build_studio_router(broker:StudioSessionBroker, authentication:AuthenticationManager|None=None)->APIRouter:
    router=APIRouter(prefix="/ui/session")
    @router.post("/mint")
    async def mint(request:Request):
        """Mint a one-use browser bootstrap from an already authenticated caller."""
        if authentication is not None:
            request.state.system_x_request_id = new_request_id()
            denied=authentication.authenticate_request(request,"system")
            if denied is not None:
                return denied
        origin=request.headers.get("origin","")
        host=request.headers.get("host","")
        if not origin.startswith(("http://","https://")) or not host:
            return JSONResponse({"error":"invalid_origin"},status_code=400)
        return {"bootstrap":broker.mint(),"expires_in":60}
    @router.post("/exchange")
    async def exchange(request:Request,response:Response):
        body=await request.json()
        try:s=broker.exchange(body.get("bootstrap",""),request.headers.get("origin",""),request.headers.get("host",""))
        except (KeyError,TypeError,ValueError):return JSONResponse({"error":"session_exchange_rejected"},status_code=401)
        response.set_cookie("system_x_studio",s.session_id,httponly=True,samesite="strict",max_age=300,path="/");return {"csrf":s.csrf,"expires_in":300,"scope":["studio:read","studio:chat"]}
    @router.post("/revoke")
    async def revoke(request:Request,response:Response):
        if (sid:=request.cookies.get("system_x_studio")):broker.revoke(sid)
        response.delete_cookie("system_x_studio",path="/");return {"revoked":True}
    @router.get("/state")
    async def state(request:Request):
        sid=request.cookies.get("system_x_studio")
        csrf=broker.csrf_for(sid) if sid else None
        return {"authenticated":bool(sid and broker.has_session(sid)),"csrf":csrf,"scope":["studio:read","studio:chat"]}
    return router
