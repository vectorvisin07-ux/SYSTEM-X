from .errors import CapabilityError
CAPABILITIES=frozenset({
    "model.read", "inference.execute", "status.read", "diagnostics.read",
    "model.activate", "alias.write", "service.read", "service.repair",
    "backup.create", "backup.verify", "backup.restore", "credential.rotate",
    "release.apply", "release.rollback", "browser_session.create",
    "browser_session.revoke", "windows_entry.write", "repair.apply",
})
def authorize(granted:list[str]|set[str],required:str,*,actor:str)->None:
    if not actor or required not in CAPABILITIES or required not in set(granted): raise CapabilityError("CAPABILITY_DENIED")
def authorize_set(granted:list[str]|set[str],required:set[str],*,actor:str)->None:
    for item in required: authorize(granted,item,actor=actor)
