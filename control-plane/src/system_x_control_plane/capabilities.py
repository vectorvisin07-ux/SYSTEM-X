from .errors import CapabilityError
CAPABILITIES=frozenset({"service.read","service.mutate","model.mutate","credential.rotate","backup.write","release.apply","browser.session","windows.entry","repair.apply"})
def authorize(granted:list[str]|set[str],required:str,*,actor:str)->None:
    if not actor or required not in CAPABILITIES or required not in set(granted): raise CapabilityError("CAPABILITY_DENIED")
def authorize_set(granted:list[str]|set[str],required:set[str],*,actor:str)->None:
    for item in required: authorize(granted,item,actor=actor)
