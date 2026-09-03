class ControlPlaneError(Exception):
    def __init__(self, reason_code: str, message: str = "control-plane rejection"):
        super().__init__(message); self.reason_code = reason_code
class SchemaError(ControlPlaneError): pass
class CapabilityError(ControlPlaneError): pass
class StateTransitionError(ControlPlaneError): pass
class IdempotencyConflict(ControlPlaneError): pass
