from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from .errors import SchemaError
class OperationType(StrEnum):
    ACTIVATE_MODEL="ActivateModel"; SET_DEFAULT_ALIAS="SetDefaultAlias"; START_SERVICE="StartService"; STOP_SERVICE="StopService"; REPAIR_SERVICE="RepairService"; REPAIR_RUNTIME="RepairRuntime"; ROTATE_CREDENTIAL="RotateCredential"; CREATE_BACKUP="CreateBackup"; VERIFY_BACKUP="VerifyBackup"; RESTORE_BACKUP="RestoreBackup"; APPLY_RELEASE="ApplyRelease"; ROLLBACK_RELEASE="RollbackRelease"; CREATE_BROWSER_SESSION="CreateBrowserSession"; REVOKE_BROWSER_SESSION="RevokeBrowserSession"; REGISTER_WINDOWS_ENTRY="RegisterWindowsEntry"; REMOVE_WINDOWS_ENTRY="RemoveWindowsEntry"
REQUIRED={"schema_version","operation_id","idempotency_key","operation_type","actor_identity","capability_set","request_hash","expected_generation","target_identity","deadline","cancellation_policy","rollback_policy","created_at","correlation_id"}
ALLOWED=REQUIRED
@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    value:dict[str,Any]
    @classmethod
    def parse(cls,value:dict[str,Any])->"CommandEnvelope":
        if not isinstance(value,dict): raise SchemaError("INVALID_COMMAND","command must be object")
        if set(value)-ALLOWED: raise SchemaError("UNKNOWN_COMMAND_FIELD")
        if REQUIRED-set(value): raise SchemaError("MISSING_COMMAND_FIELD")
        if value["schema_version"] not in ("system-x.control-command.v1",1): raise SchemaError("UNSUPPORTED_SCHEMA_VERSION")
        if value["operation_type"] not in {x.value for x in OperationType}: raise SchemaError("INVALID_OPERATION_TYPE")
        if not isinstance(value["capability_set"],list) or not all(isinstance(x,str) for x in value["capability_set"]): raise SchemaError("INVALID_CAPABILITY_SET")
        if not isinstance(value["expected_generation"],int) or value["expected_generation"]<0: raise SchemaError("INVALID_GENERATION")
        if not isinstance(value["request_hash"],str) or len(value["request_hash"])!=64: raise SchemaError("INVALID_REQUEST_HASH")
        return cls(dict(value))
    @property
    def actor(self)->str:return self.value["actor_identity"]
    @property
    def key(self)->str:return self.value["idempotency_key"]
