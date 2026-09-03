"""Deterministic Python projection of control-command.schema.json."""
SCHEMA_VERSION = "system-x.control-command.v1"
OPERATION_TYPES = ("ActivateModel","SetDefaultAlias","StartService","StopService","RepairService","RepairRuntime","RotateCredential","CreateBackup","VerifyBackup","RestoreBackup","ApplyRelease","RollbackRelease","CreateBrowserSession","RevokeBrowserSession","RegisterWindowsEntry","RemoveWindowsEntry")
REQUIRED_FIELDS = ("schema_version","operation_id","idempotency_key","operation_type","actor_identity","capability_set","request_hash","expected_generation","target_identity","deadline","cancellation_policy","rollback_policy","created_at","correlation_id")
