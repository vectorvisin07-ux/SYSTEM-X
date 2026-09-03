from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import pytest
from system_x_control_plane.capabilities import authorize
from system_x_control_plane.commands import CommandEnvelope
from system_x_control_plane.errors import CapabilityError,IdempotencyConflict,SchemaError
from system_x_control_plane.journal import ControlStore
from system_x_control_plane.states import OperationState,transition

def command(key="k",hash_value="a"*64):return {"schema_version":"system-x.control-command.v1","operation_id":"operation-1234567890","idempotency_key":key,"operation_type":"RepairService","actor_identity":"operator","capability_set":["repair.apply"],"request_hash":hash_value,"expected_generation":0,"target_identity":"system-x","deadline":"2099-01-01T00:00:00Z","cancellation_policy":"retain","rollback_policy":"safe","created_at":"2026-01-01T00:00:00Z","correlation_id":None}
def test_strict_envelope_and_unknown_rejection():
    assert CommandEnvelope.parse(command()).actor=="operator"
    with pytest.raises(SchemaError):CommandEnvelope.parse(command()|{"extra":1})
def test_capability_denial():
    with pytest.raises(CapabilityError):authorize([],"repair.apply",actor="operator")
def test_durable_idempotency_and_conflict(tmp_path):
    s=ControlStore(tmp_path/"control.sqlite3"); first=s.put_operation(operation_id="operation-1234567890",actor_id="operator",idempotency_key="k",request_hash="a"*64,operation_type="RepairService",generation=0); assert not first["reused_existing_result"]
    assert s.put_operation(operation_id="other-operation",actor_id="operator",idempotency_key="k",request_hash="a"*64,operation_type="RepairService",generation=0)["reused_existing_result"]
    with pytest.raises(IdempotencyConflict):s.put_operation(operation_id="other-operation",actor_id="operator",idempotency_key="k",request_hash="b"*64,operation_type="RepairService",generation=0)
def test_state_matrix():
    assert transition(OperationState.REQUESTED,OperationState.AUTHORIZED)==OperationState.AUTHORIZED
    with pytest.raises(Exception):transition(OperationState.COMPLETED,OperationState.RUNNING)
def test_ipc_and_integrity(tmp_path):
    s=ControlStore(tmp_path/"control.sqlite3"); assert s.integrity()
