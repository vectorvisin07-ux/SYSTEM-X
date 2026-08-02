"""Authoritative model-service lifecycle resolution.

The model-service axis is deliberately independent from process startup and
recovery transaction phases.  Callers provide current authoritative evidence;
this module applies the single deterministic precedence used by the API,
supervisor, and recovery classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelServiceState(StrEnum):
    WAITING_FOR_MODEL = "WAITING_FOR_MODEL"
    MODEL_CANDIDATE_LOADING = "MODEL_CANDIDATE_LOADING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class ModelLifecycleEvidence:
    desired_state: str = "RUNNING"
    fail_closed_latch: bool = False
    ownership_uncertain: bool = False
    control_plane_operational: bool = True
    registry_available: bool = True
    configured_default_alias: str = "default"
    resolved_default_alias: str | None = None
    resolved_public_model_id: str | None = None
    default_target_ready: bool = False
    warm_identity_present: bool = False
    exact_target_warm_healthy: bool = False
    ready_public_model_count: int = 0
    candidate_model_count: int = 0

    def __post_init__(self) -> None:
        if self.ready_public_model_count < 0:
            raise ValueError("ready_public_model_count must be non-negative")
        if self.candidate_model_count < 0:
            raise ValueError("candidate_model_count must be non-negative")
        if not self.configured_default_alias:
            raise ValueError("configured_default_alias must not be empty")


@dataclass(frozen=True, slots=True)
class ModelLifecycleResolution:
    state: ModelServiceState
    reason_code: str
    service_available: bool
    inference_ready: bool
    model_expected: bool
    recovery_eligible: bool


def resolve_model_service_state(
    evidence: ModelLifecycleEvidence,
) -> ModelLifecycleResolution:
    """Resolve current model-service state with STOPPED as strongest state."""

    if evidence.desired_state == "STOPPED":
        return _resolution(ModelServiceState.STOPPED, "STOPPED", False, False)

    if evidence.fail_closed_latch or evidence.ownership_uncertain:
        return _resolution(
            ModelServiceState.FAIL_CLOSED,
            "OWNERSHIP_UNCERTAIN"
            if evidence.ownership_uncertain
            else "FAIL_CLOSED",
            False,
            False,
        )

    if not evidence.control_plane_operational:
        return _resolution(
            ModelServiceState.DEGRADED,
            "CONTROL_PLANE_NOT_OPERATIONAL",
            False,
            _model_expected(evidence),
        )

    if not evidence.registry_available:
        return _resolution(
            ModelServiceState.DEGRADED,
            "REGISTRY_UNAVAILABLE",
            False,
            _model_expected(evidence),
        )

    if _conflicting_alias_evidence(evidence):
        return _resolution(
            ModelServiceState.FAIL_CLOSED,
            "CONFLICTING_MODEL_EVIDENCE",
            False,
            _model_expected(evidence),
        )

    if (
        evidence.default_target_ready
        and evidence.resolved_default_alias == evidence.configured_default_alias
        and evidence.resolved_public_model_id is not None
        and evidence.exact_target_warm_healthy
    ):
        return _resolution(ModelServiceState.READY, "OK", True, True)

    if (
        evidence.resolved_default_alias is not None
        or evidence.warm_identity_present
        or evidence.default_target_ready
    ):
        return _resolution(
            ModelServiceState.DEGRADED,
            "EXPECTED_MODEL_NOT_READY",
            False,
            _model_expected(evidence),
        )

    if evidence.ready_public_model_count:
        return _resolution(
            ModelServiceState.DEGRADED,
            "READY_MODEL_WITHOUT_DEFAULT",
            False,
            False,
        )

    if evidence.candidate_model_count:
        return _resolution(
            ModelServiceState.MODEL_CANDIDATE_LOADING,
            "MODEL_CANDIDATE_LOADING",
            True,
            False,
        )

    return _resolution(
        ModelServiceState.WAITING_FOR_MODEL,
        "NO_READY_MODEL",
        True,
        False,
    )


def _model_expected(evidence: ModelLifecycleEvidence) -> bool:
    """Return the strict recovery expectation predicate."""

    return evidence.warm_identity_present or (
        evidence.default_target_ready
        and evidence.resolved_default_alias == evidence.configured_default_alias
        and evidence.resolved_public_model_id is not None
    )


def _conflicting_alias_evidence(evidence: ModelLifecycleEvidence) -> bool:
    return (evidence.resolved_default_alias is None) != (
        evidence.resolved_public_model_id is None
    )


def _resolution(
    state: ModelServiceState,
    reason_code: str,
    service_available: bool,
    model_expected: bool,
) -> ModelLifecycleResolution:
    ready = state is ModelServiceState.READY
    return ModelLifecycleResolution(
        state=state,
        reason_code=reason_code,
        service_available=service_available,
        inference_ready=ready,
        model_expected=model_expected,
        recovery_eligible=(
            state is ModelServiceState.DEGRADED and model_expected
        ),
    )
