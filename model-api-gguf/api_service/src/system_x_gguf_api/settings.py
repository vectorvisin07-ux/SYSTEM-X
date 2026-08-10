"""Typed, non-secret configuration for the System X GGUF API service."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ServiceSettings(BaseModel):
    """The sole public-service and private-router configuration model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment_prefix: ClassVar[str] = "SYSTEM_X_GGUF_API_"

    service_name: str = Field(
        default="system-x-gguf-api",
        min_length=1,
        max_length=128,
    )
    service_version: str = Field(default="0.13.0", min_length=1, max_length=64)
    privacy_diagnostic_mode: Literal["off", "metadata"] = "off"
    contract_version: str = Field(
        default="system-x.gguf-api.native-inference.v1",
        min_length=1,
        max_length=128,
    )
    authentication_enabled: bool = True
    request_max_body_bytes: int = Field(default=2_097_152, ge=1024, le=16_777_216)
    request_max_total_tokens: int = Field(default=32_768, ge=1, le=1_048_576)
    request_timeout_seconds: float = Field(default=120.0, ge=0.1, le=3600.0)
    request_concurrency_limit_per_key: int = Field(default=2, ge=1, le=64)
    request_rate_limit_requests_per_key: int = Field(default=60, ge=1, le=100_000)
    request_rate_limit_window_seconds: float = Field(default=60.0, ge=0.1, le=86_400.0)
    public_host: str = Field(default="127.0.0.1")
    public_port: int | None = Field(default=None, ge=1, le=65535)
    private_backend_host: str = Field(default="127.0.0.1")
    private_backend_port: int | None = Field(default=None, ge=1, le=65535)
    private_backend_enabled: bool = False
    private_backend_models_max: int = Field(default=1, ge=1, le=1)
    private_backend_start_timeout_seconds: float = Field(
        default=30.0, gt=0.0, le=120.0
    )
    private_backend_model_timeout_seconds: float = Field(
        default=120.0, gt=0.0, le=300.0
    )
    private_backend_inference_timeout_seconds: float = Field(
        default=900.0, gt=0.0, le=3600.0
    )
    private_backend_poll_interval_seconds: float = Field(
        default=0.25, ge=0.05, le=5.0
    )
    registry_enabled: bool = False
    registry_reconcile_interval_seconds: float = Field(
        default=30.0, ge=5.0, le=3600.0
    )
    registry_watch_debounce_milliseconds: int = Field(
        default=1600, ge=100, le=10_000
    )
    registry_stability_samples: int = Field(default=3, ge=2, le=10)
    registry_stability_interval_seconds: float = Field(
        default=1.0, ge=0.1, le=10.0
    )
    registry_database_busy_timeout_milliseconds: int = Field(
        default=5000, ge=100, le=60_000
    )
    registry_default_alias: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$",
    )
    startup_model_policy: Literal["always_warm", "router_control", "registry_control", "api_only"] = "always_warm"
    automatic_recovery_enabled: bool = False
    recovery_delay_initial_seconds: float = Field(
        default=0.25, ge=0.0, le=3600.0
    )
    recovery_delay_maximum_seconds: float = Field(
        default=30.0, gt=0.0, le=3600.0
    )
    recovery_delay_multiplier: float = Field(
        default=2.0, ge=1.0, le=16.0
    )
    recovery_maximum_attempts_in_window: int = Field(
        default=3, ge=1, le=16
    )
    recovery_attempt_window_seconds: float = Field(
        default=60.0, ge=1.0, le=3600.0
    )
    recovery_stable_reset_seconds: float = Field(
        default=30.0, ge=1.0, le=3600.0
    )
    service_control_profile_identity: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    service_control_desired_state_path: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )
    external_static_enabled: bool = False
    external_static_distribution_root: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )
    external_static_mount_path: str = Field(
        default="/ui/chat",
        min_length=2,
        max_length=128,
        pattern=r"^/[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*$",
    )
    tool_max_definitions: int = Field(default=20, ge=20, le=20)
    tool_max_calls_per_turn: int = Field(default=8, ge=8, le=8)
    tool_schema_max_bytes: int = Field(default=65_536, ge=65_536, le=65_536)
    tool_schema_max_aggregate_bytes: int = Field(
        default=262_144, ge=262_144, le=262_144
    )
    tool_schema_max_depth: int = Field(default=16, ge=16, le=16)
    tool_schema_max_properties: int = Field(default=256, ge=256, le=256)
    tool_schema_max_enum_members: int = Field(default=256, ge=256, le=256)
    tool_result_max_bytes: int = Field(
        default=262_144, ge=262_144, le=262_144
    )
    tool_result_max_aggregate_bytes: int = Field(
        default=1_048_576, ge=1_048_576, le=1_048_576
    )
    docs_enabled: bool = True
    environment_name: str = Field(
        default="router-integration",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
    )

    @field_validator("service_name", "service_version", "contract_version")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("public_host")
    @classmethod
    def validate_public_loopback(cls, value: str) -> str:
        address = ipaddress.ip_address(value)
        if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
            raise ValueError("public_host must be an IPv4 loopback address")
        return address.compressed

    @field_validator("private_backend_host")
    @classmethod
    def validate_private_backend_ip(cls, value: str) -> str:
        address = ipaddress.ip_address(value)
        if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
            raise ValueError("private_backend_host must be an IPv4 loopback address")
        return address.compressed

    @field_validator("registry_default_alias", mode="before")
    @classmethod
    def normalize_registry_alias(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("external_static_distribution_root")
    @classmethod
    def validate_external_static_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip() != value or "\x00" in value or not value.startswith("/"):
            raise ValueError(
                "external_static_distribution_root must be an absolute "
                "NUL-free path without surrounding whitespace"
            )
        return value

    @field_validator("service_control_desired_state_path")
    @classmethod
    def validate_desired_state_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip() != value or "\x00" in value or not value.startswith("/"):
            raise ValueError(
                "service_control_desired_state_path must be an absolute "
                "NUL-free path without surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_private_backend_relationships(self) -> "ServiceSettings":
        if self.private_backend_models_max != 1:
            raise ValueError("private_backend_models_max must equal 1")
        if self.private_backend_enabled:
            if self.private_backend_port is None:
                raise ValueError(
                    "private_backend_port is required when private backend is enabled"
                )
            if self.public_port is None:
                raise ValueError(
                    "public_port is required when private backend is enabled"
                )
            if self.public_port == self.private_backend_port:
                raise ValueError("public and private backend ports must differ")
        if self.registry_enabled and not self.private_backend_enabled:
            raise ValueError(
                "private_backend_enabled must be true when registry is enabled"
            )
        if self.startup_model_policy == "registry_control":
            if not self.private_backend_enabled:
                raise ValueError(
                    "private_backend_enabled must be true for registry_control"
                )
            if not self.registry_enabled:
                raise ValueError(
                    "registry_enabled must be true for registry_control"
                )
            if self.automatic_recovery_enabled:
                raise ValueError(
                    "automatic_recovery_enabled must be false for registry_control"
                )
        if self.startup_model_policy == "router_control":
            if not self.private_backend_enabled:
                raise ValueError(
                    "private_backend_enabled must be true for router_control"
                )
            if self.registry_enabled:
                raise ValueError(
                    "registry_enabled must be false for router_control"
                )
            if self.automatic_recovery_enabled:
                raise ValueError(
                    "automatic_recovery_enabled must be false for router_control"
                )
        if (
            self.external_static_enabled
            and self.external_static_distribution_root is None
        ):
            raise ValueError(
                "external_static_distribution_root is required when "
                "external static serving is enabled"
            )
        if (
            self.recovery_delay_initial_seconds
            > self.recovery_delay_maximum_seconds
        ):
            raise ValueError(
                "recovery initial delay must not exceed maximum delay"
            )
        if self.automatic_recovery_enabled and (
            self.service_control_profile_identity is None
            or self.service_control_desired_state_path is None
        ):
            raise ValueError(
                "automatic recovery requires profile identity and desired-state path"
            )
        if self.request_timeout_seconds > self.private_backend_inference_timeout_seconds:
            raise ValueError(
                "request_timeout_seconds must not exceed private backend inference timeout"
            )
        return self

    @classmethod
    def from_environment(
        cls,
        base: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ServiceSettings":
        """Validate base values after applying explicitly prefixed overrides."""

        values: dict[str, Any] = dict(base or {})
        source = os.environ if environ is None else environ
        field_names = set(cls.model_fields)
        for key, value in source.items():
            if not key.startswith(cls.environment_prefix):
                continue
            field_name = key[len(cls.environment_prefix) :].lower()
            if field_name not in field_names:
                raise ValueError(f"unknown configuration environment field: {key}")
            if field_name in {
                "authentication_enabled",
                "docs_enabled",
                "private_backend_enabled",
                "registry_enabled",
                "external_static_enabled",
                "automatic_recovery_enabled",
            }:
                normalized_boolean = value.strip().lower()
                if normalized_boolean in {"1", "true", "yes", "on"}:
                    values[field_name] = True
                    continue
                if normalized_boolean in {"0", "false", "no", "off"}:
                    values[field_name] = False
                    continue
            values[field_name] = value
        return cls.model_validate(values)
