"""Fail-closed serving for one explicitly configured external distribution."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat

from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from .settings import ServiceSettings


MAXIMUM_STATIC_FILE_BYTES = 16 * 1024 * 1024
HASHED_ASSET = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "img-src 'self' data:; "
    "object-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'"
)


def _security_headers(*, cache_control: str, html: bool) -> dict[str, str]:
    headers = {
        "Cache-Control": cache_control,
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }
    if html:
        headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    return headers


def _read_regular_file(root: Path, relative: PurePosixPath) -> bytes | None:
    """Read through descriptor-relative no-follow opens for every component."""

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        parts = relative.parts
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            if not final:
                flags |= os.O_DIRECTORY
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except (FileNotFoundError, NotADirectoryError, OSError):
                return None
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAXIMUM_STATIC_FILE_BYTES
        ):
            return None
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("external static file changed during read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("external static file grew during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


class ExternalStaticDistribution:
    """Validated immutable identity of an external static distribution root."""

    def __init__(self, root_value: str, mount_path: str) -> None:
        supplied = Path(root_value)
        if not supplied.is_absolute():
            raise ValueError("external static distribution root must be absolute")
        try:
            metadata = supplied.lstat()
            resolved = supplied.resolve(strict=True)
        except OSError as exc:
            raise ValueError("external static distribution root is missing") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or resolved != supplied
        ):
            raise ValueError(
                "external static distribution root must be a physical directory "
                "without symlink traversal"
            )
        index = _read_regular_file(resolved, PurePosixPath("index.html"))
        if index is None:
            raise ValueError(
                "external static distribution root requires a regular index.html"
            )
        try:
            index.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("external static index.html must be UTF-8") from exc
        self.root = resolved
        self.mount_path = mount_path

    def install(self, application: FastAPI) -> None:
        mount = self.mount_path

        async def canonical_redirect(_request: Request) -> Response:
            return RedirectResponse(
                f"{mount}/",
                status_code=308,
                headers=_security_headers(cache_control="no-store", html=False),
            )

        async def index_response(_request: Request) -> Response:
            return self._response_for(PurePosixPath("index.html"), html=True)

        async def asset_response(
            request: Request,
            asset_path: str,
        ) -> Response:
            relative = self._validated_relative(asset_path)
            if relative is None:
                return self._not_found()
            exact = _read_regular_file(self.root, relative)
            if exact is not None:
                return self._response_for(relative, content=exact)
            accepts_html = "text/html" in request.headers.get("accept", "")
            if accepts_html and relative.suffix == "":
                return self._response_for(PurePosixPath("index.html"), html=True)
            return self._not_found()

        application.add_api_route(
            mount,
            canonical_redirect,
            methods=["GET"],
            include_in_schema=False,
            name="external_static_canonical_redirect",
        )
        application.add_api_route(
            f"{mount}/",
            index_response,
            methods=["GET"],
            include_in_schema=False,
            name="external_static_index",
        )
        application.add_api_route(
            f"{mount}/{{asset_path:path}}",
            asset_response,
            methods=["GET"],
            include_in_schema=False,
            name="external_static_asset",
        )

    @staticmethod
    def _validated_relative(value: str) -> PurePosixPath | None:
        if (
            value == ""
            or "\x00" in value
            or "\\" in value
            or len(value.encode("utf-8")) > 4096
        ):
            return None
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            return None
        return relative

    def _response_for(
        self,
        relative: PurePosixPath,
        *,
        content: bytes | None = None,
        html: bool = False,
    ) -> Response:
        value = content
        if value is None:
            value = _read_regular_file(self.root, relative)
        if value is None:
            return self._not_found()
        media_type = (
            "text/html"
            if html
            else mimetypes.guess_type(relative.name)[0]
            or "application/octet-stream"
        )
        cache_control = (
            "no-store"
            if html
            else (
                "public, max-age=31536000, immutable"
                if HASHED_ASSET.search(relative.name)
                else "no-cache"
            )
        )
        return Response(
            content=value,
            media_type=media_type,
            headers=_security_headers(
                cache_control=cache_control,
                html=html,
            ),
        )

    @staticmethod
    def _not_found() -> Response:
        return PlainTextResponse(
            "Not Found",
            status_code=404,
            headers=_security_headers(cache_control="no-store", html=False),
        )


def configured_external_static(
    settings: ServiceSettings,
) -> ExternalStaticDistribution | None:
    if not settings.external_static_enabled:
        return None
    root = settings.external_static_distribution_root
    if root is None:
        raise ValueError("external static distribution root is required")
    return ExternalStaticDistribution(root, settings.external_static_mount_path)
