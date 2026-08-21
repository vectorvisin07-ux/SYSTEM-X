"""Bounded subprocess execution used only after an operation is invoked."""

from __future__ import annotations

import grp
import os
import pwd
import re
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NoReturn, Protocol, Sequence

from .errors import BootstrapError, ErrorCode


MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_ACCOUNT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.@-]{0,30}\$?$")
ELEVATION_EXECUTABLE = Path("/usr/bin/sudo")
SYSTEMCTL_EXECUTABLE = Path("/usr/bin/systemctl")
LOGINCTL_EXECUTABLE = Path("/usr/bin/loginctl")
PYTHON_EXECUTABLE = Path("/usr/bin/python3.14")
_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_WSL_DISTRIBUTION = "Ubuntu-26.04"
_WSL_INIT = Path("/init")
_WSL_CMD_EXECUTABLE = Path("/mnt/c/Windows/System32/cmd.exe")
_WSL_EXECUTABLE = Path("/mnt/c/Windows/System32/wsl.exe")
_WSL_WINDOWS_EXECUTABLE = r"C:\Windows\System32\wsl.exe"
_SAFE_WSL_TOKEN = re.compile(r"^[A-Za-z0-9_./:-]+$")


def _running_on_wsl() -> bool:
    try:
        kernel = Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return "microsoft" in kernel and _WSL_INIT.is_file()


def _validate_interop_executable(path: Path, purpose: str) -> None:
    if path.is_symlink() or not path.is_absolute():
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{purpose} is not a safe regular file")
    try:
        info = path.stat()
    except OSError as exc:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{purpose} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or not (info.st_mode & 0o111) or (info.st_mode & 0o022):
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{purpose} is not a trusted executable")


def _validated_wsl_distribution() -> str:
    observed = os.environ.get("WSL_DISTRO_NAME", _WSL_DISTRIBUTION)
    if observed != _WSL_DISTRIBUTION:
        raise BootstrapError(
            ErrorCode.AUTHORIZATION_REQUIRED,
            "WSL elevation route is bound to the active Ubuntu-26.04 distribution",
            context={"expected_distribution": _WSL_DISTRIBUTION, "observed_distribution": observed},
        )
    return observed


def _wsl_handoff_available() -> bool:
    if not _running_on_wsl():
        return False
    _validated_wsl_distribution()
    _validate_interop_executable(_WSL_INIT, "WSL init executable")
    _validate_interop_executable(_WSL_CMD_EXECUTABLE, "Windows command interop executable")
    _validate_interop_executable(_WSL_EXECUTABLE, "Windows WSL executable")
    return True


def _wsl_reconstruct_argv(
    repository_root: Path,
    entrypoint: Path,
    user: InstallationUser,
    *,
    allow_patch_difference: bool,
) -> tuple[str, ...]:
    distribution = _validated_wsl_distribution()
    values = (str(repository_root), str(entrypoint), user.name)
    if any(not _SAFE_WSL_TOKEN.fullmatch(value) for value in values):
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "WSL elevation route received an unsafe token")
    command = " ".join(
        (
            _WSL_WINDOWS_EXECUTABLE,
            "--distribution", distribution,
            "--user", "root",
            "--cd", str(repository_root),
            "--exec", str(PYTHON_EXECUTABLE),
            "-I", "-S", "-B", str(entrypoint),
            "reconstruct", "--authorize", "--install-user", user.name,
            *(("--allow-patch-difference",) if allow_patch_difference else ()),
        )
    )
    return (str(_WSL_INIT), str(_WSL_CMD_EXECUTABLE), "/d", "/s", "/c", command)


@dataclass(frozen=True, slots=True)
class InstallationUser:
    name: str
    uid: int
    gid: int
    home: Path
    groups: tuple[int, ...]
    xdg_runtime_dir: Path
    dbus_session_bus_address: str

    @classmethod
    def from_name(cls, name: str) -> "InstallationUser":
        if (
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or name == "root"
            or not _ACCOUNT_NAME.fullmatch(name)
        ):
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "installation user must be an explicit non-root account")
        try:
            record = pwd.getpwnam(name)
        except KeyError as exc:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "installation user does not exist", context={"user": name}) from exc
        if record.pw_uid == 0 or record.pw_gid == 0:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "installation user may not be root")
        home_input = Path(record.pw_dir)
        if not home_input.is_absolute() or home_input.is_symlink():
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "installation user home is not a safe absolute directory")
        home = home_input.resolve(strict=True)
        if not home.is_dir():
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "installation user home is not a directory")
        try:
            groups = tuple(sorted(set(os.getgrouplist(record.pw_name, record.pw_gid))))
        except OSError as exc:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "installation user supplementary groups are unavailable") from exc
        return cls(record.pw_name, record.pw_uid, record.pw_gid, home, groups, Path(f"/run/user/{record.pw_uid}"), f"unix:path=/run/user/{record.pw_uid}/bus")

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "uid": self.uid, "gid": self.gid, "home": str(self.home), "groups": list(self.groups), "xdg_runtime_dir": str(self.xdg_runtime_dir), "dbus_session_bus_address": self.dbus_session_bus_address}

    def validate_repository(self, repository_root: Path) -> None:
        if repository_root.is_symlink() or not repository_root.is_dir() or not repository_root.is_absolute():
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "repository root is not a safe absolute directory")
        root = repository_root.resolve(strict=True)
        if root.stat().st_uid != self.uid:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "repository is not owned by installation user", context={"uid": root.stat().st_uid, "expected_uid": self.uid})


def _validate_executable(path: Path, purpose: str) -> None:
    if not path.is_absolute():
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{purpose} must be absolute")
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{purpose} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or not (info.st_mode & 0o111):
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{purpose} is not a trusted executable")


def build_elevated_reconstruct_argv(
    repository_root: Path,
    user: InstallationUser,
    *,
    allow_patch_difference: bool = False,
) -> tuple[str, ...]:
    entrypoint = repository_root / "bootstrap" / "run_bootstrap.py"
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "reconstruct entrypoint is not a safe regular file")
    _validate_executable(PYTHON_EXECUTABLE, "product Python executable")
    if _wsl_handoff_available():
        return _wsl_reconstruct_argv(
            repository_root,
            entrypoint,
            user,
            allow_patch_difference=allow_patch_difference,
        )
    _validate_executable(ELEVATION_EXECUTABLE, "product-owned elevation executable")
    argv = [
        str(ELEVATION_EXECUTABLE), "-n", "--", str(PYTHON_EXECUTABLE),
        "-I", "-S", "-B", str(entrypoint), "reconstruct", "--authorize",
        "--install-user", user.name,
    ]
    if allow_patch_difference:
        argv.append("--allow-patch-difference")
    return tuple(argv)


def build_elevated_reconstruct_environment(user: InstallationUser) -> dict[str, str]:
    environment = {
        "PATH": _SAFE_PATH,
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(user.home),
        "USER": user.name,
        "LOGNAME": user.name,
        "XDG_RUNTIME_DIR": str(user.xdg_runtime_dir),
        "DBUS_SESSION_BUS_ADDRESS": user.dbus_session_bus_address,
    }
    if _running_on_wsl():
        environment["WSL_DISTRO_NAME"] = _validated_wsl_distribution()
    return environment


def _user_manager_environment(user: InstallationUser) -> dict[str, str]:
    return {
        "PATH": _SAFE_PATH,
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "HOME": str(user.home),
        "USER": user.name,
        "LOGNAME": user.name,
        "XDG_RUNTIME_DIR": str(user.xdg_runtime_dir),
        "DBUS_SESSION_BUS_ADDRESS": user.dbus_session_bus_address,
        "SYSTEMD_PAGER": "cat",
        "PAGER": "cat",
    }


def user_manager_ready(user: InstallationUser, runner: Runner) -> bool:
    with installation_user_context(user):
        result = runner(
            (str(SYSTEMCTL_EXECUTABLE), "--user", "is-system-running"),
            env=_user_manager_environment(user),
            timeout=10,
        )
    return result.returncode in (0, 1) and result.stdout.strip() in {"running", "degraded"}


def ensure_user_manager(user: InstallationUser, runner: Runner) -> dict[str, object]:
    if os.geteuid() != 0:
        raise BootstrapError(
            ErrorCode.AUTHORIZATION_REQUIRED,
            "user-manager recovery requires the product-owned root continuation",
        )
    if user_manager_ready(user, runner):
        return {"changed": False, "state": "ready", "install_uid": user.uid}
    root_environment = {
        "PATH": _SAFE_PATH,
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "XDG_RUNTIME_DIR": "",
        "DBUS_SESSION_BUS_ADDRESS": "",
        "SYSTEMD_PAGER": "cat",
        "PAGER": "cat",
    }
    terminate = runner(
        (str(LOGINCTL_EXECUTABLE), "terminate-user", user.name),
        env=root_environment,
        timeout=30,
    )
    if terminate.returncode not in (0, 1):
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-session termination failed",
            context={"exit_status": terminate.returncode},
        )
    runtime_unit = f"user-runtime-dir@{user.uid}.service"
    runtime_stop = runner(
        (str(SYSTEMCTL_EXECUTABLE), "stop", runtime_unit),
        env=root_environment,
        timeout=60,
    )
    if runtime_stop.returncode not in (0, 3):
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-runtime-dir stop failed",
            context={"exit_status": runtime_stop.returncode},
        )
    daemon_reexec = runner(
        (str(SYSTEMCTL_EXECUTABLE), "daemon-reexec"),
        env=root_environment,
        timeout=120,
    )
    if daemon_reexec.returncode != 0:
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product systemd daemon-reexec failed",
            context={"exit_status": daemon_reexec.returncode},
        )
    unit = f"user@{user.uid}.service"
    kill = runner(
        (str(SYSTEMCTL_EXECUTABLE), "kill", "--kill-who=all", "--signal=SIGKILL", unit),
        env=root_environment,
        timeout=60,
    )
    if kill.returncode not in (0, 1, 3, 5):
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-manager kill failed",
            context={"exit_status": kill.returncode},
        )
    stop = runner(
        (str(SYSTEMCTL_EXECUTABLE), "--force", "--force", "stop", unit),
        env=root_environment,
        timeout=60,
    )
    if stop.returncode not in (0, 3):
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-manager stop failed",
            context={"exit_status": stop.returncode},
        )
    slice_unit = f"user-{user.uid}.slice"
    slice_stop = runner(
        (str(SYSTEMCTL_EXECUTABLE), "stop", slice_unit),
        env=root_environment,
        timeout=60,
    )
    if slice_stop.returncode not in (0, 3):
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-slice stop failed",
            context={"exit_status": slice_stop.returncode},
        )
    reset = runner(
        (str(SYSTEMCTL_EXECUTABLE), "reset-failed", unit),
        env=root_environment,
        timeout=30,
    )
    if reset.returncode != 0:
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-manager reset failed",
            context={"exit_status": reset.returncode},
        )
    start = runner(
        (str(SYSTEMCTL_EXECUTABLE), "start", unit),
        env=root_environment,
        timeout=60,
    )
    if start.returncode != 0:
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            "product user-manager start failed",
            context={"exit_status": start.returncode},
        )
    for _ in range(80):
        if user_manager_ready(user, runner):
            return {
                "changed": True,
                "state": "ready",
                "install_uid": user.uid,
                "unit": unit,
            }
        import time
        time.sleep(0.25)
    raise BootstrapError(
        ErrorCode.EXTERNAL_COMMAND_FAILED,
        "product user-manager did not reach a real user bus",
        context={"unit": unit},
    )


def exec_elevated_reconstruct(
    repository_root: Path,
    user: InstallationUser,
    *,
    allow_patch_difference: bool = False,
) -> NoReturn:
    argv = build_elevated_reconstruct_argv(
        repository_root,
        user,
        allow_patch_difference=allow_patch_difference,
    )
    environment = build_elevated_reconstruct_environment(user)
    try:
        os.execve(argv[0], list(argv), environment)
    except OSError as exc:
        raise BootstrapError(
            ErrorCode.AUTHORIZATION_REQUIRED,
            "product-owned privilege handoff could not start",
            context={"program": argv[0], "error_type": type(exc).__name__},
        ) from exc
    raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "product-owned privilege handoff returned unexpectedly")


def resolve_installation_user(name: str | None) -> InstallationUser:
    effective_uid = os.geteuid()
    if effective_uid == 0 and not name:
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "elevated entry requires --install-user")
    if name is None:
        try:
            name = pwd.getpwuid(effective_uid).pw_name
        except KeyError as exc:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "effective user has no passwd entry") from exc
    user = InstallationUser.from_name(name)
    if effective_uid != 0 and (user.uid != effective_uid or user.gid != os.getegid()):
        raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "non-root entry may only use the current installation user")
    return user


@contextmanager
def installation_user_context(user: InstallationUser):
    saved_env = {key: os.environ.get(key) for key in ("HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")}
    updates = {"HOME": str(user.home), "USER": user.name, "LOGNAME": user.name, "XDG_RUNTIME_DIR": str(user.xdg_runtime_dir), "DBUS_SESSION_BUS_ADDRESS": user.dbus_session_bus_address}
    saved_ids = (os.getresuid(), os.getresgid(), tuple(os.getgroups()))
    switched = os.geteuid() == 0 and user.uid != 0 and os.getuid() != user.uid
    try:
        if switched:
            os.setgroups(list(user.groups))
            os.setresgid(user.gid, user.gid, 0)
            os.setresuid(user.uid, user.uid, 0)
        elif os.geteuid() not in (0, user.uid):
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "cannot enter installation-user identity")
        os.environ.update(updates)
        yield
    finally:
        for key, value in saved_env.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
        if switched:
            os.setresuid(0, 0, 0)
            os.setresgid(0, 0, 0)
            os.setgroups(list(saved_ids[2]))


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class Runner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 60,
    ) -> CommandResult: ...


class SubprocessRunner:
    """Shell-free subprocess runner with a closed environment overlay."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 60,
    ) -> CommandResult:
        if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
            raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "invalid command vector")
        command_env = os.environ.copy()
        command_env.update({
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        })
        if env:
            command_env.update(env)
        try:
            completed = subprocess.run(
                list(argv),
                cwd=cwd,
                env=command_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            return CommandResult(tuple(argv), 127, "", f"{type(exc).__name__}: {exc}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BootstrapError(
                ErrorCode.EXTERNAL_COMMAND_FAILED,
                "external command could not be executed",
                context={"program": argv[0], "error_type": type(exc).__name__},
            ) from exc
        if len(completed.stdout) > MAX_CAPTURE_BYTES or len(completed.stderr) > MAX_CAPTURE_BYTES:
            raise BootstrapError(
                ErrorCode.EXTERNAL_COMMAND_FAILED,
                "external command exceeded the bounded capture envelope",
                context={"program": argv[0]},
            )
        return CommandResult(
            tuple(argv),
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )


def require_success(result: CommandResult, *, purpose: str) -> CommandResult:
    if result.returncode != 0:
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            purpose,
            context={
                "program": result.argv[0],
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-1000:],
            },
        )
    return result
