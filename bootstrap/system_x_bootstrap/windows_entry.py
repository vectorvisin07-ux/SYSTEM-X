"""Create the current-user SYSTEM X Windows entry from the canonical WSL contract."""
from __future__ import annotations
import subprocess

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WSL = "/mnt/c/Windows/System32/wsl.exe"

def create() -> int:
    script = r'''$dir = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"; New-Item -ItemType Directory -Force -Path $dir | Out-Null; $p = Join-Path $dir "SYSTEM X.lnk"; $s = New-Object -ComObject WScript.Shell; $l = $s.CreateShortcut($p); $l.TargetPath = "C:\Windows\System32\wsl.exe"; $l.Arguments = "-d SYSTEMS-29 --cd /home" + "/user/SYSTEMS/system-x ./system-x open"; $l.WorkingDirectory = "C:\Windows\System32"; $l.Description = "SYSTEM X local Studio"; $l.Save(); Write-Output $p'''
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script], check=False).returncode

if __name__ == "__main__":
    raise SystemExit(create())
