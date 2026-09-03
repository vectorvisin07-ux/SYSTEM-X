from __future__ import annotations
import json,os,socket,struct
from pathlib import Path
MAX_FRAME=64*1024
class ControlSocket:
    def __init__(self,path:Path):self.path=Path(path)
    def bind(self):
        self.path.parent.mkdir(parents=True,exist_ok=True); os.chmod(self.path.parent,0o700)
        if self.path.exists():self.path.unlink()
        s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(str(self.path)); os.chmod(self.path,0o600); s.listen(8); return s
    @staticmethod
    def validate_peer(conn:socket.socket,uid:int|None=None)->bool:
        if not hasattr(socket,"SO_PEERCRED"):return False
        _,peer_uid,_=struct.unpack("3i",conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12)); return peer_uid==(os.getuid() if uid is None else uid)
    @staticmethod
    def decode(frame:bytes)->dict:
        if len(frame)>MAX_FRAME:raise ValueError("FRAME_TOO_LARGE")
        value=json.loads(frame)
        if not isinstance(value,dict):raise ValueError("INVALID_FRAME")
        return value
