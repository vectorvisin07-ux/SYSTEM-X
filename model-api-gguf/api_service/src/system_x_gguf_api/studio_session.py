"""Ephemeral same-origin browser sessions; raw secrets never persist."""
from __future__ import annotations
import hashlib,hmac,secrets,time
from dataclasses import dataclass
@dataclass(frozen=True)
class Session:
    session_id:str; csrf:str; digest:str; expires:float
class StudioSessionBroker:
    def __init__(self,ttl:int=300):self.ttl=ttl;self._boot={};self._sessions={}
    @staticmethod
    def digest(value:str)->str:return hashlib.sha256(value.encode()).hexdigest()
    def mint(self)->str:
        token=secrets.token_urlsafe(32);self._boot[self.digest(token)]=time.monotonic()+self.ttl;return token
    def exchange(self,token:str,origin:str,host:str)->Session:
        if origin not in {f"http://{host}",f"https://{host}"}:raise ValueError("origin rejected")
        expiry=self._boot.pop(self.digest(token),0)
        if expiry<time.monotonic():raise ValueError("bootstrap rejected")
        sid=secrets.token_urlsafe(32);s=Session(sid,secrets.token_urlsafe(24),self.digest(sid),time.monotonic()+self.ttl);self._sessions[s.digest]=s;return s
    def valid(self,sid:str,csrf:str)->bool:
        s=self._sessions.get(self.digest(sid));return bool(s and s.expires>time.monotonic() and hmac.compare_digest(s.csrf,csrf))
    def has_session(self,sid:str)->bool:
        s=self._sessions.get(self.digest(sid));return bool(s and s.expires>time.monotonic())
    def revoke(self,sid:str)->None:self._sessions.pop(self.digest(sid),None)
