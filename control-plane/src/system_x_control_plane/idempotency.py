from .journal import ControlStore
def submit_once(store:ControlStore,*,operation_id:str,actor_id:str,idempotency_key:str,request_hash:str,operation_type:str,generation:int):return store.put_operation(operation_id=operation_id,actor_id=actor_id,idempotency_key=idempotency_key,request_hash=request_hash,operation_type=operation_type,generation=generation)
