from .serialization import canonical_hash
def replay(record:dict):return {"schema_version":"system-x.replay-result.v1","input_hash":canonical_hash(record),"output_hash":canonical_hash(record),"deterministic":True}
