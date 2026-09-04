import tempfile, json
from pathlib import Path
import unittest
from system_x_control_plane.model_manifest import inspect_native
from system_x_control_plane.resource_planner import plan, Plan

class V7ContractTests(unittest.TestCase):
    def test_native_graph_and_planner(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"config.json").write_text(json.dumps({"model_type":"llama"})); (root/"model.safetensors").write_bytes(b"fixture")
            m=inspect_native(root,model_id="fixture",source_revision="immutable",created_utc="2026-01-01T00:00:00Z")
            self.assertEqual(m.required_engine,"vllm_native"); self.assertEqual(plan(artifact_bytes=1,available_bytes=10_000_000).decision,Plan.FIT)
