import asyncio
import json
import sys
import types
import unittest

fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        pass

    def add_middleware(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func


class _HTTPExceptionStub(Exception):
    def __init__(self, status_code=None, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.File = lambda *args, **kwargs: None
fastapi_stub.UploadFile = object
fastapi_stub.HTTPException = _HTTPExceptionStub
sys.modules.setdefault("fastapi", fastapi_stub)

cors_stub = types.ModuleType("fastapi.middleware.cors")
cors_stub.CORSMiddleware = object
sys.modules.setdefault("fastapi.middleware", types.ModuleType("fastapi.middleware"))
sys.modules.setdefault("fastapi.middleware.cors", cors_stub)

responses_stub = types.ModuleType("fastapi.responses")
responses_stub.JSONResponse = object
responses_stub.StreamingResponse = object
sys.modules.setdefault("fastapi.responses", responses_stub)

pandas_stub = types.ModuleType("pandas")
pandas_stub.read_csv = lambda *args, **kwargs: None
pandas_stub.notna = lambda value: value is not None
pandas_stub.DataFrame = object
sys.modules.setdefault("pandas", pandas_stub)

import backend_server


class BackendServerKnowledgeBaseTests(unittest.TestCase):
    def test_process_users_uses_knowledge_base_module(self):
        calls = []

        fake_wirelessagent = types.SimpleNamespace()

        def load_user_data_from_csv(csv_path):
            calls.append(("load", csv_path))
            return [
                {
                    "user_id": "rx-1",
                    "location": "(1.0, 2.0, 1.5)",
                    "request": "I need to stream 4K video",
                    "cqi": 12,
                    "ground_truth": "eMBB",
                }
            ]

        def reset_network_state():
            calls.append(("reset", None))

        def process_user_request(**kwargs):
            calls.append(("process", kwargs))
            return {
                "user_id": kwargs["user_id"],
                "request": kwargs["request"],
                "cqi": kwargs["cqi"],
                "slice_type": "eMBB",
                "bandwidth": 72.0,
                "rate": 100.0,
                "allocation_failed": False,
            }

        fake_wirelessagent.load_user_data_from_csv = load_user_data_from_csv
        fake_wirelessagent.reset_network_state = reset_network_state
        fake_wirelessagent.process_user_request = process_user_request

        original_loader = backend_server.get_wirelessagent_module
        backend_server.get_wirelessagent_module = lambda: fake_wirelessagent
        try:
            df = object()
            logs = backend_server.LogCapture()
            results = asyncio.run(
                backend_server.process_users_async(df, logs, "uploaded.csv")
            )
        finally:
            backend_server.get_wirelessagent_module = original_loader

        self.assertEqual(results[0]["slice_type"], "eMBB")
        self.assertEqual(calls[0], ("load", "uploaded.csv"))
        self.assertEqual(calls[1], ("reset", None))
        self.assertEqual(calls[2][0], "process")
        self.assertEqual(calls[2][1]["ground_truth"], "eMBB")

    def test_stream_process_users_emits_live_events(self):
        fake_wirelessagent = types.SimpleNamespace()

        fake_wirelessagent.load_user_data_from_csv = lambda csv_path: [
            {
                "user_id": "rx-1",
                "location": "(1.0, 2.0, 1.5)",
                "request": "I need to stream 4K video",
                "cqi": 12,
                "ground_truth": "eMBB",
            }
        ]
        fake_wirelessagent.reset_network_state = lambda: None
        fake_wirelessagent.process_user_request = lambda **kwargs: {
            "user_id": kwargs["user_id"],
            "request": kwargs["request"],
            "cqi": kwargs["cqi"],
            "slice_type": "eMBB",
            "bandwidth": 72.0,
            "rate": 100.0,
            "latency": 40.0,
            "allocation_failed": False,
            "adjustments_made": False,
        }

        original_loader = backend_server.get_wirelessagent_module
        backend_server.get_wirelessagent_module = lambda: fake_wirelessagent
        try:
            events = list(backend_server.iter_process_user_events("uploaded.csv"))
        finally:
            backend_server.get_wirelessagent_module = original_loader

        event_types = [event["type"] for event in events]
        self.assertIn("log", event_types)
        self.assertIn("result", event_types)
        self.assertIn("progress", event_types)
        self.assertEqual(events[-1]["type"], "complete")

        result_event = next(event for event in events if event["type"] == "result")
        self.assertEqual(result_event["result"]["slice_type"], "eMBB")

    def test_encode_stream_event_uses_ndjson(self):
        encoded = backend_server.encode_stream_event({"type": "complete", "total": 1})
        self.assertTrue(encoded.endswith("\n"))
        self.assertEqual(json.loads(encoded), {"type": "complete", "total": 1})


if __name__ == "__main__":
    unittest.main()
