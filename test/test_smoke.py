import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from smoke import Smoke


class SmokeEvidenceTest(unittest.TestCase):
    def test_request_evidence_matches_sent_body_after_caller_mutates_messages(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        with tempfile.TemporaryDirectory() as directory, HTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.handle_request, daemon=True)
            worker.start()
            smoke = Smoke(SimpleNamespace(port=server.server_port, request_timeout=5,
                                          evidence=Path(directory)))
            messages = [{"role": "user", "content": "Remember ORANGE731."}]
            smoke.request("/v1/chat/completions", {"messages": messages})
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            messages.append({"role": "user", "content": "What code?"})
            smoke.save()
            saved = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual(saved["exchanges"][0]["request"], received[0])


if __name__ == "__main__":
    unittest.main()
