#!/usr/bin/env python3
import http.server
import json
import os
import sys


def register_with_reaper():
    # Cross-process registry entry for the test-side reaper; the file layout
    # is the protocol (see tests/support/mod.rs). Only a detached group
    # leader registers: anything else dies with its parent.
    registry = os.environ.get("FIXTURE_REAPER_REGISTRY")
    if not registry or os.getpgid(0) != os.getpid():
        return
    pgid = os.getpid()
    with open(f"/proc/{pgid}/stat") as stat:
        starttime = stat.read().rsplit(")", 1)[1].split()[19]
    entry = "\n".join(
        [
            os.environ["FIXTURE_REAPER_OWNER"],
            starttime,
            os.environ["FIXTURE_REAPER_WORKSPACE"],
        ]
    )
    path = os.path.join(registry, f"{pgid}.grp")
    temp = f"{path}.tmp.{pgid}"
    with open(temp, "w") as handle:
        handle.write(entry)
    os.rename(temp, path)


register_with_reaper()
host, port = sys.argv[1], int(sys.argv[2])
print("FIXTURE_PASS HF_TOKEN=" + os.environ.get("HF_TOKEN", "UNSET"), flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path in ["/health", "/v1/models"] else 404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "id": "fixture-completion",
                "object": "text_completion",
                "model": request["model"],
                "choices": [{"index": 0, "text": " San Francisco", "finish_reason": "stop"}],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


http.server.HTTPServer((host, port), Handler).serve_forever()
