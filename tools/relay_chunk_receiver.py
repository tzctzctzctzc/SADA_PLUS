#!/usr/bin/env python3
"""Minimal private HTTP PUT receiver for parallel video chunk relay."""

import os
import re
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.environ.get("RELAY_RECEIVER_ROOT", "/vepfs/vlms/data/EPIC_KITCHENS/videos/.chunked")
TOKEN = os.environ.get("RELAY_RECEIVER_TOKEN", "")
HOST = os.environ.get("RELAY_RECEIVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("RELAY_RECEIVER_PORT", "18080"))
VIDEO_RE = re.compile(r"^/chunk/(P[0-9]+_[0-9]+)/([0-9]{2}\.part)$")


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # The sender opens many chunk streams at once; avoid a tiny listen queue
    # becoming the next bottleneck on the userspace Tailscale path.
    request_queue_size = 256


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _target(self):
        match = VIDEO_RE.match(unquote(urlparse(self.path).path))
        if not match:
            self.send_error(404)
            return None
        video_id, index = match.groups()
        directory = os.path.join(ROOT, video_id)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, index)

    def _authorized(self):
        if TOKEN and self.headers.get("X-Relay-Token", "") != TOKEN:
            self.send_error(403)
            return False
        return True

    def do_HEAD(self):
        if not self._authorized():
            return
        target = self._target()
        if target is None:
            return
        if os.path.isfile(target):
            self.send_response(200)
            self.send_header("Content-Length", str(os.path.getsize(target)))
            self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        if not self._authorized():
            return
        if urlparse(self.path).path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")
            return
        self.send_error(404)

    def do_PUT(self):
        if not self._authorized():
            return
        target = self._target()
        if target is None:
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0:
            self.send_error(411)
            return
        written = 0
        try:
            with open(target, "wb") as output:
                while written < length:
                    # read1() returns available bytes instead of waiting for the
                    # entire requested block, which is important on the shaped
                    # Tailscale path used by the storage server.
                    block = self.rfile.read1(min(256 * 1024, length - written))
                    if not block:
                        break
                    output.write(block)
                    written += len(block)
        except OSError as exc:
            self.send_error(500, str(exc))
            return
        if written != length:
            self.send_error(400, "incomplete upload")
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"OK")


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("relay receiver listening on %s:%d root=%s" % (HOST, PORT, ROOT), flush=True)
    server.serve_forever()
