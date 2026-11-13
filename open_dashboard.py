"""Open Formation Review on a local server with video seeking."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import webbrowser


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        self.serve(send_body=False)

    def do_GET(self):
        self.serve(send_body=True)

    def serve(self, send_body):
        name = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            self.send_error(404)
            return

        with path.open("rb") as stream:
            size = path.stat().st_size
            start, stop = 0, size - 1
            byte_range = self.headers.get("Range")
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range or "")
            if match and size:
                first, last = match.groups()
                if first:
                    start = int(first)
                    stop = min(int(last), stop) if last else stop
                elif last:
                    start = max(0, size - int(last))
                if start > stop:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

            partial_response = bool(match and size and any(match.groups()))
            self.send_response(206 if partial_response else 200)
            content_type = {".js": "text/javascript", ".mp4": "video/mp4"}.get(path.suffix.lower())
            self.send_header("Content-Type", content_type or guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(stop - start + 1))
            if partial_response:
                self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
            self.end_headers()
            if send_body:
                stream.seek(start)
                remaining = stop - start + 1
                try:
                    while remaining:
                        block = stream.read(min(remaining, 64 * 1024))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # Browsers may cancel a range request when the playhead moves.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=Path(__file__).resolve().parent / "dashboard")
    parser.add_argument("--port", type=int, default=0, help="Loopback port. Defaults to an available port.")
    parser.add_argument("--no-open", action="store_true", help="Print the address for manual browser use.")
    args = parser.parse_args()
    root = args.dashboard.resolve()
    if not (root / "index.html").is_file():
        parser.error("Keep the launcher beside the extracted dashboard folder.")
    handler = partial(DashboardHandler, root=root)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Formation Review is open at {url}", flush=True)
        print("Keep this window open during the review. Press Ctrl+C to stop.", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
