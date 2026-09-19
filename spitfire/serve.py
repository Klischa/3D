# -*- coding: utf-8 -*-
"""
Мини-сервер для просмотра модели в браузере (только stdlib).

Отдаёт:
    /            -> viewer/index.html
    /models/*    -> models/*          (spitfire.glb и другие форматы)
    /renders/*   -> renders/*         (готовые картинки)
    /*           -> viewer/*          (three.js и вьюер)

Запуск:
    python3 serve.py            # http://localhost:8000
    python3 serve.py --port 8080 --host 0.0.0.0
"""

import argparse
import os
import posixpath
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = (
    ("/models/", os.path.join(HERE, "models")),
    ("/renders/", os.path.join(HERE, "renders")),
    ("/", os.path.join(HERE, "viewer")),
)
MIME = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "text/plain; charset=utf-8",
    ".mtl": "text/plain; charset=utf-8",
    ".stl": "model/stl",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def resolve(url_path):
    path = posixpath.normpath(url_path.split("?", 1)[0].split("#", 1)[0])
    for prefix, root in ROOTS:
        if path == prefix.rstrip("/") or path.startswith(prefix):
            rel = path[len(prefix):].lstrip("/")
            full = os.path.normpath(os.path.join(root, rel))
            if not full.startswith(root):
                return None
            if os.path.isdir(full):
                full = os.path.join(full, "index.html")
            return full
    return None


class Handler(SimpleHTTPRequestHandler):
    server_version = "SpitfireViewer/1.0"

    def translate_path(self, path):
        return resolve(path) or os.path.join(HERE, "viewer", "404.html")

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        return MIME.get(ext, "application/octet-stream")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Spitfire viewer: http://%s:%d/  (Ctrl+C — стоп)" % (args.host, args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
