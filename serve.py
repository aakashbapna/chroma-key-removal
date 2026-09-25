"""Local demo server with cross-origin isolation for the TFLite WASM runtime."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import os
class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        super().end_headers()
if __name__=='__main__':
    os.chdir(Path(__file__).resolve().parent)
    print('Open http://localhost:8766/web/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
