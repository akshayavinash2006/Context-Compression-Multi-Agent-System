"""python -m server [--port 8000] [--reload]"""
import argparse
import warnings

import uvicorn

warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    print(f"GraphMem factory server: http://{a.host}:{a.port}  (UI dev server: http://localhost:5173)")
    uvicorn.run("server.app:app", host=a.host, port=a.port, reload=a.reload)
