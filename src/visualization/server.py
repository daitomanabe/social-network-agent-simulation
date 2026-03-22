"""Server startup script."""

from __future__ import annotations

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel World Simulator API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n🌍 Parallel World Simulator API")
    print(f"   http://{args.host}:{args.port}")
    print(f"   Swagger: http://{args.host}:{args.port}/docs")
    print(f"   WebSocket: ws://{args.host}:{args.port}/ws/live\n")

    uvicorn.run(
        "src.visualization.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
