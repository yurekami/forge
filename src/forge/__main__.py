"""FORGE CLI entry point.

Usage:
    forge                     # Start the web server
    forge --port 8420         # Custom port
    forge --provider openai   # Force a specific provider
    forge --db ./my.db        # Custom database path
    python -m forge           # Alternative invocation
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="FORGE: Multi-Model Adversarial Code Arena",
    )
    parser.add_argument(
        "--port", type=int, default=8420,
        help="Port to serve on (default: 8420)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--provider", type=str, default="auto",
        help="Force a provider: openai, anthropic, ollama, demo, auto (default: auto)",
    )
    parser.add_argument(
        "--db", type=str, default="forge.db",
        help="Path to SQLite database (default: forge.db)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    banner = """
    +-------------------------------------------------------+
    |                                                       |
    |     FFFFF  OOO  RRRR   GGG  EEEEE                    |
    |     F     O   O R   R G     E                         |
    |     FFF   O   O RRRR  G  GG EEE                       |
    |     F     O   O R  R  G   G E                         |
    |     F      OOO  R   R  GGG  EEEEE                    |
    |                                                       |
    |     Multi-Model Adversarial Code Arena                |
    |     Where AI perspectives compete, review, and forge  |
    |                                                       |
    +-------------------------------------------------------+
    """
    print(banner)
    print(f"  Starting FORGE on http://{args.host}:{args.port}")
    print(f"  Provider: {args.provider}")
    print(f"  Database: {args.db}")
    print()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "forge.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
