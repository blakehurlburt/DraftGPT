#!/usr/bin/env python3
"""Launch the Draft Assistant web server."""

import os

import uvicorn

if __name__ == "__main__":
    # CR opus: host="0.0.0.0" binds to all network interfaces, making the server
    # CR opus: accessible from any machine on the network. For local development,
    # CR opus: "127.0.0.1" would be safer. Consider making this configurable via env var.
    uvicorn.run(
        "draftassist.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
