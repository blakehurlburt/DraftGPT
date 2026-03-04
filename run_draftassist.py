#!/usr/bin/env python3
"""Launch the Draft Assistant web server."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "draftassist.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
