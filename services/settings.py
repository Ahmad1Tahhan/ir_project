"""Service registry: host + ports for every microservice.

Override the host with the IRSYS_HOST env var if needed. Ports can be moved
here in one place; the gateway and launcher both read from this registry.
"""
from __future__ import annotations

import os

HOST = os.environ.get("IRSYS_HOST", "127.0.0.1")

PORTS = {
    "gateway": 8000,
    "retrieval": 8001,
    "preprocessing": 8002,
    "refinement": 8003,
    "evaluation": 8004,
    "rag": 8005,
}


def url(service: str) -> str:
    return f"http://{HOST}:{PORTS[service]}"
