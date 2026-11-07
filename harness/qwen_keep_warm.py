"""Ping the Qwen vLLM server so a Kaggle T4 notebook stays warm.

Kaggle and similar free GPU runtimes idle-stop when nothing hits the
endpoint. Run this in a second terminal (or notebook cell loop) while
``run_qa_matrix.py`` uses any ``qa-qwen-*`` config:

    export QWEN_API_BASE=https://.../v1
    python -m harness.qwen_keep_warm

Every ``--interval`` seconds it sends a one-token chat completion and
prints the latency. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

import certifi
from dotenv import load_dotenv

load_dotenv()


def ping(api_base: str, model: str, api_key: str, timeout: float) -> float:
    """POST a tiny chat completion; return elapsed seconds."""
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    # macOS Python.org installs often lack system CA certs; use certifi.
    ctx = ssl.create_default_context(cafile=certifi.where())
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        resp.read()
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep the Qwen vLLM endpoint warm.")
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="seconds between pings (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per ping in seconds (default: 120)",
    )
    args = parser.parse_args()

    api_base = os.environ.get("QWEN_API_BASE", "").strip()
    if not api_base:
        print(
            "QWEN_API_BASE is not set. Point it at the OpenAI-compatible "
            "vLLM URL from the Kaggle notebook.",
            file=sys.stderr,
        )
        sys.exit(1)
    model = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B-AWQ")
    api_key = os.environ.get("QWEN_API_KEY", "not-needed")

    print(f"keeping warm: {api_base} model={model} every {args.interval:.0f}s")
    while True:
        try:
            elapsed = ping(api_base, model, api_key, args.timeout)
            print(f"ok {elapsed:.2f}s", flush=True)
        except urllib.error.URLError as exc:
            print(f"ping failed: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep looping on any ping error
            print(f"ping failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
