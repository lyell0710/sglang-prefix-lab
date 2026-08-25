#!/usr/bin/env python3
"""Wait for an HTTP endpoint without adding a client dependency."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    print(f"ready url={args.url} status={response.status}")
                    return 0
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = repr(exc)
        time.sleep(args.interval)

    print(f"timeout url={args.url} last_error={last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

