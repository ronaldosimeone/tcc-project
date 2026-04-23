"""
Load test — SSE sensor stream (RNF-28: ≥ 50 simultaneous clients).

Two modes
---------
1. Locust (default):
       pip install locust
       locust -f locust_sse.py --host http://localhost --headless \
              -u 60 -r 10 --run-time 60s

2. Standalone asyncio (no extra deps beyond httpx which is in requirements.txt):
       python locust_sse.py [--url http://localhost/api] \
                            [--clients 60] [--duration 30]

The standalone mode is used to validate RNF-28 in CI without a Locust server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import httpx
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Standalone asyncio runner
# ---------------------------------------------------------------------------

REQUIRED_SENSOR_FIELDS = {
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Oil_level",
}


@dataclass
class ClientResult:
    client_id: int
    events_received: int = 0
    validation_errors: int = 0
    error: str = ""
    elapsed: float = 0.0


async def _run_sse_client(
    client: httpx.AsyncClient,
    url: str,
    duration: float,
    client_id: int,
) -> ClientResult:
    result = ClientResult(client_id=client_id)
    start = time.monotonic()

    try:
        async with client.stream("GET", f"{url}/stream/sensors") as response:
            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}"
                return result

            deadline = start + duration

            async for raw_line in response.aiter_lines():
                if time.monotonic() >= deadline:
                    break

                if not raw_line.startswith("data: "):
                    continue

                try:
                    payload = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    result.validation_errors += 1
                    continue

                missing = REQUIRED_SENSOR_FIELDS - payload.keys()
                if missing:
                    result.validation_errors += 1
                else:
                    result.events_received += 1

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)

    result.elapsed = time.monotonic() - start
    return result


async def _standalone(url: str, n_clients: int, duration: float) -> None:
    print(
        f"Starting {n_clients} SSE clients for {duration}s against {url}/stream/sensors"
    )

    limits = httpx.Limits(
        max_connections=n_clients + 10, max_keepalive_connections=n_clients + 10
    )
    async with httpx.AsyncClient(timeout=duration + 15, limits=limits) as client:
        tasks = [_run_sse_client(client, url, duration, i) for i in range(n_clients)]
        results: list[ClientResult] = await asyncio.gather(*tasks)  # type: ignore[assignment]

    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    total_events = sum(r.events_received for r in ok)
    validation_failures = sum(r.validation_errors for r in ok)

    print(f"\n{'='*55}")
    print(f"  Clients launched   : {n_clients}")
    print(f"  Successful         : {len(ok)}")
    print(f"  Failed             : {len(failed)}")
    print(f"  Total events recv  : {total_events}")
    print(
        f"  Avg events/client  : {total_events / max(len(ok), 1):.1f}  (expected ~{duration:.0f})"
    )
    print(f"  Validation errors  : {validation_failures}")

    if failed:
        print("\n  First 5 failures:")
        for r in failed[:5]:
            print(f"    client {r.client_id}: {r.error}")

    rnf28_pass = len(ok) >= 50
    print(
        f"\n  RNF-28  ≥50 clients : {'PASS ✓' if rnf28_pass else 'FAIL ✗'}  ({len(ok)} successful)"
    )

    rf12_avg = total_events / max(len(ok), 1)
    rf12_pass = rf12_avg >= duration * 0.8  # allow 20% tolerance for test startup
    print(
        f"  RF-12  ~1 event/s   : {'PASS ✓' if rf12_pass else 'FAIL ✗'}  ({rf12_avg:.1f} avg events over {duration:.0f}s)"
    )
    print(f"{'='*55}\n")

    if not (rnf28_pass and rf12_pass):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Locust user class (imported when locust is the runner)
# ---------------------------------------------------------------------------

try:
    from locust import HttpUser, between, task as locust_task

    class SSEUser(HttpUser):
        """
        Each virtual user connects to /api/stream/sensors and reads events for
        ~30 seconds, then reconnects immediately (wait_time = 0).
        """

        wait_time = between(0, 0)

        @locust_task
        def stream_sensors(self) -> None:
            events_read = 0
            deadline = time.monotonic() + 30

            with self.client.get(
                "/api/stream/sensors",
                stream=True,
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                catch_response=True,
                timeout=45,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                    return

                for raw_line in response.iter_lines(decode_unicode=True):
                    if time.monotonic() >= deadline:
                        break

                    if not raw_line.startswith("data: "):
                        continue

                    try:
                        payload = json.loads(raw_line[6:])
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON in data line")
                        return

                    missing = REQUIRED_SENSOR_FIELDS - payload.keys()
                    if missing:
                        response.failure(f"Missing sensor fields: {missing}")
                        return

                    events_read += 1

                # RF-12 sanity: expect ~25 events in 30 s (20% tolerance for ramp-up)
                if events_read < 24:
                    response.failure(f"Too few events: {events_read} < 24")
                else:
                    response.success()

except ImportError:
    pass  # locust not installed — standalone mode only


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone SSE load test (RNF-28 validator)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost/api",
        help="Base URL for the API (default: http://localhost/api)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=60,
        help="Number of concurrent SSE clients (default: 60)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="How long each client reads events in seconds (default: 30)",
    )
    args = parser.parse_args()

    asyncio.run(_standalone(args.url, args.clients, args.duration))
