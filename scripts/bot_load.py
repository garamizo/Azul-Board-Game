#!/usr/bin/env python3
"""S4: N all-bot 4-player games on the dev stack. Passes when GET /api/games/{id}
p95 is under 200 ms and every unfinished game advanced at least every 15 s."""
import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request


def call(base, method, path, body=None, user="load@example.com"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={
        "X-Dev-User": user, "Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return None if res.status == 204 else json.load(res)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:29649")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--container", default="azul-dev-app-1")
    a = p.parse_args()

    ids = []
    for _ in range(a.games):
        g = call(a.base, "POST", "/api/games", {"players": 4})
        call(a.base, "POST", f"/api/games/{g['id']}/start", {})
        call(a.base, "POST", f"/api/games/{g['id']}/seats/0/to-bot", {})
        ids.append(g["id"])

    stop = time.monotonic() + a.seconds
    cpu = []

    def sample_cpu():
        while time.monotonic() < stop:
            out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", a.container],
                                 capture_output=True, text=True).stdout.strip().rstrip("%")
            try:
                cpu.append(float(out))
            except ValueError:
                pass

    threading.Thread(target=sample_cpu, daemon=True).start()

    latencies = []
    versions = {i: None for i in ids}
    changed = {i: time.monotonic() for i in ids}
    worst_gap = {i: 0.0 for i in ids}
    finished = set()
    while time.monotonic() < stop:
        for i in ids:
            if i in finished:
                continue
            t0 = time.monotonic()
            v = call(a.base, "GET", f"/api/games/{i}")
            now = time.monotonic()
            latencies.append((now - t0) * 1000)
            if v["status"] == "finished":
                finished.add(i)
                continue
            worst_gap[i] = max(worst_gap[i], now - changed[i])
            if v["version"] != versions[i]:
                versions[i], changed[i] = v["version"], now
        time.sleep(0.2)

    p95 = statistics.quantiles(latencies, n=20)[18]
    report = {
        "games": ids,
        "requests": len(latencies),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(p95, 1),
        "worst_gap_s": {i: round(g, 1) for i, g in worst_gap.items()},
        "finished": sorted(finished),
        "cpu_percent_mean": round(statistics.mean(cpu), 1) if cpu else None,
        "cpu_percent_max": max(cpu) if cpu else None,
    }
    print(json.dumps(report, indent=2))
    # A game that finished still counts its stalls before it finished.
    ok = p95 < 200 and all(g <= 15 for g in worst_gap.values())
    print("S4 PASS" if ok else "S4 FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
