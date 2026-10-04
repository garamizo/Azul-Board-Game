#!/usr/bin/env python3
"""S2: kill -9 the dev stack's app (A) right after a human move and (B) while the
bot is thinking; after it comes back the game continues from the last committed
version and the bot's turn completes."""
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:29649"
CONTAINER = "azul-dev-app-1"
UP = ["docker", "compose", "-f", "docker-compose.dev.yml", "up", "-d", "--wait", "app"]


def call(method, path, body=None, user="alice@example.com"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={
        "X-Dev-User": user, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.load(res)


def wait(gid, done, seconds=60):
    until = time.monotonic() + seconds
    while True:
        try:
            v = call("GET", f"/api/games/{gid}")
            if done(v):
                return v
        except (urllib.error.URLError, ConnectionError):
            pass
        if time.monotonic() > until:
            raise SystemExit(f"FAIL: game {gid} did not get there in {seconds}s")
        time.sleep(0.3)


def scenario(name, pause_before_kill):
    g = call("POST", "/api/games", {"players": 2})
    gid = g["id"]
    call("POST", f"/api/games/{gid}/start", {})
    v = wait(gid, lambda x: x["legal"] is not None)
    take = v["legal"]["takes"][0]
    after = call("POST", f"/api/games/{gid}/moves", {
        "version": v["version"], "requestId": str(uuid.uuid4()), "kind": "take",
        "factory": take[0], "color": take[1], "row": take[2]})
    committed = after["version"]
    time.sleep(pause_before_kill)
    subprocess.run(["docker", "kill", "-s", "KILL", CONTAINER], check=True, capture_output=True)
    subprocess.run(UP, check=True, capture_output=True)
    back = wait(gid, lambda x: x["version"] >= committed)
    assert back["version"] >= committed, "a committed move was lost"
    resumed = wait(gid, lambda x: x["version"] > committed and (x["board"]["activeSeat"] == 0 or x["status"] == "finished"))
    print(f"ok    {name}: committed v{committed}, resumed to v{resumed['version']}")


if __name__ == "__main__":
    scenario("A: killed right after a human move", 0.0)
    scenario("B: killed while the bot is thinking", 1.0)
    print("S2 PASS")
