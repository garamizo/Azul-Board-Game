# Azul Web — Plan 4 of 4: Serve Stack and Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the game at https://azul.signalwave.dev behind Cloudflare Tunnel and Access (the `players` rule group), prove restart safety (S2) and bot load (S4) on this machine, document operations, and move Catan's Access application to the same group.

**Architecture:** One image (web bundle + published server on the ASP.NET runtime), a serve compose stack (`app` + `cloudflared`, named volume, loopback port 29648) and a dev compose stack (dev mode, port 29649) for load, crash and backup tests. The Cloudflare side is done through the API with the user's `publishing-behind-cloudflare` skill (`cf_publish.py`), then a small script switches Catan's policy with a saved rollback copy.

**Tech Stack:** Docker / Compose, `mcr.microsoft.com/dotnet/aspnet:10.0`, `node:24-alpine`, `cloudflare/cloudflared:latest`, bash + curl, Python 3 standard library.

**Spec:** `docs/superpowers/specs/2026-10-03-azul-web-design.md` (§6, §7, S1–S4). Requires Plans 1–3.

## Global Constraints

Everything in Plans 1–3's Global Constraints applies. In addition:

- Follow `~/.claude/skills/publishing-behind-cloudflare/SKILL.md` and its `reference.md`. Read both before Task 20.
- Compose project names: `azul-serve` (production), `azul-dev` (tests). Every serve command passes `--env-file .env.serve`; `.env.serve` is git-ignored and mode 600.
- The app listens on 8080 in the container; published only on `127.0.0.1:29648` (serve) and `127.0.0.1:29649` (dev). `cloudflared` reaches it as `http://app:8080`. Never install a connector on the host.
- Secrets (tunnel token, API token) never appear in a terminal, a log, a commit or a doc.
- The Access application exists before the tunnel route (`cf_publish.py` guarantees the order).
- Every Cloudflare dashboard path written into a doc is fetched today from developers.cloudflare.com and carries the verification date; otherwise the doc gives the setting's name plus the link.

### Deviations from the spec found while planning

1. The load script is `scripts/bot_load.py` (Python, stdlib), not `bot-load.sh`: it parses JSON and computes percentiles. A crash script `scripts/crash_test.py` and an Access policy script `scripts/access_set_policy.py` are added.
2. Backups use a throwaway Alpine container with `sqlite3 .backup` on the named volume (the runtime image has no `sqlite3`).
3. `serve-check.sh --local` runs only the loopback probes, for use before `cloudflared` starts.

## Review Focus

1. The tunnel token or AUD missing or empty in `.env.serve`: compose must refuse to start rather than run an unguarded app → Task 20 Step 6 (`${VAR:?}` check with an empty file).
2. A restore that leaves `azul.db-wal`/`-shm` files from the old database, or files owned by the wrong user: the app must start on the restored data → Task 21 `make backup` / `make restore` round trip.
3. Redeploy while a game is in progress (`git pull && make serve`): the game continues → Task 21 crash test scenario B covers kill/restart; Task 23 Step 9 repeats a redeploy on the live stack.
4. Cloudflare answers a public `/` with something other than an Access login redirect (route published without Access, or Access bypass): serve-check must fail → Task 21 Step 2 runs probe 4 against a host that does not redirect and expects FAIL.
5. Catan's update accidentally recreating the app (new AUD) or dropping fields: Catan users would be locked out → Task 24 tests plus the post-update AUD check.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `Dockerfile`, `.dockerignore` | The image. |
| `docker-compose.serve.yml`, `docker-compose.dev.yml` | Production stack; local dev-mode stack for tests. |
| `.env.serve.example` | Template for `.env.serve`. |
| `Makefile` | `serve`, `serve-app`, `serve-check`, `serve-check-local`, `serve-logs`, `serve-down`, `dev-stack`, `dev-stack-down`, `load-test`, `crash-test`, `backup`, `restore`. |
| `scripts/serve-check.sh` | The skill's probes. |
| `scripts/bot_load.py` | S4 measurement. |
| `scripts/crash_test.py` | S2 measurement. |
| `scripts/access_set_policy.py`, `scripts/test_access_set_policy.py` | Catan policy switch + rollback, with tests. |
| `docs/deploy.md` | Publishing and operations. |
| `README.md` | Link to the web version and `docs/deploy.md`. |

---

### Task 20: Image and compose stacks

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.serve.yml`, `docker-compose.dev.yml`, `.env.serve.example`
- Modify: `Makefile`

- [ ] **Step 1: Read the skill**

Run: `cat ~/.claude/skills/publishing-behind-cloudflare/SKILL.md ~/.claude/skills/publishing-behind-cloudflare/reference.md`
Expected: you know the architecture, the probes and the traps table before writing the files below.

- [ ] **Step 2: Write the image**

`Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1
# Web bundle + published server on the ASP.NET runtime, one origin on 8080.

FROM node:24-alpine AS web
WORKDIR /src
COPY assets/ assets/
COPY web/package.json web/package-lock.json web/
RUN cd web && npm ci --no-audit --no-fund
COPY web/ web/
RUN cd web && npm run build

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY AzulLibrary/AzulLibrary.csproj AzulLibrary/
COPY server/AzulServer/AzulServer.csproj server/AzulServer/
RUN dotnet restore server/AzulServer/AzulServer.csproj
COPY AzulLibrary/ AzulLibrary/
COPY server/AzulServer/ server/AzulServer/
RUN dotnet publish server/AzulServer/AzulServer.csproj -c Release -o /out --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /out/ /app/
COPY --from=web /src/web/dist/ /app/wwwroot/
USER root
RUN mkdir -p /data && chown app:app /data
USER app
ENV ASPNETCORE_HTTP_PORTS=8080 AZUL_DATA_DIR=/data AZUL_WEB_ROOT=/app/wwwroot
EXPOSE 8080
ENTRYPOINT ["/app/AzulServer"]
```

`.dockerignore`:

```gitignore
.git
**/bin
**/obj
**/node_modules
web/dist
web/public/assets
web/test-results
web/playwright-report
web/.env*
.data
.env*
*.ipynb
*.pt
img
runs
outputs
tileclassifier
azul
tools
docs
AzulBench
AzulLibrary.Tests
server/AzulServer.Tests
```

- [ ] **Step 3: Write the compose files and the env template**

`docker-compose.serve.yml`:

```yaml
# Production: https://azul.signalwave.dev through a Cloudflare Tunnel with
# Cloudflare Access in front (docs/deploy.md). The app also checks Access's
# JWT itself on every path but /api/health, so the loopback port below is for
# probes only. cloudflared reaches the app as http://app:8080; never run a
# connector on the host for this tunnel (two connectors split traffic).
name: azul-serve
x-logging: &logging
  driver: json-file
  options: { max-size: "10m", max-file: "3" }
services:
  app:
    build: .
    restart: unless-stopped
    environment:
      AZUL_ACCESS_TEAM_DOMAIN: ${AZUL_ACCESS_TEAM_DOMAIN:?set in .env.serve (docs/deploy.md)}
      AZUL_ACCESS_AUD: ${AZUL_ACCESS_AUD:?set in .env.serve (docs/deploy.md)}
      AZUL_PUBLIC_ORIGIN: ${AZUL_PUBLIC_ORIGIN:-https://azul.signalwave.dev}
      AZUL_BOT_WORKERS: ${AZUL_BOT_WORKERS:-2}
      AZUL_BOT_THINK_SECONDS: ${AZUL_BOT_THINK_SECONDS:-3}
    ports: ["127.0.0.1:${AZUL_SERVE_PORT:-29648}:8080"]
    volumes: ["azul-data:/data"]
    healthcheck:
      test: ["CMD", "/app/AzulServer", "--healthcheck"]
      interval: 10s
      timeout: 5s
      retries: 6
      start_period: 10s
    logging: *logging
  cloudflared:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel --no-autoupdate run
    environment:
      TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN:?set in .env.serve (docs/deploy.md)}
    depends_on:
      app: { condition: service_healthy }
    logging: *logging
volumes:
  azul-data: {}
```

`docker-compose.dev.yml`:

```yaml
# Dev mode (no Access; identity from X-Dev-User) on 127.0.0.1:29649, for the
# load, crash and backup tests. Never publish this stack.
name: azul-dev
services:
  app:
    build: .
    restart: unless-stopped
    environment:
      AZUL_BOT_WORKERS: ${AZUL_BOT_WORKERS:-2}
      AZUL_BOT_THINK_SECONDS: ${AZUL_BOT_THINK_SECONDS:-3}
    ports: ["127.0.0.1:${AZUL_DEV_PORT:-29649}:8080"]
    volumes: ["azul-dev-data:/data"]
    healthcheck:
      test: ["CMD", "/app/AzulServer", "--healthcheck"]
      interval: 5s
      timeout: 5s
      retries: 12
volumes:
  azul-dev-data: {}
```

`.env.serve.example`:

```dotenv
# Copy to .env.serve (git-ignored, chmod 600). Every `make serve*` command
# reads it (docker compose --env-file .env.serve). docs/deploy.md has the steps.

# Printed by cf_publish.py: the Zero Trust team domain and the Access
# application's Audience (AUD) tag. Both or neither; compose refuses to start
# the app without them.
AZUL_ACCESS_TEAM_DOMAIN=
AZUL_ACCESS_AUD=

# The tunnel token cf_publish.py wrote to ~/.config/cloudflare/azul-tunnel-token.
# Append it without printing it (docs/deploy.md).
CLOUDFLARE_TUNNEL_TOKEN=

AZUL_PUBLIC_ORIGIN=https://azul.signalwave.dev

# Optional
# AZUL_SERVE_PORT=29648
# AZUL_BOT_WORKERS=2
# AZUL_BOT_THINK_SECONDS=3
```

- [ ] **Step 4: Makefile targets**

Add (and list in `.PHONY`):

```make
SERVE := docker compose -f docker-compose.serve.yml --env-file .env.serve
DEV := docker compose -f docker-compose.dev.yml

serve:
	$(SERVE) up -d --build --wait app
	$(SERVE) up -d cloudflared

serve-app:
	$(SERVE) up -d --build --wait app

serve-check:
	set -a; . ./.env.serve; set +a; scripts/serve-check.sh

serve-check-local:
	set -a; . ./.env.serve; set +a; scripts/serve-check.sh --local

serve-logs:
	$(SERVE) logs -f --tail=200 $(SERVICE)

serve-down:
	$(SERVE) down

dev-stack:
	$(DEV) up -d --build --wait

dev-stack-down:
	$(DEV) down -v
```

- [ ] **Step 5: Build and run the dev stack**

Run:

```bash
cd /home/garamizo/Azul-Board-Game-web
make dev-stack
curl -s http://127.0.0.1:29649/api/health
curl -s -o /dev/null -w '%{http_code}\n' -H 'X-Dev-User: a@x' http://127.0.0.1:29649/
docker run --rm --entrypoint id azul-dev-app -u app
docker compose -f docker-compose.dev.yml ps --format '{{.Name}} {{.Health}}'
```

Expected: `{"ok":true}`; `200` (the SPA shell in dev mode); `1654` (the image's `app` uid, used by `make restore`); `azul-dev-app-1 healthy`. If the uid differs, use that value in Task 21's `restore` target.

- [ ] **Step 6: Check that compose refuses an empty configuration**

Run:

```bash
printf 'AZUL_ACCESS_TEAM_DOMAIN=\nAZUL_ACCESS_AUD=\nCLOUDFLARE_TUNNEL_TOKEN=\n' > /tmp/claude-empty.env
docker compose -f docker-compose.serve.yml --env-file /tmp/claude-empty.env config >/dev/null; echo "exit=$?"
rm /tmp/claude-empty.env
```

Expected: an error naming `AZUL_ACCESS_TEAM_DOMAIN` (`:?` treats empty as unset) and `exit=1`.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.serve.yml docker-compose.dev.yml .env.serve.example Makefile
git commit -m "deploy: image, serve and dev compose stacks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 21: serve-check, load test (S4), crash test (S2), backup and restore

**Files:**
- Create: `scripts/serve-check.sh`, `scripts/bot_load.py`, `scripts/crash_test.py`
- Modify: `Makefile` (`load-test`, `crash-test`, `backup`, `restore`)

- [ ] **Step 1: Write serve-check**

`scripts/serve-check.sh` (`chmod +x`):

```bash
#!/usr/bin/env bash
# The publishing-behind-cloudflare probes for azul.signalwave.dev:
#   1. loopback /api/health -> 200
#   2. loopback guarded paths (/api/games and the SPA at /) without a token -> 401
#   3. machine-to-machine endpoints without their key -> 401: there are none
#   4. public / -> a redirect to https://<team domain>/cdn-cgi/access/login...
# `--local` runs 1-2 only (before cloudflared starts).
set -uo pipefail
PORT="${AZUL_SERVE_PORT:-29648}"
HOST="${AZUL_PUBLIC_HOST:-azul.signalwave.dev}"
TEAM="${AZUL_ACCESS_TEAM_DOMAIN:?AZUL_ACCESS_TEAM_DOMAIN not set; run through make, which reads .env.serve}"
TEAM="${TEAM#https://}"
TEAM="${TEAM%/}"
fail=0
ok() { printf 'ok    %s\n' "$1"; }
bad() { printf 'FAIL  %s\n' "$1"; fail=1; }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@"; }

c=$(code "http://127.0.0.1:$PORT/api/health")
[ "$c" = 200 ] && ok "1. loopback /api/health is 200" || bad "1. loopback /api/health is $c, expected 200"
for path in /api/games /; do
  c=$(code "http://127.0.0.1:$PORT$path")
  [ "$c" = 401 ] && ok "2. loopback $path without a token is 401" || bad "2. loopback $path without a token is $c, expected 401"
done
echo "note  3. no machine-to-machine endpoints in this app"

if [ "${1:-}" != "--local" ]; then
  read -r status location < <(curl -s -o /dev/null --max-time 15 -w '%{http_code} %{redirect_url}' "https://$HOST/")
  parts=$(python3 -c 'import sys, urllib.parse as u; p = u.urlsplit(sys.argv[1]); print(p.hostname or "-", p.path or "-")' "${location:-}")
  loc_host=${parts% *}
  loc_path=${parts#* }
  if [[ "$status" =~ ^30[1278]$ && "$loc_host" == "$TEAM" && "$loc_path" == /cdn-cgi/access/login* ]]; then
    ok "4. public / redirects to Access login on $TEAM"
  else
    bad "4. public / gave $status -> '${location:-}', expected a redirect to https://$TEAM/cdn-cgi/access/login..."
  fi
fi
exit $fail
```

- [ ] **Step 2: Test serve-check against a local Access-mode app**

The serve stack's `app` with dummy Access values answers 401 to everything but health, without contacting Cloudflare (no token means no key lookup):

```bash
cd /home/garamizo/Azul-Board-Game-web
printf 'AZUL_ACCESS_TEAM_DOMAIN=example.invalid\nAZUL_ACCESS_AUD=dummy\nCLOUDFLARE_TUNNEL_TOKEN=dummy\n' > .env.serve.test
docker compose -f docker-compose.serve.yml --env-file .env.serve.test -p azul-servecheck up -d --build --wait app
AZUL_ACCESS_TEAM_DOMAIN=example.invalid scripts/serve-check.sh --local; echo "exit=$?"
AZUL_ACCESS_TEAM_DOMAIN=example.invalid AZUL_PUBLIC_HOST=example.com scripts/serve-check.sh; echo "exit=$?"
docker compose -f docker-compose.serve.yml --env-file .env.serve.test -p azul-servecheck down -v
rm .env.serve.test
```

Expected: first run three `ok` lines and `exit=0`; second run the same plus `FAIL  4. public / gave 200 -> ''...` and `exit=1` (example.com does not redirect to Access: the probe catches a missing Access application).

- [ ] **Step 3: Write the load test**

`scripts/bot_load.py`:

```python
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
    ok = p95 < 200 and all(g <= 15 for i, g in worst_gap.items() if i not in finished)
    print("S4 PASS" if ok else "S4 FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the crash test**

`scripts/crash_test.py`:

```python
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
```

- [ ] **Step 5: Makefile targets for load, crash, backup, restore**

Add (and list in `.PHONY`):

```make
load-test: dev-stack
	python3 scripts/bot_load.py

crash-test: dev-stack
	python3 scripts/crash_test.py

# Backups of a stack's database (STACK=serve or dev), WAL-safe via sqlite3 .backup.
STACK ?= serve
BACKUP_DIR ?= $(HOME)/backups/azul
backup:
	mkdir -p $(BACKUP_DIR)
	docker run --rm -v azul-$(STACK)_azul$(if $(filter dev,$(STACK)),-dev,)-data:/data -v $(BACKUP_DIR):/out alpine \
		sh -c 'f=/out/azul-$(STACK)-$$(date +%Y%m%d-%H%M%S).db && apk add -q sqlite && sqlite3 /data/azul.db ".backup $$f" && chown $(UID):$(GID) $$f && echo $$f'

# make restore FILE=~/backups/azul/azul-serve-....db [STACK=serve]; stops the app first.
restore:
	test -f "$(FILE)"
	docker compose -f docker-compose.$(STACK).yml $(if $(filter serve,$(STACK)),--env-file .env.serve,) stop app
	docker run --rm -v azul-$(STACK)_azul$(if $(filter dev,$(STACK)),-dev,)-data:/data -v $(abspath $(FILE)):/in.db:ro alpine \
		sh -c 'rm -f /data/azul.db-wal /data/azul.db-shm && cp /in.db /data/azul.db && chown 1654:1654 /data/azul.db'
	docker compose -f docker-compose.$(STACK).yml $(if $(filter serve,$(STACK)),--env-file .env.serve,) up -d --wait app
```

- [ ] **Step 6: Run S4, S2 and a backup/restore round trip**

```bash
cd /home/garamizo/Azul-Board-Game-web
make load-test
make crash-test
# backup / restore on the dev stack
ID=$(curl -s -H 'X-Dev-User: a@x' -H 'Content-Type: application/json' -d '{"players":2}' http://127.0.0.1:29649/api/games | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
make backup STACK=dev BACKUP_DIR=$PWD/.data/backups
curl -s -X DELETE -H 'X-Dev-User: a@x' http://127.0.0.1:29649/api/games/$ID -o /dev/null -w '%{http_code}\n'
make restore STACK=dev FILE=$(ls -1t .data/backups/*.db | head -1)
curl -s -o /dev/null -w '%{http_code}\n' -H 'X-Dev-User: a@x' http://127.0.0.1:29649/api/games/$ID
make dev-stack-down
```

Expected: `S4 PASS` with the JSON report (keep it for the hand-off: p50, p95, CPU, worst gaps); `S2 PASS`; `204` for the delete, then `200` after the restore (the game is back).

If S4 fails on latency, record the report and lower the default `AZUL_BOT_WORKERS` only if CPU is saturated; if it fails on gaps, report it rather than loosening the 15 s bound.

- [ ] **Step 7: Commit**

```bash
git add scripts/serve-check.sh scripts/bot_load.py scripts/crash_test.py Makefile
git commit -m "deploy: serve-check probes, load and crash tests, backup/restore

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 22: docs/deploy.md and README

**Files:**
- Create: `docs/deploy.md`
- Modify: `README.md`

- [ ] **Step 1: Verify the dashboard pages for today's date**

For each URL in the skill's `reference.md` → *Where to verify* that the doc will link (tunnels, self-hosted app, policies, rule groups, validating JWT), fetch the raw Markdown and grep the labels the doc names:

```bash
for u in \
  https://developers.cloudflare.com/cloudflare-one/access-controls/policies/groups/ \
  https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/ \
  https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/ ; do
  echo "== $u"; curl -sL "${u}index.md" | grep -nE '\*\*[^*]+\*\*' | head -20
done
```

Expected: the bold labels (for example **Rule groups**, **Access controls**, **Policies**) as they are today. Use exactly those words in the doc. If a page 404s, search developers.cloudflare.com for its title and use the new URL.

- [ ] **Step 2: Write `docs/deploy.md`**

Write the document with these sections, using the commands from this plan verbatim and the menu wording from Step 1. Content:

```markdown
# Publishing at azul.signalwave.dev

The serve stack (`docker-compose.serve.yml`) runs the game and a `cloudflared` connector. Cloudflare
Access admits the people in the Access rule group `players`, the same group as
catan.signalwave.dev. The server also verifies Access's signed JWT on every request except
`/api/health`, so the loopback port `127.0.0.1:29648` answers 401.

| Thing | Where |
| --- | --- |
| Who may play | Access rule group `players` (shared with Catan) |
| Team domain, AUD, tunnel token | `.env.serve` (git-ignored, mode 600) |
| Games | docker volume `azul-serve_azul-data` |
| Logs | `make serve-logs [SERVICE=app\|cloudflared]` |

## One-time setup

Dashboard steps verified against developers.cloudflare.com on <DATE OF STEP 1>.

1. **API token**: the one in `~/.config/cloudflare/api-token` (permissions table in the
   `publishing-behind-cloudflare` skill). Check: <the skill's curl verify command>.
2. **Access application, tunnel, route, DNS** with `cf_publish.py`:
   <dry-run command>, review, then <real command with --token-out ~/.config/cloudflare/azul-tunnel-token>.
   It reuses the `players` group, creates the `Players` policy if missing, and creates the Access
   application before the route.
3. **`.env.serve`**: `install -m 600 .env.serve.example .env.serve`; fill the team domain and AUD
   printed by `cf_publish.py`; append the token without printing it:
   `printf 'CLOUDFLARE_TUNNEL_TOKEN=%s\n' "$(cat ~/.config/cloudflare/azul-tunnel-token)" >> .env.serve`
   (delete the empty `CLOUDFLARE_TUNNEL_TOKEN=` line first).
4. **Start**: `make serve-app && make serve-check-local`, then `make serve && make serve-check`.

## Inviting someone

Add their email to the rule group `players` (<menu path from Step 1>, [docs](<groups URL>)). This
also admits them to Catan. No restart. Removing someone takes effect when their Access session
(24 h) ends.

## Day to day

- Deploy a change: `git pull && make serve` (games survive; bots resume).
- Backup: `make backup` (files in `~/backups/azul`). Restore: `make restore FILE=...`.
- Rotate the tunnel token: <the tunnel page's refresh-token wording from Step 1>, update `.env.serve`, `make serve`.
- Move to another machine: copy `.env.serve` and a backup, `make serve` there, `make restore FILE=...`,
  then `make serve-down` here.

## Checking it live

- An invited email reaches the lobby; someone else is refused at Cloudflare's page.
- Sign out in another tab (`/cdn-cgi/access/logout`) with a game open: the page reloads into the
  sign-in page once, no loop.
- Two phones play a game to the end.
```

Replace every `<...>` with the real value from this plan or Step 1 before committing (`grep -n '<' docs/deploy.md` must show only HTML-free Markdown).

- [ ] **Step 3: README**

Append to `README.md`:

```markdown
## Play online

The game runs as a web app for invited friends at https://azul.signalwave.dev (server in
`server/`, browser client in `web/`). Publishing and operations: [docs/deploy.md](docs/deploy.md).
Local development: `make test`, `make dev-server`, `cd web && npm run dev`.
```

- [ ] **Step 4: Commit**

```bash
git add docs/deploy.md README.md
git commit -m "docs: deploy and operations guide

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 23: Publish through Cloudflare

The user asked for this publication; it creates live Cloudflare objects. Show the dry-run output in the progress report before the real run.

**Files:**
- Create: `.env.serve` (local only, never committed)

- [ ] **Step 1: Check the API token**

```bash
printf 'Authorization: Bearer %s\n' "$(cat ~/.config/cloudflare/api-token)" |
  curl -s -H @- https://api.cloudflare.com/client/v4/user/tokens/verify | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["status"])'
```

Expected: `active`.

- [ ] **Step 2: Dry run**

```bash
S=~/.claude/skills/publishing-behind-cloudflare/scripts
python3 $S/cf_publish.py --hostname azul.signalwave.dev --service http://app:8080 \
  --tunnel azul --app-name Azul --rule-group players --policy-name Players --session 24h --dry-run
```

Expected: it reuses rule group `players`, plans to create policy `Players`, Access application `Azul`, tunnel `azul`, the route `azul.signalwave.dev → http://app:8080`, and a proxied CNAME; no "refusing" line. If it refuses, stop and report the reason; change nothing by hand.

- [ ] **Step 3: Real run**

```bash
python3 $S/cf_publish.py --hostname azul.signalwave.dev --service http://app:8080 \
  --tunnel azul --app-name Azul --rule-group players --policy-name Players --session 24h \
  --token-out ~/.config/cloudflare/azul-tunnel-token
ls -l ~/.config/cloudflare/azul-tunnel-token
```

Expected: the team domain and AUD printed; the token file exists with mode `-rw-------`.

- [ ] **Step 4: Write `.env.serve`**

```bash
cd /home/garamizo/Azul-Board-Game-web
install -m 600 .env.serve.example .env.serve
sed -i '/^CLOUDFLARE_TUNNEL_TOKEN=$/d' .env.serve
sed -i "s|^AZUL_ACCESS_TEAM_DOMAIN=.*|AZUL_ACCESS_TEAM_DOMAIN=<team domain from Step 3>|; s|^AZUL_ACCESS_AUD=.*|AZUL_ACCESS_AUD=<AUD from Step 3>|" .env.serve
printf 'CLOUDFLARE_TUNNEL_TOKEN=%s\n' "$(cat ~/.config/cloudflare/azul-tunnel-token)" >> .env.serve
grep -c '^CLOUDFLARE_TUNNEL_TOKEN=.\+' .env.serve
git status --short .env.serve
```

Expected: `1`; `git status` prints nothing (ignored).

- [ ] **Step 5: Start the app and check the origin**

Run: `make serve-app && make serve-check-local`
Expected: three `ok` lines, exit 0.

- [ ] **Step 6: Start the connector and check Access in front**

Run: `make serve && make serve-check`
Expected: probes 1, 2 and 4 `ok`; `make serve-logs SERVICE=cloudflared` shows `Registered tunnel connection`.

- [ ] **Step 7: Hand the live check to the user**

Ask the user to open https://azul.signalwave.dev on a phone and a desktop with two invited emails, start a 3-player game (two people + bot) and play a few rounds, and to try one email outside the group. Record their answer.

- [ ] **Step 8: CSRF through the tunnel**

From the user's live session, a successful move proves POSTs pass the Origin check behind cloudflared (Plan 2 Review Focus 2). If moves fail with 403 `bad-origin`, check `AZUL_PUBLIC_ORIGIN` in `.env.serve` equals `https://azul.signalwave.dev` and `make serve` again.

- [ ] **Step 9: Redeploy during a game**

With a game in progress: `make serve` (rebuild + restart). Expected: the game page shows "Reconnecting…" briefly and continues; a pending bot turn completes.

---

### Task 24: Move Catan to the shared `players` group

**Files:**
- Create: `scripts/access_set_policy.py`, `scripts/test_access_set_policy.py`

- [ ] **Step 1: Write the failing tests**

`scripts/test_access_set_policy.py`:

```python
import json
import os
import stat
import tempfile
import unittest

import access_set_policy as asp


class FakeApi:
    def __init__(self, apps, policies):
        self.apps = apps
        self.policies = policies
        self.writes = []

    def get(self, path, **query):
        if path.endswith("/accounts"):
            return [{"id": "acct"}]
        if path.endswith("/access/apps"):
            return self.apps
        if path.endswith("/access/policies"):
            return self.policies
        for app in self.apps:
            if path.endswith(f"/access/apps/{app['id']}"):
                return app
        raise KeyError(path)

    def put(self, path, body):
        self.writes.append((path, body))
        app_id = path.rsplit("/", 1)[1]
        for app in self.apps:
            if app["id"] == app_id:
                app.update({k: v for k, v in body.items() if k != "policies"})
                app["policies"] = [{"id": pid, "precedence": i + 1} for i, pid in enumerate(body["policies"])]
                return app
        raise KeyError(path)


def catan(**extra):
    app = {"id": "app1", "uid": "app1", "aud": "AUD-CATAN", "name": "catan", "domain": "catan.signalwave.dev",
           "type": "self_hosted", "session_duration": "24h", "created_at": "t", "updated_at": "t",
           "policies": [{"id": "friends", "name": "Friends", "precedence": 1}]}
    app.update(extra)
    return app


PLAYERS = {"id": "players-policy", "name": "Players", "decision": "allow"}


class SetPolicyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.backup = os.path.join(self.dir, "catan-before.json")

    def test_dry_run_writes_nothing(self):
        api = FakeApi([catan()], [PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=True, log=lambda *_: None)
        self.assertEqual(api.writes, [])
        self.assertFalse(os.path.exists(self.backup))

    def test_backup_first_then_only_the_policy_changes(self):
        api = FakeApi([catan()], [PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(stat.S_IMODE(os.stat(self.backup).st_mode), 0o600)
        with open(self.backup) as f:
            self.assertEqual(json.load(f)["policies"][0]["id"], "friends")
        path, body = api.writes[0]
        self.assertTrue(path.endswith("/access/apps/app1"))
        self.assertEqual(body["policies"], ["players-policy"])
        for key in asp.READ_ONLY:
            self.assertNotIn(key, body)
        self.assertEqual(body["domain"], "catan.signalwave.dev")
        self.assertEqual(body["session_duration"], "24h")

    def test_refuses_ambiguous_or_missing_targets(self):
        for apps, policies in [
            ([catan(), catan(id="app2")], [PLAYERS]),
            ([], [PLAYERS]),
            ([catan(type="ssh")], [PLAYERS]),
            ([catan()], []),
            ([catan()], [PLAYERS, dict(PLAYERS, id="dup")]),
            ([catan()], [dict(PLAYERS, decision="bypass")]),
        ]:
            with self.subTest(apps=len(apps), policies=policies):
                api = FakeApi(apps, policies)
                with self.assertRaises(asp.Refused):
                    asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
                self.assertEqual(api.writes, [])

    def test_aud_change_is_detected(self):
        api = FakeApi([catan()], [PLAYERS])
        original_put = api.put
        api.put = lambda path, body: dict(original_put(path, body), aud="NEW")
        with self.assertRaises(asp.Refused):
            asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)

    def test_restore_puts_back_the_saved_policies(self):
        api = FakeApi([catan()], [PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        asp.restore(api, "catan.signalwave.dev", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(api.writes[-1][1]["policies"], ["friends"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd scripts && python3 -m unittest test_access_set_policy -v`
Expected: `ModuleNotFoundError: No module named 'access_set_policy'`.

- [ ] **Step 3: Check the update payload against Cloudflare's schema**

```bash
curl -sL https://raw.githubusercontent.com/cloudflare/api-schemas/main/openapi.json -o /tmp/claude-cf-openapi.json
python3 - <<'EOF'
import json
s = json.load(open('/tmp/claude-cf-openapi.json'))
op = s['paths']['/accounts/{account_id}/access/apps/{app_id}']['put']
print(json.dumps(op['requestBody']['content']['application/json']['schema'], indent=1)[:3000])
EOF
rm /tmp/claude-cf-openapi.json
```

Expected: the request body is the self-hosted application schema with `policies` as a list of ids or `{id, precedence}` objects. Note every property marked `readOnly` there; `READ_ONLY` below must contain at least those (plus `policies`, which the script sets itself). Adjust the constant if the schema lists more.

- [ ] **Step 4: Implement**

`scripts/access_set_policy.py`:

```python
#!/usr/bin/env python3
"""Point an existing self-hosted Access application at exactly one reusable
Allow policy, saving the application's JSON first so it can be restored.

  python3 scripts/access_set_policy.py --hostname catan.signalwave.dev --policy-name Players \\
      --backup ~/.config/cloudflare/catan-app-before.json [--dry-run]
  python3 scripts/access_set_policy.py --hostname catan.signalwave.dev \\
      --restore ~/.config/cloudflare/catan-app-before.json [--dry-run]

Token: CLOUDFLARE_API_TOKEN or ~/.config/cloudflare/api-token (mode 600), as cf_publish.py.
Refuses unless exactly one self-hosted application has exactly that domain and
exactly one reusable policy has that name with decision allow. The application
is updated in place (its AUD tag must not change), never recreated.
"""
import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.cloudflare.com/client/v4"
TOKEN_FILE = Path.home() / ".config" / "cloudflare" / "api-token"
# Server-managed fields not sent back on update (checked against
# github.com/cloudflare/api-schemas, see Plan 4 Task 24 Step 3).
READ_ONLY = ("id", "uid", "aud", "created_at", "updated_at", "policies")


class Refused(Exception):
    pass


class Api:
    def __init__(self, token):
        self.token = token

    def _send(self, method, path, body=None, query=None):
        url = API + path + ("?" + urllib.parse.urlencode(query) if query else "")
        req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                payload = json.load(res)
        except urllib.error.HTTPError as e:
            payload = json.load(e)
        if not payload.get("success"):
            raise Refused(f"{method} {path}: {payload.get('errors')}")
        return payload["result"]

    def get(self, path, **query):
        return self._send("GET", path, query=query or None)

    def put(self, path, body):
        return self._send("PUT", path, body)


def load_token():
    if os.environ.get("CLOUDFLARE_API_TOKEN"):
        return os.environ["CLOUDFLARE_API_TOKEN"].strip()
    if stat.S_IMODE(TOKEN_FILE.stat().st_mode) & 0o077:
        raise Refused(f"{TOKEN_FILE} must be chmod 600")
    return TOKEN_FILE.read_text().strip()


def account(api):
    if os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        return os.environ["CLOUDFLARE_ACCOUNT_ID"]
    accounts = api.get("/accounts")
    if len(accounts) != 1:
        raise Refused(f"token sees {len(accounts)} accounts; set CLOUDFLARE_ACCOUNT_ID")
    return accounts[0]["id"]


def find_app(api, acct, hostname):
    apps = [a for a in api.get(f"/accounts/{acct}/access/apps")
            if a.get("type") == "self_hosted" and a.get("domain") == hostname]
    if len(apps) != 1:
        raise Refused(f"expected one self-hosted application for {hostname}, found {len(apps)}")
    return api.get(f"/accounts/{acct}/access/apps/{apps[0]['id']}")


def find_policy(api, acct, name):
    policies = [p for p in api.get(f"/accounts/{acct}/access/policies") if p.get("name") == name]
    if len(policies) != 1:
        raise Refused(f"expected one reusable policy named {name!r}, found {len(policies)}")
    if policies[0].get("decision") != "allow":
        raise Refused(f"policy {name!r} is {policies[0].get('decision')!r}, not allow")
    return policies[0]


def write_backup(path, app):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(app, f, indent=2)


def update(api, acct, app, policy_ids, dry_run, log):
    body = {k: v for k, v in app.items() if k not in READ_ONLY}
    body["policies"] = policy_ids
    current = [p["id"] for p in sorted(app.get("policies") or [], key=lambda p: p.get("precedence", 0))]
    log(f"{app['domain']}: policies {current} -> {policy_ids}" + (" (dry run)" if dry_run else ""))
    if dry_run:
        return app
    result = api.put(f"/accounts/{acct}/access/apps/{app['id']}", body)
    if result.get("aud") != app.get("aud"):
        raise Refused(f"the application's AUD changed ({app.get('aud')} -> {result.get('aud')}); restore from the backup")
    got = [p["id"] for p in sorted(result.get("policies") or [], key=lambda p: p.get("precedence", 0))]
    if got != policy_ids:
        raise Refused(f"policies are {got}, expected {policy_ids}")
    log(f"{app['domain']}: updated; AUD unchanged")
    return result


def set_policy(api, hostname, policy_name, backup, dry_run, log=print):
    acct = account(api)
    app = find_app(api, acct, hostname)
    policy = find_policy(api, acct, policy_name)
    if not dry_run:
        write_backup(backup, app)
        log(f"saved {hostname} to {backup}")
    return update(api, acct, app, [policy["id"]], dry_run, log)


def restore(api, hostname, backup, dry_run, log=print):
    saved = json.loads(Path(backup).expanduser().read_text())
    acct = account(api)
    app = find_app(api, acct, hostname)
    ids = [p["id"] for p in sorted(saved.get("policies") or [], key=lambda p: p.get("precedence", 0))]
    return update(api, acct, app, ids, dry_run, log)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hostname", required=True)
    p.add_argument("--policy-name")
    p.add_argument("--backup", help="where to save the application before changing it")
    p.add_argument("--restore", help="a backup written by an earlier run")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        api = Api(load_token())
        if a.restore:
            restore(api, a.hostname, a.restore, a.dry_run)
        elif a.policy_name and a.backup:
            set_policy(api, a.hostname, a.policy_name, a.backup, a.dry_run)
        else:
            p.error("give --policy-name and --backup, or --restore")
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `cd scripts && python3 -m unittest test_access_set_policy -v`
Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/access_set_policy.py scripts/test_access_set_policy.py
git commit -m "deploy: script to point an Access app at one policy, with rollback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Switch Catan (live)**

The user chose this (players for both). Dry run, then real run:

```bash
cd /home/garamizo/Azul-Board-Game-web
python3 scripts/access_set_policy.py --hostname catan.signalwave.dev --policy-name Players \
  --backup ~/.config/cloudflare/catan-app-before.json --dry-run
python3 scripts/access_set_policy.py --hostname catan.signalwave.dev --policy-name Players \
  --backup ~/.config/cloudflare/catan-app-before.json
```

Expected: `catan.signalwave.dev: policies ['f51aadbf…'] -> ['<Players id>']`, then `updated; AUD unchanged`.

- [ ] **Step 8: Acceptance**

1. API check (no emails printed): both applications list exactly the `Players` policy, and Catan's AUD equals the one in `~/catanatron/.env` (`CATAN_ACCESS_AUD`, compare without printing: `python3 -c` reading both and printing `same`/`different`).
2. Ask the user to have one `players` member who was not on Catan's old list open https://catan.signalwave.dev (should get in), and to try an email outside the group on both hostnames (refused).
3. Record the rollback command in the hand-off: `python3 scripts/access_set_policy.py --hostname catan.signalwave.dev --restore ~/.config/cloudflare/catan-app-before.json`.

- [ ] **Step 9: Catan's own docs (only with the user's go-ahead)**

`~/catanatron/documentation/deploy.md` still says the policy is "Invited → Include Emails". Ask the user whether to update it. If yes: `git -C ~/catanatron worktree add ../catanatron-players-policy -b docs/players-policy`, edit the Access paragraph there to say Catan's application uses the reusable policy `Players` (Include → rule group `players`, shared with azul.signalwave.dev), commit, and leave merging to the user.

---

### Task 25: Final verification

- [ ] **Step 1: Everything green**

```bash
cd /home/garamizo/Azul-Board-Game-web
make test
make web-test
make e2e
make desktop-smoke
(cd scripts && python3 -m unittest -v)
make serve-check
```

Expected: every command exits 0. Record the counts (tests passed) for the hand-off.

- [ ] **Step 2: Hand-off notes**

Collect for the final report: S4 numbers (p50, p95, CPU, worst gap), S2 result, the serve-check output, the user's live-check answers (Task 23 Step 7, Task 24 Step 8), the Catan rollback command, and the fixed ports that collide across worktrees (5080, 5081, 29648, 29649).
