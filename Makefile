# Every .NET command runs in the pinned SDK container as the host user, so
# nothing is installed on the host and no root-owned files land in the tree.
SHELL := /bin/bash
UID := $(shell id -u)
GID := $(shell id -g)
SDK_IMAGE := mcr.microsoft.com/dotnet/sdk:10.0
NUGET_DIR := $(HOME)/.nuget/packages
DOTNET := docker run --rm -i --user $(UID):$(GID) \
	-e HOME=/tmp -e DOTNET_CLI_HOME=/tmp -e NUGET_PACKAGES=/nuget \
	-e DOTNET_NOLOGO=1 -e DOTNET_CLI_TELEMETRY_OPTOUT=1 -e DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
	-v $(CURDIR):/src -v $(NUGET_DIR):/nuget -w /src $(SDK_IMAGE) dotnet

.PHONY: dotnet build test desktop-smoke dev-server web-test e2e-publish e2e-server-start e2e-server-restart e2e-server-stop e2e serve serve-app serve-check serve-check-local serve-logs serve-down dev-stack dev-stack-down

$(NUGET_DIR):
	mkdir -p $@

dotnet: | $(NUGET_DIR)
	$(DOTNET) $(ARGS)

build: | $(NUGET_DIR)
	$(DOTNET) build Azul.slnx -c Release

test: | $(NUGET_DIR)
	$(DOTNET) test Azul.slnx $(if $(FILTER),--filter "$(FILTER)",)

desktop-smoke: | $(NUGET_DIR)
	$(DOTNET) build AzulLibrary/AzulLibrary.csproj -c Release
	docker build -q -t azul-desktop-smoke tools/desktop-smoke
	docker run --rm --user $(UID):$(GID) -e HOME=/tmp -v $(CURDIR):/src -w /src \
		azul-desktop-smoke /venv/bin/python tools/desktop-smoke/smoke.py

# The server in dev mode (no Access; identity from X-Dev-User or the
# azul_dev_user cookie) on http://127.0.0.1:5080, serving web/dist if built.
dev-server: | $(NUGET_DIR)
	mkdir -p .data/dev
	docker run --rm -i --user $(UID):$(GID) -p 127.0.0.1:5080:8080 \
		-e HOME=/tmp -e DOTNET_CLI_HOME=/tmp -e NUGET_PACKAGES=/nuget -e DOTNET_NOLOGO=1 \
		-e ASPNETCORE_URLS=http://0.0.0.0:8080 -e AZUL_DATA_DIR=/src/.data/dev \
		-e AZUL_WEB_ROOT=/src/web/dist -e AZUL_BOT_THINK_SECONDS=$${AZUL_BOT_THINK_SECONDS:-1} \
		-v $(CURDIR):/src -v $(NUGET_DIR):/nuget -w /src $(SDK_IMAGE) \
		dotnet run --project server/AzulServer --no-launch-profile

web-test:
	cd web && npm run check && npm test

# End-to-end: the published server in dev mode, in the ASP.NET runtime image,
# on 127.0.0.1:5081 with a fresh database. Fixed port and container name per
# checkout directory; two worktrees running e2e at once collide on the port.
E2E_NAME := azul-e2e-$(notdir $(CURDIR))
E2E_WAIT = for i in $$(seq 1 60); do curl -sf http://127.0.0.1:5081/api/health >/dev/null && exit 0; sleep 1; done; docker logs $(E2E_NAME); exit 1

e2e-publish: | $(NUGET_DIR)
	$(DOTNET) publish server/AzulServer/AzulServer.csproj -c Release -o /src/.data/e2e-server

e2e-server-start: e2e-publish
	rm -rf .data/e2e-db && mkdir -p .data/e2e-db
	docker rm -f $(E2E_NAME) >/dev/null 2>&1 || true
	docker run -d --name $(E2E_NAME) --user $(UID):$(GID) -p 127.0.0.1:5081:8080 \
		-e ASPNETCORE_HTTP_PORTS=8080 -e AZUL_DATA_DIR=/data -e AZUL_WEB_ROOT=/web \
		-e AZUL_BOT_THINK_SECONDS=0.2 -e AZUL_MIN_MOVE_DELAY_SECONDS=0.2 \
		-v $(CURDIR)/.data/e2e-server:/app:ro -v $(CURDIR)/.data/e2e-db:/data -v $(CURDIR)/web/dist:/web:ro \
		mcr.microsoft.com/dotnet/aspnet:10.0 dotnet /app/AzulServer.dll >/dev/null
	@$(E2E_WAIT)

e2e-server-restart:
	docker restart $(E2E_NAME) >/dev/null
	@$(E2E_WAIT)

e2e-server-stop:
	docker rm -f $(E2E_NAME) >/dev/null 2>&1 || true

e2e:
	cd web && npm run build && npx playwright test

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
