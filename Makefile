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

.PHONY: dotnet build test desktop-smoke dev-server

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
