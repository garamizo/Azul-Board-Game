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

.PHONY: dotnet build test desktop-smoke

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
