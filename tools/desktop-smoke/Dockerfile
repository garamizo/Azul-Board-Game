# The desktop app loads AzulLibrary through pythonnet; this image has the
# .NET 10 runtime (from the SDK image) plus Python and pythonnet, no display.
FROM mcr.microsoft.com/dotnet/sdk:10.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /venv \
 && /venv/bin/pip install --no-cache-dir pythonnet==3.2.0
