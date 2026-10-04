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
