#!/usr/bin/env bash
# Restores a stack's database from a backup made by `make backup`.
#   scripts/restore.sh <serve|dev> <backup file>
# The backup is checked first (integrity, schema_version, games table) in a
# throwaway container; an invalid file is refused before the app stops. The
# current database is kept as /data/azul.db.before-restore-<timestamp>, and
# the backup replaces azul.db with a rename, so a failure part way never
# leaves a half-copied azul.db.
set -euo pipefail

stack=${1:-}
file=${2:-}
case "$stack" in
  serve) vol=azul-serve_azul-data; compose=(docker compose -f docker-compose.serve.yml --env-file .env.serve) ;;
  dev) vol=azul-dev_azul-dev-data; compose=(docker compose -f docker-compose.dev.yml) ;;
  *) echo "restore: STACK must be serve or dev" >&2; exit 2 ;;
esac
if [[ -z "$file" || ! -f "$file" ]]; then
  echo "restore: FILE=... must name an existing backup file" >&2
  exit 2
fi
file=$(realpath "$file")

echo "Checking $file ..."
if ! docker run --rm -v "$file:/in.db:ro" alpine sh -c '
    set -e
    apk add -q sqlite
    cp /in.db /tmp/check.db
    [ "$(sqlite3 -bail /tmp/check.db "PRAGMA integrity_check;")" = ok ]
    v=$(sqlite3 -bail /tmp/check.db "SELECT MAX(v) FROM schema_version;")
    [ -n "$v" ]
    sqlite3 -bail /tmp/check.db "SELECT COUNT(*) FROM games;" >/dev/null
    echo "backup ok: schema version $v"'; then
  echo "restore: $file is not a valid Azul database; nothing was changed" >&2
  exit 1
fi

ts=$(date -u +%Y%m%d-%H%M%S)  # UTC, like the backup names
safety="azul.db.before-restore-$ts"
"${compose[@]}" stop app
# The safety copy goes through sqlite3 .backup so that anything still in
# the old -wal file is included. The old -wal/-shm are removed before the
# rename: a WAL left next to a different database could be replayed into it.
docker run --rm -v "$vol:/data" -v "$file:/in.db:ro" -e SAFETY="$safety" alpine sh -c '
    set -e
    apk add -q sqlite
    if [ -f /data/azul.db ]; then
      sqlite3 -bail /data/azul.db ".backup /data/$SAFETY"
      chown 1654:1654 "/data/$SAFETY"
    fi
    cp /in.db /data/azul.db.new
    chown 1654:1654 /data/azul.db.new
    rm -f /data/azul.db-wal /data/azul.db-shm
    mv /data/azul.db.new /data/azul.db'
"${compose[@]}" up -d --wait app
echo "Restored $file. The previous database is kept in the volume $vol as $safety."
