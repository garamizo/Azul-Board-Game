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
