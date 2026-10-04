# Publishing at azul.signalwave.dev

The serve stack (`docker-compose.serve.yml`) runs the game and a `cloudflared` connector. Cloudflare
Access admits the people in the Access rule group `players`, the same group as
catan.signalwave.dev. The server also verifies Access's signed JWT on every request except
`/api/health`, so the loopback port `127.0.0.1:29648` answers 401.

| Thing | Where |
| --- | --- |
| Who may play | Access rule group `players` (shared with Catan), reusable policy `Players` |
| Team domain, AUD, tunnel token | `.env.serve` (git-ignored, mode 600) |
| Games | docker volume `azul-serve_azul-data` |
| Logs | `make serve-logs [SERVICE=app\|cloudflared]` |

## One-time setup

Dashboard steps verified against developers.cloudflare.com on 2026-10-03. If a menu has moved,
re-check the linked page and update this section.

1. **API token**: the one in `~/.config/cloudflare/api-token` (permissions table in the
   `publishing-behind-cloudflare` skill; create it from
   [Create API token](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/)).
   Check it is accepted (expect `active`):

   ```bash
   printf 'Authorization: Bearer %s\n' "$(cat ~/.config/cloudflare/api-token)" |
     curl -s -H @- https://api.cloudflare.com/client/v4/user/tokens/verify |
     python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["status"])'
   ```

2. **Access application, tunnel, route, DNS** with `cf_publish.py` (the API, not dashboard
   click-steps). Dry run first and review it:

   ```bash
   S=~/.claude/skills/publishing-behind-cloudflare/scripts
   python3 $S/cf_publish.py --hostname azul.signalwave.dev --service http://app:8080 \
     --tunnel azul --app-name Azul --rule-group players --policy-name Players --session 24h --dry-run
   ```

   Then the real run, which writes the tunnel token to a file (mode 600), never to the terminal:

   ```bash
   python3 $S/cf_publish.py --hostname azul.signalwave.dev --service http://app:8080 \
     --tunnel azul --app-name Azul --rule-group players --policy-name Players --session 24h \
     --token-out ~/.config/cloudflare/azul-tunnel-token
   ```

   It reuses the `players` group, creates the `Players` policy if missing, and creates the Access
   application before the route. It prints the team domain and the AUD tag. To find them again, see
   [Validate the JWT](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/):
   go to **Zero Trust** > **Access controls** > **Applications**, select **Configure** for the
   application, and from **Additional settings** copy the **Application Audience (AUD) Tag**.
3. **`.env.serve`**: `install -m 600 .env.serve.example .env.serve`; fill the team domain and AUD
   printed by `cf_publish.py`; append the token without printing it (delete the empty
   `CLOUDFLARE_TUNNEL_TOKEN=` line first):

   ```bash
   sed -i '/^CLOUDFLARE_TUNNEL_TOKEN=$/d' .env.serve
   printf 'CLOUDFLARE_TUNNEL_TOKEN=%s\n' "$(cat ~/.config/cloudflare/azul-tunnel-token)" >> .env.serve
   ```

4. **Start**: `make serve-app && make serve-check-local`, then `make serve && make serve-check`.

## Inviting someone

Add their email to the rule group `players`: go to **Zero Trust** > **Access controls** >
**Policies** and select the **Rule groups** tab
([docs](https://developers.cloudflare.com/cloudflare-one/access-controls/policies/groups/)); the
page covers creating a group, so find editing a group's rules from it. This also admits them to
Catan. No restart. Removing someone takes effect when their Access session (24 h) ends.

## Day to day

- Deploy a change: `git pull && make serve` (games survive; bots resume).
- Backup: `make backup` (files in `~/backups/azul`). Restore: `make restore FILE=...`. It checks
  the file first (SQLite integrity, schema version, `games` table) and refuses an invalid one
  before stopping anything. It then stops the app, keeps the current database in the volume as
  `azul.db.before-restore-<timestamp>` (the name is printed; delete old ones by hand), swaps the
  backup in and starts the app. To undo it, copy the safety copy out
  (`docker run --rm -v azul-serve_azul-data:/data -v ~/backups/azul:/out alpine cp /data/azul.db.before-restore-<timestamp> /out/`)
  and `make restore FILE=` that file.
- Rotate the tunnel token
  ([docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/remote-tunnel-permissions/#rotate-a-token-without-service-disruption)):
  go to **Networking** > **Tunnels**, select the tunnel, and on the **Overview** tab select
  **Refresh token**. Copy the new token from the installation command it shows (do not run the
  command; the token is the `eyJ...` string). Put it in `CLOUDFLARE_TUNNEL_TOKEN` in `.env.serve`
  without printing it (for example `$EDITOR .env.serve`), then `make serve`. After the refresh,
  `cloudflared` can no longer open new connections with the old token, but existing connectors keep
  running and the tunnel keeps serving, so the running connector stays up until `make serve`
  replaces it. The page's no-disruption rotation assumes at least two replicas; with one connector
  expect a brief gap while it restarts. If the token leaked, follow
  [Rotate a compromised token](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/remote-tunnel-permissions/#rotate-a-compromised-token)
  (it also deletes the existing connections).
- Move to another machine (never run both at once: two connectors would split players between
  two databases): here `make serve-down`, then `make backup` (the backup container reads the
  stopped stack's volume directly). Copy `.env.serve` and the backup to the new machine; there
  `make serve-app`, `make restore FILE=...`, `make serve-check-local`, then `make serve`.

## Checking it live

- An invited email reaches the lobby; someone else is refused at Cloudflare's page.
- Sign out in another tab (`/cdn-cgi/access/logout`) with a game open: the page reloads into the
  sign-in page once, no loop.
- Two phones play a game to the end.
