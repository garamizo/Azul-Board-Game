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
PER_PAGE = 50
TOKEN_FILE = Path.home() / ".config" / "cloudflare" / "api-token"
# Server-managed fields not sent back on update (checked against
# github.com/cloudflare/api-schemas, see Plan 4 Task 24 Step 3).
READ_ONLY = ("id", "uid", "aud", "created_at", "updated_at", "policies")


class Refused(Exception):
    pass


class Api:
    def __init__(self, token):
        self.token = token

    def _call(self, method, path, body=None, query=None):
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
        return payload

    def get(self, path, **query):
        return self._call("GET", path, query=query or None)["result"]

    def list(self, path, **query):
        """Every page of a list endpoint (result_info.total_pages)."""
        items, page = [], 1
        while True:
            payload = self._call("GET", path, query={**query, "page": page, "per_page": PER_PAGE})
            items.extend(payload.get("result") or [])
            info = payload.get("result_info") or {}
            if page >= int(info.get("total_pages") or 1):
                return items
            page += 1

    def put(self, path, body):
        return self._call("PUT", path, body)["result"]


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
    apps = [a for a in api.list(f"/accounts/{acct}/access/apps")
            if a.get("type") == "self_hosted" and a.get("domain") == hostname]
    if len(apps) != 1:
        raise Refused(f"expected one self-hosted application for {hostname}, found {len(apps)}")
    return api.get(f"/accounts/{acct}/access/apps/{apps[0]['id']}")


def find_policy(api, acct, name, all_policies):
    policies = [p for p in all_policies if p.get("name") == name]
    if len(policies) != 1:
        raise Refused(f"expected one reusable policy named {name!r}, found {len(policies)}")
    if policies[0].get("decision") != "allow":
        raise Refused(f"policy {name!r} is {policies[0].get('decision')!r}, not allow")
    return policies[0]


def check_removable(app, all_policies):
    """Cloudflare deletes a legacy (app-scoped) policy once it leaves its application,
    so the rollback could not bring it back; only reusable policies may be swapped out."""
    by_id = {p["id"]: p for p in all_policies}
    for pid in policy_ids(app):
        pol = by_id.get(pid)
        if pol is None or pol.get("reusable") is False:
            name = (pol or {}).get("name") or next((p.get("name") for p in app.get("policies") or [] if p.get("id") == pid), pid)
            raise Refused(f"policy {name!r} ({pid}) on {app.get('domain')} is not reusable, so removing it would delete it "
                          f"and the rollback could not restore it; make it reusable first "
                          f"(PUT /accounts/<acct>/access/policies/{pid}/make_reusable) or switch by hand")


def write_backup(path, app):
    """Creates the backup; never replaces one (it holds the original configuration)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise Refused(f"{path} already exists and may hold the original configuration; move it away first")
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


def policy_ids(app):
    return [p["id"] for p in sorted(app.get("policies") or [], key=lambda p: p.get("precedence", 0))]


def set_policy(api, hostname, policy_name, backup, dry_run, log=print):
    acct = account(api)
    app = find_app(api, acct, hostname)
    all_policies = api.list(f"/accounts/{acct}/access/policies")
    policy = find_policy(api, acct, policy_name, all_policies)
    check_removable(app, all_policies)
    if policy_ids(app) == [policy["id"]]:
        log(f"{hostname} already uses only {policy_name!r}; nothing to do")
        return app
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
