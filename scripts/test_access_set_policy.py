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

    def list(self, path, **query):
        if path.endswith("/access/apps"):
            return self.apps
        if path.endswith("/access/policies"):
            return self.policies
        raise KeyError(path)

    def get(self, path, **query):
        if path.endswith("/accounts"):
            return [{"id": "acct"}]
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


PLAYERS = {"id": "players-policy", "name": "Players", "decision": "allow", "reusable": True}
FRIENDS = {"id": "friends", "name": "Friends", "decision": "allow", "reusable": True}


class SetPolicyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.backup = os.path.join(self.dir, "catan-before.json")

    def test_dry_run_writes_nothing(self):
        api = FakeApi([catan()], [FRIENDS, PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=True, log=lambda *_: None)
        self.assertEqual(api.writes, [])
        self.assertFalse(os.path.exists(self.backup))

    def test_backup_first_then_only_the_policy_changes(self):
        api = FakeApi([catan()], [FRIENDS, PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(stat.S_IMODE(os.stat(self.backup).st_mode), 0o600)
        with open(self.backup) as f:
            self.assertEqual(json.load(f)["policies"][0]["id"], "friends")
        path, body = api.writes[0]
        self.assertTrue(path.endswith("/access/apps/app1"))
        self.assertEqual(body["policies"], ["players-policy"])
        for key in asp.READ_ONLY:
            if key != "policies":  # stripped, then set by the script
                self.assertNotIn(key, body)
        self.assertEqual(body["domain"], "catan.signalwave.dev")
        self.assertEqual(body["session_duration"], "24h")

    def test_refuses_ambiguous_or_missing_targets(self):
        for apps, policies in [
            ([catan(), catan(id="app2")], [FRIENDS, PLAYERS]),
            ([], [FRIENDS, PLAYERS]),
            ([catan(type="ssh")], [FRIENDS, PLAYERS]),
            ([catan()], [FRIENDS]),
            ([catan()], [FRIENDS, PLAYERS, dict(PLAYERS, id="dup")]),
            ([catan()], [FRIENDS, dict(PLAYERS, decision="bypass")]),
        ]:
            with self.subTest(apps=len(apps), policies=policies):
                api = FakeApi(apps, policies)
                with self.assertRaises(asp.Refused):
                    asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
                self.assertEqual(api.writes, [])

    def test_aud_change_is_detected(self):
        api = FakeApi([catan()], [FRIENDS, PLAYERS])
        original_put = api.put
        api.put = lambda path, body: dict(original_put(path, body), aud="NEW")
        with self.assertRaises(asp.Refused):
            asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)

    def test_rerun_is_a_no_op_and_keeps_the_original_backup(self):
        api = FakeApi([catan()], [FRIENDS, PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(len(api.writes), 1)
        with open(self.backup) as f:
            self.assertEqual(json.load(f)["policies"][0]["id"], "friends")

    def test_an_existing_backup_is_never_overwritten(self):
        with open(self.backup, "w") as f:
            f.write("{}")
        api = FakeApi([catan()], [FRIENDS, PLAYERS])
        with self.assertRaises(asp.Refused):
            asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(api.writes, [])

    def test_restore_puts_back_the_saved_policies(self):
        api = FakeApi([catan()], [FRIENDS, PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        asp.restore(api, "catan.signalwave.dev", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(api.writes[-1][1]["policies"], ["friends"])

    def test_refuses_a_policy_that_is_not_reusable_or_not_listed(self):
        for policies in ([PLAYERS], [dict(FRIENDS, reusable=False), PLAYERS]):
            with self.subTest(policies=len(policies)):
                api = FakeApi([catan()], policies)
                with self.assertRaises(asp.Refused) as cm:
                    asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
                self.assertIn("make_reusable", str(cm.exception))
                self.assertFalse(os.path.exists(self.backup))
                self.assertEqual(api.writes, [])


class PagedApi(asp.Api):
    """Serves list endpoints two items per page, to exercise Api.list."""

    def __init__(self, items):
        super().__init__("t")
        self.items = items

    def _call(self, method, path, body=None, query=None):
        page, per = query["page"], 2
        chunk = self.items[(page - 1) * per: page * per]
        return {"success": True, "result": chunk, "result_info": {"total_pages": -(-len(self.items) // per)}}


class PaginationTests(unittest.TestCase):
    def test_list_reads_every_page(self):
        api = PagedApi([{"id": str(i)} for i in range(5)])
        self.assertEqual(len(api.list("/x")), 5)

    def test_duplicate_app_on_page_two_is_refused(self):
        apps = [catan(id="a", domain="other1"), catan(id="b", domain="other2"),
                catan(id="c"), catan(id="d", domain="other3"), catan(id="e")]
        with self.assertRaises(asp.Refused):
            asp.find_app(PagedApi(apps), "acct", "catan.signalwave.dev")


if __name__ == "__main__":
    unittest.main()
