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

    def get(self, path, **query):
        if path.endswith("/accounts"):
            return [{"id": "acct"}]
        if path.endswith("/access/apps"):
            return self.apps
        if path.endswith("/access/policies"):
            return self.policies
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


PLAYERS = {"id": "players-policy", "name": "Players", "decision": "allow"}


class SetPolicyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.backup = os.path.join(self.dir, "catan-before.json")

    def test_dry_run_writes_nothing(self):
        api = FakeApi([catan()], [PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=True, log=lambda *_: None)
        self.assertEqual(api.writes, [])
        self.assertFalse(os.path.exists(self.backup))

    def test_backup_first_then_only_the_policy_changes(self):
        api = FakeApi([catan()], [PLAYERS])
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
            ([catan(), catan(id="app2")], [PLAYERS]),
            ([], [PLAYERS]),
            ([catan(type="ssh")], [PLAYERS]),
            ([catan()], []),
            ([catan()], [PLAYERS, dict(PLAYERS, id="dup")]),
            ([catan()], [dict(PLAYERS, decision="bypass")]),
        ]:
            with self.subTest(apps=len(apps), policies=policies):
                api = FakeApi(apps, policies)
                with self.assertRaises(asp.Refused):
                    asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
                self.assertEqual(api.writes, [])

    def test_aud_change_is_detected(self):
        api = FakeApi([catan()], [PLAYERS])
        original_put = api.put
        api.put = lambda path, body: dict(original_put(path, body), aud="NEW")
        with self.assertRaises(asp.Refused):
            asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)

    def test_rerun_is_a_no_op_and_keeps_the_original_backup(self):
        api = FakeApi([catan()], [PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(len(api.writes), 1)
        with open(self.backup) as f:
            self.assertEqual(json.load(f)["policies"][0]["id"], "friends")

    def test_an_existing_backup_is_never_overwritten(self):
        with open(self.backup, "w") as f:
            f.write("{}")
        api = FakeApi([catan()], [PLAYERS])
        with self.assertRaises(asp.Refused):
            asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(api.writes, [])

    def test_restore_puts_back_the_saved_policies(self):
        api = FakeApi([catan()], [PLAYERS])
        asp.set_policy(api, "catan.signalwave.dev", "Players", self.backup, dry_run=False, log=lambda *_: None)
        asp.restore(api, "catan.signalwave.dev", self.backup, dry_run=False, log=lambda *_: None)
        self.assertEqual(api.writes[-1][1]["policies"], ["friends"])


if __name__ == "__main__":
    unittest.main()
