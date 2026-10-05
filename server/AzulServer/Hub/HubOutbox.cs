using AzulServer.Data;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;

namespace AzulServer.Hub;

public sealed record HubRow(string GameId, string Body, string Status, int Attempts, string NextAttemptAt,
    int? LastStatus, string? LastError, string CreatedAt, string? SentAt, string? AlertedAt, string? LeaseId);

public sealed record HubClaim(string Body, int Attempts, string LeaseId);

public sealed record HubFailure(string GameId, int? LastStatus, string? LastError);

public sealed record HubSummary(IReadOnlyDictionary<string, int> Counts, IReadOnlyList<HubFailure> Failed, string? OldestPendingCreatedAt);

/// Every statement on hub_reports. Timestamps are GameService's "O" UTC
/// strings, which compare correctly as text. Outcomes are written only under
/// the lease that claimed the row (spec 6.3, lease ownership). A retry or a
/// rebuild refuses a row whose lease is live, since the hub may already have
/// accepted the body in flight; once the lease has expired it clears
/// lease_id, so a late answer cannot land on a reset or replaced row.
public static class HubOutbox
{
    static SqliteCommand Command(SqliteConnection c, SqliteTransaction? tx, string sql, params (string Name, object? Value)[] args)
    {
        var cmd = c.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        foreach (var (name, value) in args)
            cmd.Parameters.AddWithValue(name, value ?? DBNull.Value);
        return cmd;
    }

    static List<string> Ids(SqliteConnection c, SqliteTransaction? tx, string sql, params (string, object?)[] args)
    {
        using var cmd = Command(c, tx, sql, args);
        using var r = cmd.ExecuteReader();
        var ids = new List<string>();
        while (r.Read()) ids.Add(r.GetString(0));
        return ids;
    }

    static int Exec(SqliteConnection c, SqliteTransaction? tx, string sql, params (string, object?)[] args)
    {
        using var cmd = Command(c, tx, sql, args);
        return cmd.ExecuteNonQuery();
    }

    /// ON CONFLICT DO NOTHING (not INSERT OR IGNORE): only the primary key
    /// conflict is ignored; any other failure aborts the finishing commit.
    public static bool Queue(SqliteConnection c, SqliteTransaction tx, string gameId, string body, string now) =>
        Exec(c, tx, "INSERT INTO hub_reports(game_id, body, status, attempts, next_attempt_at, created_at) " +
                    "VALUES ($g, $b, 'pending', 0, $now, $now) ON CONFLICT(game_id) DO NOTHING",
            ("$g", gameId), ("$b", body), ("$now", now)) == 1;

    public static bool Exists(SqliteConnection c, SqliteTransaction? tx, string gameId)
    {
        using var cmd = Command(c, tx, "SELECT 1 FROM hub_reports WHERE game_id = $g", ("$g", gameId));
        return cmd.ExecuteScalar() is not null;
    }

    public static HubRow? Get(SqliteConnection c, SqliteTransaction? tx, string gameId)
    {
        using var cmd = Command(c, tx,
            "SELECT game_id, body, status, attempts, next_attempt_at, last_status, last_error, created_at, sent_at, alerted_at, lease_id " +
            "FROM hub_reports WHERE game_id = $g", ("$g", gameId));
        using var r = cmd.ExecuteReader();
        if (!r.Read()) return null;
        string? S(int i) => r.IsDBNull(i) ? null : r.GetString(i);
        return new HubRow(r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), r.GetString(4),
            r.IsDBNull(5) ? null : r.GetInt32(5), S(6), r.GetString(7), S(8), S(9), S(10));
    }

    public static List<string> MissingIds(SqliteConnection c) => Ids(c, null,
        "SELECT id FROM games WHERE status = 'finished' AND hub_tracked = 1 " +
        "AND id NOT IN (SELECT game_id FROM hub_reports) ORDER BY id");

    /// Reconcile (spec 6.2): build and queue each finished tracked game that
    /// has no report. One BEGIN IMMEDIATE transaction per game (deferred:
    /// false) reads it and writes the row, so a concurrent Delete cannot
    /// interleave. With `logged`, each game's failure is an error the first
    /// time only (later cycles log it at debug), so a game that stays
    /// unbuildable does not log an error every cycle.
    public static int QueueMissing(Db db, string? publicOrigin, string now, ILogger log, ISet<string>? logged = null)
    {
        LogLevel Level(string id) => logged is null || logged.Add(id) ? LogLevel.Error : LogLevel.Debug;
        List<string> ids;
        using (var c = db.Open()) ids = MissingIds(c);
        int queued = 0;
        foreach (var id in ids)
        {
            try
            {
                using var c = db.Open();
                using var tx = c.BeginTransaction(deferred: false);
                var g = GameStore.Load(c, id, tx);
                if (g is null || Exists(c, tx, id)) continue;
                var built = HubReport.Build(g, HubReport.BotPlayedSeats(c, tx, id, null), publicOrigin);
                if (built.Body is null)
                {
                    log.Log(Level(id), "hub report for game {Game} cannot be built: {Reason}", id, built.Skip);
                    continue;
                }
                if (Queue(c, tx, id, built.Body, now)) queued++;
                tx.Commit();
            }
            catch (Exception e)
            {
                log.Log(Level(id), e, "hub report for game {Game} could not be queued", id);
            }
        }
        return queued;
    }

    public static List<string> DueIds(SqliteConnection c, string now, int limit) => Ids(c, null,
        "SELECT game_id FROM hub_reports WHERE status = 'pending' AND next_attempt_at <= $now " +
        "ORDER BY next_attempt_at LIMIT $n", ("$now", now), ("$n", limit));

    /// Takes a lease: the row is not due again before `leaseUntil`, and only
    /// the returned LeaseId may record this attempt's outcome.
    public static HubClaim? TryClaim(SqliteConnection c, string gameId, string now, string leaseUntil)
    {
        var leaseId = Guid.NewGuid().ToString("N");
        using var tx = c.BeginTransaction(deferred: false);
        if (Exec(c, tx, "UPDATE hub_reports SET next_attempt_at = $until, lease_id = $lease " +
                        "WHERE game_id = $g AND status = 'pending' AND next_attempt_at <= $now",
                ("$until", leaseUntil), ("$lease", leaseId), ("$g", gameId), ("$now", now)) != 1)
            return null;
        var row = Get(c, tx, gameId)!;
        tx.Commit();
        return new HubClaim(row.Body, row.Attempts, leaseId);
    }

    public static bool MarkSent(SqliteConnection c, string gameId, string leaseId, int status, string now) =>
        Exec(c, null, "UPDATE hub_reports SET status = 'sent', attempts = attempts + 1, last_status = $s, " +
                      "last_error = NULL, sent_at = $now, lease_id = NULL WHERE game_id = $g AND lease_id = $l",
            ("$s", status), ("$now", now), ("$g", gameId), ("$l", leaseId)) == 1;

    public static bool MarkFailed(SqliteConnection c, string gameId, string leaseId, int status, string? error) =>
        Exec(c, null, "UPDATE hub_reports SET status = 'failed', attempts = attempts + 1, last_status = $s, " +
                      "last_error = $e, lease_id = NULL WHERE game_id = $g AND lease_id = $l",
            ("$s", status), ("$e", error), ("$g", gameId), ("$l", leaseId)) == 1;

    public static bool MarkRetry(SqliteConnection c, string gameId, string leaseId, int? status, string? error, string nextAttemptAt) =>
        Exec(c, null, "UPDATE hub_reports SET attempts = attempts + 1, last_status = $s, last_error = $e, " +
                      "next_attempt_at = $next, lease_id = NULL WHERE game_id = $g AND lease_id = $l",
            ("$s", status), ("$e", error), ("$next", nextAttemptAt), ("$g", gameId), ("$l", leaseId)) == 1;

    public static List<string> StuckIds(SqliteConnection c, string createdBefore, string alertedBefore) => Ids(c, null,
        "SELECT game_id FROM hub_reports WHERE status = 'pending' AND created_at < $created " +
        "AND (alerted_at IS NULL OR alerted_at < $alerted) ORDER BY created_at",
        ("$created", createdBefore), ("$alerted", alertedBefore));

    public static void MarkAlerted(SqliteConnection c, string gameId, string now) =>
        Exec(c, null, "UPDATE hub_reports SET alerted_at = $now WHERE game_id = $g", ("$now", now), ("$g", gameId));

    public static HubSummary Summary(SqliteConnection c)
    {
        var counts = new Dictionary<string, int> { ["pending"] = 0, ["sent"] = 0, ["failed"] = 0 };
        using (var cmd = Command(c, null, "SELECT status, COUNT(*) FROM hub_reports GROUP BY status"))
        using (var r = cmd.ExecuteReader())
            while (r.Read()) counts[r.GetString(0)] = r.GetInt32(1);
        var failed = new List<HubFailure>();
        using (var cmd = Command(c, null, "SELECT game_id, last_status, last_error FROM hub_reports " +
                                          "WHERE status = 'failed' ORDER BY created_at DESC LIMIT 20"))
        using (var r = cmd.ExecuteReader())
            while (r.Read())
                failed.Add(new HubFailure(r.GetString(0), r.IsDBNull(1) ? null : r.GetInt32(1), r.IsDBNull(2) ? null : r.GetString(2)));
        using var oldest = Command(c, null, "SELECT MIN(created_at) FROM hub_reports WHERE status = 'pending'");
        return new HubSummary(counts, failed, oldest.ExecuteScalar() as string);
    }

    public static List<string> FailedIds(SqliteConnection c) =>
        Ids(c, null, "SELECT game_id FROM hub_reports WHERE status = 'failed' ORDER BY created_at");

    /// The outcome of a retry or rebuild that finds a send in flight.
    public const string InFlight = "in flight; retry later";

    /// A row whose lease is live: claimed, and its lease (next_attempt_at)
    /// not yet expired. Retry and rebuild leave it alone (spec 6.3).
    const string NotLeased = "NOT (hub_reports.lease_id IS NOT NULL AND hub_reports.next_attempt_at > $now)";

    /// One conditional statement (no read-then-update). It skips a sent row
    /// and a row with a live lease; it clears an expired lease, which drops
    /// the late answer of that send, and restarts the backoff (attempts = 0).
    public static string ResetForRetry(SqliteConnection c, string gameId, string now)
    {
        if (Exec(c, null, "UPDATE hub_reports SET status = 'pending', attempts = 0, lease_id = NULL, next_attempt_at = $now " +
                          $"WHERE game_id = $g AND status <> 'sent' AND {NotLeased}", ("$now", now), ("$g", gameId)) == 1)
            return "pending";
        return Unchanged(Get(c, null, gameId));   // only to say why nothing changed
    }

    static string Unchanged(HubRow? row) => row switch
    {
        null => "no report",
        { Status: "sent" } => "already sent",
        _ => InFlight,
    };

    /// Rebuild from the stored game and replace the row only when the build
    /// succeeds; a sent row, a row with a live lease and a report whose game
    /// is gone are never touched (spec 6.4). One BEGIN IMMEDIATE transaction
    /// reads the game and writes the row; the replacing statement clears an
    /// expired lease_id.
    public static string Rebuild(Db db, string gameId, string? publicOrigin, string now)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction(deferred: false);
        var g = GameStore.Load(c, gameId, tx);
        if (g is null) return "game deleted";
        var built = HubReport.Build(g, HubReport.BotPlayedSeats(c, tx, gameId, null), publicOrigin);
        if (built.Body is null) return $"unbuildable: {built.Skip}";
        int changed = Exec(c, tx,
            "INSERT INTO hub_reports(game_id, body, status, attempts, next_attempt_at, created_at) " +
            "VALUES ($g, $b, 'pending', 0, $now, $now) " +
            "ON CONFLICT(game_id) DO UPDATE SET body = excluded.body, status = 'pending', attempts = 0, " +
            "next_attempt_at = excluded.next_attempt_at, last_status = NULL, last_error = NULL, sent_at = NULL, " +
            $"alerted_at = NULL, lease_id = NULL WHERE hub_reports.status <> 'sent' AND {NotLeased}",
            ("$g", gameId), ("$b", built.Body), ("$now", now));
        if (changed == 0) return Unchanged(Get(c, tx, gameId));
        tx.Commit();
        return "rebuilt";
    }
}
