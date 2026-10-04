using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubOutboxTests : IDisposable
{
    readonly Db db = NewDb();
    public void Dispose() => SqliteConnection.ClearAllPools();

    const string T0 = "2026-10-04T19:00:00.0000000Z";
    const string T1 = "2026-10-04T19:01:00.0000000Z";
    const string T5 = "2026-10-04T19:05:00.0000000Z";
    const string T9 = "2026-10-04T19:09:00.0000000Z";
    static readonly SeatRecord[] Two = [Human(0, "alice@example.com"), Bot(1)];

    void InsertGame(GameRecord g)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        GameStore.Insert(c, tx, g);
        tx.Commit();
    }

    void QueueRow(string id, string body = "{\"x\":1}", string now = T0)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        HubOutbox.Queue(c, tx, id, body, now);
        tx.Commit();
    }

    HubRow Row(string id)
    {
        using var c = db.Open();
        return HubOutbox.Get(c, null, id)!;
    }

    /// Claims at T0 (lease to T5) and returns the lease id.
    string Claim(string id)
    {
        using var c = db.Open();
        return HubOutbox.TryClaim(c, id, T0, T5)!.LeaseId;
    }

    [Fact]
    public void QueueIsInsertOnce()
    {
        QueueRow("g", "{\"first\":1}");
        QueueRow("g", "{\"second\":1}");
        var row = Row("g");
        Assert.Equal(("{\"first\":1}", "pending", 0, T0, (string?)null),
            (row.Body, row.Status, row.Attempts, row.NextAttemptAt, row.LeaseId));
    }

    [Fact]
    public void AClaimIsALeaseWithItsOwnId()
    {
        QueueRow("g");
        using var c = db.Open();
        Assert.Equal(["g"], HubOutbox.DueIds(c, T0, 10));
        var first = HubOutbox.TryClaim(c, "g", T0, T5)!;
        Assert.Equal(("{\"x\":1}", 0), (first.Body, first.Attempts));
        Assert.Equal(first.LeaseId, HubOutbox.Get(c, null, "g")!.LeaseId);
        Assert.Null(HubOutbox.TryClaim(c, "g", T0, T5));   // leased
        Assert.Empty(HubOutbox.DueIds(c, T1, 10));
        var second = HubOutbox.TryClaim(c, "g", T5, T9)!;   // lease expired
        Assert.NotEqual(first.LeaseId, second.LeaseId);
        Assert.False(HubOutbox.MarkSent(c, "g", first.LeaseId, 201, T5));   // the expired claim's outcome is stale
        Assert.True(HubOutbox.MarkSent(c, "g", second.LeaseId, 201, T5));
    }

    [Fact]
    public void OutcomesAreRecordedUnderTheLease()
    {
        QueueRow("a"); QueueRow("b"); QueueRow("c");
        string la = Claim("a"), lb = Claim("b"), lc = Claim("c");
        using var c = db.Open();
        Assert.True(HubOutbox.MarkSent(c, "a", la, 201, T1));
        Assert.True(HubOutbox.MarkFailed(c, "b", lb, 422, "bad body"));
        Assert.True(HubOutbox.MarkRetry(c, "c", lc, null, "connection refused", T5));
        var a = Row("a");
        Assert.Equal(("sent", 1, (int?)201, (string?)T1, (string?)null), (a.Status, a.Attempts, a.LastStatus, a.SentAt, a.LeaseId));
        var b = Row("b");
        Assert.Equal(("failed", (int?)422, (string?)"bad body", (string?)null), (b.Status, b.LastStatus, b.LastError, b.LeaseId));
        var r = Row("c");
        Assert.Equal(("pending", 1, (int?)null, T5, (string?)null), (r.Status, r.Attempts, r.LastStatus, r.NextAttemptAt, r.LeaseId));
        var summary = HubOutbox.Summary(c);
        Assert.Equal((1, 1, 1), (summary.Counts["sent"], summary.Counts["failed"], summary.Counts["pending"]));
        Assert.Equal([new HubFailure("b", 422, "bad body")], summary.Failed);
        Assert.Equal(T0, summary.OldestPendingCreatedAt);
    }

    [Fact]
    public void ARetryInvalidatesTheInFlightOutcome()
    {
        QueueRow("g");
        var lease = Claim("g");
        using var c = db.Open();
        Assert.Equal("pending", HubOutbox.ResetForRetry(c, "g", T1));
        var reset = Row("g");
        Assert.Equal(((string?)null, T1), (reset.LeaseId, reset.NextAttemptAt));
        Assert.False(HubOutbox.MarkSent(c, "g", lease, 201, T1));
        Assert.False(HubOutbox.MarkFailed(c, "g", lease, 422, "late"));
        Assert.False(HubOutbox.MarkRetry(c, "g", lease, 500, "late", T9));
        var after = Row("g");
        Assert.Equal(("pending", 0, T1), (after.Status, after.Attempts, after.NextAttemptAt));
    }

    [Fact]
    public void StuckRowsAreFoundUntilAlerted()
    {
        QueueRow("g", now: T0);
        using var c = db.Open();
        Assert.Equal(["g"], HubOutbox.StuckIds(c, T1, T1));
        HubOutbox.MarkAlerted(c, "g", T1);
        Assert.Empty(HubOutbox.StuckIds(c, T5, T0));
        Assert.Equal(["g"], HubOutbox.StuckIds(c, T5, T5));
    }

    [Fact]
    public void ReconcileQueuesFinishedTrackedGamesWithoutAReport()
    {
        InsertGame(Finished("done", Two));
        InsertGame(Finished("old", Two) with { HubTracked = false });
        InsertGame(Finished("live", Two) with { Status = Status.Playing });
        InsertGame(Finished("keyless", Two) with { BotKey = null });
        var log = new ListLogger<HubOutboxTests>();
        Assert.Equal(1, HubOutbox.QueueMissing(db, null, T0, log));
        using var c = db.Open();
        Assert.True(HubOutbox.Exists(c, null, "done"));
        Assert.False(HubOutbox.Exists(c, null, "old"));
        Assert.False(HubOutbox.Exists(c, null, "live"));
        Assert.False(HubOutbox.Exists(c, null, "keyless"));
        Assert.Equal(1, log.Count(LogLevel.Error, "keyless"));
        Assert.Equal(["keyless"], HubOutbox.MissingIds(c));
    }

    [Fact]
    public void RetryPutsUnsentRowsBack()
    {
        QueueRow("f"); QueueRow("s");
        string lf = Claim("f"), ls = Claim("s");
        using var c = db.Open();
        HubOutbox.MarkFailed(c, "f", lf, 409, "dup");
        HubOutbox.MarkSent(c, "s", ls, 200, T1);
        Assert.Equal("pending", HubOutbox.ResetForRetry(c, "f", T5));
        Assert.Equal(("pending", T5), (Row("f").Status, Row("f").NextAttemptAt));
        Assert.Equal("already sent", HubOutbox.ResetForRetry(c, "s", T5));
        Assert.Equal("sent", Row("s").Status);
        Assert.Equal("no report", HubOutbox.ResetForRetry(c, "nope", T5));
        Assert.Empty(HubOutbox.FailedIds(c));   // f is pending again
    }

    [Fact]
    public void RebuildReplacesOnlyOnSuccessAndNeverLosesAReport()
    {
        InsertGame(Finished("g", Two));
        QueueRow("g", "{\"stale\":true}");
        QueueRow("gone", "{\"keep\":true}");
        var lease = Claim("g");   // a send is in flight
        Assert.Equal("rebuilt", HubOutbox.Rebuild(db, "g", null, T5));
        Assert.Equal("game deleted", HubOutbox.Rebuild(db, "gone", null, T5));
        var g = Row("g");
        Assert.Equal(("pending", 0, T5, (string?)null), (g.Status, g.Attempts, g.NextAttemptAt, g.LeaseId));
        Assert.Contains("\"external_id\":\"g\"", g.Body);
        using (var c = db.Open()) Assert.False(HubOutbox.MarkSent(c, "g", lease, 201, T5));   // the old body's answer is dropped
        Assert.Equal("pending", Row("g").Status);
        Assert.Equal("{\"keep\":true}", Row("gone").Body);

        InsertGame(Finished("nokey", Two) with { BotKey = null });
        QueueRow("nokey", "{\"orig\":1}");
        Assert.Equal("unbuildable: no bot key", HubOutbox.Rebuild(db, "nokey", null, T5));
        Assert.Equal("{\"orig\":1}", Row("nokey").Body);

        InsertGame(Finished("done", Two));
        QueueRow("done", "{\"sent\":1}");
        using (var c = db.Open()) HubOutbox.MarkSent(c, "done", Claim("done"), 201, T1);
        Assert.Equal("already sent", HubOutbox.Rebuild(db, "done", null, T5));
        Assert.Equal("{\"sent\":1}", Row("done").Body);
    }

    /// Reconcile and rebuild read the game and write the row in one
    /// transaction opened with BeginTransaction(deferred: false). That must be
    /// BEGIN IMMEDIATE: the write lock is taken at BEGIN, before the read.
    [Fact]
    public void AnImmediateTransactionHoldsTheWriteLockFromBegin()
    {
        using var c1 = db.Open();
        using var tx = c1.BeginTransaction(deferred: false);   // no statement run yet
        using var c2 = new SqliteConnection(db.ConnectionString + ";Default Timeout=1;Pooling=False");
        c2.Open();
        var busy = Assert.Throws<SqliteException>(() => c2.BeginTransaction(deferred: false));
        Assert.Equal(5, busy.SqliteErrorCode);   // SQLITE_BUSY
    }
}
