using System.Net;
using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Time.Testing;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubSenderTests : IDisposable
{
    readonly Db db = NewDb();
    readonly FakeHub hubHttp = new();
    readonly FakeTimeProvider time = new(DateTimeOffset.Parse("2026-10-04T19:00:00Z"));
    readonly ListLogger<HubSender> log = new();
    public void Dispose() => SqliteConnection.ClearAllPools();

    static readonly HubOptions Configured = new() { Url = "http://hub.test", Key = "phk_test" };

    HubSender Sender(HubOptions? hub = null) =>
        new(db, new AzulOptions { Hub = hub ?? Configured }, new HubSignal(), hubHttp, time, log);

    string Now => time.GetUtcNow().UtcDateTime.ToString("O");

    void QueueRow(string id, string body = "{\"external_id\":\"x\",\"é\":\"ü\"}")
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        HubOutbox.Queue(c, tx, id, body, Now);
        tx.Commit();
    }

    HubRow Row(string id)
    {
        using var c = db.Open();
        return HubOutbox.Get(c, null, id)!;
    }

    [Theory]
    [InlineData(200, HubSender.Outcome.Sent)]
    [InlineData(201, HubSender.Outcome.Sent)]
    [InlineData(409, HubSender.Outcome.Failed)]
    [InlineData(422, HubSender.Outcome.Failed)]
    [InlineData(401, HubSender.Outcome.RetryKey)]
    [InlineData(500, HubSender.Outcome.RetryTransient)]
    [InlineData(503, HubSender.Outcome.RetryTransient)]
    [InlineData(null, HubSender.Outcome.RetryTransient)]
    [InlineData(400, HubSender.Outcome.RetryConfig)]
    [InlineData(403, HubSender.Outcome.RetryConfig)]
    [InlineData(404, HubSender.Outcome.RetryConfig)]
    [InlineData(413, HubSender.Outcome.RetryConfig)]
    public void StatusesMapToTheHubContract(int? status, HubSender.Outcome expected) =>
        Assert.Equal(expected, HubSender.Classify(status));

    [Theory]
    [InlineData(1, 60)]
    [InlineData(2, 120)]
    [InlineData(6, 1920)]
    [InlineData(7, 3600)]
    [InlineData(30, 3600)]
    public void BackoffDoublesToAnHourPlusTenPercentJitter(int attempts, int baseSeconds)
    {
        Assert.Equal(TimeSpan.FromSeconds(baseSeconds), HubSender.Backoff(attempts, 0));
        Assert.Equal(TimeSpan.FromSeconds(baseSeconds * 1.1), HubSender.Backoff(attempts, 1));
    }

    [Fact]
    public async Task TheStoredBodyIsSentByteForByteWithBothHeaders()
    {
        QueueRow("g");
        Assert.Equal(1, await Sender().RunCycleAsync(default));
        var (req, body) = Assert.Single(hubHttp.Requests);
        Assert.Equal(HttpMethod.Post, req.Method);
        Assert.Equal("http://hub.test/api/v1/results", req.RequestUri!.ToString());
        Assert.Equal("{\"external_id\":\"x\",\"é\":\"ü\"}", body);
        Assert.Equal("application/json", req.Content!.Headers.ContentType!.MediaType);
        Assert.Equal(("Bearer", "phk_test"), (req.Headers.Authorization!.Scheme, req.Headers.Authorization.Parameter));
        Assert.Equal(("sent", 1, 201), (Row("g").Status, Row("g").Attempts, Row("g").LastStatus));
    }

    [Fact]
    public async Task AnyTwoHundredIsSentWhateverTheBody()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("<html>proxy</html>") };
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        Assert.Equal("sent", Row("g").Status);
    }

    [Fact]
    public async Task UnprocessableIsPermanentWithItsBodyKept()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.UnprocessableEntity)
            { Content = new StringContent("{\"error\":\"players: must be 1-12\"}" + new string('x', 600)) };
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        var row = Row("g");
        Assert.Equal(("failed", 422), (row.Status, row.LastStatus));
        Assert.StartsWith("{\"error\":\"players", row.LastError);
        Assert.Equal(500, row.LastError!.Length);
        Assert.Equal(1, log.Count(LogLevel.Error, "permanently"));
        await Sender().RunCycleAsync(default);
        Assert.Single(hubHttp.Requests);   // never resent
    }

    [Fact]
    public async Task AWrongKeyStaysPendingWithBackoffAndAnError()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.Unauthorized);
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        var row = Row("g");
        Assert.Equal(("pending", 1, 401), (row.Status, row.Attempts, row.LastStatus));
        var next = DateTimeOffset.Parse(row.NextAttemptAt) - time.GetUtcNow();
        Assert.InRange(next.TotalSeconds, 60, 66);
        Assert.Equal(1, log.Count(LogLevel.Error, "AZUL_HUB_KEY"));
        await Sender().RunCycleAsync(default);
        Assert.Single(hubHttp.Requests);   // not due yet
    }

    [Fact]
    public async Task ANetworkErrorStaysPendingWithAWarning()
    {
        hubHttp.Throw = new HttpRequestException("connection refused");
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        var row = Row("g");
        Assert.Equal(("pending", (int?)null, "connection refused"), (row.Status, row.LastStatus, row.LastError));
        Assert.Equal(1, log.Count(LogLevel.Warning, "unavailable"));
    }

    [Fact]
    public async Task AConfigFaultStaysPendingWithAnError()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.NotFound);
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        Assert.Equal(("pending", 404), (Row("g").Status, Row("g").LastStatus));
        Assert.Equal(1, log.Count(LogLevel.Error, "AZUL_HUB_URL"));
    }

    [Fact]
    public async Task AStuckRowAlertsOnceAnHour()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.ServiceUnavailable);
        QueueRow("g");
        var sender = Sender();
        await sender.RunCycleAsync(default);
        time.Advance(TimeSpan.FromHours(2));
        await sender.RunCycleAsync(default);
        time.Advance(TimeSpan.FromMinutes(10));
        await sender.RunCycleAsync(default);
        Assert.Equal(1, log.Count(LogLevel.Error, "over an hour"));
        time.Advance(TimeSpan.FromHours(1));
        await sender.RunCycleAsync(default);
        Assert.Equal(2, log.Count(LogLevel.Error, "over an hour"));
    }

    [Fact]
    public async Task ReconcileRunsAtTheStartOfACycle()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("lost", [Human(0, "alice@example.com"), Bot(1)]));
            tx.Commit();
        }
        Assert.Equal(1, await Sender().RunCycleAsync(default));
        Assert.Contains("\"external_id\":\"lost\"", Assert.Single(hubHttp.Requests).Body);
    }

    [Fact]
    public async Task ReportOfADeletedGameIsStillSent()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("del", [Human(0, "alice@example.com"), Bot(1)]));
            HubOutbox.Queue(c, tx, "del", "{\"external_id\":\"del\"}", Now);
            GameStore.Delete(c, tx, "del");
            tx.Commit();
        }
        await Sender().RunCycleAsync(default);
        Assert.Equal("sent", Row("del").Status);
    }

    [Fact]
    public async Task WithoutBothSettingsTheSenderStaysOff()
    {
        QueueRow("g");
        var off = Sender(new HubOptions { Url = "http://hub.test" });
        await off.StartAsync(default);
        await off.ExecuteTask!;
        Assert.Empty(hubHttp.Requests);
        Assert.Equal(1, log.Count(LogLevel.Warning, "both AZUL_HUB_URL and AZUL_HUB_KEY"));
    }

    [Fact]
    public async Task ADatabaseErrorDoesNotStopTheLoop()
    {
        var signal = new HubSignal();
        var sender = new HubSender(db, new AzulOptions { Hub = Configured }, signal, hubHttp, TimeProvider.System, log);
        using (var c = db.Open()) { using var cmd = c.CreateCommand(); cmd.CommandText = "ALTER TABLE hub_reports RENAME TO hub_reports_away"; cmd.ExecuteNonQuery(); }
        await sender.StartAsync(default);
        await WaitUntil(() => log.Count(LogLevel.Error, "cycle failed") >= 1);
        using (var c = db.Open()) { using var cmd = c.CreateCommand(); cmd.CommandText = "ALTER TABLE hub_reports_away RENAME TO hub_reports"; cmd.ExecuteNonQuery(); }
        QueueRow("g");
        signal.Wake();
        await WaitUntil(() => hubHttp.Requests.Count == 1);
        await sender.StopAsync(default);
    }

    static async Task WaitUntil(Func<bool> done, int seconds = 10)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (!done())
        {
            if (DateTime.UtcNow > until) throw new TimeoutException();
            await Task.Delay(20);
        }
    }

    [Fact]
    public async Task ARebuildDuringASendWinsOverTheOldOutcome()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("g", [Human(0, "alice@example.com"), Bot(1)]));
            HubOutbox.Queue(c, tx, "g", "{\"old\":1}", Now);
            tx.Commit();
        }
        hubHttp.Gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var sender = Sender();
        var cycle = sender.RunCycleAsync(default);
        await WaitUntil(() => hubHttp.Requests.Count == 1);   // the old body is in flight
        Assert.Equal("{\"old\":1}", hubHttp.Requests[0].Body);
        Assert.Equal("rebuilt", HubOutbox.Rebuild(db, "g", null, Now));
        hubHttp.Gate.SetResult();   // the hub now answers 201 to the OLD body
        await cycle;
        var row = Row("g");
        Assert.Equal(("pending", (string?)null), (row.Status, row.LeaseId));
        Assert.Contains("\"external_id\":\"g\"", row.Body);
        Assert.Equal(1, log.Count(LogLevel.Information, "stale hub outcome"));

        hubHttp.Gate = null;
        Assert.Equal(1, await sender.RunCycleAsync(default));   // the new body goes out fresh
        Assert.Equal(2, hubHttp.Requests.Count);
        Assert.Contains("\"external_id\":\"g\"", hubHttp.Requests[1].Body);
        Assert.Equal("sent", Row("g").Status);
    }

    // ---------- through the real app ----------

    [Fact]
    public async Task ASenderThatCannotReachItsDatabaseDoesNotStopPlay()
    {
        var senderLog = new ListLogger<HubSender>();
        var fake = new FakeHub();
        var notADirectory = Path.GetTempFileName();   // every Db.Open() below it throws
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, Hub = Configured }, s =>
        {
            s.AddHttpClient(HubSender.HttpName).ConfigurePrimaryHttpMessageHandler(() => fake);
            s.Remove(s.Single(d => d.ServiceType == typeof(IHostedService) && d.ImplementationType == typeof(HubSender)));
            s.AddHostedService(sp => new HubSender(new Db(new AzulOptions { DataDir = notADirectory }),
                sp.GetRequiredService<AzulOptions>(), sp.GetRequiredService<HubSignal>(),
                sp.GetRequiredService<IHttpClientFactory>(), TimeProvider.System, senderLog));
        });
        const string ann = "ann@example.com";
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        var seats = Enumerable.Range(0, 2).Select(i => i == seat ? Human(i, ann) : Bot(i)).ToArray();
        HubPlay.Insert(app.Service<Db>(), "play", seats, state, creator: ann);
        var client = app.Client(ann);

        var res = await client.Post("/api/games/play/moves", HubPlay.Body(5, finishing));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        await WaitUntil(() => senderLog.Count(LogLevel.Error, "cycle failed") >= 1);
        Assert.Equal(Status.Finished, (await Play.Get(client, "play")).Status);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/api/health")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/api/games")).StatusCode);
        using var c = app.Service<Db>().Open();
        Assert.Equal("pending", HubOutbox.Get(c, null, "play")!.Status);   // queued by the move, waiting
        Assert.Empty(fake.Requests);
    }

    static TestApp Wired(FakeHub fake, string? dataDir = null) => new(
        new AzulOptions { BotWorkers = 1, MinMoveDelaySeconds = 0, SweepSeconds = 0.2, Hub = Configured },
        s => s.AddHttpClient(HubSender.HttpName).ConfigurePrimaryHttpMessageHandler(() => fake),
        dataDir);

    [Fact]
    public async Task AFinishedGameReachesTheHub()
    {
        var fake = new FakeHub();
        using var app = Wired(fake);
        var alice = app.Client("alice@example.com");
        var id = await Play.Started(2, alice);
        await Play.ToEnd(alice, id);
        await WaitUntil(() => fake.Requests.Count == 1);
        Assert.Contains($"\"external_id\":\"{id}\"", fake.Requests[0].Body);
    }

    [Fact]
    public async Task PendingRowsAreSentAfterAStart()
    {
        var dir = Directory.CreateTempSubdirectory("azul-restart-").FullName;
        var backlog = new Db(new AzulOptions { DataDir = dir });
        backlog.Migrate();
        using (var c = backlog.Open())
        using (var tx = c.BeginTransaction())
        {
            HubOutbox.Queue(c, tx, "waiting", "{\"external_id\":\"waiting\"}", DateTime.UtcNow.ToString("O"));
            tx.Commit();
        }
        var fake = new FakeHub();
        using var app = Wired(fake, dir);
        _ = app.Client();   // starts the host and its hosted services
        await WaitUntil(() => fake.Requests.Count == 1);
    }

    [Fact]
    public async Task ATrailingSlashOnTheUrlIsIgnored()
    {
        QueueRow("g");
        await Sender(new HubOptions { Url = "http://hub.test/", Key = "phk_test" }).RunCycleAsync(default);
        Assert.Equal("http://hub.test/api/v1/results", Assert.Single(hubHttp.Requests).Request.RequestUri!.ToString());
    }

    [Fact]
    public async Task ATimeoutStaysPendingWithAWarning()
    {
        hubHttp.Throw = new TaskCanceledException("The request was canceled due to the configured HttpClient.Timeout of 10 seconds elapsing.");
        QueueRow("g");
        Assert.Equal(0, await Sender().RunCycleAsync(default));
        var row = Row("g");
        Assert.Equal(("pending", 1, (int?)null), (row.Status, row.Attempts, row.LastStatus));
        Assert.Contains("Timeout", row.LastError);
        Assert.Equal(1, log.Count(LogLevel.Warning, "unavailable"));
    }

    [Fact]
    public void TheHubClientTimesOutAfterTenSeconds()
    {
        using var app = Wired(new FakeHub());
        Assert.Equal(TimeSpan.FromSeconds(10), app.Service<IHttpClientFactory>().CreateClient(HubSender.HttpName).Timeout);
    }

    [Fact]
    public async Task AFinishingCommitWakesASenderWaitingOnALongIdle()
    {
        // A row queued before the start: once the hub has seen it, the first
        // cycle has passed its due query, and with an hour-long idle wait only
        // the finishing commit's wake can send the next report in time.
        var dir = Directory.CreateTempSubdirectory("azul-wake-").FullName;
        var backlog = new Db(new AzulOptions { DataDir = dir });
        backlog.Migrate();
        using (var c = backlog.Open())
        using (var tx = c.BeginTransaction())
        {
            HubOutbox.Queue(c, tx, "warmup", "{\"external_id\":\"warmup\"}", DateTime.UtcNow.ToString("O"));
            tx.Commit();
        }
        var fake = new FakeHub();
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, Hub = Configured }, s =>
        {
            s.AddHttpClient(HubSender.HttpName).ConfigurePrimaryHttpMessageHandler(() => fake);
            s.Remove(s.Single(d => d.ServiceType == typeof(IHostedService) && d.ImplementationType == typeof(HubSender)));
            s.AddHostedService(sp => new HubSender(sp.GetRequiredService<Db>(), sp.GetRequiredService<AzulOptions>(),
                sp.GetRequiredService<HubSignal>(), sp.GetRequiredService<IHttpClientFactory>(), TimeProvider.System,
                new ListLogger<HubSender>()) { IdleWait = TimeSpan.FromHours(1) });
        }, dir);
        const string ann = "ann@example.com";
        var client = app.Client(ann);   // starts the host and the sender
        await WaitUntil(() => fake.Requests.Count == 1);
        Assert.Contains("warmup", fake.Requests[0].Body);

        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        var seats = Enumerable.Range(0, 2).Select(i => i == seat ? Human(i, ann) : Bot(i)).ToArray();
        HubPlay.Insert(app.Service<Db>(), "play", seats, state, creator: ann);
        Assert.Equal(HttpStatusCode.OK, (await client.Post("/api/games/play/moves", HubPlay.Body(5, finishing))).StatusCode);
        await WaitUntil(() => fake.Requests.Count == 2, seconds: 5);
        Assert.Contains("\"external_id\":\"play\"", fake.Requests[1].Body);
    }
}
