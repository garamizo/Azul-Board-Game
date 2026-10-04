using System.Net;
using System.Text.Json.Nodes;
using AzulServer.Data;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubCommandsTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("azul-cmd-").FullName;
    readonly Db db;
    public HubCommandsTests()
    {
        db = new Db(new AzulOptions { DataDir = dir });
        db.Migrate();
    }
    public void Dispose() => SqliteConnection.ClearAllPools();

    Dictionary<string, string> env => new() { ["AZUL_DATA_DIR"] = dir, ["AZUL_HUB_URL"] = "http://hub.test", ["AZUL_HUB_KEY"] = "phk_test" };

    async Task<(int Code, JsonObject Out)> Run(params string[] args) => await Run(null, args);

    async Task<(int Code, JsonObject Out)> Run(HttpMessageHandler? handler, params string[] args)
    {
        var w = new StringWriter();
        var e = env;
        int code = await HubCommands.RunAsync(args, w, k => e.GetValueOrDefault(k), handler);
        return (code, JsonNode.Parse(w.ToString())!.AsObject());
    }

    void Queue(string id, string body = "{}")
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        HubOutbox.Queue(c, tx, id, body, "2026-10-04T19:00:00.0000000Z");
        tx.Commit();
    }

    /// Outcomes need the claim's lease (Task 5).
    void Fail(string id, int status, string error)
    {
        using var c = db.Open();
        var claim = HubOutbox.TryClaim(c, id, "2026-10-04T19:00:00.0000000Z", "2026-10-04T19:05:00.0000000Z")!;
        Assert.True(HubOutbox.MarkFailed(c, id, claim.LeaseId, status, error));
    }

    [Fact]
    public async Task BotKeyNeedsNoDatabase()
    {
        var (code, o) = await Run("bot-key");
        Assert.Equal(0, code);
        Assert.Equal(BotIdentity.For(AzulOptions.FromEnvironment(k => env.GetValueOrDefault(k))).Key, (string?)o["botKey"]);
    }

    [Fact]
    public async Task StatusCountsAndListsFailures()
    {
        Queue("a"); Queue("b");
        Fail("b", 422, "bad");
        var (code, o) = await Run("status");
        Assert.Equal(0, code);
        Assert.Equal(1, (int)o["counts"]!["pending"]!);
        Assert.Equal(1, (int)o["counts"]!["failed"]!);
        Assert.Equal("b", (string?)o["failed"]![0]!["gameId"]);
        Assert.Equal("2026-10-04T19:00:00.0000000Z", (string?)o["oldestPendingCreatedAt"]);
        Assert.NotNull(o["botKey"]);
    }

    [Fact]
    public async Task RetryOneOrAllFailed()
    {
        Queue("a"); Queue("b");
        Fail("a", 409, "x");
        Fail("b", 422, "y");
        var (code, one) = await Run("retry", "a");
        Assert.Equal(0, code);
        Assert.Equal("pending", (string?)one["results"]![0]!["outcome"]);
        var (_, all) = await Run("retry", "--all-failed");
        Assert.Equal(["b"], all["results"]!.AsArray().Select(r => (string)r!["gameId"]!));
    }

    [Fact]
    public async Task RebuildKeepsAReportWhoseGameIsGone()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("here", [Human(0, "alice@example.com"), Bot(1)]));
            tx.Commit();
        }
        Queue("here", "{\"old\":1}"); Queue("gone", "{\"keep\":1}");
        var (_, o) = await Run("retry", "here", "--rebuild");
        Assert.Equal("rebuilt", (string?)o["results"]![0]!["outcome"]);
        var (_, g) = await Run("retry", "gone", "--rebuild");
        Assert.Equal("game deleted", (string?)g["results"]![0]!["outcome"]);
        using var c2 = db.Open();
        Assert.Equal("{\"keep\":1}", HubOutbox.Get(c2, null, "gone")!.Body);
    }

    [Theory]
    [InlineData(422, 0)]
    [InlineData(401, 1)]
    [InlineData(500, 1)]
    public async Task PingPassesOnlyOnUnprocessable(int status, int expected)
    {
        var fake = new FakeHub { Respond = () => new HttpResponseMessage((HttpStatusCode)status) };
        var (code, o) = await Run(fake, "ping");
        Assert.Equal(expected, code);
        Assert.Equal(status, (int)o["status"]!);
        var (req, body) = Assert.Single(fake.Requests);
        Assert.Equal("{}", body);
        Assert.Equal("phk_test", req.Headers.Authorization!.Parameter);
    }

    [Fact]
    public async Task BadUsageExitsTwo()
    {
        Assert.Equal(2, (await Run("nope")).Code);
        Assert.Equal(2, (await Run("retry")).Code);
        Assert.Equal(2, (await Run()).Code);
    }
}
