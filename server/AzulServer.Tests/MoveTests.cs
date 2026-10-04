using System.Net;
using System.Text.Json;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

public sealed class MoveTests : IDisposable
{
    readonly TestApp app = new();
    readonly HttpClient alice, bob;

    public MoveTests()
    {
        alice = app.Client("alice@x.com");
        bob = app.Client("bob@x.com");
    }

    public void Dispose() => app.Dispose();

    async Task<(string Id, GameView View)> TwoHumans()
    {
        var id = await Play.Started(2, alice, bob);
        return (id, await Play.Get(alice, id));
    }

    Task<HttpResponseMessage> Move(HttpClient c, string id, object body) => c.Post($"/api/games/{id}/moves", body);

    [Fact]
    public async Task LegalMoveAdvancesTheGame()
    {
        var (id, v) = await TwoHumans();
        var res = await Move(alice, id, Play.AnyLegal(v));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var after = await res.Read<GameView>();
        Assert.Equal(v.Version + 1, after.Version);
        Assert.Equal(1, after.Board!.ActiveSeat);
        Assert.Equal("take", after.LastMove!.Kind);
        Assert.NotNull((await Play.Get(bob, id)).Legal);
    }

    [Fact]
    public async Task NotYourTurnIs403()
    {
        var (id, v) = await TwoHumans();
        Assert.Equal(HttpStatusCode.Forbidden, (await Move(bob, id, Play.Take(v, v.Legal!.Takes![0]))).StatusCode);
    }

    [Fact]
    public async Task StaleVersionIs409WithTheCurrentView()
    {
        var (id, v) = await TwoHumans();
        await Move(alice, id, Play.AnyLegal(v));
        var res = await Move(alice, id, Play.AnyLegal(v));
        Assert.Equal(HttpStatusCode.Conflict, res.StatusCode);
        var body = await res.Read<JsonElement>();
        Assert.Equal("stale", body.GetProperty("error").GetString());
        Assert.Equal(v.Version + 1, body.GetProperty("view").GetProperty("version").GetInt64());
    }

    [Fact]
    public async Task IllegalAndWrongPhaseAre400()
    {
        var (id, v) = await TwoHumans();
        // A colour the factory does not hold.
        int f = 0, color = Enumerable.Range(0, 5).First(c => v.Board!.Factories[0][c] == 0);
        Assert.Equal(HttpStatusCode.BadRequest, (await Move(alice, id, Play.Take(v, [f, color, 5]))).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await Move(alice, id, Play.Wall(v, [-1, -1, -1, -1, -1]))).StatusCode);
    }

    // VERSION is replaced with the game's current version, so the checks that
    // need the game (factory upper bound, FIRST only from the centre) are reached.
    public static TheoryData<string> BadShapes => new()
    {
        """{"version":VERSION,"kind":"take","factory":0,"color":0,"row":0}""",
        """{"version":VERSION,"requestId":"nope","kind":"take","factory":0,"color":0,"row":0}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":0,"color":7,"row":0}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":0,"color":0,"row":6}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":99,"color":0,"row":0}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":0,"color":5,"row":5}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"wall","columns":[1,2,3,4]}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"wall","columns":[6,-1,-1,-1,-1]}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"dance"}""",
        """{"version":"two"}""",
    };

    [Theory]
    [MemberData(nameof(BadShapes))]
    public async Task BadShapesAre400(string json)
    {
        var (id, v) = await TwoHumans();
        var body = json.Replace("VERSION", v.Version.ToString());
        var res = await alice.PostAsync($"/api/games/{id}/moves", new StringContent(body, System.Text.Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Equal(v.Version, (await Play.Get(alice, id)).Version);
    }

    [Fact]
    public async Task TakingOnlyTheFirstMarker()
    {
        var (id, v) = await TwoHumans();
        int centre = v.Board!.Factories.Length;
        Assert.Contains(v.Legal!.Takes!, t => t.SequenceEqual(new[] { centre, 5, 5 }));
        var res = await Move(alice, id, Play.Take(v, [centre, 5, 5]));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var after = await res.Read<GameView>();
        Assert.True(after.Board!.Players[0].HasFirst);
        Assert.False(after.Board.CenterHasFirst);
    }

    [Fact]
    public async Task RetryAfterTheTurnMovedOnReturnsTheStoredResult()
    {
        var (id, v) = await TwoHumans();
        var rid = Guid.NewGuid().ToString();
        var first = await (await Move(alice, id, Play.Take(v, v.Legal!.Takes![0], rid))).Read<GameView>();
        var bobView = await Play.Get(bob, id);
        Assert.Equal(HttpStatusCode.OK, (await Move(bob, id, Play.AnyLegal(bobView))).StatusCode);
        var retry = await Move(alice, id, Play.Take(v, v.Legal!.Takes![0], rid));
        Assert.Equal(HttpStatusCode.OK, retry.StatusCode);
        Assert.Equal(first.Version, (await retry.Read<GameView>()).Version);
        Assert.Equal(v.Version + 2, (await Play.Get(alice, id)).Version);  // nothing applied twice
    }

    [Fact]
    public async Task ReusedRequestIdWithOtherContentIs409()
    {
        var (id, v) = await TwoHumans();
        var rid = Guid.NewGuid().ToString();
        await Move(alice, id, Play.Take(v, v.Legal!.Takes![0], rid));
        var res = await Move(alice, id, Play.Take(v, v.Legal!.Takes![1], rid));
        Assert.Equal(HttpStatusCode.Conflict, res.StatusCode);
        Assert.Equal("request-id-reused", (await res.Read<JsonElement>()).GetProperty("error").GetString());
        var bobView = await Play.Get(bob, id);
        var asBob = await Move(bob, id, Play.Take(bobView, bobView.Legal!.Takes![0], rid));
        Assert.Equal(HttpStatusCode.Conflict, asBob.StatusCode);
    }

    [Fact]
    public async Task SameRequestTwiceAtOnceMovesOnce()
    {
        var (id, v) = await TwoHumans();
        var body = Play.Take(v, v.Legal!.Takes![0], Guid.NewGuid().ToString());
        var results = await Task.WhenAll(Move(alice, id, body), Move(alice, id, body));
        Assert.All(results, r => Assert.Equal(HttpStatusCode.OK, r.StatusCode));
        Assert.Equal(v.Version + 1, (await Play.Get(alice, id)).Version);
    }

    [Fact]
    public async Task TwoDevicesOneVersionOneWins()
    {
        var (id, v) = await TwoHumans();
        var phone = app.Client("alice@x.com");
        var results = await Task.WhenAll(
            Move(alice, id, Play.Take(v, v.Legal!.Takes![0])),
            Move(phone, id, Play.Take(v, v.Legal!.Takes![0])));
        Assert.Equal(new[] { HttpStatusCode.OK, HttpStatusCode.Conflict }, results.Select(r => r.StatusCode).Order());
    }

    [Fact]
    public async Task FinishedGameAcceptsOnlyRetries()
    {
        // alice vs greedy bot, with bots on and no delay.
        using var bots = new TestApp(new AzulOptions { BotWorkers = 1, MinMoveDelaySeconds = 0, SweepSeconds = 0.2 });
        var a = bots.Client("alice@x.com");
        var id = await Play.Started(2, a);
        object? lastAccepted = null;
        GameView v;
        while (true)
        {
            v = await Play.WaitFor(a, id, x => x.Status == Status.Finished || x.Legal is not null);
            if (v.Status == Status.Finished) break;
            var body = Play.AnyLegal(v);
            var r = await a.Post($"/api/games/{id}/moves", body);
            // 409 when a forced move was auto-played first.
            Assert.True(r.StatusCode is HttpStatusCode.OK or HttpStatusCode.Conflict, $"{r.StatusCode}");
            if (r.StatusCode == HttpStatusCode.OK) lastAccepted = body;
        }
        Assert.NotNull(v.Result);
        Assert.NotNull(lastAccepted);
        var retry = await a.Post($"/api/games/{id}/moves", lastAccepted);
        Assert.Equal(HttpStatusCode.OK, retry.StatusCode);
        var fresh = await a.Post($"/api/games/{id}/moves", Play.Wall(v, [-1, -1, -1, -1, -1]));
        Assert.Equal(HttpStatusCode.Conflict, fresh.StatusCode);
    }
}
