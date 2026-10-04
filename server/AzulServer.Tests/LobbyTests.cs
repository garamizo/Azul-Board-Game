using System.Net;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

public sealed class LobbyTests : IDisposable
{
    readonly TestApp app = new();
    readonly HttpClient alice, bob, carol;

    public LobbyTests()
    {
        alice = app.Client("alice@x.com");
        bob = app.Client("bob@x.com");
        carol = app.Client("carol@x.com");
    }

    public void Dispose() => app.Dispose();

    static async Task<HttpStatusCode> S(Task<HttpResponseMessage> r) => (await r).StatusCode;

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public async Task CreateSeatsTheCreator(int players)
    {
        var g = await Play.Create(alice, players);
        Assert.Equal(Status.Lobby, g.Status);
        Assert.Equal(players, g.Seats.Length);
        Assert.Equal(new SeatView(0, SeatKind.Human, "alice@x.com"), g.Seats[0]);
        Assert.All(g.Seats.Skip(1), s => Assert.Equal(SeatKind.Open, s.Kind));
        Assert.Equal(0, g.You.Seat);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public async Task CreateRejectsBadPlayerCounts(int players) =>
        Assert.Equal(HttpStatusCode.BadRequest, await S(alice.Post("/api/games", new { players })));

    [Fact]
    public async Task ClaimRules()
    {
        var g = await Play.Create(alice, 3);
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{g.Id}/seats/1/claim")));
        Assert.Equal(HttpStatusCode.Conflict, await S(carol.Post($"/api/games/{g.Id}/seats/1/claim")));  // taken
        Assert.Equal(HttpStatusCode.Conflict, await S(bob.Post($"/api/games/{g.Id}/seats/2/claim")));    // already seated
        Assert.Equal(HttpStatusCode.NotFound, await S(carol.Post($"/api/games/{g.Id}/seats/7/claim")));
        Assert.Equal(HttpStatusCode.NotFound, await S(carol.Post("/api/games/nope/seats/1/claim")));
    }

    [Fact]
    public async Task ClaimIsCaseInsensitiveInDevMode()
    {
        var g = await Play.Create(alice, 3);
        var shouty = app.Client("ALICE@X.COM");
        Assert.Equal(HttpStatusCode.Conflict, await S(shouty.Post($"/api/games/{g.Id}/seats/1/claim")));
    }

    [Fact]
    public async Task ReleaseRules()
    {
        var g = await Play.Create(alice, 3);
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        await carol.Post($"/api/games/{g.Id}/seats/2/claim");
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{g.Id}/seats/1/release")));
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{g.Id}/seats/1/release")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/2/release")));  // creator removes carol
        var v = await Play.Get(alice, g.Id);
        Assert.Equal(SeatKind.Open, v.Seats[1].Kind);
        Assert.Null(v.Seats[2].Email);
    }

    [Fact]
    public async Task KindRules()
    {
        var g = await Play.Create(alice, 3);
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        Assert.Equal(HttpStatusCode.Forbidden, await S(bob.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "bot" })));
        Assert.Equal(HttpStatusCode.Conflict, await S(alice.Post($"/api/games/{g.Id}/seats/1/kind", new { kind = "bot" })));
        Assert.Equal(HttpStatusCode.BadRequest, await S(alice.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "human" })));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "bot" })));
        Assert.Equal(SeatKind.Bot, (await Play.Get(alice, g.Id)).Seats[2].Kind);
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "open" })));
    }

    [Fact]
    public async Task StartRules()
    {
        var g = await Play.Create(alice, 3);
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        Assert.Equal(HttpStatusCode.Forbidden, await S(bob.Post($"/api/games/{g.Id}/start")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/start")));
        var v = await Play.Get(alice, g.Id);
        Assert.Equal(Status.Playing, v.Status);
        Assert.Equal(SeatKind.Bot, v.Seats[2].Kind);
        Assert.NotNull(v.Board);
        Assert.Equal(0, v.Board!.ActiveSeat);
        Assert.Equal(HttpStatusCode.Conflict, await S(alice.Post($"/api/games/{g.Id}/start")));
        Assert.Equal(HttpStatusCode.Conflict, await S(carol.Post($"/api/games/{g.Id}/seats/2/claim")));
    }

    [Fact]
    public async Task UnseatedCreatorCanStartButNotWithoutHumans()
    {
        var g = await Play.Create(alice, 2);
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/0/release")));
        Assert.Equal(HttpStatusCode.Conflict, await S(alice.Post($"/api/games/{g.Id}/start")));  // no human
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/start")));
        Assert.Null((await Play.Get(alice, g.Id)).You.Seat);
    }

    [Fact]
    public async Task HandToBotAndTakeBack()
    {
        var id = await Play.Started(3, alice, bob, carol);
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{id}/seats/1/to-bot")));
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{id}/seats/1/to-bot")));
        var v = await Play.Get(bob, id);
        Assert.Equal(new SeatView(1, SeatKind.Bot, "bob@x.com"), v.Seats[1]);
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{id}/seats/1/take-back")));
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{id}/seats/1/take-back")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{id}/seats/2/to-bot")));  // creator hands carol's seat
        Assert.Equal(HttpStatusCode.OK, await S(carol.Post($"/api/games/{id}/seats/2/take-back")));
    }

    [Fact]
    public async Task NeverHumanBotCannotBeTaken()
    {
        var id = await Play.Started(3, alice, bob);
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{id}/seats/2/take-back")));
        Assert.Equal(HttpStatusCode.Forbidden, await S(alice.Post($"/api/games/{id}/seats/2/take-back")));
    }

    [Fact]
    public async Task DeleteRules()
    {
        var id = await Play.Started(2, alice, bob);
        Assert.Equal(HttpStatusCode.Forbidden, await S(bob.SendAsync(new HttpRequestMessage(HttpMethod.Delete, $"/api/games/{id}"))));
        Assert.Equal(HttpStatusCode.NoContent, await S(alice.SendAsync(new HttpRequestMessage(HttpMethod.Delete, $"/api/games/{id}"))));
        Assert.Equal(HttpStatusCode.NotFound, await S(alice.GetAsync($"/api/games/{id}")));
    }

    [Fact]
    public async Task ListAndSpectating()
    {
        var id = await Play.Started(2, alice, bob);
        var list = await (await carol.GetAsync("/api/games")).Read<GameSummary[]>();
        Assert.Contains(list, s => s.Id == id && s.Round == 1);
        var watch = await Play.Get(carol, id);
        Assert.Null(watch.You.Seat);
        Assert.NotNull(watch.Board);
    }

    [Fact]
    public async Task CorruptGameDoesNotBreakTheListOrTheSweep()
    {
        var good = await Play.Started(2, alice, bob);
        var bad = await Play.Started(2, alice, bob);
        using (var c = app.Service<Db>().Open())
        using (var cmd = c.CreateCommand())
        {
            cmd.CommandText = "UPDATE games SET state_json = '{\"format\":1}' WHERE id = $id";
            cmd.Parameters.AddWithValue("$id", bad);
            cmd.ExecuteNonQuery();
        }
        Assert.Equal(HttpStatusCode.OK, await S(alice.GetAsync("/api/games")));
        Assert.Equal(HttpStatusCode.InternalServerError, await S(alice.GetAsync($"/api/games/{bad}")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.GetAsync($"/api/games/{good}")));
        Assert.DoesNotContain(bad, app.Service<GameService>().GamesNeedingServerMove());
    }

    [Fact]
    public async Task SweepSkipsAGameWhoseStateDoesNotFitItsSeats()
    {
        // Both games have a bot to move (alice handed seat 0 to a bot).
        var good = await Play.Started(2, alice, bob);
        var bad = await Play.Started(2, alice, bob);
        foreach (var id in new[] { good, bad })
            Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{id}/seats/0/to-bot")));
        // A valid 4-player snapshot with seat 3 to move, on a 2-seat game.
        var s = new Azul.Game(4, new Random(1)).ToSnapshot() with { ActivePlayer = 3 };
        using (var c = app.Service<Db>().Open())
        using (var cmd = c.CreateCommand())
        {
            cmd.CommandText = "UPDATE games SET state_json = $s WHERE id = $id";
            cmd.Parameters.AddWithValue("$s", Json.Serialize(s));
            cmd.Parameters.AddWithValue("$id", bad);
            cmd.ExecuteNonQuery();
        }
        var ids = app.Service<GameService>().GamesNeedingServerMove();
        Assert.Contains(good, ids);
        Assert.DoesNotContain(bad, ids);
    }
}
