using System.Net;
using AzulServer.Games;

namespace AzulServer.Tests;

/// API-level helpers shared by the lobby, move, SSE and bot tests.
public static class Play
{
    public static async Task<GameView> Create(HttpClient c, int players)
    {
        var res = await c.Post("/api/games", new { players });
        Assert.Equal(HttpStatusCode.Created, res.StatusCode);
        return await res.Read<GameView>();
    }

    public static async Task<GameView> Get(HttpClient c, string id) =>
        await (await c.GetAsync($"/api/games/{id}")).Read<GameView>();

    /// Creator = clients[0] in seat 0; clients[i] claims seat i; remaining
    /// seats are bots; started.
    public static async Task<string> Started(int players, params HttpClient[] clients)
    {
        var g = await Create(clients[0], players);
        for (int i = 1; i < clients.Length; i++)
            Assert.Equal(HttpStatusCode.OK, (await clients[i].Post($"/api/games/{g.Id}/seats/{i}/claim")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await clients[0].Post($"/api/games/{g.Id}/start")).StatusCode);
        return g.Id;
    }

    public static object Take(GameView v, int[] take, string? requestId = null) => new
    {
        version = v.Version,
        requestId = requestId ?? Guid.NewGuid().ToString(),
        kind = "take",
        factory = take[0],
        color = take[1],
        row = take[2],
    };

    public static object Wall(GameView v, int[] columns, string? requestId = null) => new
    {
        version = v.Version,
        requestId = requestId ?? Guid.NewGuid().ToString(),
        kind = "wall",
        columns,
    };

    /// A legal move for the viewer: the first take, or every completed line to
    /// the floor (always legal; first targets could clash on a wall column).
    public static object AnyLegal(GameView v) => v.Legal!.Takes is { } takes
        ? Take(v, takes[0])
        : Wall(v, v.Legal.Wall!.Select(r => r is null ? -1 : 5).ToArray());

    /// Plays `c`'s turns with any legal move until the game is over (bots on).
    public static async Task<GameView> ToEnd(HttpClient c, string id, int seconds = 120)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (true)
        {
            var v = await WaitFor(c, id, x => x.Status == Status.Finished || x.Legal is not null, seconds);
            if (v.Status == Status.Finished) return v;
            if (DateTime.UtcNow > until) throw new TimeoutException($"game {id} did not finish");
            await c.Post($"/api/games/{id}/moves", AnyLegal(v));  // 409 when a forced move landed first
        }
    }

    public static async Task<GameView> WaitFor(HttpClient c, string id, Func<GameView, bool> done, int seconds = 30)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (true)
        {
            var v = await Get(c, id);
            if (done(v)) return v;
            if (DateTime.UtcNow > until) throw new TimeoutException($"game {id} stuck at version {v.Version}");
            await Task.Delay(50);
        }
    }
}
