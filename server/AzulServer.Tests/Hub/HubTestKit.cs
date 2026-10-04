using System.Net;
using Azul;
using AzulServer.Data;
using AzulServer.Games;
using Microsoft.Extensions.Logging;

namespace AzulServer.Tests;

public static class HubFixtures
{
    public const string Started = "2026-10-04T18:02:11.0000000Z";
    public const string FinishedAt = "2026-10-04T18:41:52.1239999Z";

    public static Db NewDb()
    {
        var db = new Db(new AzulOptions { DataDir = Directory.CreateTempSubdirectory("azul-hub-").FullName });
        db.Migrate();
        return db;
    }

    static int[][] Grid(int fullRows) => Enumerable.Range(0, 5)
        .Select(r => Enumerable.Range(0, 5).Select(col => r < fullRows ? (r + col) % 5 : -1).ToArray())
        .ToArray();

    /// A finished, hub-tracked record. `players` overrides each seat's
    /// (score, complete wall rows); otherwise a greedy game's real result.
    public static GameRecord Finished(string id, IReadOnlyList<SeatRecord> seats,
        (int Score, int Rows)[]? players = null, string reason = "normal", int seed = 4)
    {
        var game = new Game(seats.Count, new Random(seed));
        while (!game.IsGameOver()) game.Play(game.GetGreedyMove());
        var snap = game.ToSnapshot();
        if (players is not null)
            snap = snap with
            {
                Players = snap.Players.Select((p, i) => p with { Score = players[i].Score, Grid = Grid(players[i].Rows) }).ToArray(),
            };
        return new GameRecord(id, "alice@example.com", Status.Finished, seats.Count, 99, Json.Serialize(snap), reason,
            "2026-10-04T18:00:00.0000000Z", FinishedAt, seats)
        {
            StartedAt = Started, FinishedAt = FinishedAt, HubTracked = true, BotKey = "mcts@0123456789ab",
        };
    }

    public static SeatRecord Human(int idx, string email) => new(idx, SeatKind.Human, email);
    public static SeatRecord Bot(int idx, string? owner = null) => new(idx, SeatKind.Bot, owner);
}

/// Records every request; answers with `Respond()` or throws `Throw`.
public sealed class FakeHub : HttpMessageHandler, IHttpClientFactory
{
    public readonly List<(HttpRequestMessage Request, string Body)> Requests = new();
    public Func<HttpResponseMessage> Respond = () => new HttpResponseMessage(HttpStatusCode.Created) { Content = new StringContent("{}") };
    public Exception? Throw;
    /// When set, every request is recorded and then held until the test completes it.
    public TaskCompletionSource? Gate;

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(ct);
        lock (Requests) Requests.Add((request, body));
        if (Gate is { } gate) await gate.Task.WaitAsync(ct);
        if (Throw is not null) throw Throw;
        return Respond();
    }

    public HttpClient CreateClient(string name) => new(this, disposeHandler: false);
}

public sealed class ListLogger<T> : ILogger<T>
{
    public readonly List<(LogLevel Level, string Message)> Entries = new();
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(LogLevel level) => true;
    public void Log<TState>(LogLevel level, EventId id, TState state, Exception? e, Func<TState, Exception?, string> format)
    {
        lock (Entries) Entries.Add((level, format(state, e)));
    }
    public int Count(LogLevel level, string contains)
    {
        lock (Entries) return Entries.Count(x => x.Level == level && x.Message.Contains(contains));
    }
}
