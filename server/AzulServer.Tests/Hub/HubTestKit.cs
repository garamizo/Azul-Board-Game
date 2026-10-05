using System.Net;
using System.Text.Json.Nodes;
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

/// Records every request; answers with `Respond()` when set, throws `Throw`,
/// or else answers as the hub's ingest does (~/playhub ingest.ts): the first
/// body for an external_id is recorded (201), the same body again is 200, a
/// different body for a recorded external_id is 409, no external_id is 422.
public sealed class FakeHub : HttpMessageHandler, IHttpClientFactory
{
    public readonly List<(HttpRequestMessage Request, string Body)> Requests = new();
    public Func<HttpResponseMessage>? Respond;
    public Exception? Throw;
    /// When set, every request is recorded and then held until the test completes it.
    public TaskCompletionSource? Gate;
    /// The body the hub recorded for each external_id.
    public readonly Dictionary<string, string> Accepted = new();

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(ct);
        lock (Requests) Requests.Add((request, body));
        if (Gate is { } gate) await gate.Task.WaitAsync(ct);
        if (Throw is not null) throw Throw;
        return Respond is { } respond ? respond() : Ingest(body);
    }

    HttpResponseMessage Ingest(string body)
    {
        string? id = null;
        try { id = JsonNode.Parse(body)?["external_id"]?.GetValue<string>(); } catch (Exception) { }
        lock (Accepted)
        {
            if (id is null) return Answer(HttpStatusCode.UnprocessableEntity, "{\"error\":\"external_id: Required\"}");
            if (!Accepted.TryGetValue(id, out var recorded))
            {
                Accepted[id] = body;
                return Answer(HttpStatusCode.Created, "{}");
            }
            return recorded == body
                ? Answer(HttpStatusCode.OK, "{}")
                : Answer(HttpStatusCode.Conflict, "{\"error\":\"external_id already recorded with a different result.\"}");
        }
    }

    static HttpResponseMessage Answer(HttpStatusCode status, string json) =>
        new(status) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") };

    public HttpClient CreateClient(string name) => new(this, disposeHandler: false);
}

/// A response body whose bytes never come (`Stall`) or break off after a
/// first chunk (`Truncated`).
public sealed class BrokenBody(bool stall, string prefix = "") : Stream
{
    public static HttpContent Stall() => new StreamContent(new BrokenBody(true));
    public static HttpContent Truncated(string prefix) => new StreamContent(new BrokenBody(false, prefix));

    byte[]? first = System.Text.Encoding.UTF8.GetBytes(prefix);

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default)
    {
        if (stall) { await Task.Delay(Timeout.Infinite, ct); return 0; }
        if (first is { Length: > 0 } chunk)
        {
            first = null;
            int n = Math.Min(chunk.Length, buffer.Length);
            chunk.AsMemory(0, n).CopyTo(buffer);
            return n;
        }
        throw new IOException("The response ended prematurely.");
    }

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken ct) =>
        ReadAsync(buffer.AsMemory(offset, count), ct).AsTask();
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
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
