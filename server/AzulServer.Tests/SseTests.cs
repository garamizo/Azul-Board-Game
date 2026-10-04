using System.Net;
using System.Text;
using AzulServer.Games;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace AzulServer.Tests;

public sealed class SseTests : IDisposable
{
    readonly TestApp app = new(new AzulOptions { BotWorkers = 0, SseHeartbeatSeconds = 0.5 });

    public void Dispose() => app.Dispose();

    sealed class Stream(StreamReader reader, HttpResponseMessage response) : IDisposable
    {
        public HttpResponseMessage Response { get; } = response;

        /// Next event, skipping pings unless asked for.
        public async Task<(string Event, string Data)> Next(bool pings = false, int seconds = 10)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
            string? ev = null;
            var data = new StringBuilder();
            while (true)
            {
                var line = await reader.ReadLineAsync(cts.Token) ?? throw new EndOfStreamException();
                if (line.StartsWith(':'))
                {
                    if (pings) return (":", line);
                    continue;
                }
                if (line.Length == 0)
                {
                    if (ev is not null) return (ev, data.ToString());
                    continue;
                }
                if (line.StartsWith("event: ")) ev = line[7..];
                else if (line.StartsWith("data: ")) data.Append(line[6..]);
            }
        }

        public async Task<bool> Ended(int seconds = 5)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
            try { return await reader.ReadLineAsync(cts.Token) is null; }
            catch (OperationCanceledException) { return false; }
        }

        public void Dispose() { reader.Dispose(); Response.Dispose(); }
    }

    async Task<Stream> Open(HttpClient c, string id)
    {
        var res = await c.GetAsync($"/api/games/{id}/events", HttpCompletionOption.ResponseHeadersRead);
        return new Stream(new StreamReader(await res.Content.ReadAsStreamAsync()), res);
    }

    [Fact]
    public async Task FirstEventIsTheSnapshotThenHeartbeats()
    {
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice, app.Client("bob@x.com"));
        using var s = await Open(alice, id);
        Assert.Equal("text/event-stream", s.Response.Content.Headers.ContentType!.MediaType);
        var (ev, data) = await s.Next();
        Assert.Equal("state", ev);
        Assert.Equal(id, Json.Deserialize<GameView>(data).Id);
        Assert.Equal(":", (await s.Next(pings: true)).Event);
    }

    [Fact]
    public async Task EachViewerGetsTheirOwnView()
    {
        var alice = app.Client("alice@x.com");
        var bob = app.Client("bob@x.com");
        var carol = app.Client("carol@x.com");  // spectator
        var dave = app.Client("dave@x.com");
        var id = await Play.Started(3, alice, bob, dave);
        await dave.Post($"/api/games/{id}/seats/2/to-bot");
        var streams = new Dictionary<string, Stream>();
        foreach (var (name, c) in new[] { ("alice", alice), ("bob", bob), ("carol", carol), ("dave", dave) })
        {
            streams[name] = await Open(c, id);
            await streams[name].Next();  // snapshot
        }
        var v = await Play.Get(alice, id);
        Assert.Equal(HttpStatusCode.OK, (await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v))).StatusCode);

        var views = new Dictionary<string, GameView>();
        foreach (var (name, s) in streams)
        {
            var (ev, data) = await s.Next();
            Assert.Equal("state", ev);
            views[name] = Json.Deserialize<GameView>(data);
            Assert.Equal(v.Version + 1, views[name].Version);
        }
        Assert.Equal(0, views["alice"].You.Seat);
        Assert.Null(views["alice"].Legal);
        Assert.Equal(1, views["bob"].You.Seat);
        Assert.NotNull(views["bob"].Legal);  // bob is next
        Assert.Null(views["carol"].You.Seat);
        Assert.Null(views["carol"].Legal);
        Assert.Equal(2, views["dave"].You.Seat);
        Assert.Null(views["dave"].Legal);
        foreach (var s in streams.Values) s.Dispose();
    }

    sealed class LoseThirdNotification : IFaultInjector
    {
        int commits;
        public void AfterCommit(string gameId)
        {
            // claim = 1, start = 2, alice's move = 3
            if (Interlocked.Increment(ref commits) == 3)
                throw new InvalidOperationException("notification lost");
        }
    }

    [Fact]
    public async Task HeartbeatRepairsALostNotification()
    {
        using var faulty = new TestApp(new AzulOptions { BotWorkers = 0, SseHeartbeatSeconds = 0.5 },
            s => s.AddSingleton<IFaultInjector, LoseThirdNotification>());
        var alice = faulty.Client("alice@x.com");
        var bob = faulty.Client("bob@x.com");
        var id = await Play.Started(2, alice, bob);
        using var s = await Open(bob, id);
        var first = Json.Deserialize<GameView>((await s.Next()).Data);
        await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(await Play.Get(alice, id)));
        var (ev, data) = await s.Next(seconds: 5);
        Assert.Equal("state", ev);
        var next = Json.Deserialize<GameView>(data);
        Assert.Equal(first.Version + 1, next.Version);
        Assert.NotNull(next.Legal);  // bob's turn, an ordinary human turn
    }

    [Fact]
    public async Task DeleteEndsTheStream()
    {
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice, app.Client("bob@x.com"));
        using var s = await Open(alice, id);
        await s.Next();
        await alice.SendAsync(new HttpRequestMessage(HttpMethod.Delete, $"/api/games/{id}"));
        Assert.Equal("deleted", (await s.Next()).Event);
        Assert.True(await s.Ended());
    }

    [Fact]
    public async Task UnknownGameIs404()
    {
        var res = await app.Client().GetAsync("/api/games/nope/events", HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.NotFound, res.StatusCode);
    }

    [Fact]
    public void SlowSubscriberSeesOnlyTheLatestVersion()
    {
        var hub = new EventHub();
        var (_, reader) = hub.Subscribe("g");
        hub.Publish("g", 5);
        hub.Publish("g", 6);
        hub.Publish("g", 7);
        Assert.True(reader.TryRead(out var v));
        Assert.Equal(7, v);
        Assert.False(reader.TryRead(out _));
    }

    [Fact]
    public async Task StreamsUnsubscribeWhenClientsLeave()
    {
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice, app.Client("bob@x.com"));
        var s = await Open(alice, id);
        await s.Next();
        Assert.Equal(1, app.Service<EventHub>().SubscriberCount(id));
        s.Dispose();
        var until = DateTime.UtcNow.AddSeconds(5);
        while (app.Service<EventHub>().SubscriberCount(id) > 0 && DateTime.UtcNow < until)
            await Task.Delay(50);
        Assert.Equal(0, app.Service<EventHub>().SubscriberCount(id));
    }

    [Fact]
    public async Task StreamsEndWhenTheServerStops()
    {
        // A long heartbeat, so only shutdown can end the stream within the test.
        using var slow = new TestApp(new AzulOptions { BotWorkers = 0, SseHeartbeatSeconds = 60 });
        var alice = slow.Client("alice@x.com");
        var id = await Play.Started(2, alice, slow.Client("bob@x.com"));
        using var s = await Open(alice, id);
        await s.Next();
        var hub = slow.Service<EventHub>();  // the host disposes its services once stopped
        slow.Service<IHostApplicationLifetime>().StopApplication();
        Assert.True(await s.Ended());
        var until = DateTime.UtcNow.AddSeconds(5);
        while (hub.SubscriberCount(id) > 0 && DateTime.UtcNow < until)
            await Task.Delay(50);
        Assert.Equal(0, hub.SubscriberCount(id));
    }
}
