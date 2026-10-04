using System.Net;
using Azul;
using AzulServer.Data;
using AzulServer.Games;
using Microsoft.Extensions.DependencyInjection;

namespace AzulServer.Tests;

public class BotTests
{
    static AzulOptions Fast(int workers = 1) =>
        new() { BotWorkers = workers, MinMoveDelaySeconds = 0, SweepSeconds = 0.2 };

    static long Count(TestApp app, string sql)
    {
        using var c = app.Service<Db>().Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        return (long)cmd.ExecuteScalar()!;
    }

    [Fact]
    public async Task BotsAnswerAfterAHumanMove()
    {
        using var app = new TestApp(Fast());
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(3, alice);
        var v = await Play.Get(alice, id);
        await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v));
        var back = await Play.WaitFor(alice, id, x => x.Board!.ActiveSeat == 0 || x.Status == Status.Finished);
        Assert.True(back.Version >= v.Version + 3);
        Assert.True(Count(app, $"SELECT COUNT(*) FROM moves WHERE game_id='{id}' AND actor='bot'") >= 2);
    }

    [Fact]
    public async Task ForcedWallTurnsArePlayedForTheHuman()
    {
        using var app = new TestApp(Fast());
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice);
        // alice always sends everything to the floor, so she never completes a
        // line and every wall turn of hers is forced.
        while (true)
        {
            // Wait for a take turn of alice's; her (forced) wall turns are the server's.
            var v = await Play.WaitFor(alice, id, x => x.Legal?.Takes is not null || x.Board!.Round >= 2 || x.Status == Status.Finished);
            if (v.Board!.Round >= 2 || v.Status == Status.Finished) break;
            var floorTake = v.Legal!.Takes!.First(t => t[2] == 5);
            await alice.Post($"/api/games/{id}/moves", Play.Take(v, floorTake));  // 409 if a forced take beat her
        }
        Assert.True(Count(app, $"SELECT COUNT(*) FROM moves WHERE game_id='{id}' AND actor='auto'") >= 1);
    }

    sealed class BlockingBrain : IBotBrain
    {
        public readonly SemaphoreSlim Started = new(0);
        public Move ChooseMove(Game game, CancellationToken ct)
        {
            Started.Release();
            ct.WaitHandle.WaitOne(TimeSpan.FromSeconds(10));
            return game.GetGreedyMove();
        }
    }

    [Fact]
    public async Task TakeBackWhileTheBotThinks()
    {
        var brain = new BlockingBrain();
        using var app = new TestApp(Fast(), s => s.AddSingleton<IBotBrain>(brain));
        var alice = app.Client("alice@x.com");
        var bob = app.Client("bob@x.com");
        var id = await Play.Started(2, alice, bob);
        await bob.Post($"/api/games/{id}/seats/1/to-bot");
        var v = await Play.Get(alice, id);
        await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v));
        Assert.True(await brain.Started.WaitAsync(TimeSpan.FromSeconds(10)), "bot never started thinking");
        Assert.Equal(HttpStatusCode.OK, (await bob.Post($"/api/games/{id}/seats/1/take-back")).StatusCode);
        var mine = await Play.WaitFor(bob, id, x => x.Legal is not null, 10);
        Assert.Equal(1, mine.Board!.ActiveSeat);
        await Task.Delay(300);  // give a stale bot result time to (wrongly) land
        Assert.Equal(0L, Count(app, $"SELECT COUNT(*) FROM moves WHERE game_id='{id}' AND actor='bot'"));
        Assert.Equal(mine.Version, (await Play.Get(bob, id)).Version);
    }

    /// Throws after the second commit of the test (the start is the first,
    /// alice's move the second): the bot's turn is committed but never poked.
    sealed class CrashAfterSecondCommit : IFaultInjector
    {
        int commits;
        public void AfterCommit(string gameId)
        {
            if (Interlocked.Increment(ref commits) == 2)
                throw new InvalidOperationException("simulated crash after commit");
        }
    }

    [Fact]
    public async Task SweepRepairsALostPoke()
    {
        using var app = new TestApp(Fast(), s => s.AddSingleton<IFaultInjector, CrashAfterSecondCommit>());
        var alice = app.Client("alice@x.com");
        var g = await Play.Create(alice, 2);
        await alice.Post($"/api/games/{g.Id}/start");
        var v = await Play.Get(alice, g.Id);
        await alice.Post($"/api/games/{g.Id}/moves", Play.AnyLegal(v));
        await Play.WaitFor(alice, g.Id, x => x.Board!.ActiveSeat == 0 && x.Version >= v.Version + 2);
    }

    [Fact]
    public async Task EveryGameKeepsMovingWithOneWorker()
    {
        using var app = new TestApp(Fast(workers: 1));
        var alice = app.Client("alice@x.com");
        var ids = new List<string>();
        for (int i = 0; i < 6; i++)
        {
            var id = await Play.Started(4, alice);
            await alice.Post($"/api/games/{id}/seats/0/to-bot");  // all bots now
            ids.Add(id);
        }
        var start = new Dictionary<string, long>();
        foreach (var id in ids) start[id] = (await Play.Get(alice, id)).Version;
        await Task.Delay(TimeSpan.FromSeconds(3));
        foreach (var id in ids)
        {
            var v = await Play.Get(alice, id);
            Assert.True(v.Version >= start[id] + 3 || v.Status == Status.Finished, $"game {id} stalled at {v.Version}");
        }
    }

    sealed class CountingBrain : IBotBrain
    {
        int current, max;
        public int Max => max;
        public Move ChooseMove(Game game, CancellationToken ct)
        {
            int n = Interlocked.Increment(ref current);
            int seen;
            do { seen = max; } while (n > seen && Interlocked.CompareExchange(ref max, n, seen) != seen);
            Thread.Sleep(150);
            Interlocked.Decrement(ref current);
            return game.GetGreedyMove();
        }
    }

    [Fact]
    public async Task OneSearchPerGameEvenWithEagerSweeps()
    {
        var brain = new CountingBrain();
        using var app = new TestApp(new AzulOptions { BotWorkers = 3, MinMoveDelaySeconds = 0, SweepSeconds = 0.02 },
            s => s.AddSingleton<IBotBrain>(brain));
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(4, alice);
        await alice.Post($"/api/games/{id}/seats/0/to-bot");
        var start = (await Play.Get(alice, id)).Version;
        await Task.Delay(TimeSpan.FromSeconds(2));
        Assert.Equal(1, brain.Max);
        Assert.True((await Play.Get(alice, id)).Version >= start + 3);
    }

    [Fact]
    public async Task PendingBotTurnResumesAfterARestart()
    {
        var dir = Directory.CreateTempSubdirectory("azul-restart-").FullName;
        string id;
        long version;
        using (var first = new TestApp(new AzulOptions { BotWorkers = 0 }, dataDir: dir))
        {
            var alice = first.Client("alice@x.com");
            id = await Play.Started(2, alice);
            var v = await Play.Get(alice, id);
            await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v));  // now the bot's turn; no workers
            version = (await Play.Get(alice, id)).Version;
        }
        using var second = new TestApp(Fast(), dataDir: dir);
        var after = await Play.WaitFor(second.Client("alice@x.com"), id, x => x.Version > version);
        Assert.Equal(Status.Playing, after.Status);
    }

    [Fact]
    public void MctsBrainProposesALegalMove()
    {
        var brain = new MctsBrain(new AzulOptions { BotThinkSeconds = 0.3 });
        var game = new Game(3, new Random(1));
        Assert.True(game.IsValid(brain.ChooseMove(game, CancellationToken.None)));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.True(game.IsValid(brain.ChooseMove(game, cancelled.Token)));
    }
}
