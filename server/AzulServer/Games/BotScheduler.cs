using System.Diagnostics;
using Ai;
using Azul;

namespace AzulServer.Games;

public interface IBotBrain
{
    Move ChooseMove(Game game, CancellationToken ct);
}

/// Deterministic and instant; used by tests.
public sealed class GreedyBrain : IBotBrain
{
    public Move ChooseMove(Game game, CancellationToken ct) => game.GetGreedyMove();
}

/// The desktop game's bot: MCTS_Stochastic for AZUL_BOT_THINK_SECONDS on its
/// own clone, greedy when the search found nothing (game.py:261).
public sealed class MctsBrain(AzulOptions options) : IBotBrain
{
    /// The search stops here even with think time left.
    public const int RolloutCap = 300_000;
    /// Part of the bot's hub key (Hub/BotIdentity.cs). Bump it whenever this
    /// class's logic changes how a move is chosen (the greedy fallback, the
    /// stopping rule, the final pick), so the hub rates the changed bot as a
    /// new account.
    public const int BrainRevision = 1;

    public Move ChooseMove(Game game, CancellationToken ct)
    {
        var root = new MCTS_Stochastic<Game, Move>(game, 0.0f);  // clones the game
        var clock = Stopwatch.StartNew();
        while (!ct.IsCancellationRequested
               && clock.Elapsed.TotalSeconds < options.BotThinkSeconds
               && root.numRolls < RolloutCap)
            root.Grow();
        if (root.actions.Count == 0) return game.GetGreedyMove();
        int best = root.GetBestActionIdx();
        var move = root.NumRolls(best) > 0 && root.WinRatio(best) > 0 ? root.actions[best] : game.GetGreedyMove();
        return game.IsValid(move) ? move : game.GetGreedyMove();
    }
}

public sealed class BotScheduler(ServerMoveQueue queue, GameService games, IBotBrain brain,
    AzulOptions options, ILogger<BotScheduler> log) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken ct)
    {
        var loops = Enumerable.Range(0, Math.Max(0, options.BotWorkers))
            .Select(_ => Task.Run(() => Worker(ct), ct))
            .Append(Sweep(ct));
        return Task.WhenAll(loops);
    }

    async Task Sweep(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                foreach (var id in games.GamesNeedingServerMove())
                    queue.Poke(id);
            }
            catch (Exception e)
            {
                log.LogError(e, "sweep failed");
            }
            try { await Task.Delay(TimeSpan.FromSeconds(options.SweepSeconds), ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    async Task Worker(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            string id;
            try { id = await queue.TakeAsync(ct); }
            catch (OperationCanceledException) { return; }
            if (!queue.TryStart(id)) continue;
            bool hadTurn = false;
            try
            {
                hadTurn = await PlayOne(id, ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                return;
            }
            catch (Exception e)
            {
                // Dropped; the sweep brings the game back.
                log.LogError(e, "server move failed for game {Game}", id);
            }
            finally
            {
                queue.Finish(id);
                // A poke taken while this game was active was skipped; look
                // again. Bounded: the next pass plays a move or finds no turn.
                if (hadTurn && !ct.IsCancellationRequested) queue.Poke(id);
            }
        }
    }

    async Task<bool> PlayOne(string id, CancellationToken ct)
    {
        var turn = games.GetServerTurn(id);
        if (turn is null) return false;
        var clock = Stopwatch.StartNew();
        Move move;
        if (turn.Forced is { } forced)
        {
            move = forced;
        }
        else
        {
            var search = queue.BeginSearch(id, ct);
            try { move = brain.ChooseMove(turn.Game, search.Token); }
            finally { queue.EndSearch(id, search); }
        }
        var wait = TimeSpan.FromSeconds(options.MinMoveDelaySeconds) - clock.Elapsed;
        if (wait > TimeSpan.Zero) await Task.Delay(wait, ct);
        await games.ApplyServerMove(id, turn.Version, move, turn.Actor);  // 409 = something changed; dropped
        return true;
    }
}
