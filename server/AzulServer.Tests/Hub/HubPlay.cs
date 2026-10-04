using Azul;
using AzulServer.Api;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

/// Deterministic end-of-game states: the hub tests need to choose who makes
/// the finishing move and how (person, bot), which a live game cannot.
public static class HubPlay
{
    /// The state one greedy move before a normal finish (a full wall row),
    /// the move, and the seat that makes it. Tries seeds from `seed` up, so a
    /// seed whose greedy game stalemates is skipped deterministically.
    public static (GameSnapshot State, Move Finishing, int Seat) Penultimate(int players, int seed)
    {
        for (int s = seed; s < seed + 50; s++)
        {
            var game = new Game(players, new Random(s));
            for (int step = 0; step < 2000 && !game.IsFinished; step++)
            {
                var state = game.ToSnapshot();
                var move = game.GetGreedyMove();
                var probe = Game.FromSnapshot(state, new Random(s));
                probe.Play(new Move(move, probe));   // a copy: the returned move is never one that was played
                if (probe.IsFinished)
                {
                    bool normal = probe.players.Any(p => Enumerable.Range(0, 5).Any(r => Enumerable.Range(0, 5).All(c => p.grid[r, c] >= 0)));
                    if (!normal) break;   // stalemate: next seed
                    return (state, new Move(move, Game.FromSnapshot(state)), state.ActivePlayer);
                }
                game.Play(move);
            }
        }
        throw new InvalidOperationException($"no normal finish for seeds {seed}..{seed + 49}");
    }

    /// A playing, hub-tracked game at version 5 with no stored moves.
    public static void Insert(Db db, string id, SeatRecord[] seats, GameSnapshot state, string creator, bool tracked = true)
    {
        var g = new GameRecord(id, creator, Status.Playing, seats.Length, 5, Json.Serialize(state), null,
            "2026-10-04T18:00:00.0000000Z", "2026-10-04T18:00:00.0000000Z", seats)
        {
            StartedAt = "2026-10-04T18:00:01.0000000Z",
            HubTracked = tracked,
            BotKey = "mcts@0123456789ab",
        };
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        GameStore.Insert(c, tx, g);
        tx.Commit();
    }

    static bool IsTake(Move m) => m.colIdx[0] == Move.NOT_SET;

    public static MoveRequest Request(long version, Move m) => IsTake(m)
        ? new MoveRequest(version, Guid.NewGuid().ToString(), "take", m.factoryIdx, m.color, m.row, null)
        : new MoveRequest(version, Guid.NewGuid().ToString(), "wall", null, null, null, (int[])m.colIdx.Clone());

    /// The same move as an HTTP body for POST /api/games/{id}/moves.
    public static object Body(long version, Move m) => IsTake(m)
        ? new { version, requestId = Guid.NewGuid().ToString(), kind = "take", factory = m.factoryIdx, color = m.color, row = m.row }
        : new { version, requestId = Guid.NewGuid().ToString(), kind = "wall", columns = (int[])m.colIdx.Clone() };
}
