using Azul;
using AzulServer.Data;

namespace AzulServer.Games;

public static class Projection
{
    const int First = 5;

    public static GameSnapshot Snapshot(GameRecord g) => Json.Deserialize<GameSnapshot>(g.StateJson!);

    public static GameSummary Summary(GameRecord g) => new(
        g.Id, g.Status, g.NumPlayers, g.Creator, SeatViews(g),
        g.StateJson is null ? null : Snapshot(g).RoundIdx + 1, g.UpdatedAt);

    public static GameView Project(GameRecord g, MoveRecord? last, string viewer)
    {
        int? seat = g.Seats.FirstOrDefault(s => s.Email == viewer)?.Idx;
        BoardView? board = null;
        LegalView? legal = null;
        ResultView? result = null;
        if (g.StateJson is not null)
        {
            var game = Game.FromSnapshot(Snapshot(g));
            board = Board(game, g.Status);
            if (g.Status == Status.Playing && seat is { } s && g.Seats[s].Kind == SeatKind.Human && game.activePlayer == s)
                legal = Legal(game);
            if (g.Status == Status.Finished)
                result = Result(game, g.FinishReason ?? "normal");
        }
        return new GameView(g.Id, g.Status, g.Version, g.NumPlayers, g.Creator, new ViewerInfo(viewer, seat),
            SeatViews(g), board, legal, LastMove(last), result);
    }

    public static MoveJson Describe(Move m) => m.colIdx[0] == Move.NOT_SET
        ? new MoveJson("take", m.factoryIdx, m.color, m.row, m.numTiles, null)
        : new MoveJson("wall", null, null, null, null, (int[])m.colIdx.Clone());

    static SeatView[] SeatViews(GameRecord g) => g.Seats.Select(s => new SeatView(s.Idx, s.Kind, s.Email)).ToArray();

    static BoardView Board(Game game, string status)
    {
        int nf = game.numFactories;
        var center = game.factories[nf];
        return new BoardView(
            game.roundIdx + 1,
            status == Status.Finished ? "over" : game.isRegularPhase ? "take" : "wall",
            game.activePlayer,
            game.factories.Take(nf).Select(f => (int[])f.Clone()).ToArray(),
            center.Take(5).ToArray(),
            center[First] > 0,
            (int[])game.bag.Clone(),
            (int[])game.discarded.Clone(),
            game.players.Select(PlayerOf).ToArray());
    }

    static PlayerView PlayerOf(Player p)
    {
        var lines = new int[]?[5];
        for (int row = 0; row < 5; row++)
            for (int c = 0; c < 5; c++)
                if (p.line[row, c] > 0)
                {
                    lines[row] = [c, p.line[row, c]];
                    break;
                }
        var wall = Enumerable.Range(0, 5)
            .Select(r => Enumerable.Range(0, 5).Select(col => p.grid[r, col]).ToArray()).ToArray();
        // The engine keeps counts, not arrival order: expand in colour order.
        var floor = new List<int>();
        for (int c = 0; c < 5; c++)
            for (int i = 0; i < p.floor[c]; i++)
                floor.Add(c);
        return new PlayerView(p.score, lines, wall, floor.ToArray(), p.floor[First] > 0);
    }

    static LegalView Legal(Game game) => game.isRegularPhase
        ? new LegalView(game.LegalTakes().Select(t => new[] { t.Factory, t.Color, t.Row }).ToArray(), null)
        : new LegalView(null, game.WallOptions()!.Select(o => o is null ? null : new WallRowView(o.Color, o.Targets)).ToArray());

    static ResultView Result(Game game, string reason)
    {
        var scores = game.players.Select(p => p.score).ToArray();
        int max = scores.Max();
        return new ResultView(scores, Enumerable.Range(0, scores.Length).Where(i => scores[i] == max).ToArray(), reason);
    }

    static LastMoveView? LastMove(MoveRecord? m)
    {
        if (m is null) return null;
        var j = Json.Deserialize<MoveJson>(m.MoveJson);
        return new LastMoveView(m.Version, m.Seat, j.Kind, j.Factory, j.Color, j.Row, j.Tiles, j.Columns);
    }
}
