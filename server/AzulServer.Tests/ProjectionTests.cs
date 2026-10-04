using Azul;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

public class ProjectionTests
{
    static GameRecord Playing(Game game, params SeatRecord[] seats) => new(
        "g", "a@x", Status.Playing, game.numPlayers, 7, Json.Serialize(game.ToSnapshot()), null, "t", "t", seats);

    static readonly SeatRecord[] ThreeSeats =
        [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Human, "b@x"), new(2, SeatKind.Bot, "c@x")];

    [Fact]
    public void BoardMirrorsTheEngine()
    {
        var game = new Game(3, new Random(1));
        var s = game.ToSnapshot();
        // Player 1: two red on line 2, one blue and the FIRST marker on the floor.
        s.Players[1].Line[2][2] = 2; s.Bag[2] -= 2;
        s.Players[1].Floor[0] = 1; s.Bag[0] -= 1;
        s.Players[1].Floor[5] = 1; s.Factories[^1][5] = 0;
        var restored = Game.FromSnapshot(s);
        var v = Projection.Project(Playing(restored, ThreeSeats), null, "b@x");

        Assert.Equal(restored.numFactories, v.Board!.Factories.Length);
        Assert.Equal(5, v.Board.Center.Length);
        Assert.False(v.Board.CenterHasFirst);
        var p1 = v.Board.Players[1];
        Assert.Equal(new[] { 2, 2 }, p1.Lines[2]);
        Assert.Null(p1.Lines[0]);
        Assert.Equal(new[] { 0 }, p1.Floor);
        Assert.True(p1.HasFirst);
        Assert.Equal("take", v.Board.Phase);
        Assert.Equal(1, v.Board.Round);
    }

    [Fact]
    public void LegalOnlyForTheActiveHuman()
    {
        var game = new Game(3, new Random(1));  // seat 0 to move
        var rec = Playing(game, ThreeSeats);
        var a = Projection.Project(rec, null, "a@x");
        Assert.Equal(0, a.You.Seat);
        Assert.NotNull(a.Legal!.Takes);
        Assert.Null(a.Legal.Wall);
        Assert.Null(Projection.Project(rec, null, "b@x").Legal);
        var spectator = Projection.Project(rec, null, "z@x");
        Assert.Null(spectator.You.Seat);
        Assert.Null(spectator.Legal);
    }

    /// A 3-player wall phase with a completed line for the active seat;
    /// `clearLines` empties that seat's lines so its only move is all -1.
    static Game WallTurn(bool clearLines)
    {
        for (int seed = 1; seed < 200; seed++)
        {
            var g = new Game(3, new Random(seed));
            while (!g.IsGameOver())
            {
                if (!g.isRegularPhase && g.WallOptions()!.Any(o => o is not null && o.Targets.Length > 1))
                {
                    if (!clearLines) return g;
                    var s = g.ToSnapshot();
                    var lines = s.Players[s.ActivePlayer].Line;
                    for (int row = 0; row < 5; row++)
                        for (int c = 0; c < 5; c++) { s.Bag[c] += lines[row][c]; lines[row][c] = 0; }
                    return Game.FromSnapshot(s);
                }
                g.Play(g.GetGreedyMove());
            }
        }
        throw new InvalidOperationException("no wall turn with a choice found");
    }

    static SeatRecord[] AllHuman(Game g) =>
        Enumerable.Range(0, 3).Select(i => new SeatRecord(i, SeatKind.Human, i == g.activePlayer ? "me@x" : $"p{i}@x")).ToArray();

    [Fact]
    public void ForcedWallTurnIsAutoPlayNotLegal()
    {
        var game = WallTurn(clearLines: true);
        Assert.NotNull(game.ForcedMove());
        var v = Projection.Project(Playing(game, AllHuman(game)), null, "me@x");
        Assert.Equal("wall", v.Board!.Phase);
        Assert.Null(v.Legal);
        Assert.True(v.AutoPlay);
        // Other viewers see a plain board.
        Assert.False(Projection.Project(Playing(game, AllHuman(game)), null, "z@x").AutoPlay);
    }

    [Fact]
    public void WallTurnWithAChoiceIsNotAutoPlay()
    {
        var game = WallTurn(clearLines: false);
        Assert.Null(game.ForcedMove());
        var v = Projection.Project(Playing(game, AllHuman(game)), null, "me@x");
        Assert.NotNull(v.Legal!.Wall);
        Assert.False(v.AutoPlay);
        var normal = Projection.Project(Playing(new Game(3, new Random(1)), ThreeSeats), null, "a@x");
        Assert.NotNull(normal.Legal);
        Assert.False(normal.AutoPlay);
    }

    [Fact]
    public void HandedToBotSeatStillBelongsToItsEmail()
    {
        var game = new Game(3, new Random(1));
        game.Play(game.GetGreedyMove());
        game.Play(game.GetGreedyMove());  // seat 2 (the bot seat) to move
        var v = Projection.Project(Playing(game, ThreeSeats), null, "c@x");
        Assert.Equal(2, v.You.Seat);
        Assert.Null(v.Legal);  // a bot plays that seat until it is taken back
        Assert.Equal("c@x", v.Seats[2].Email);
    }

    [Fact]
    public void FinishedGameHasAResultAndTies()
    {
        var game = new Game(2, new Random(4));
        while (!game.IsGameOver()) game.Play(game.GetGreedyMove());
        var rec = Playing(game, [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Bot, null)])
            with { Status = Status.Finished, FinishReason = "normal" };
        var v = Projection.Project(rec, null, "a@x");
        Assert.Equal("over", v.Board!.Phase);
        Assert.Equal(game.players.Select(p => p.score), v.Result!.Scores);
        int max = v.Result.Scores.Max();
        Assert.Equal(Enumerable.Range(0, 2).Where(i => v.Result.Scores[i] == max), v.Result.Winners);
        Assert.Null(v.Legal);
    }

    [Fact]
    public void LobbyHasNoBoard()
    {
        var rec = new GameRecord("g", "a@x", Status.Lobby, 2, 1, null, null, "t", "t",
            [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Open, null)]);
        var v = Projection.Project(rec, null, "a@x");
        Assert.Null(v.Board);
        Assert.Equal(0, v.You.Seat);
        Assert.Null(Projection.Summary(rec).Round);
    }

    [Fact]
    public void LastMoveAndDescribe()
    {
        var game = new Game(2, new Random(1));
        var move = new Move(new Move { factoryIdx = game.numFactories, color = 5, row = 5 }, game);
        var json = Json.Serialize(Projection.Describe(move));
        var last = new MoveRecord("g", 8, 0, "a@x", json, null, null, null, "t");
        var v = Projection.Project(Playing(game, [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Bot, null)]), last, "a@x");
        Assert.Equal(new LastMoveView(8, 0, "take", game.numFactories, 5, 5, 1, null), v.LastMove);
    }
}
