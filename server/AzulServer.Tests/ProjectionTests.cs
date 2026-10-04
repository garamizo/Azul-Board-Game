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
