using Azul;

namespace AzulLibrary.Tests;

public class ValidationTests
{
    static Move TakeRequest(Game g, int factory, int color, int row) =>
        new Move(new Move { factoryIdx = factory, color = color, row = row }, g);

    [Fact]
    public void WallColumnSixIsRejected()
    {
        var g = Scenarios.WallPhaseWithCompletedLine(2);
        int row = Scenarios.CompletedRows(g)[0];
        var cols = Enumerable.Repeat(-1, 5).ToArray();
        foreach (var r in Scenarios.CompletedRows(g)) cols[r] = 5;  // floor: valid
        Assert.True(g.IsValid(new Move(cols, g)));
        cols[row] = 6;
        Assert.False(g.IsValid(new Move(cols, g)));
        cols[row] = -2;
        Assert.False(g.IsValid(new Move(cols, g)));
    }

    [Fact]
    public void WallColorsOutOfRangeRejected()
    {
        var g = Scenarios.WallPhaseWithCompletedLine(2);
        var cols = Enumerable.Repeat(-1, 5).ToArray();
        var colors = new[] { 6, -1, -1, -1, -1 };
        Assert.False(g.IsValid(cols, colors));
        colors[0] = -2;
        Assert.False(g.IsValid(cols, colors));
        Assert.False(g.IsValid(new int[4], new int[5]));
    }

    [Fact]
    public void OutOfRangeTakesAreRejectedWithoutThrowing()
    {
        var g = new Game(2, new Random(1));
        Assert.False(g.IsValid(-1, 0, 0));
        Assert.False(g.IsValid(g.numFactories + 1, 0, 0));
        Assert.False(g.IsValid(0, -1, 0));
        Assert.False(g.IsValid(0, 6, 0));
        Assert.False(g.IsValid(0, 0, -1));
        Assert.False(g.IsValid(0, 0, 6));
    }

    [Fact]
    public void WrongPhaseMovesAreRejected()
    {
        var take = new Game(2, new Random(1));
        var wallMove = new Move(Enumerable.Repeat(-1, 5).ToArray(), new int[] { -1, -1, -1, -1, -1 }, take.activePlayer);
        Assert.False(take.IsValid(wallMove));

        var wall = Scenarios.WallPhaseWithCompletedLine(2);
        var takeMove = new Move(0, 0, 5, 1, false, wall.activePlayer);
        Assert.False(wall.IsValid(takeMove));
    }

    [Fact]
    public void MoveForAnotherPlayerIsRejected()
    {
        var g = new Game(2, new Random(1));
        var legal = g.GetGreedyMove();
        Assert.True(g.IsValid(legal));
        legal.playerIdx = 1 - g.activePlayer;
        Assert.False(g.IsValid(legal));
    }

    [Fact]
    public void TakingOnlyTheFirstMarkerIsLegal()
    {
        var g = new Game(2, new Random(1));
        int player = g.activePlayer;
        var move = TakeRequest(g, g.CENTER, 5, 5);
        Assert.True(move.isFirst);
        Assert.Equal(1, move.numTiles);
        Assert.True(g.IsValid(move));
        g.Play(move);
        Assert.Equal(1, g.players[player].floor[5]);
        Assert.Equal(0, g.factories[g.CENTER][5]);
        Assert.NotEqual(player, g.activePlayer);
    }

    [Fact]
    public void FirstMarkerOnlyFromTheCentreAndOnlyToTheFloor()
    {
        var g = new Game(2, new Random(1));
        Assert.False(g.IsValid(0, 5, 5));
        Assert.False(g.IsValid(g.CENTER, 5, 0));
    }

    [Theory]
    [InlineData(2, 4)]
    [InlineData(3, 5)]
    [InlineData(4, 6)]
    public void FinishedGameRefusesFurtherPlay(int numPlayers, int seed)
    {
        var g = Scenarios.FinishedGame(numPlayers, seed);
        Assert.True(g.IsFinished);
        var scores = g.players.Select(p => p.score).ToArray();
        var wallMove = new Move(Enumerable.Repeat(-1, 5).ToArray(), g);
        Assert.False(g.IsValid(wallMove));
        Assert.Throws<InvalidOperationException>(() => g.Play(wallMove));
        Assert.Equal(scores, g.players.Select(p => p.score).ToArray());
    }

    [Fact]
    public void IsFinishedOnlyAfterTheLastMove()
    {
        var g = new Game(2, new Random(9));
        while (!g.IsGameOver())
        {
            Assert.False(g.IsFinished);
            g.Play(g.GetGreedyMove());
        }
        Assert.True(g.IsFinished);
        Assert.True(g.Clone().IsFinished);
    }
}
