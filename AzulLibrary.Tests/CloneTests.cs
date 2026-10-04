using Ai;
using Azul;

namespace AzulLibrary.Tests;

public class CloneTests
{
    [Theory]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(4, 3)]
    public void CloneEqualsOriginalAfterEveryMove(int numPlayers, int seed)
    {
        var game = new Game(numPlayers, new Random(seed));
        while (!game.IsGameOver())
        {
            var clone = game.Clone();
            GameAssert.Equal(game, clone);
            GameAssert.SharesNothing(game, clone);
            game.Play(game.GetGreedyMove());
        }
        GameAssert.Equal(game, game.Clone());
    }

    [Fact]
    public void PlayingTheCloneLeavesTheOriginalAlone()
    {
        var game = new Game(3, new Random(5));
        var before = game.Clone();
        var clone = game.Clone();
        while (!clone.IsGameOver())
            clone.Play(clone.GetGreedyMove());
        GameAssert.Equal(before, game);
    }

    [Fact]
    public void CloningDoesNotAdvanceTheOriginalsRandomness()
    {
        var a = new Game(2, new Random(11));
        var b = new Game(2, new Random(11));
        var clone = a.Clone();
        while (!clone.IsGameOver())
            clone.Play(clone.GetRandomMove());
        Assert.Equal(b.rng.Next(), a.rng.Next());
    }

    [Fact]
    public void MctsStillProposesValidMoves()
    {
        var game = new Game(3, new Random(3));
        var tree = new MCTS_Stochastic<Game, Move>(game, 0f);
        for (int i = 0; i < 300; i++) tree.Grow();  // a fixed count, not a time budget
        Assert.Equal(300, tree.numRolls);
        Assert.True(game.IsValid(tree.GetBestAction()));
    }

    [Fact]
    public void ParallelMctsSearchesDoNotInterfere()
    {
        // Each search owns its game and tree; run 8 at once and check every
        // proposed move is valid for its own game.
        var games = Enumerable.Range(1, 8).Select(s => new Game(4, new Random(s))).ToArray();
        var moves = new Move[games.Length];
        Parallel.For(0, games.Length, i =>
        {
            var tree = new MCTS_Stochastic<Game, Move>(games[i], 0f);
            for (int r = 0; r < 400; r++) tree.Grow();
            moves[i] = tree.GetBestAction();
        });
        for (int i = 0; i < games.Length; i++)
            Assert.True(games[i].IsValid(moves[i]), $"game {i}: {moves[i]}");
    }

    [Fact]
    public void TicTacToeGreedyStillWorks()
    {
        var ttt = new TicTacToe.Game();
        var move = ttt.GetGreedyMove();
        Assert.True(ttt.IsValid(move));
        var copy = ttt.Clone();
        copy.Play(move);
        Assert.True(ttt.IsValid(move), "playing the clone changed the original");
    }
}
