using Azul;

namespace AzulLibrary.Tests;

public class GameplayTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void GreedyGameFinishes(int numPlayers)
    {
        var game = new Game(numPlayers);
        int moves = 0;
        while (!game.IsGameOver())
        {
            var move = game.GetGreedyMove();
            Assert.True(game.IsValid(move), $"greedy move {move} invalid");
            game.Play(move);
            Assert.True(++moves < 2000, "game did not end");
        }
        Assert.All(game.players, p => Assert.True(p.score >= 0));
    }
}
