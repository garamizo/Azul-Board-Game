using Azul;

namespace AzulLibrary.Tests;

public class DeterminismTests
{
    internal static Game PlayGreedy(int numPlayers, int seed)
    {
        var game = new Game(numPlayers, new Random(seed));
        while (!game.IsGameOver())
            game.Play(game.GetGreedyMove());
        return game;
    }

    internal static string Fingerprint(Game g) =>
        g.ToString()
        + "|" + string.Join(",", g.players.Select(p => p.score))
        + "|" + string.Join(",", g.bag) + "|" + string.Join(",", g.discarded)
        + "|" + g.step + "|" + g.roundIdx;

    [Fact]
    public void SameSeedSameGame()
    {
        Assert.Equal(Fingerprint(PlayGreedy(3, 7)), Fingerprint(PlayGreedy(3, 7)));
    }

    [Fact]
    public void ParallelGamesMatchSequentialGames()
    {
        int[] seeds = Enumerable.Range(1, 16).ToArray();
        string[] sequential = seeds.Select(s => Fingerprint(PlayGreedy(4, s))).ToArray();
        string[] parallel = new string[seeds.Length];
        Parallel.For(0, seeds.Length, new ParallelOptions { MaxDegreeOfParallelism = 8 },
            i => parallel[i] = Fingerprint(PlayGreedy(4, seeds[i])));
        Assert.Equal(sequential, parallel);
    }
}
