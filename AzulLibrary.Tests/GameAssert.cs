using Azul;

namespace AzulLibrary.Tests;

/// Field-by-field comparison. Game.Equals is not used: it ignores bag,
/// discard, phase and more (Logic.cs:679).
internal static class GameAssert
{
    public static void Equal(Game expected, Game actual)
    {
        Assert.Equal(expected.numPlayers, actual.numPlayers);
        Assert.Equal(expected.activePlayer, actual.activePlayer);
        Assert.Equal(expected.step, actual.step);
        Assert.Equal(expected.chanceHash, actual.chanceHash);
        Assert.Equal(expected.roundIdx, actual.roundIdx);
        Assert.Equal(expected.isRegularPhase, actual.isRegularPhase);
        Assert.Equal(expected.newRoundPlayer, actual.newRoundPlayer);
        Assert.Equal(expected.countPlayerClearedRound, actual.countPlayerClearedRound);
        Assert.Equal(expected.numFactories, actual.numFactories);
        Assert.Equal(expected.CENTER, actual.CENTER);
        Assert.Equal(expected.factories.Length, actual.factories.Length);
        for (int i = 0; i < expected.factories.Length; i++)
            Assert.Equal(expected.factories[i], actual.factories[i]);
        Assert.Equal(expected.bag, actual.bag);
        Assert.Equal(expected.discarded, actual.discarded);
        Assert.Equal(expected.players.Length, actual.players.Length);
        for (int i = 0; i < expected.players.Length; i++)
        {
            var e = expected.players[i];
            var a = actual.players[i];
            Assert.Equal(e.score, a.score);
            Assert.Equal(e.grid.Cast<int>(), a.grid.Cast<int>());
            Assert.Equal(e.line.Cast<int>(), a.line.Cast<int>());
            Assert.Equal(e.floor, a.floor);
        }
    }

    public static void SharesNothing(Game a, Game b)
    {
        Assert.NotSame(a.factories, b.factories);
        for (int i = 0; i < a.factories.Length; i++)
            Assert.NotSame(a.factories[i], b.factories[i]);
        Assert.NotSame(a.bag, b.bag);
        Assert.NotSame(a.discarded, b.discarded);
        Assert.NotSame(a.players, b.players);
        for (int i = 0; i < a.players.Length; i++)
        {
            Assert.NotSame(a.players[i], b.players[i]);
            Assert.NotSame(a.players[i].grid, b.players[i].grid);
            Assert.NotSame(a.players[i].line, b.players[i].line);
            Assert.NotSame(a.players[i].floor, b.players[i].floor);
        }
        Assert.NotSame(a.rng, b.rng);
    }
}
