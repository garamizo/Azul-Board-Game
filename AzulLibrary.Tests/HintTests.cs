using Azul;

namespace AzulLibrary.Tests;

public class HintTests
{
    static IEnumerable<Game> States(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        while (!g.IsGameOver())
        {
            yield return g.Clone(new Random(seed));
            g.Play(g.GetGreedyMove());
        }
    }

    static IEnumerable<int[]> Expand(WallRowOption?[] opts)
    {
        IEnumerable<int[]> acc = new[] { Array.Empty<int>() };
        for (int row = 0; row < 5; row++)
        {
            var targets = opts[row]?.Targets ?? new[] { -1 };
            acc = acc.SelectMany(prefix => targets.Select(t => prefix.Append(t).ToArray()));
        }
        // The one cross-row rule: same colour, same wall column.
        return acc.Where(cols => !Enumerable.Range(0, 5).Any(i => Enumerable.Range(0, i).Any(j =>
            cols[i] is >= 0 and < 5 && cols[i] == cols[j] && opts[i]!.Color == opts[j]!.Color)));
    }

    [Theory]
    [InlineData(2, 31)]
    [InlineData(3, 32)]
    [InlineData(4, 33)]
    public void HintsMatchTheEnginesGenerators(int numPlayers, int seed)
    {
        foreach (var g in States(numPlayers, seed))
        {
            if (g.isRegularPhase)
            {
                var expected = g.GetPossibleActions().Select(m => (m.factoryIdx, m.color, m.row)).ToHashSet();
                Assert.Equal(expected, g.LegalTakes().Select(t => (t.Factory, t.Color, t.Row)).ToHashSet());
                Assert.Null(g.WallOptions());
            }
            else
            {
                var expected = g.GetColIdxMoves().Select(m => string.Join(",", m.colIdx)).ToHashSet();
                var opts = g.WallOptions()!;
                Assert.Equal(expected, Expand(opts).Select(c => string.Join(",", c)).ToHashSet());
                Assert.Empty(g.LegalTakes());
                foreach (var o in opts.Where(o => o is not null))
                    Assert.Equal(5, o!.Targets[^1]);
            }
        }
    }

    [Fact]
    public void HintsDoNotTouchTheRandomness()
    {
        foreach (var (a, b) in States(3, 40).Zip(States(3, 40)))
        {
            a.LegalTakes();
            a.WallOptions();
            a.ForcedMove();
            Assert.Equal(b.rng.Next(), a.rng.Next());
        }
    }

    [Fact]
    public void OnlyTheFirstMarkerLeftIsForced()
    {
        var s = new Game(2, new Random(1)).ToSnapshot();
        for (int f = 0; f < s.Factories.Length; f++)
            for (int c = 0; c < 5; c++) { s.Bag[c] += s.Factories[f][c]; s.Factories[f][c] = 0; }
        var g = Game.FromSnapshot(s);
        var forced = g.ForcedMove();
        Assert.NotNull(forced);
        Assert.Equal((g.CENTER, 5, 5), (forced!.factoryIdx, forced.color, forced.row));
        Assert.True(g.IsValid(forced));
    }

    [Fact]
    public void WallTurnWithNothingToPlaceIsForced()
    {
        var g = Scenarios.WallPhaseWithCompletedLine(2);
        var s = g.ToSnapshot();
        var lines = s.Players[s.ActivePlayer].Line;
        for (int row = 0; row < 5; row++)
            for (int c = 0; c < 5; c++) { s.Bag[c] += lines[row][c]; lines[row][c] = 0; }
        var empty = Game.FromSnapshot(s);
        var forced = empty.ForcedMove();
        Assert.NotNull(forced);
        Assert.All(forced!.colIdx, c => Assert.Equal(-1, c));
        Assert.True(empty.IsValid(forced));
    }

    [Fact]
    public void NormalTurnsAreNotForced()
    {
        Assert.Null(new Game(3, new Random(1)).ForcedMove());
        // A wall turn with a line that can go on the wall has a choice.
        var wall = Scenarios.WallPhaseWithCompletedLine(2);
        if (wall.WallOptions()!.Any(o => o is not null && o.Targets.Length > 1))
            Assert.Null(wall.ForcedMove());
    }

    [Fact]
    public void ForcedMoveIsNullWhenFinished()
    {
        Assert.Null(Scenarios.FinishedGame(2, 4).ForcedMove());
    }
}
