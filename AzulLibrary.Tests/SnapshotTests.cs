using System.Text.Json;
using Azul;

namespace AzulLibrary.Tests;

public class SnapshotTests
{
    static GameSnapshot ViaJson(GameSnapshot s) =>
        JsonSerializer.Deserialize<GameSnapshot>(JsonSerializer.Serialize(s))!;

    [Theory]
    [InlineData(2, 21)]
    [InlineData(3, 22)]
    [InlineData(4, 23)]
    public void RoundTripsAfterEveryMove(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        while (true)
        {
            var restored = Game.FromSnapshot(ViaJson(g.ToSnapshot()));
            GameAssert.Equal(g, restored);
            if (g.IsGameOver()) break;
            g.Play(g.GetGreedyMove());
        }
        Assert.True(Game.FromSnapshot(g.ToSnapshot()).IsFinished);
    }

    [Fact]
    public void RestoredGameKeepsPlaying()
    {
        var g = new Game(3, new Random(4));
        for (int i = 0; i < 10; i++) g.Play(g.GetGreedyMove());
        var r = Game.FromSnapshot(g.ToSnapshot(), new Random(1));
        while (!r.IsGameOver()) r.Play(r.GetGreedyMove());
        Assert.True(r.IsFinished);
    }

    [Fact]
    public void ExhaustedBagRoundTrips()
    {
        // Move every tile from the factories, bag and discard onto player 0's
        // floor: the next deal finds nothing to draw and leaves factories empty.
        var s = new Game(4, new Random(2)).ToSnapshot();
        var floor = s.Players[0].Floor;
        for (int c = 0; c < 5; c++)
        {
            floor[c] += s.Bag[c] + s.Discarded[c];
            s.Bag[c] = 0;
            s.Discarded[c] = 0;
            foreach (var f in s.Factories) { floor[c] += f[c]; f[c] = 0; }
        }
        var g = Game.FromSnapshot(s);
        g.FillFactories();
        Assert.Equal(6, g.factories[g.CENTER].Length);
        Assert.Equal(1, g.factories[g.CENTER][5]);
        GameAssert.Equal(g, Game.FromSnapshot(ViaJson(g.ToSnapshot())));
    }

    public static TheoryData<string, Func<GameSnapshot, GameSnapshot>> Corruptions => new()
    {
        { "format", s => s with { Format = 2 } },
        { "players", s => s with { NumPlayers = 5 } },
        { "factories length", s => s with { Factories = s.Factories[..^1] } },
        { "active player", s => s with { ActivePlayer = s.NumPlayers } },
        { "negative bag", s => { var b = (int[])s.Bag.Clone(); b[0] = -1; return s with { Bag = b }; } },
        { "tile missing", s => { var b = (int[])s.Bag.Clone(); b[0] -= 1; return s with { Bag = b }; } },
        { "two colours on a line", s => {
            var p = s.Players[0];
            p.Line[1][0] = 1; p.Line[1][2] = 1;
            var b = (int[])s.Bag.Clone(); b[0] -= 1; b[2] -= 1;
            return s with { Bag = b }; } },
        { "line overfull", s => {
            var p = s.Players[0];
            p.Line[0][3] = 2;
            var b = (int[])s.Bag.Clone(); b[3] -= 2;
            return s with { Bag = b }; } },
        { "duplicate wall colour in a row", s => {
            var p = s.Players[0];
            p.Grid[0][0] = 1; p.Grid[0][1] = 1;
            var b = (int[])s.Bag.Clone(); b[1] -= 2;
            return s with { Bag = b }; } },
        { "duplicate wall colour in a column", s => {
            var p = s.Players[0];
            p.Grid[0][0] = 4; p.Grid[1][0] = 4;
            var b = (int[])s.Bag.Clone(); b[4] -= 2;
            return s with { Bag = b }; } },
        { "wall cell out of range", s => { s.Players[0].Grid[2][2] = 7; return s; } },
        { "two first markers", s => { s.Players[0].Floor[5] = 1; return s; } },
        { "finished in take phase", s => s with { IsFinished = true } },
    };

    [Theory]
    [MemberData(nameof(Corruptions))]
    public void CorruptSnapshotsAreRejected(string what, Func<GameSnapshot, GameSnapshot> corrupt)
    {
        var s = corrupt(new Game(2, new Random(1)).ToSnapshot());
        var ex = Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s));
        Assert.False(string.IsNullOrEmpty(ex.Message), what);
    }

    [Fact]
    public void NullAndRaggedArraysAreRejected()
    {
        var s = new Game(2, new Random(1)).ToSnapshot();
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Bag = null! }));
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Players = null! }));
        var ragged = s.Factories.Select(f => (int[])f.Clone()).ToArray();
        ragged[0] = new int[3];
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Factories = ragged }));
        var p = s.Players[0] with { Grid = s.Players[0].Grid[..4] };
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Players = new[] { p, s.Players[1] } }));
        // A row whose JSON has "Discarded": null deserializes to a null array.
        var fromJson = JsonSerializer.Deserialize<GameSnapshot>(JsonSerializer.Serialize(s with { Discarded = null! }))!;
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(fromJson));
    }
}
