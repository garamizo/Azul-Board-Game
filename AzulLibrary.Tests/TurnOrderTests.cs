using Azul;

namespace AzulLibrary.Tests;

public class TurnOrderTests
{
    [Theory]
    [InlineData(2, 52)]
    [InlineData(3, 52)]
    [InlineData(4, 53)]
    public void WallPhaseFollowsTheEnginesOrder(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        bool sawEmptyWallTurn = false;
        while (!g.IsGameOver())
        {
            // Take phase.
            int lastTaker = -1;
            while (g.isRegularPhase)
            {
                lastTaker = g.activePlayer;
                // Random takes: greedy play never leaves a player with nothing to place.
                g.Play(g.GetRandomMove());
            }
            Assert.Equal((lastTaker + 1) % numPlayers, g.activePlayer);
            int holder = Array.FindIndex(g.players, p => p.floor[5] == 1);
            Assert.True(holder >= 0, "someone holds the FIRST marker after the take phase");

            // Wall phase: every seat once, in turn, including seats with nothing to place.
            var seen = new List<int>();
            int round = g.roundIdx;
            for (int i = 0; i < numPlayers; i++)
            {
                Assert.False(g.isRegularPhase);
                Assert.False(g.IsFinished);
                if (Scenarios.CompletedRows(g).Count == 0)
                {
                    sawEmptyWallTurn = true;
                    var forced = g.ForcedMove();
                    Assert.NotNull(forced);
                    Assert.All(forced!.colIdx, c => Assert.Equal(-1, c));
                }
                seen.Add(g.activePlayer);
                g.Play(g.GetGreedyMove());
            }
            Assert.Equal(Enumerable.Range(0, numPlayers).ToHashSet(), seen.ToHashSet());
            if (!g.IsGameOver())
            {
                Assert.True(g.isRegularPhase);
                Assert.Equal(round + 1, g.roundIdx);
                Assert.Equal(holder, g.activePlayer);
            }
        }
        Assert.True(g.IsFinished);
        Assert.True(sawEmptyWallTurn, "seed never produced a wall turn with nothing to place; pick another seed");
    }
}
