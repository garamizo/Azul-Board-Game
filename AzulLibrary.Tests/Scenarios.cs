using Azul;

namespace AzulLibrary.Tests;

internal static class Scenarios
{
    /// Rows of the active player's pattern lines that are complete.
    public static List<int> CompletedRows(Game g)
    {
        var rows = new List<int>();
        var p = g.players[g.activePlayer];
        for (int row = 0; row < 5; row++)
            for (int color = 0; color < 5; color++)
                if (p.line[row, color] > row) rows.Add(row);
        return rows;
    }

    /// Plays seeded greedy games until the active player is in the wall
    /// phase with at least one completed line.
    public static Game WallPhaseWithCompletedLine(int numPlayers)
    {
        for (int seed = 1; seed < 200; seed++)
        {
            var g = new Game(numPlayers, new Random(seed));
            while (!g.IsGameOver())
            {
                if (!g.isRegularPhase && CompletedRows(g).Count > 0)
                    return g;
                g.Play(g.GetGreedyMove());
            }
        }
        throw new InvalidOperationException("no wall-phase state with a completed line found");
    }

    public static Game FinishedGame(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        while (!g.IsGameOver())
            g.Play(g.GetGreedyMove());
        return g;
    }
}
