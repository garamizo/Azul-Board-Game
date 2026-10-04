namespace Azul
{
    using System;
    using System.Linq;

    public sealed record PlayerSnapshot(int Score, int[][] Grid, int[][] Line, int[] Floor);

    /// Everything a game needs to continue, in a JSON-friendly shape (jagged
    /// arrays; System.Text.Json cannot serialize int[,]). The RNG is not part
    /// of it: a restored game draws from a fresh Random.
    public sealed record GameSnapshot(
        int Format, int NumPlayers, int ActivePlayer, int Step, ulong ChanceHash,
        int RoundIdx, bool IsRegularPhase, int NewRoundPlayer, int CountPlayerClearedRound,
        bool IsFinished, int[][] Factories, int[] Bag, int[] Discarded, PlayerSnapshot[] Players)
    {
        public const int CurrentFormat = 1;
    }

    public sealed class InvalidSnapshotException(string message) : Exception(message);

    public partial class Game
    {
        const int FIRST = 5;

        public GameSnapshot ToSnapshot() => new(
            GameSnapshot.CurrentFormat, numPlayers, activePlayer, step, chanceHash,
            roundIdx, isRegularPhase, newRoundPlayer, countPlayerClearedRound, IsFinished,
            factories.Select(f => (int[])f.Clone()).ToArray(),
            (int[])bag.Clone(), (int[])discarded.Clone(),
            players.Select(p => new PlayerSnapshot(p.score, ToJagged(p.grid), ToJagged(p.line), (int[])p.floor.Clone())).ToArray());

        /// Rebuilds a game without dealing; throws InvalidSnapshotException
        /// for any shape or rule violation.
        public static Game FromSnapshot(GameSnapshot s, Random? rng = null)
        {
            ValidateShape(s);
            int nf = FactoriesVsPlayer(s.NumPlayers);
            var g = new Game(CloneTag.Instance)
            {
                rng = rng ?? new Random(),
                numPlayers = s.NumPlayers,
                activePlayer = s.ActivePlayer,
                step = s.Step,
                chanceHash = s.ChanceHash,
                roundIdx = s.RoundIdx,
                isRegularPhase = s.IsRegularPhase,
                newRoundPlayer = s.NewRoundPlayer,
                countPlayerClearedRound = s.CountPlayerClearedRound,
                IsFinished = s.IsFinished,
                numFactories = nf,
                CENTER = nf,
                factories = s.Factories.Select(f => (int[])f.Clone()).ToArray(),
                bag = (int[])s.Bag.Clone(),
                discarded = (int[])s.Discarded.Clone(),
                factoryIdxArray = Enumerable.Range(0, nf + 1).ToArray(),
                players = s.Players.Select(p => new Player
                {
                    score = p.Score,
                    grid = To2D(p.Grid),
                    line = To2D(p.Line),
                    floor = (int[])p.Floor.Clone(),
                }).ToArray(),
            };
            g.ValidateRules();
            return g;
        }

        static int[][] ToJagged(int[,] a) =>
            Enumerable.Range(0, a.GetLength(0))
                .Select(r => Enumerable.Range(0, a.GetLength(1)).Select(c => a[r, c]).ToArray())
                .ToArray();

        static int[,] To2D(int[][] a)
        {
            var r = new int[a.Length, a[0].Length];
            for (int i = 0; i < a.Length; i++)
                for (int j = 0; j < a[i].Length; j++)
                    r[i, j] = a[i][j];
            return r;
        }

        static void Require(bool ok, string what)
        {
            if (!ok) throw new InvalidSnapshotException(what);
        }

        static bool Shape(int[]? a, int length) => a is not null && a.Length == length;

        static bool Shape(int[][]? a, int rows, int cols) =>
            a is not null && a.Length == rows && a.All(r => Shape(r, cols));

        static void ValidateShape(GameSnapshot? s)
        {
            Require(s is not null, "snapshot is null");
            Require(s!.Format == GameSnapshot.CurrentFormat, $"unknown format {s.Format}");
            Require(s.NumPlayers is >= 2 and <= 4, "numPlayers must be 2..4");
            int nf = FactoriesVsPlayer(s.NumPlayers);
            Require(s.Factories is not null && s.Factories.Length == nf + 1, "factories length");
            for (int i = 0; i < nf; i++)
                Require(Shape(s.Factories![i], Constants.numColors), $"factory {i} shape");
            Require(Shape(s.Factories![nf], Constants.numColors + 1), "centre shape");
            Require(Shape(s.Bag, Constants.numColors), "bag shape");
            Require(Shape(s.Discarded, Constants.numColors), "discard shape");
            Require(s.Players is not null && s.Players.Length == s.NumPlayers, "players length");
            foreach (var p in s.Players!)
            {
                Require(p is not null, "player is null");
                Require(Shape(p!.Grid, 5, 5), "wall shape");
                Require(Shape(p.Line, 5, 5), "pattern lines shape");
                Require(Shape(p.Floor, Constants.numColors + 1), "floor shape");
            }
            Require(s.ActivePlayer >= 0 && s.ActivePlayer < s.NumPlayers, "activePlayer out of range");
            Require(s.NewRoundPlayer >= 0 && s.NewRoundPlayer < s.NumPlayers, "newRoundPlayer out of range");
            // The engine resets this only when the wall phase starts, so it
            // stays at numPlayers through the following take phase.
            Require(s.CountPlayerClearedRound >= 0 && s.CountPlayerClearedRound <= s.NumPlayers,
                "countPlayerClearedRound out of range");
            Require(s.Step >= 0 && s.RoundIdx >= 0, "negative counters");
        }

        void ValidateRules()
        {
            foreach (var f in factories)
                Require(f.All(v => v >= 0), "negative factory count");
            Require(bag.All(v => v >= 0) && discarded.All(v => v >= 0), "negative bag or discard");

            var total = new int[Constants.numColors];
            for (int c = 0; c < Constants.numColors; c++)
                total[c] = bag[c] + discarded[c] + factories.Sum(f => f[c]);

            int firstMarkers = factories[CENTER][FIRST];
            Require(factories[CENTER][FIRST] is 0 or 1, "centre FIRST slot");
            foreach (var p in players)
            {
                Require(p.score >= 0, "negative score");
                Require(p.floor.All(v => v >= 0), "negative floor count");
                Require(p.floor[FIRST] is 0 or 1, "floor FIRST slot");
                firstMarkers += p.floor[FIRST];
                for (int c = 0; c < Constants.numColors; c++)
                    total[c] += p.floor[c];

                for (int row = 0; row < 5; row++)
                {
                    int lineColors = 0;
                    for (int c = 0; c < Constants.numColors; c++)
                    {
                        int n = p.line[row, c];
                        Require(n >= 0, "negative line count");
                        if (n == 0) continue;
                        lineColors++;
                        Require(n <= row + 1, $"line {row} overfull");
                        for (int col = 0; col < 5; col++)
                            Require(p.grid[row, col] != c, $"line {row} colour already on that wall row");
                        total[c] += n;
                    }
                    Require(lineColors <= 1, $"line {row} holds two colours");

                    for (int col = 0; col < 5; col++)
                    {
                        int v = p.grid[row, col];
                        Require(v >= -1 && v < Constants.numColors, "wall cell out of range");
                        if (v < 0) continue;
                        total[v]++;
                        for (int other = col + 1; other < 5; other++)
                            Require(p.grid[row, other] != v, "colour twice in a wall row");
                        for (int otherRow = row + 1; otherRow < 5; otherRow++)
                            Require(p.grid[otherRow, col] != v, "colour twice in a wall column");
                    }
                }
            }

            for (int c = 0; c < Constants.numColors; c++)
                Require(total[c] == Constants.totalTilesPerColor, $"colour {c} has {total[c]} tiles, not 20");

            if (isRegularPhase)
                Require(firstMarkers == 1, "take phase needs exactly one FIRST marker");
            else
                Require(firstMarkers <= 1, "more than one FIRST marker");

            if (IsFinished)
                Require(!isRegularPhase && countPlayerClearedRound == numPlayers && IsGameOver(),
                    "finished flag on a game that is not over");
        }
    }
}
