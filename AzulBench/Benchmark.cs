namespace GameUtils;
using System.Diagnostics;
using System.IO;
using CsvHelper;
using System.Globalization;

// Benchmark<G, M> moved here from AzulLibrary/Utils.cs (it needs CsvHelper).

/*
    Evaluate one agent against standard policies
    Players are [agent, 0greedy, 10greedy, 100greedy]
    All combinations are tested (ie for v2, v3 and v4), and all orders are tested 
    Final scores and orders are recorded into file.
    Columns are [IndexP0, IndexP1, IndexP2, IndexP3, ScoreP0, ScoreP1, ScoreP2, ScoreP3]
    where IndexPN is the index order in the game player N started 
*/
class Benchmark<G, M>
    where G : Game<M>, new()
{
    int numCycles;
    String filename;
    String comment;
    public List<Func<G, M>> policies = new();
    // Func<G, M> policy;
    Random RandomSeed = new();
    public int maxNumPlayers; // => policies.Length;
    int minNumPlayers = 4;
    public int numAgents => policies.Count;

    public Benchmark(int minNumPlayers, int maxNumPlayers, int numCycles = 3, String filename = @"benchmark.csv", String comment = "")
    {
        // default agents
        // policies.Add((G g) => g.GetEGreedyMove(0.0f));  // random
        // policies.Add((G g) => g.GetEGreedyMove(0.0f));  // random
        // policies.Add((G g) => g.GetEGreedyMove(0.1f));  // 10% random, 90% greedy
        // policies.Add((G g) => g.GetEGreedyMove(0.0f));  // 100% greedy

        // this.policy = policy;
        this.minNumPlayers = minNumPlayers;
        this.maxNumPlayers = maxNumPlayers;
        this.numCycles = numCycles;
        this.filename = filename;
        this.comment = comment;
    }

    public void WriteHeader(CsvWriter csv)
    {
        for (int p = 0; p < numAgents; p++)
            csv.WriteField($"IndexP{p}");
        for (int p = 0; p < numAgents; p++)
            csv.WriteField($"ScoreP{p}");
        csv.NextRecord();
    }

    public void WriteRecord(CsvWriter csv, int?[] pindex, float?[] scores)
    {
        for (int p = 0; p < numAgents; p++)
            csv.WriteField(pindex[p]);
        for (int p = 0; p < numAgents; p++)
            csv.WriteField(scores[p]);
        csv.NextRecord();
    }

    (int?[], float?[]) ReverseIndex(int[] pindex, float[] scores)
    {
        var scores_ = new float?[numAgents];
        var pindexInv = new int?[numAgents];
        for (int p = 0; p < numAgents; p++)
        {
            scores_[p] = null;
            pindexInv[p] = null;
        }
        for (int p = 0; p < scores.Length; p++)
        {
            scores_[pindex[p]] = scores[p];
            pindexInv[pindex[p]] = p;
        }
        return (pindexInv, scores_);
    }

    public void Run()
    {
        Debug.Assert(maxNumPlayers <= numAgents, "Not enough policies");

        int count = 0;
        int totalIters = 0;
        for (int nPlayers = minNumPlayers; nPlayers <= maxNumPlayers; nPlayers++)
            totalIters += numCycles * GameMath.Factorial(numAgents) /
                GameMath.Factorial(numAgents - nPlayers);


        Stopwatch stopWatch = new();
        stopWatch.Start();
        using (var writer = new StreamWriter(filename))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteComment(comment);
            csv.NextRecord();
            WriteHeader(csv);
            for (int numPlayers = minNumPlayers; numPlayers <= maxNumPlayers; numPlayers++)
            {
                Console.WriteLine($"Benchmarking {numPlayers} players -----------------");
                int[] playCount = new int[numAgents];
                int[] winCount = new int[numAgents];
                for (int reps = 0; reps < numCycles; reps++)
                {
                    var pindexList = GameMath.GetPermutations(numAgents, numPlayers);
                    GameMath.Shuffle<int[]>(RandomSeed, pindexList);
                    for (int iters = 0; iters < pindexList.Count; iters++)
                    {
                        var pindex = pindexList[iters];
                        var game = (G)new G().Reset(numPlayers);

                        while (game.IsGameOver() == false)
                        {
                            var action = policies[pindex[game.activePlayer]](game);
                            Debug.Assert(game.IsValid(action));
                            if (game.IsValid(action) == false)
                                throw new Exception("Invalid move");
                            game.Play(action);
                        }

                        var (pindexOrd, scoresOrd) = ReverseIndex(pindex, game.GetRewards());
                        WriteRecord(csv, pindexOrd, scoresOrd);
                        writer.Flush();

                        TimeSpan ts = stopWatch.Elapsed;
                        Console.Write($"\tRunTime {(ts.TotalMinutes).ToString("F2")} min, " +
                            $"\t{count}/{totalIters} reps, " +
                            $"\t{(ts.TotalMinutes / (count + 1)).ToString("F1")} min/game, " +
                            $"\t{(ts.TotalMinutes / (count + 1) * (totalIters - count)).ToString("F1")} min remaining");

                        // update play and win count per policy
                        for (int p = 0; p < numAgents; p++)
                        {
                            if (scoresOrd[p] != null)
                            {
                                playCount[p]++;
                                if (scoresOrd[p] == 1.0f)
                                    winCount[p]++;
                            }
                            Console.Write($"\t{100 * winCount[p] / (1e-5f + playCount[p]):F1}");
                        }
                        Console.WriteLine();
                        count++;
                    }
                }
            }
        }
    }
}