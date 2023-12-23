namespace GameUtils;
using System.Diagnostics;
using System.IO;
using CsvHelper;
using System.Globalization;
// using Azul;

class GameMath
{

    public static float Sigmoid(float value)
    {
        return 1.0f / (1.0f + (float)MathF.Exp(-value));
    }
    public static int SampleWeightedDiscrete(Random rng, int[] weights)
    {
        int x = rng.Next(0, weights.Sum());

        int index = 0; // so you know what to do next
        foreach (int w in weights)
        {
            index++;
            if ((x -= w) < 0)
                break;
        }
        return index - 1;
    }

    public static void Shuffle<T>(Random rng, T[] array)
    {
        int n = array.Length;
        while (n > 1)
        {
            int k = rng.Next(n--);
            (array[k], array[n]) = (array[n], array[k]);
        }
    }

    public static void Shuffle<T>(Random rng, List<T> array)
    {
        int n = array.Count;
        while (n > 1)
        {
            int k = rng.Next(n--);
            (array[k], array[n]) = (array[n], array[k]);
        }
    }

    public static List<T> Shuffled<T>(Random rng, List<T> array)
    {
        int n = array.Count;
        while (n > 1)
        {
            int k = rng.Next(n--);
            (array[k], array[n]) = (array[n], array[k]);
        }
        return array;
    }

    public static void PrintArray<T>(T[] array)
    {
        Console.Write("[");
        for (int i = 0; i < array.Length; i++)
        {
            Console.Write($"{array[i]}");
            if (i < array.Length - 1)
                Console.Write(", ");
        }
        Console.WriteLine("]");
    }

    public static void PrintList<T>(List<T> array)
    {
        Console.Write("[");
        for (int i = 0; i < array.Count; i++)
        {
            Console.Write($"{array[i]}");
            if (i < array.Count - 1)
                Console.Write(", ");
        }
        Console.WriteLine("]");
    }


    // public static List<T[]> GetPermutations<T>(T[] input)
    // {
    //     var result = new List<T[]>();

    //     void RecursiveAlgorithm(List<T> element, List<T> bag)
    //     {
    //         if (bag.Count == 0)
    //             result.Add(element.ToArray());
    //         else
    //             for (int i = 0; i < bag.Count; i++)
    //             {
    //                 List<T> bagNew = new(bag.Where((e, idx) => idx != i));
    //                 element.Add(bag[i]);
    //                 RecursiveAlgorithm(element, bagNew);
    //                 element.RemoveAt(element.Count - 1);
    //             }
    //     }
    //     RecursiveAlgorithm(new List<T>(), input.ToList());
    //     return result;
    // }

    public static List<T[]> GetPermutations<T>(T[] input, int len)
    {
        var result = new List<T[]>();

        void RecursiveAlgorithm(List<T> element, List<T> bag)
        {
            if (bag.Count <= input.Length - len)
                result.Add(element.ToArray());
            else
                for (int i = 0; i < bag.Count; i++)
                {
                    List<T> bagNew = new(bag.Where((e, idx) => idx != i));
                    element.Add(bag[i]);
                    RecursiveAlgorithm(element, bagNew);
                    element.RemoveAt(element.Count - 1);
                }
        }
        RecursiveAlgorithm(new List<T>(), input.ToList());
        return result;
    }
    public static List<T[]> GetPermutations<T>(T[] input) => GetPermutations(input, input.Length);
    public static List<int[]> GetPermutations(int numItems, int len) =>
        GetPermutations(Enumerable.Range(0, numItems).ToArray(), len);

    public static int NChooseK(int n, int k)
    {
        if (k == 0)
            return 1;
        return (n * NChooseK(n - 1, k - 1)) / k;
    }

    public static int Factorial(int n)
    {
        if (n == 0)
            return 1;
        return n * Factorial(n - 1);
    }
}
class RewardMap
{
    public static float[] Passthrough(float[] scores)
    {
        // float scoreMax = scores.Max() + 0.01f;
        // float[] reward = new float[scores.Length];
        // for (int i = 0; i < scores.Length; i++)
        //     reward[i] = scores[i] / scoreMax;
        // return reward;
        return scores;
    }

    public static float[] MinMax(float[] scores)
    {
        float scoreMax = scores.Max();
        float scoreMin = scores.Min();
        float[] reward = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            reward[i] = (scores[i] - scoreMin) / (scoreMax - scoreMin + 0.01f);
        return reward;
    }

    public static float[] Linear(float[] scores)
    {
        float scoreSum = scores.Sum() + 0.01f;
        float[] reward = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            reward[i] = scores[i] / scoreSum;
        return reward;
    }

    public static float[] Sigmoid(float[] scores)
    {
        float scoreSum = scores.Sum() + 0.01f;
        float[] reward = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            reward[i] = 2 * GameMath.Sigmoid(5 * scores[i] / scoreSum) - 1.0f;
        return reward;
    }

    public static float[] WinLose(float[] scores)
    {
        float scoreMax = scores.Max();
        int ties = 0;
        for (int i = 0; i < scores.Length; i++)
            if (scores[i] == scoreMax)
                ties++;

        float[] reward = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            reward[i] = scores[i] == scoreMax ? 1.0f / ties : 0.0f;
        return reward;
    }

    public static float[] WinLosePlus(float[] scores)
    {
        float scoreMax = scores.Max();
        int ties = 0;
        for (int i = 0; i < scores.Length; i++)
            if (scores[i] == scoreMax)
                ties++;

        float[] reward = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            reward[i] = (scores[i] == scoreMax ? 1.0f / ties : 0.0f) + scores[i] / 100_000.0f;
        return reward;
    }
}

interface IGame<TMove>
{
    public int ActivePlayer { get; }
    public int NumPlayers { get; }
    public static Random rng = new();
    public Random RandomSeed { get => rng; }
    // public bool paranoid;
    public bool IsGameOver();
    // public abstract bool IsTerminal();
    public bool IsValid(TMove action);
    public bool Play(TMove action);
    public bool IsEqual(IGame<TMove> game);
    public List<TMove> GetPossibleActions();

    // [0f, 1f], 0f: sure loss, 1f: sure win, use probability to be in between
    public float[] GetHeuristics();  // assumes game is NOT over

    // 0f: loss, 1f: win, 1/numPlayers: tie
    public float[] GetRewards();  // assumes game is over
    // public abstract int[] GetScores();

    public TMove GetEGreedyMove(float epsilon);
}

public abstract class Game<M>
{
    public static Random rng = new();
    public int activePlayer;
    public int numPlayers;
    public int step;
    public UInt64 chanceHash;

    public Random RandomSeed { get => rng; }
    public abstract Game<M> Reset(int numPlayers);
    // public bool paranoid;
    public abstract bool IsGameOver();
    // public abstract bool IsTerminal();
    public abstract bool IsValid(M action);
    public abstract bool Play(M action);
    public abstract bool Equals(Game<M> game);
    public abstract List<M> GetPossibleActions(bool sort = false);
    public abstract M GetRandomMove();
    public abstract M GetGreedyMove();

    // [0f, 1f], 0f: sure loss, 1f: sure win, use probability to be in between
    public abstract float[] GetHeuristics();  // assumes game is NOT over

    // 0f: loss, 1f: win, 1/numPlayers: tie
    public abstract float[] GetRewards();  // assumes game is over
    public abstract float[] GetScores();  // scores, not win/lose

    public M GetEGreedyMove(float epsilon)
    {
        /* 
            epsilon-greedy policy
            epsilon = 0.0 -> greedy
            epsilon = 1.0 -> random
        */
        if (RandomSeed.NextDouble() < epsilon)
            return GetRandomMove();
        else
            return GetGreedyMove();
    }
}

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