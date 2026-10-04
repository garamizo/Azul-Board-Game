namespace GameUtils;
using System.Diagnostics;
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
    // Per game, never shared: games and MCTS clones run on different threads.
    public Random rng = new();
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
