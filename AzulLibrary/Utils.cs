using System.Diagnostics;

namespace Utils;


class GameUtils
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


    public static List<T[]> GetPermutations<T>(T[] input)
    {
        var result = new List<T[]>();

        void RecursiveAlgorithm(List<T> element, List<T> bag)
        {
            if (bag.Count == 0)
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
}
class RewardMap
{
    public static float[] Passthrough(float[] scores)
    {
        float scoreMax = scores.Max() + 0.01f;
        float[] reward = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            reward[i] = scores[i] / scoreMax;
        return reward;
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
            reward[i] = 2 * GameUtils.Sigmoid(5 * scores[i] / scoreSum) - 1.0f;
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
