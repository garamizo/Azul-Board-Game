using System.Diagnostics;

namespace Utils
{
    class GameUtils
    {
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
    }
}