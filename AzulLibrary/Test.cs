using System.Diagnostics;
using System;
using System.Collections;
using DeepCopy;  // dotnet add package DeepCopy

using Utils;
using Azul;  // Game, GameAction
using Ai;  // MCTS


// // Game g = new(4);
// Game g = Game.LoadCustomGame();
// Console.WriteLine(g);
// Game gOld = new(3);

// g.players[0].score = 50;

// g.bag = new int[] { 9 * 4 - 2, 3, 0, 0, 0 };
// g.discarded = new int[] { 18, 20, 20, 20, 20 };

// // g.Print();
// while (g.IsGameOver() == false)
// {
//     GameAction a = Game.DefaultPolicy(g);
//     Debug.Assert(g.IsValid(a));
//     gOld = DeepCopier.Copy(g);
//     if (g.Play(ref a))  // if round is over
//         break;
// }

// // gOld.Print();
// MCTS_Stochastic r = new(gOld);
// r.GrowWhile(10.0f);


// GameAction a2 = r.GetBestAction();
// // var r2 = r.SearchNode(g, 3);

// return;

// // Game g = Game.GenerateLastMoveGame();
// Game g = new(3);
// MCTS_Stochastic r = new(g);

// List<GameAction> actions = g.GetPossibleActions();
// foreach (GameAction a in actions)
//     a.Print();

// g.Print();
// r.GrowWhile(5.0f);

// Console.WriteLine($"numRolls: {r.numRolls}");
// Console.WriteLine($"numChild: {r.childs.Count}");

// return;

// // moves per sec) 10: 78k 73k 77k, 15: 54k 56k 56k, 20: 49k 47k 46k
// Func<Game, GameAction>[] policies_ = new Func<Game, GameAction>[] {
//     // RandomPolicy,  // 486_598 moves/sec
//     // RandomPolicy,  // with shuffle method, 615_415 moves/sec
//     // RandomPolicy,
//     // RandomPolicy,
//     Game.DefaultPolicy2,  // 20_278 moves/sec
//     Game.DefaultPolicy2,
//     Game.DefaultPolicy2,
//     Game.DefaultPolicy2,
// };

// // calculate time
// int numMoves = 0;
// Stopwatch stopWatch = new();
// stopWatch.Start();
// for (int i = 0; i < 5000; i++)
// {
//     Game g = new(4);
//     while (g.IsGameOver() == false)
//     {
//         GameAction a = policies_[g.activePlayer](g);
//         Debug.Assert(g.IsValid(a));
//         g.Play(ref a);
//         numMoves++;
//     }

//     // Console.Write($"Final score: [");
//     // for (int j = 0; j < g.numPlayers; j++)
//     //     Console.Write($"{g.players[j].score},");
//     // Console.WriteLine($"] on round {g.roundIdx} with {g.step} steps.");
// }
// stopWatch.Stop();

// TimeSpan ts = stopWatch.Elapsed;
// Console.WriteLine($"RunTime {ts.TotalSeconds} sec, {numMoves} moves, {numMoves / ts.TotalSeconds} moves/sec");

// return;

// Defined end of game =================
// // Game game = Game.GenerateLastMoveGame();
// Game game = new(3);
// // var action = new GameAction(0, 0, 0, game);
// game.Print();
// // game.Play(ref action);
// MCTS? root = new(game);
// root.Grow(1000);
// root.Print();
// // root.Rollout();

// // root.state.Print();
// for (int i = 0; i < root.childs.Count; i++)
//     GameUtils.PrintArray<float>(root.childs[i].state.GetReward());

// Console.WriteLine($"gameOver: {game.IsGameOver()} roundOver: {game.IsRoundOver()}");
// float[] rewards = game.GetReward();
// for (int i = 0; i < rewards.Length; i++)
//     Console.WriteLine($"Player {i} score={game.players[i].score}, reward={rewards[i]}");

Console.WriteLine("Benchmark or agents =====================");
Func<float[], float[]>[] rewardMaps = new Func<float[], float[]>[] {
    RewardMap.Passthrough,
    RewardMap.MinMax, // lost [0, 6, 0, 14] to LinearMap
    RewardMap.Linear,  // [pass:17, linear:15, sigmoid:40, winlose:28] for duration=1.0
    RewardMap.Sigmoid,  // [pass:54, linear:4, sigmoid:14, winlose:27] for 24 reps, duration=5
    RewardMap.WinLose,  // [minmax:25, pass:19, wl:25, wl+:31]
    RewardMap.WinLosePlus
};
float[] durations = new float[] { 1.0f, 1.0f, 10.0f };
float[] eGreedies = new float[] { -1.0f, 0.0f, 0.05f, 0.1f, 0.2f, 1.0f };

GameAction RandomPolicy(Game game)
{
    return game.GetRandomAction();
}

// Console.WriteLine("det 10greedy, det 20greedy, stoc 10greedy, stoc 20greedy | winlose"); // [37.5, 16.666666, 35.416664, 10.416666]
Console.WriteLine("det 10greedy wl, stoc 10greedy pass, stoc 10greedy minmax, stoc 10greedy lin"); // [37.5, 16.666666, 35.416664, 10.416666]
AgentGenerator agent0 = new(durations[1], RewardMap.WinLose, 0.10f, false, true); // deterministic, 10greedy
AgentGenerator agent1 = new(durations[1], RewardMap.WinLose, 0.10f, false, false);  // deterministic, 20greedy
AgentGenerator agent2 = new(durations[1], RewardMap.WinLose, 0.10f, true, true);  // stochastic, 10reedy
AgentGenerator agent3 = new(durations[1], RewardMap.WinLose, 0.10f, true, false);  // stochastic, 20greedy

float[] CalculateWinStats(int[,] scores, int reps, int numPlayers)
{
    float[] wins = new float[numPlayers];
    for (int r = 0; r < reps; r++)
    {
        int maxScore = 0;
        int ties = 0;
        for (int p = 0; p < numPlayers; p++)
        {
            if (scores[r, p] == maxScore)
                ties++;
            else if (scores[r, p] > maxScore)
            {
                maxScore = scores[r, p];
                ties = 1;
            }
        }

        for (int p = 0; p < numPlayers; p++)
            if (scores[r, p] == maxScore)
                wins[p] += 100.0f / ties / reps;
    }
    return wins;
}

// moves per sec) 10: 78k 73k 77k, 15: 54k 56k 56k, 20: 49k 47k 46k
// win percent: [x, control, x, control]
//  10: [ 6, 44,  5, 45]
//  15: [12, 37, 10, 40]
//  20: [14, 35, 16, 34]
var policies = new Func<Game, GameAction>[] {
    // RandomPolicy,
    // Game.DefaultPolicy2,
    // Game.DefaultPolicy,
    agent0.Policy,
    agent1.Policy,
    agent2.Policy,
    // (Game g) => Game.EGreedyPolicy(g, 0.0f),
    // (Game g) => Game.EGreedyPolicy(g, 0.1f),
    // (Game g) => Game.EGreedyPolicy(g, 0.2f),
    // (Game g) => Game.EGreedyPolicy(g, 0.4f),
    // RandomPolicy,
    // RandomPolicy,
    // RandomPolicy,
    // Game.DefaultPolicy,
    // Game.DefaultPolicy,
    // Game.DefaultPolicy,
    agent3.Policy,
};

// Simulate game =====================
int numPlayers = 4;

int[] pindex = Enumerable.Range(0, numPlayers).ToArray();
List<int[]> pindexList = GameUtils.GetPermutations<int>(pindex);
GameUtils.Shuffle<int[]>(Game.rng, pindexList);

int cycles = 3;
int[,] results = new int[cycles * pindexList.Count, numPlayers];
int count = 0;

Stopwatch stopWatch = new();
stopWatch.Start();
for (int reps = 0; reps < cycles; reps++)
    for (int iters = 0; iters < pindexList.Count; iters++)
    {
        // System.Threading.Thread.Sleep(200);

        pindex = pindexList[iters];
        Game game = new(numPlayers);

        while (game.IsGameOver() == false)
        {
            GameAction action = policies[pindex[game.activePlayer]](game);
            Debug.Assert(game.IsValid(action));

            if ((game.Play(ref action) || game.IsGameOver()) && cycles * pindexList.Count < 30)
            {
                int[] tmp = new int[numPlayers];
                for (int p = 0; p < numPlayers; p++)
                    tmp[pindex[p]] = game.players[p].score;

                Console.Write($"\tRound {game.roundIdx}) [");
                for (int i = 0; i < numPlayers; i++)
                    Console.Write($" {tmp[i]}" + (i == pindex[0] ? "*," : ","));
                Console.WriteLine("]");
            }
        }

        for (int p = 0; p < numPlayers; p++)
            results[count, pindex[p]] = game.players[p].score;

        Console.Write($"Score of rep {count}: [");
        for (int p = 0; p < numPlayers; p++)
            Console.Write($" {results[count, p]}" + (p == pindex[0] ? "*," : ","));
        Console.Write($"],\tWins: [");
        foreach (var s in CalculateWinStats(results, count + 1, numPlayers))
            Console.Write($" {s.ToString("F0")},");
        Console.WriteLine("] %");
        // GameUtils.PrintArray<float>(CalculateWinStats(results, count + 1, numPlayers));

        TimeSpan ts = stopWatch.Elapsed;
        Console.WriteLine($"\tRunTime {(ts.TotalMinutes).ToString("F1")} min, \t{count}/{cycles * pindexList.Count} reps, " +
            $"\t{(ts.TotalMinutes / (count + 1)).ToString("F1")} min/game, " +
            $"\t{(ts.TotalMinutes / (count + 1) * (cycles * pindexList.Count - count)).ToString("F1")} min remaining");

        count++;
    }

// // Measure time =====================
// Stopwatch stopWatch = new();
// stopWatch.Start();
// for (int j = 0; j < 10; j++)
// {
//     Game game = new(4);
//     MCTS root = new(game);
//     for (int i = 0; i < 1_000; i++)
//         root.Rollout();
// }
// stopWatch.Stop();
// // 51 secs for 1_000_000 games
// TimeSpan ts = stopWatch.Elapsed;
// Console.WriteLine($"RunTime {ts.TotalSeconds} sec");

// Test MCTS =====================
// Game game = new(2);
// MCTS root = new(game);

// root.GrowWhile(10.0f, 500);
// int actionIdx = root.GetBestAction();
// GameAction action = root.actions[actionIdx];
// float wRatio = root.childs[actionIdx].numWins / (float)root.childs[actionIdx].numRolls;
// int numRolls = root.childs[actionIdx].numRolls;

// // root.Print();
// root.state.Print();
// Console.WriteLine($"Selected action ({action.factoryIdx}, {action.color}, {action.row}) " +
//     $"with win rate of {wRatio}, {numRolls} simulations, out of " +
//     $"{root.actions.Count} possible actions.");

// solution had 330 simulations, in 10 secs
// custom copy, 100 sims

// Selected action (5, 2, 3) with win rate of 0.59322035, 59 simulations, out of 97 possible actions

// // Test operators =====================
// Game game = new(2);
// Game game2 = DeepCopier.Copy(game);
// Game game3 = new(3);

// Console.WriteLine(game != game2);
// Console.WriteLine(game != game);
// Console.WriteLine(game2 != game);

// Console.WriteLine(game != game3);
// Console.WriteLine(game3 != game3);
// Console.WriteLine(game3 != game);

public class AgentGenerator
{
    MCTS? root;
    MCTS_Stochastic? root2;
    Func<float[], float[]>? rewardMap;
    float durationMax = 5.0f;
    float eGreedy = 0.0f;
    bool stochastic = false;
    bool paranoid = false;

    public AgentGenerator(float durationMax, Func<float[], float[]>? rewardMap,
        float eGreedy, bool stochastic, bool paranoid)
    {
        this.rewardMap = rewardMap;
        this.durationMax = durationMax;
        this.eGreedy = eGreedy;
        this.stochastic = stochastic;
        this.paranoid = paranoid;
    }

    public GameAction Policy(Game game)
    {
        if (stochastic)
            return PolicyStochastic(game);

        if (root == null)
            root = new(game, rewardMap: rewardMap, eGreedy: eGreedy, paranoid: paranoid);

        float duration = (game.step < 3 * game.numPlayers) ? durationMax : (durationMax / 2);
        root = root.SearchNode(game);
        if (root == null)
            root = new(game, rewardMap: rewardMap, eGreedy: eGreedy);

        root.GrowWhile(duration);
        return root.GetBestAction();
    }

    public GameAction PolicyStochastic(Game game)
    {
        if (root2 == null)
            root2 = new(game, rewardMap: rewardMap, eGreedy: eGreedy, paranoid: paranoid);

        float duration = (game.step < 3 * game.numPlayers) ? durationMax : (durationMax / 2);
        root2 = root2.SearchNode(game);
        if (root2 == null)
            root2 = new(game, rewardMap: rewardMap, eGreedy: eGreedy);

        root2.GrowWhile(duration);
        return root2.GetBestAction();
    }

}