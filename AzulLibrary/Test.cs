// using Azul;
// using TicTacToe;
using Azul;
using Ai;
using System.IO;
using CsvHelper;
using System.Globalization;
using System.Diagnostics;
using GameMath = GameUtils.GameMath;
using DeepCopy;

// // 2600 ms to 650 ms for GetGreedyMove
// // 842-935 to 3855-3699 rolls from 2s growth
// // var watch = Stopwatch.StartNew();
// var numRolls = new List<int>();
// for (int j = 0; j < 10; j++)
// {
//     var root_ = new MCTS_Stochastic<Game, Move>(
//         new Game(4), eGreedy: 0.1f);
//     root_.GrowWhile(2f);
//     numRolls.Add(root_.numRolls);
//     // Console.WriteLine("Old:" + root_.state.GetGreedyMove2());
//     // Console.WriteLine("New:" + root_.state.GetGreedyMove() + "\n\n");
// }
// // watch.Stop();
// // Console.WriteLine($"Time elapsed: {watch.ElapsedMilliseconds} ms");
// Console.WriteLine($"Average numRolls: {numRolls.Average()}");


// // Console.WriteLine(game.GetGreedyMove());
// // Console.WriteLine(game.GetGreedyMove2());

// return;

// // var game = Game.GenerateLastMoveGame();
// var game = new Game(4);
// // // // var game = Game.LoadCustomGame();
// var mcts_ = new MCTS<Game, Move>(game, eGreedy: 0.1f);
// for (int i = 0; i < 50; i++)
// {
//     mcts_.GrowWhile(1f);
//     int actionIdx = mcts_.GetBestActionIdx();
//     var action = mcts_.actions[actionIdx];
//     int numRolls = mcts_.childs[actionIdx].numRolls;
//     var winRatio = mcts_.childs[actionIdx].numWins / numRolls;
//     Console.WriteLine($"{action} => {winRatio} ({numRolls})");
// }

// // var mcts2_ = new MCTS_Stochastic<Game, Move>(game, eGreedy: 0.1f);
// // mcts2_.GrowWhile(1f);

// Console.WriteLine("Stochastic ======================");
// var mcts2_ = new MCTS_Stochastic<Game, Move>(game, eGreedy: 0.1f);
// for (int i = 0; i < 50; i++)
// {
//     mcts2_.GrowWhile(1f);
//     int actionIdx = mcts2_.GetBestActionIdx();
//     var action = mcts2_.actions[actionIdx];
//     int numRolls = 0;
//     float numWins = 0f;
//     foreach (var child in mcts2_.childs[actionIdx])
//     {
//         numRolls += child.numRolls;
//         numWins += child.numWins;
//     }
//     Console.WriteLine($"{action} => {numWins / numRolls} ({numRolls})");
// }

// var timer = new Stopwatch();
// timer.Start();
// for (int i = 0; i < 100_000; i++)
//     game = DeepCopier.Copy(game);
// timer.Stop();
// Console.WriteLine($"Time elapsed: {timer.ElapsedMilliseconds} ms");




// var mcts = new MCTS<Game, Move>(new Game(2), eGreedy: 0.1f, paranoid: false);
// Func<Game, Move> mypolicy = (Game g) =>
// {
//     (mcts, var found) = mcts.SearchNode(g);
//     mcts.GrowWhile(0.1f);
//     return mcts.GetBestAction();
// };

// var game = new Game(3);
// var a = mypolicy(game);
// a = mypolicy(game);
// a = mypolicy(game);
// // a = mypolicy(game);

// return;

// var game = new Game(3);
// game.grid = new int[,] {
//     { 1, z, z, z },
//     { z, z, z, z },
//     { z, z, 0, z },
// };
// game.activePlayer = 1;

// // var game = new Game(3);
// // game.grid = new int[,] {
// //     { 0, z, z },
// //     { z, 1, z },
// //     { 0, z, z },
// // };
// // game.activePlayer = 2;

// Console.WriteLine(game);
// // get time duration
// var watch = Stopwatch.StartNew();
// watch.Start();
// var solver = Maxn<Game, Move>.GetBestMove(game);
// // var solver = Paranoid2<Game, Move>.GetBestMove(game);
// // var solver = Paranoid2<Game, Move>.GetBestMove(game);
// watch.Stop();
// Console.WriteLine($"Time elapsed: {watch.ElapsedMilliseconds} ms");
// Console.WriteLine(solver);

// GameMath.PrintArray(game.GetHeuristics());
// return;


// var game = new Game(3);
// var actions = game.GetPossibleActions();
// List<float[]> rewards = new();
// for (int i = 0; i < actions.Count; i++)
// {
//     var gameNew = DeepCopier.Copy(game);
//     gameNew.Play(actions[i]);
//     rewards.Add(gameNew.GetHeuristics());

//     Console.Write("\n" + actions[i] + " => ");
//     for (int j = 0; j < game.numPlayers; j++)
//         Console.Write($"{rewards[i][j]} ");
// }
// Console.WriteLine(game);
// return;

// play against the machine ==================
// var game = new Game(3);
// var mcts_ = new MCTS_Stochastic<Game, Move>(game, eGreedy: 0.1f);
// Func<Game, Move> mypolicy = (Game g) =>
// {
//     (mcts_, var found) = mcts_.SearchNode(g);
//     mcts_.GrowWhile(2.0f);
//     return mcts_.GetBestAction();
// };
// while (game.IsGameOver() == false)
// {
//     // Move a = game.GetGreedyMove();
//     // var a = (game.activePlayer == 0) ? game.GetUserMove() :
//     //     ParanoidID<Game, Move>.GetBestMove(game, 8, 2.0f);
//     var a = (game.activePlayer == 0) ? game.GetUserMove() :
//         mypolicy(game);
//     Debug.Assert(game.IsValid(a));

//     Console.WriteLine(a);
//     if (game.Play(a))  // if round is over
//         Console.WriteLine(".");
//     // break;
// }
// Console.WriteLine(game);
// return;

// var agent = new AgentGenerator(
//     durationMax: 0.01f,
//     rewardMap: GameUtils.RewardMap.WinLosePlus,
//     eGreedy: 0.1f,  // greedy 90% of the time
//     stochastic: false,
//     paranoid: false);


// var game = Game.GenerateLastMoveGame();
// Console.WriteLine(game);

// var a = Maxn<Game, Move>.GetBestMove(game, depth: 3, timeout: 1.0f);

// return;

// var game = new Game().Reset(4);
// Console.WriteLine(game);

// var agent = new AgentGenerator(
//     durationMax: 0.01f,
//     rewardMap: GameUtils.RewardMap.WinLosePlus,
//     eGreedy: 0.1f,  // greedy 90% of the time
//     stochastic: false,
//     paranoid: false);
// String description = agent.description;

// var policy = (Func<Game, Move>)((Game game) =>
// {
//     var a = Maxn<Game, Move>.GetBestMove(game, depth: 1, timeout: 0.1f);
//     return a;
// });

// var game = Game.GenerateLastMoveGame();
// Console.WriteLine(Maxn<Game, Move>.count);
// var a = policy(game);
// Console.WriteLine(Maxn<Game, Move>.count);
// a = policy(game);
// a = policy(game);
// Console.WriteLine(Maxn<Game, Move>.count);

float timeout = 3f;
var bench = new GameUtils.Benchmark<Game, Move>(
    minNumPlayers: 4,
    maxNumPlayers: 4,
    numCycles: 10,
    filename: @"benchmark3.csv",
    comment: "e10, e10, Paranoid"
);
// bench.policies.Add((Game g) => g.GetEGreedyMove(0.05f));
// bench.policies.Add((Game g) => g.GetEGreedyMove(0.05f));
// bench.policies.Add((Game g) => g.GetGreedyMove2());
// bench.policies.Add((Game g) => g.GetGreedyMove2());
// bench.policies.Add((Game g) => g.GetRandomMove());
// bench.policies.Add((Game g) => MaxnID<Game, Move>.GetBestMove(g, timeout: timeout));
// bench.policies.Add((Game g) => ParanoidID<Game, Move>.GetBestMove(g, timeout: timeout));

var mcts_stochastic = new MCTS_Stochastic<Game, Move>(new Game(4), eGreedy: 0.01f);
bench.policies.Add((Game g) =>
{
    (mcts_stochastic, var found) = mcts_stochastic.SearchNode(g);
    mcts_stochastic.GrowWhile(timeout);
    int idx = mcts_stochastic.GetBestActionIdx();
    if (mcts_stochastic.WinRatio(idx) == 0.0)
        return g.GetGreedyMove();
    return mcts_stochastic.actions[idx];
}); // 3.3     40.0    53.3         for 30 games
// 0.0     40.0    56.4     for 55 games

var mcts = new MCTS_Stochastic<Game, Move>(new Game(4), eGreedy: 0.02f);
bench.policies.Add((Game g) =>
{
    (mcts, var found) = mcts.SearchNode(g);
    mcts.GrowWhile(timeout);
    return mcts.GetBestAction();
});

var mcts3 = new MCTS_Stochastic<Game, Move>(new Game(4), eGreedy: 0.05f);
bench.policies.Add((Game g) =>
{
    (mcts3, var found) = mcts3.SearchNode(g);
    mcts3.GrowWhile(timeout);
    return mcts3.GetBestAction();
});

var mcts4 = new MCTS_Stochastic<Game, Move>(new Game(4), eGreedy: 0.1f);
bench.policies.Add((Game g) =>
{
    (mcts4, var found) = mcts4.SearchNode(g);
    mcts4.GrowWhile(timeout);
    return mcts4.GetBestAction();
});

// 2p: 31.0    31.0    88.0,        3p: 24.3    22.3    50.7        2.43 min runtime
// 30.5    35.0    84.5         27.3    26.7    44.3
// 4x4 grid,    2p: 32.0    33.0    85.0,   3p: 14.3    15.3    70.3

// bench.policies.Add((Game g) => Paranoid<Game, Move>.GetBestMove(g, depth: 16, timeout: 0.1f));
// 2p: 38.0    40.0    72.0,        3p: 23.7    29.7    43.0        0.89 min runtime
// 4x4 grid,    2p:38.5    35.5    76.0         3p: 30.3    24.7    45.0

// bench.policies.Add((Game g) => MaxnID<Game, Move>.GetBestMove(g, timeout: 0.1f));
// 2p: 31.5    29.0    89.5,        3p: 24.0    34.0    36.3,       2.39 min runtime

bench.Run();

return;



public class AgentGenerator
{
    MCTS<Game, Move>? root;
    MCTS_Stochastic<Game, Move>? root2;
    Func<float[], float[]>? rewardMap;
    float durationMax;
    float eGreedy;
    bool stochastic;
    bool paranoid;
    public String description;

    public AgentGenerator(float durationMax, Func<float[], float[]>? rewardMap,
        float eGreedy, bool stochastic, bool paranoid)
    {
        this.rewardMap = rewardMap;
        this.durationMax = durationMax;
        this.eGreedy = eGreedy;
        this.stochastic = stochastic;
        this.paranoid = paranoid;
        description = $"durationMax={durationMax:F2}, eGreedy={eGreedy:F2}, stochastic={stochastic}, paranoid={paranoid}";
    }

    public Move Policy(Game game)
    {
        // var rootActive = stochastic ? root2 : root;
        if (stochastic) PolicyStochastic(game);

        if (root == null)
            root = new(game, eGreedy: eGreedy);

        float duration = (game.step < 3 * game.numPlayers) ? durationMax : (durationMax / 2);
        (root, var found) = root.SearchNode(game);
        // root = rootNew;
        root.GrowWhile(duration);

        return root.GetBestAction();
    }

    public Move PolicyStochastic(Game game)
    {
        if (root2 == null)
            root2 = new(game, eGreedy: eGreedy);

        float duration = (game.step < 3 * game.numPlayers) ? durationMax : (durationMax / 2);
        var (rootNew, found) = root2.SearchNode(game);
        root2 = rootNew;

        root2.GrowWhile(duration);
        return root2.GetBestAction();
    }
}


class Tests
{
    public static void TestGamePlay()
    {
        for (int numPlayers = 2; numPlayers <= 2; numPlayers++)
            for (int reps = 0; reps < 5; reps++)
            {
                Game g = new(numPlayers);
                // Game g = Game.LoadCustomGame();
                // Console.WriteLine(g);
                Game gOld = new(numPlayers);

                // g.players[0].score = 50;

                // g.bag = new int[] { 9 * 4 - 2, 3, 0, 0, 0 };
                // g.discarded = new int[] { 18, 20, 20, 20, 20 };

                // Console.WriteLine(g);

                while (g.IsGameOver() == false)
                {
                    // Move a = g.GetGreedyMove();
                    Move a = g.GetRandomMove();
                    Debug.Assert(g.IsValid(a));
                    // Console.WriteLine(a);
                    gOld = DeepCopier.Copy(g);
                    if (g.Play(a))  // if round is over
                        Console.WriteLine(".");
                    // break;
                }

                Console.WriteLine(g);
            }
    }
}