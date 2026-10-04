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




// // // g.Play(g.GetGreedyMove());
// Console.WriteLine(g);
// // // var m = new Move(new int[] { 3, 2, 2, -1, -1 }, g);

// var ai = new MCTS<Game, Move>(g, eGreedy: 1.0f);

// ai.GrowWhile(6f);
// // Child 3) Player=2, Factory=7, Color=WHITE:4, Row=FLOOR:5, NumTiles=2, IsFirst=False, Rolls: 38295, winRatio: 0.51298255
// // g.Play(g.GetGreedyMove());

// // // Console.WriteLine(ai.ToString(1));
// // Console.WriteLine(ai);

// // (ai, var found) = ai.SearchNode(g);


// // Console.WriteLine($"{g}\nFound: {found}\n{ai}\n{ai.state}");

// // // Active player 2, numRolls 12488, numChilds 216
// // // Child 136) Player 2) ColIdx=[0, 2, 2, -1, -1], Rolls: 132, winRatio: 0.51893938

// for (int i = 0; i < 1; i++)
// {
//     var g = Game.LoadCustomGame();
//     var ai = new MCTS_Stochastic<Game, Move>(g, eGreedy: 0.1f);
//     ai.GrowWhile(10f);
//     Console.WriteLine($"Iter {i}: {ai.ToString(5)}");
// }
// var g = Game.LoadCustomGame();
// Console.WriteLine($"Old method {g.GetPossibleActions().Count}");
// foreach (var a in g.GetPossibleActions())
//     Console.WriteLine(a);

// Console.WriteLine($"New method {g.GetPossibleActions2().Count}");
// foreach (var a in g.GetPossibleActions2())
//     Console.WriteLine(a);

// return;


// var g = Game.LoadCustomGame();

// foreach (var m in g.GetPossibleActions())
//     Console.WriteLine(m);

// // var ms = g.GetPossibleActions();
// Console.WriteLine(g);
// // g.GetGreedyMove();
// // Console.WriteLine(g.GetGreedyMove());

// return;


// // Compare two implementations ====================
// var g = Game.LoadCustomGame();
// Stopwatch watch = new();

// watch.Restart();
// var ms = g.GetPossibleActions();
// watch.Stop();
// foreach (var m in ms)
//     Console.WriteLine(m);
// Console.WriteLine($"Time elapsed: {watch.Elapsed.TotalMilliseconds * 1000} ms");

// watch.Restart();
// ms = g.GetColIdxMoves2();
// watch.Stop();
// foreach (var m in ms)
//     Console.WriteLine(m);
// Console.WriteLine($"Time elapsed: {watch.Elapsed.TotalMilliseconds * 1e3} ms");

// return;

// Calc num rolls per second -------------------

int NumRolls = 0;
const int NUM_ITERS = 10;

for (int i = 0; i < NUM_ITERS; i++)
{
    var game = new Game(4);
    // var game = Game.LoadCustomGame();

    var mcts_ = new MCTS_Stochastic<Game, Move>(game, eGreedy: 0f);
    // var mcts_ = new MCTS<Game, Move>(game, eGreedy: 0.1f);
    mcts_.GrowWhile(1f);
    NumRolls += mcts_.numRolls;

    // Stopwatch watch = new();
    // watch.Start();
    // var m = game.GetGreedyMove();
    // watch.Stop();
    // duration += (float)watch.Elapsed.TotalSeconds;
    Console.Write(".");
}

Console.WriteLine($"\n {NumRolls / (float)NUM_ITERS}");
// Console.WriteLine(1e6 * duration / (float)NUM_ITERS);
// Console.WriteLine(mcts_);

return;  // before 3549
// random: before-2020, after-2018
// // greedy move (regular): 391 us

// fix RandomMove: 4387
// no copy: 50.8
// clone: 26.8 rolls per 3s
// Array.Copy: 22.4 rolls per 3s
// forfor: 14.4 rools per 3s
// DeepCopier: 15.2

// // play against the machine ==================
// var game = new Game(3);
// // var game = Game.LoadCustomGame();
// Move a = new();
// float timeout_ = 3.0f;

// var mcts_ = new MCTS<Game, Move>(game, eGreedy: 0.0f);
// Func<Game, Move> policy_mcts = (Game g) =>
// {
//     (mcts_, var found) = mcts_.SearchNode(g);
//     mcts_.GrowWhile(timeout_, 300_000);
//     int idx = mcts_.GetBestActionIdx();
//     Console.WriteLine($"\tWinRatio: {mcts_.WinRatio(idx)}, numRolls: {mcts_.NumRolls(idx)}");
//     if (mcts_.WinRatio(idx) < 0.0001f || mcts_.NumRolls(idx) < 100)  // to avoid giving-up-move
//         return g.GetGreedyMove();
//     return mcts_.actions[idx];
// };

// var mcts2_ = new MCTS_Stochastic<Game, Move>(game, eGreedy: 0.0f);
// Func<Game, Move> policy_mcts_stochastic = (Game g) =>
// {
//     (mcts2_, var found) = mcts2_.SearchNode(g);
//     mcts2_.GrowWhile(timeout_, 300_000);
//     int idx = mcts2_.GetBestActionIdx();
//     Console.WriteLine($"\tWinRatio: {mcts2_.WinRatio(idx)}, numRolls: {mcts2_.NumRolls(idx)}, {mcts2_.actions[idx]}");
//     if (mcts2_.WinRatio(idx) < 0.0001f || mcts2_.NumRolls(idx) < 100)  // to avoid giving-up-move
//         return g.GetGreedyMove();
//     // else if (mcts2_.WinRatio(idx) > 0.999f && mcts2_.NumRolls(idx) > 5_000)  // maximize score
//     // {
//     //     var idx2 = mcts2_.GetParanoidActionIdx();
//     //     if (idx2 > -1) idx = idx2;
//     //     Console.WriteLine("\tParanoid action: " + (idx2 > -1 ? mcts2_.actions[idx] : "Not found"));
//     // }
//     return mcts2_.actions[idx];
// };

// Func<Game, Move>[] policies =
// {
//     // (Game g) => g.GetGreedyMove(),
//     policy_mcts,
//     (Game g) => g.GetUserMove(),
//     // (Game g) => g.GetRandomMove(),
//     // (Game g) => g.GetGreedyMove(),
//     // mypolicy,
//     policy_mcts_stochastic,
// };

// while (game.IsGameOver() == false)
// {
//     a = policies[game.activePlayer](DeepCopier.Copy(game));
//     Console.WriteLine(a);
//     Debug.Assert(game.IsValid(a), "Invalid move");
//     if (game.IsValid(a) == false)
//     {
//         Console.WriteLine("Invalid move");
//         Console.WriteLine(Game.ToScript(game));
//         continue;
//     }

//     if (game.Play(a))  // if round is over
//         Console.WriteLine(game);
// }
// Console.WriteLine(game);
// return;

// ========================================
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

float timeout = 3f;
var bench = new GameUtils.Benchmark<Game, Move>(
    minNumPlayers: 3,
    maxNumPlayers: 3,
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
// bench.policies.Add((Game g) => g.GetGreedyMove());

var mcts_stochastic = new MCTS_Stochastic<Game, Move>(new Game(3), eGreedy: 0.0f);
bench.policies.Add((Game g) =>
{
    (mcts_stochastic, var found) = mcts_stochastic.SearchNode(g);
    mcts_stochastic.GrowWhile(timeout, 300_000);
    int idx = mcts_stochastic.GetBestActionIdx();
    if (mcts_stochastic.WinRatio(idx) < 0.00001 || mcts_stochastic.NumRolls(idx) < 100)
        return g.GetGreedyMove();
    return mcts_stochastic.actions[idx];
});

var mcts = new MCTS_Stochastic<Game, Move>(new Game(3), eGreedy: 0.01f);
bench.policies.Add((Game g) =>
{
    (mcts, var found) = mcts.SearchNode(g);
    mcts.GrowWhile(timeout, 300_000);
    int idx = mcts.GetBestActionIdx();
    if (mcts.WinRatio(idx) < 0.00001 || mcts.NumRolls(idx) < 100)
        return g.GetGreedyMove();
    return mcts.actions[idx];
});

var mcts3 = new MCTS<Game, Move>(new Game(4), eGreedy: 0.05f);
bench.policies.Add((Game g) =>
{
    (mcts3, var found) = mcts3.SearchNode(g);
    mcts3.GrowWhile(timeout, 300_000);
    var idx = mcts3.GetBestActionIdx();
    if (mcts3.WinRatio(idx) < 0.00001 || mcts3.NumRolls(idx) < 100)
        return g.GetGreedyMove();
    return mcts3.actions[idx];
});

// var mcts4 = new MCTS<Game, Move>(new Game(4), eGreedy: 0.2f);
// bench.policies.Add((Game g) =>
// {
//     (mcts4, var found) = mcts4.SearchNode(g);
//     mcts4.GrowWhile(timeout, 300_000);
//     var idx = mcts4.GetBestActionIdx();
//     if (mcts4.WinRatio(idx) < 0.0001 || mcts4.NumRolls(idx) < 100)
//         return g.GetGreedyMove();
//     return mcts4.GetBestAction();
// });

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