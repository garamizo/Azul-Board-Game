using System.Diagnostics;
using System;
using System.Collections;
using Utils;
using Azul;  // Game, GameAction
using Ai;  // MCTS


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


Func<float[], float[]>[] rewardMaps = new Func<float[], float[]>[] {
    GameUtils.PassthroughMap,
    GameUtils.MinMaxMap,
    GameUtils.LinearMap,
    GameUtils.SigmoidMap,
    GameUtils.WinLoseMap,
    GameUtils.WinLosePlusMap
};
int[] rolloutReps = new int[] { 1, 10, 100 };
float[] durations = new float[] { 1.0f, 5.0f, 10.0f };

// Simulate game =====================
Game game = new(4);

AgentGenerator agent0 = new(game, rolloutReps[0], durations[1], rewardMaps[2]);
AgentGenerator agent1 = new(game, rolloutReps[0], durations[1], rewardMaps[1]);

GameAction RandomPolicy(Game game)
{
    return game.GetRandomAction();
}

Func<Game, GameAction>[] policies = new Func<Game, GameAction>[] {
    RandomPolicy,
    agent0.Policy,
    RandomPolicy,
    agent1.Policy,
};

GameAction action = new();
while (game.IsGameOver() == false)
{
    action = policies[game.activePlayer](game);
    Debug.Assert(game.IsValid(action));

    if (game.Play(ref action))
    {
        Console.Write($"Round {game.roundIdx}) [");
        for (int i = 0; i < game.numPlayers; i++)
            Console.Write($"{game.players[i].score},");
        Console.WriteLine("]");
    }
}

Console.Write($"Final score: [");
for (int i = 0; i < game.numPlayers; i++)
    Console.Write($"{game.players[i].score},");
Console.WriteLine($"] on round {game.roundIdx} with {game.step} steps.");


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
    float durationMax = 5.0f;
    public AgentGenerator(Game game, int rolloutReps,
        float durationMax, Func<float[], float[]>? rewardMap)
    {
        root = new(game, rolloutReps: rolloutReps, rewardMap: rewardMap);
        this.durationMax = durationMax;
    }

    public GameAction Policy(Game game)
    {
        // float duration = MathF.Max(durationMax / 5,
        //     durationMax - game.step * durationMax / 20);
        float duration = (game.step < 3 * game.numPlayers) ? durationMax : (durationMax / 5);
        root = root.SearchNode(game);
        if (root == null)
            root = new(game);

        root.GrowWhile(duration);
        return root.GetBestAction();
    }
}