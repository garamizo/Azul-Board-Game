namespace Ai;
using System.Diagnostics;
using Azul;
using DeepCopy;  // dotnet add package DeepCopy
using Utils;

public class MCTS
{
    public List<MCTS> childs = new();
    public MCTS? parent;
    public Game state;
    public List<GameAction> actions = new();
    public int numRolls = 0;
    public int numWins = 0;
    Random rng = new();
    static float c = MathF.Sqrt(2.0f);


    public MCTS(Game state_, MCTS? parent_ = null)
    {
        state = state_;
        parent = parent_;

        if (state.IsTerminal() == false)
        {
            actions = state.GetPossibleActions();
            GameUtils.Shuffle(rng, actions);
        }
    }

    public void Backpropagate(float[] reward)
    {
        var node = this;
        while (node.parent != null)
        {
            node.numRolls++;
            node.numWins += (int)reward[node.state.activePlayer];
            node = node.parent;
        }
        node.numRolls++;
    }

    public float[] Rollout()
    {
        int REPS = 10;

        float[] rewardAcc = new float[state.numPlayers];
        for (int i = 0; i < REPS; i++)
        {
            var stateNext = DeepCopier.Copy(state);
            while (stateNext.IsGameOver() == false)
            {
                var action = stateNext.GetRandomAction();
                stateNext.Play(ref action);
            }
            var reward = stateNext.GetReward();
            for (int j = 0; j < state.numPlayers; j++)
                rewardAcc[j] += reward[j] / REPS;
        }
        return rewardAcc;
    }

    public void Expand(int actionIdx)
    {
        Debug.Assert(actions.Count > 0);
        // Debug.Assert(childs != null);

        var action = actions[actionIdx];
        var stateChild = DeepCopier.Copy(state);
        stateChild.Play(ref action);
        childs.Add(new MCTS(stateChild, this));
    }

    public (MCTS, int?) Select()
    {
        Debug.Assert(actions.Count > 0);  // not terminal node

        var node = this;
        while (node.childs.Count == node.actions.Count)
        {
            float valBest = -100.0f;
            float k = MCTS.c * MathF.Sqrt(MathF.Log(node.numRolls));

            MCTS nodeBest = node;
            for (int i = 0; i < node.childs.Count; i++)
            {
                var child = node.childs[i];
                float val = child.numWins / (float)child.numRolls + k / MathF.Sqrt(child.numRolls);
                if (val > valBest)
                {
                    valBest = val;
                    nodeBest = child;
                }
            }
            node = nodeBest;
            if (node.actions == null)  // best node is terminal
                return (node, null);
        }
        // unexplored child node
        return (node, node.childs != null ? node.childs.Count : 0);
    }

    public void Grow()
    {
        var (node, actionIdx) = Select();
        if (actionIdx != null)
        {
            node.Expand((int)actionIdx);
            node = node.childs.Last();
        }
        var reward = node.Rollout();
        node.Backpropagate(reward);
    }

    public void GrowWhile(float seconds)
    {
        Stopwatch stopWatch = new();
        stopWatch.Start();
        while (stopWatch.Elapsed.TotalSeconds < seconds)
            Grow();
        stopWatch.Stop();
    }

    public int GetBestAction()
    {
        Debug.Assert(actions.Count > 0);  // not terminal node

        float valBest = -100.0f;
        int idxBest = 0;
        for (int i = 0; i < childs.Count; i++)
        {
            float val = childs[i].numWins / (float)childs[i].numRolls;
            if (val > valBest)
            {
                valBest = val;
                idxBest = i;
            }
        }
        // return (actions[idxBest], valBest, childs[idxBest].numRolls);
        return idxBest;
    }

    public void Print()
    {
        Console.WriteLine($"Active player {state.activePlayer}, numRolls {numRolls}");
        for (int i = 0; i < childs.Count; i++)
            Console.WriteLine($"Child {i}) {childs[i].numRolls} {childs[i].numWins / (float)childs[i].numRolls}");
    }
}

