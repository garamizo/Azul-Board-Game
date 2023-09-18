namespace Ai;
using System.Diagnostics;
using System.Threading.Channels;
using Azul;  // Game, GameAction
using DeepCopy;  // dotnet add package DeepCopy
using Utils;

public class MCTS
{
    public List<MCTS> childs = new();
    public MCTS? parent;
    public Game state;
    public List<GameAction> actions = new();
    public int numRolls = 0;
    public float numWins = 0;
    // Random rng = new();
    static float c = MathF.Sqrt(2.0f);
    bool isChance = false;
    UInt32 chanceHash;
    int rolloutReps;
    Func<float[], float[]>? rewardMap;


    public MCTS(Game state, MCTS? parent = null, bool copy = true,
        int rolloutReps = 1, Func<float[], float[]>? rewardMap = null)
    {
        this.state = copy ? DeepCopier.Copy(state) : state;
        this.parent = parent;
        this.rolloutReps = rolloutReps;
        this.rewardMap = rewardMap == null ? GameUtils.MinMaxMap : rewardMap;

        if (this.state.IsTerminal() == false)
        {
            actions = this.state.GetPossibleActions();
            GameUtils.Shuffle(Game.rng, actions);
        }
    }



    public void Backpropagate(float[] reward)
    {
        // Debug.Assert(actions.Count == 0);
        var node = this;
        while (node.parent != null)  // not root
        {
            node.numRolls++;
            node.numWins += reward[node.parent.state.activePlayer];
            node = node.parent;
        }
        node.numRolls++;
    }

    public float[] Rollout()
    {
        // int REPS = 100;

        float[] rewardAcc = new float[state.numPlayers];
        for (int i = 0; i < rolloutReps; i++)
        {
            var stateNext = DeepCopier.Copy(state);
            while (stateNext.IsGameOver() == false)
            {
                var action = stateNext.GetRandomAction();
                stateNext.Play(ref action);
            }
            var reward = rewardMap != null ? rewardMap(stateNext.GetReward()) : stateNext.GetReward();
            for (int j = 0; j < state.numPlayers; j++)
                rewardAcc[j] += reward[j] / rolloutReps;
        }
        return rewardAcc;
    }

    public MCTS Expand(int actionIdx)
    {
        Debug.Assert(actions.Count > 0);
        // Debug.Assert(childs != null);

        var action = actions[actionIdx];
        var stateChild = DeepCopier.Copy(state);
        bool playIsChance = stateChild.Play(ref action);
        childs.Add(new MCTS(stateChild, this, copy: false));
        return childs.Last();
    }

    public (MCTS, int) Select()
    {
        // return node to expand
        // Debug.Assert(actions.Count > 0);  // not terminal node

        var node = this;
        int idxBest = 0;
        // stop searching when reach explored or chance nodes 
        // chance nodes have repeated actions, but different childs
        while (node.childs.Count == node.actions.Count && node.isChance == false)
        {
            float k = c * MathF.Sqrt(MathF.Log(node.numRolls));

            float valBest = -100.0f;
            MCTS nodeBest = node;
            for (int i = 0; i < node.childs.Count; i++)
            {
                var child = node.childs[i];
                float val = child.numWins / child.numRolls + k / MathF.Sqrt(child.numRolls);
                if (val > valBest)
                {
                    valBest = val;
                    nodeBest = child;
                    idxBest = i;
                }
            }
            node = nodeBest;
            if (node.actions.Count == 0)  // best node is terminal
                return (node, -1);
        }
        // select parent node, repeat action, but spawn a new child
        if (node.isChance && node.parent != null)
        {
            node = node.parent;
            node.actions.Add(node.actions[idxBest]);
            return (node, node.actions.Count - 1);
        }
        // unexplored child node
        return (node, node.childs.Count);
    }

    public void Grow(int reps = 1)
    {
        for (int i = 0; i < reps; i++)
        {
            var (node, actionIdx) = Select();
            if (actionIdx >= 0)  // if not terminal, get unexplored
                node = node.Expand((int)actionIdx);
            var reward = node.Rollout();
            node.Backpropagate(reward);
        }
    }

    public void GrowWhile(float seconds, int maxRolls = 1_000_000)
    {
        Stopwatch stopWatch = new();
        stopWatch.Start();
        while (stopWatch.Elapsed.TotalSeconds < seconds && numRolls < maxRolls)
            Grow();
    }

    public int GetBestActionIdx()
    {
        Debug.Assert(actions.Count > 0);  // not terminal node

        float valBest = -100.0f;
        int idxBest = 0;
        for (int i = 0; i < childs.Count; i++)
        {
            float val = childs[i].numWins / childs[i].numRolls;
            if (val > valBest)
            {
                valBest = val;
                idxBest = i;
            }
        }
        // return (actions[idxBest], valBest, childs[idxBest].numRolls);
        return idxBest;
    }

    public GameAction GetBestAction() => actions[GetBestActionIdx()];

    public MCTS? SearchNode(Game state, int maxDepth = 4)
    {
        if (this.state == state)
        {
            parent = null;
            isChance = false;
            return this;
        }
        if (maxDepth <= 0)
            return null;
        // search breath first to find node
        for (int i = 0; i < childs.Count; i++)
        {
            MCTS? node = childs[i].SearchNode(state, maxDepth - 1);
            if (node != null)
                return node;
        }
        return null;
    }

    public void Print()
    {
        Console.WriteLine($"Active player {state.activePlayer}, numRolls {numRolls}, numChilds {childs.Count}");
        for (int i = 0; i < childs.Count; i++)
            Console.WriteLine($"Child {i}) " + actions[i].Print(toScreen: false) +
                $", nRolls: {childs[i].numRolls}, wRatio: {childs[i].numWins / (float)childs[i].numRolls}");
    }
}

