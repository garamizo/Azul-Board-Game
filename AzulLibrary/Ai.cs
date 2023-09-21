namespace Ai;

using System.Diagnostics;  // Debug.Assert, Stopwatch
using Azul;  // Game, GameAction
using DeepCopy;  // DeepCopier (dotnet add package DeepCopy)
using Utils;


public class MCTS_Stochastic
{
    public List<MCTS_Stochastic> childs = new();
    public MCTS_Stochastic? parent;
    public Game state;
    public List<GameAction> actions = new();
    public int actionIdx;
    public int numRolls = 0;
    public float numWins = 0;
    public static float c = MathF.Sqrt(2.0f);
    Func<float[], float[]>? rewardMap;
    float eGreedy;
    public int rootPlayer = 0;
    bool paranoid;  // if true, tree cannot be shared among many players

    // Construct root
    public MCTS_Stochastic(Game state, Func<float[], float[]>? rewardMap = null,
        float eGreedy = 0.1f, bool paranoid = true)
    {
        this.state = DeepCopier.Copy(state);
        this.parent = null;
        this.rewardMap = rewardMap == null ? RewardMap.WinLosePlus : rewardMap;
        this.eGreedy = eGreedy;
        rootPlayer = state.activePlayer;
        this.paranoid = paranoid;

        actions = this.state.GetPossibleActions();
        GameUtils.Shuffle(Game.rng, actions);
    }

    // construct child
    public MCTS_Stochastic(MCTS_Stochastic parent, int actionIdx)
    {
        this.state = DeepCopier.Copy(parent.state);
        this.actionIdx = actionIdx;
        this.parent = parent;
        this.rewardMap = parent.rewardMap;
        this.eGreedy = parent.eGreedy;
        this.rootPlayer = parent.rootPlayer;
        this.paranoid = parent.paranoid;

        var action = parent.actions[actionIdx];
        this.state.Play(ref action);

        if (this.state.IsTerminal() == false)
        {
            actions = this.state.GetPossibleActions();
            GameUtils.Shuffle(Game.rng, actions);
        }
    }

    public bool IsLeaf() => actions.Count == 0;
    public bool IsRoot() => parent == null;
    // public bool IsFullyExpanded() => childs.Count == actions.Count; // a leaf => fully expanded
    public bool IsChance() => state.chanceHash > 0;
    public UInt64 GetChanceHash() => state.chanceHash;


    public bool IsFullyExpanded()
    {
        for (int i = actions.Count - 1; i >= 0; i--)
        {
            bool actionIsExplored = false;
            foreach (var c in childs)
                if (c.actionIdx == i)
                {
                    actionIsExplored = true;
                    break;
                }
            if (actionIsExplored == false)
                return false;
        }
        return true;
    }


    public void Backpropagate(float[] reward)
    {
        var node = this;
        while (node.IsRoot() == false)  // not root
        {
            if (node.IsChance())
                // update first node with same actionIdx
                for (int i = 0; i < node.parent.childs.Count; i++)
                    if (node.parent.childs[i].actionIdx == node.actionIdx)
                    {
                        // node.numRolls++;
                        node = node.parent.childs[i];
                        break;
                    }

            node.numRolls++;
            node.numWins += reward[node.parent.state.activePlayer];  // #ok
            node = node.parent;
        }
        node.numRolls++;
    }

    public float[] Rollout()
    {
        var stateNext = DeepCopier.Copy(state);
        while (stateNext.IsGameOver() == false)
        {
            // var action = stateNext.GetRandomAction();
            // var action = Game.DefaultPolicy(stateNext);
            var action = Game.EGreedyPolicy(stateNext, eGreedy);
            stateNext.Play(ref action);
        }
        var reward = rewardMap != null ? rewardMap(stateNext.GetReward()) : stateNext.GetReward();
        // paranoid: root vs rest
        //      root loses => reward or rest == 1
        if (paranoid)
        {
            float rewardRoot = reward[rootPlayer];
            for (int i = 0; i < reward.Length; i++)
                if (i != rootPlayer)
                    reward[i] = MathF.Max(0.0f, 1.0f - rewardRoot);
        }
        return reward;
    }

    public MCTS_Stochastic Expand(int actionIdx = -1)
    {   // allow for expanding previously explored nodes
        childs.Add(new MCTS_Stochastic(this, actionIdx == -1 ? childs.Count : actionIdx));
        return childs.Last();
    }


    public MCTS_Stochastic Select(bool debug = false)
    {
        // return node to expand, these nodes can be:
        //   - not fully,
        //   - fully expanded but terminal
        //   - or fully expanded but parent of the best node, which is chance node
        var node = this;
        while (node.IsFullyExpanded() && node.IsLeaf() == false)
        {
            float k = c * MathF.Sqrt(MathF.Log(node.numRolls));

            float valBest = -100.0f;
            var nodeBest = node;
            for (int i = 0; i < node.childs.Count; i++)
            {
                var child = node.childs[i];
                float val = child.numWins / child.numRolls + k / MathF.Sqrt(child.numRolls);
                if (val > valBest)
                {
                    valBest = val;
                    nodeBest = child;
                }
            }
            if (nodeBest.IsChance())
            {
                // roll dice to get new child or return an existing one
                // node, actionIdx => nodeBest
                var nodeNew = node.Expand(nodeBest.actionIdx);

                // check for existing chance hash
                for (int i = 0; i < node.childs.Count - 1; i++)
                    if (node.childs[i].actionIdx == nodeNew.actionIdx &&
                        node.childs[i].GetChanceHash() == nodeNew.GetChanceHash())
                    {   // GetChanceHash is faster than state==
                        node.childs.Remove(nodeNew);
                        nodeNew = node.childs[i];
                        // existing node will continue to be searched
                        break;
                    }
                // newly created node will exit the loop to be expanded
                nodeBest = nodeNew;
            }
            node = nodeBest;
        }
        return node;
    }

    public void Grow(int reps = 1)
    {
        for (int i = 0; i < reps; i++)
        {
            var node = Select();
            if (node.IsLeaf() == false)  // if not terminal, get unexplored
                node = node.Expand();
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
        return childs[idxBest].actionIdx;
    }
    public GameAction GetBestAction() => actions[GetBestActionIdx()];

    public MCTS_Stochastic? SearchNode(Game state, int maxDepth = 4)
    {
        if (this.state == state)
        {
            parent = null;
            rootPlayer = state.activePlayer;
            this.state.chanceHash = 0;
            return this;
        }
        if (maxDepth <= 0)
            return null;
        // search breath first to find node
        for (int i = 0; i < childs.Count; i++)
        {
            var node = childs[i].SearchNode(state, maxDepth - 1);
            if (node != null)
                return node;
        }
        return null;
    }

    public void Print()
    {
        Console.WriteLine($"Active player {state.activePlayer}, numRolls {numRolls}, numChilds {childs.Count}");
        for (int i = 0; i < childs.Count; i++)
            Console.WriteLine($"Child {i}) " + actions[childs[i].actionIdx].Print(toScreen: false) +
                $", nRolls: {childs[i].numRolls}, wRatio: {childs[i].numWins / (float)childs[i].numRolls}");
    }

    // convert Print to ToString
    public override string ToString()
    {
        string s = $"Active player {state.activePlayer}, numRolls {numRolls}, numChilds {childs.Count}\n";
        for (int i = 0; i < childs.Count; i++)
            s += $"Child {i}) " + actions[childs[i].actionIdx].Print(toScreen: false) +
                $", nRolls: {childs[i].numRolls}, wRatio: {childs[i].numWins / (float)childs[i].numRolls}\n";
        return s;
    }
}



public class MCTS
{
    public List<MCTS> childs = new();
    public MCTS? parent;
    public Game state;
    public List<GameAction> actions = new();
    public int actionIdx;
    public int numRolls = 0;
    public float numWins = 0;
    public static float c = MathF.Sqrt(2.0f);
    Func<float[], float[]>? rewardMap;
    float eGreedy;
    bool paranoid;
    public int rootPlayer;


    // Construct root
    public MCTS(Game state, Func<float[], float[]>? rewardMap = null,
        float eGreedy = 0.1f, bool paranoid = true)
    {
        this.state = DeepCopier.Copy(state);
        this.parent = null;
        this.rewardMap = rewardMap == null ? RewardMap.WinLose : rewardMap;
        this.eGreedy = eGreedy;
        this.paranoid = paranoid;
        rootPlayer = state.activePlayer;

        actions = this.state.GetPossibleActions();
        GameUtils.Shuffle(Game.rng, actions);
    }

    // construct child
    public MCTS(MCTS parent, int actionIdx)
    {
        this.state = DeepCopier.Copy(parent.state);
        this.actionIdx = actionIdx;
        this.parent = parent;
        this.rewardMap = parent.rewardMap;
        this.eGreedy = parent.eGreedy;
        this.paranoid = parent.paranoid;
        this.rootPlayer = parent.rootPlayer;

        var action = parent.actions[actionIdx];
        bool isChance = this.state.Play(ref action);

        if (isChance == false && this.state.IsGameOver() == false)
        {
            actions = this.state.GetPossibleActions();
            GameUtils.Shuffle(Game.rng, actions);
        }
    }

    public bool IsLeaf() => actions.Count == 0;
    public bool IsRoot() => parent == null;
    public bool IsFullyExpanded() => childs.Count == actions.Count; // a leaf => fully expanded

    public void Backpropagate(float[] reward)
    {
        var node = this;
        while (node.IsRoot() == false)  // not root
        {
            node.numRolls++;
            node.numWins += reward[node.parent.state.activePlayer];  // #ok
            node = node.parent;
        }
        node.numRolls++;
    }

    public float[] Rollout()
    {
        var stateNext = DeepCopier.Copy(state);
        while (stateNext.IsGameOver() == false)
        {
            // var action = stateNext.GetRandomAction();
            // var action = Game.DefaultPolicy(stateNext);
            var action = Game.EGreedyPolicy(stateNext, eGreedy);
            stateNext.Play(ref action);
        }
        var reward = rewardMap != null ? rewardMap(stateNext.GetReward()) : stateNext.GetReward();
        if (paranoid)
        {
            float rewardRoot = reward[rootPlayer];
            for (int i = 0; i < reward.Length; i++)
                if (i != rootPlayer)
                    reward[i] = MathF.Max(0.0f, 1.0f - rewardRoot);
        }
        return reward;
    }

    public virtual MCTS Expand()
    {
        // var stateChild = DeepCopier.Copy(state);
        // var action = actions[childs.Count];
        // stateChild.Play(ref action);
        // childs.Add(new MCTS(stateChild, this, copy: false));
        childs.Add(new MCTS(this, childs.Count));
        return childs.Last();
    }

    public virtual MCTS Select()
    {
        // return node to expand
        var node = this;
        // stop searching when reach explored or chance nodes 
        // childs.Count == actions.Count && actions.Count > 0
        while (node.IsFullyExpanded() && node.IsLeaf() == false)
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
                }
            }
            node = nodeBest;
        }
        // unexplored child node
        return node;
    }

    public void Grow(int reps = 1)
    {
        for (int i = 0; i < reps; i++)
        {
            var node = Select();
            if (node.IsLeaf() == false)  // if not terminal, get unexplored
                node = node.Expand();
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
        return idxBest;
    }
    public GameAction GetBestAction() => actions[GetBestActionIdx()];

    public MCTS? SearchNode(Game state, int maxDepth = 4)
    {
        if (this.state == state)
        {
            parent = null;
            return this;
        }
        if (maxDepth > 0)
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

