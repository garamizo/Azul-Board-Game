namespace Ai
{
    using System.Diagnostics;  // Debug.Assert, Stopwatch
    using System.Collections.Generic;
    using DeepCopy;  // DeepCopier (dotnet add package DeepCopy)
    using GameMath = GameUtils.GameMath;
    using RewardMap = GameUtils.RewardMap;
    using System;
    using System.Linq;  // Sum, Max, Min
    using Azul;


    /* Paranoid search with Iteractive Deepening*/
    public class ParanoidID<TGame, TMove>
        where TGame : GameUtils.Game<TMove>
    {
        const float MIN_REWARD = -1000.0f;
        const float MAX_REWARD = 1000.0f;
        static Stopwatch stopWatch = new();
        static float _timeout;
        static int rootPlayer;
        static int count = 0;

        public static TMove GetBestMove(TGame state, int depth = int.MaxValue, float timeout = float.MaxValue)
        {
            stopWatch.Restart();
            _timeout = timeout;
            rootPlayer = state.activePlayer;

            var actions = GameMath.Shuffled<TMove>(state.RandomSeed, state.GetPossibleActions());
            for (int d = 1; d <= depth; d++)
            {
                int idxBest = 0;
                var valBest = -1f;
                for (int i = 0; i < actions.Count; i++)
                {
                    var stateNext = DeepCopier.Copy(state);
                    stateNext.Play(actions[i]);
                    float score = -GetGameValue(stateNext, d - 1, -1f, -valBest);

                    if (score > valBest)
                    {
                        valBest = score;
                        idxBest = i;
                    }
                }
                if (stopWatch.Elapsed.TotalSeconds > _timeout)
                    break;  // incomplete loop - do not update best action

                // insert best move to front 
                var actionBest = actions[idxBest];
                actions.RemoveAt(idxBest);
                actions.Insert(0, actionBest);
            }
            return actions[0];
        }

        // rootPlayer tries to maximize: scores[rootPlayer]
        // other players try to minimize: scores[rootPlayer] - scores[activePlayer]
        static float GetGameValue(TGame state, int depth, float alpha, float beta)
        {
            bool isGameOver = state.IsGameOver();
            if (depth < 0 || isGameOver || stopWatch.Elapsed.TotalSeconds > _timeout)
            {
                count++;
                float[] scores = isGameOver ? state.GetRewards() : state.GetHeuristics(); // 0.0 - 1.0
                float scoreSum = 0f, scoreMax = -100f;
                int numTies = 0;
                for (int i = 0; i < scores.Length; i++)
                {
                    scoreSum += scores[i];
                    if (scores[i] > scoreMax)
                    {
                        scoreMax = scores[i];
                        numTies = 1;
                    }
                    else if (scores[i] == scoreMax)
                        numTies++;
                }

                if (scores[rootPlayer] == scoreMax && numTies > 1)  // tie
                    return 0f;
                return (2 * scores[rootPlayer] - scoreSum) *
                    (state.activePlayer == rootPlayer ? 1 : -1) / (1 + state.step / 100f);
            }

            foreach (var action in state.GetPossibleActions())
            {
                var stateNext = DeepCopier.Copy(state);
                stateNext.Play(action);

                float val;
                if ((stateNext.activePlayer == rootPlayer && state.activePlayer != rootPlayer) ||
                    (stateNext.activePlayer != rootPlayer && state.activePlayer == rootPlayer))
                    val = -GetGameValue(stateNext, depth - 1, -beta, -alpha);
                else
                    val = GetGameValue(stateNext, depth - 1, alpha, beta);

                if (val > alpha) alpha = val;
                if (alpha >= beta || stopWatch.Elapsed.TotalSeconds > _timeout)
                    return beta;
            }
            return alpha;
        }
    }

    public class MaxnID<TGame, TMove>
        where TGame : GameUtils.Game<TMove>
    {
        const float MIN_REWARD = -1000.0f;
        const float MAX_REWARD = 1000.0f;
        static Stopwatch stopWatch = new();
        static float _timeout;
        static int rootPlayer;
        static int count = 0;

        public static TMove GetBestMove(TGame state, int depth = int.MaxValue, float timeout = float.MaxValue)
        {
            stopWatch.Restart();
            _timeout = timeout;
            rootPlayer = state.activePlayer;

            var actions = GameMath.Shuffled<TMove>(state.RandomSeed, state.GetPossibleActions());
            for (int d = 1; d <= depth; d++)
            {
                int idxBest = 0;
                var valBest = -1f;
                for (int i = 0; i < actions.Count; i++)
                {
                    var score = GetGameMoveValue(state, actions[i], d - 1)[state.activePlayer];
                    if (score > valBest)
                    {
                        valBest = score;
                        idxBest = i;
                    }
                }
                if (stopWatch.Elapsed.TotalSeconds > _timeout)
                    break;  // incomplete loop - do not update best action

                // insert best move to front 
                var actionBest = actions[idxBest];
                actions.RemoveAt(idxBest);
                actions.Insert(0, actionBest);
            }
            return actions[0];
        }

        static float[] GetGameMoveValue(TGame state, TMove act, int depth)
        {
            var stateNext = DeepCopier.Copy(state);
            stateNext.Play(act);
            bool isGameOver = stateNext.IsGameOver();
            if (depth < 0 || isGameOver || stopWatch.Elapsed.TotalSeconds > _timeout)
            {
                count++;
                return isGameOver ? stateNext.GetRewards() : stateNext.GetHeuristics(); // 0.0 - 1.0
            }

            // float[] alpha = -float.MaxValue;
            // get float array of -1.0f in single line 
            float[] alpha = Enumerable.Repeat(-10.0f, stateNext.numPlayers).ToArray();
            int at = stateNext.activePlayer;
            foreach (var action in stateNext.GetPossibleActions())
            {
                float[] scores = GetGameMoveValue(stateNext, action, depth - 1);

                if (scores[at] > alpha[at])
                    alpha = scores;
                if (stopWatch.Elapsed.TotalSeconds > _timeout) break;
            }
            return alpha;
        }
    }

    public class MCTS_Stochastic<TGame, TMove>
        where TGame : GameUtils.Game<TMove>
    {
        public List<MCTS_Stochastic<TGame, TMove>>[] childs;
        public MCTS_Stochastic<TGame, TMove>? parent;  // null for root
        public TGame state;
        public List<TMove> actions = new();  // all possible actions
        public int actionIdx;  // parent + actionIdx => this
        public int numRolls = 0;
        public float numWins = 0;  // wins for parent.activePlayer
        public static float c = MathF.Sqrt(2.0f);
        // Func<float[], float[]> rewardMap = RewardMap.Passthrough;
        float eGreedy;
        public int rootPlayer = 0;
        // bool paranoid;  // if true, tree cannot be shared among many players
        Random RandomSeed { get => state.RandomSeed; }
        const bool SORT_ACTIONS = false;
        public float WinRatio(int idx)
        {
            int numRolls = 0;
            float numWins = 0f;
            foreach (var child in childs[idx])
            {
                numRolls += child.numRolls;
                numWins += child.numWins;
            }
            return numWins / numRolls;
        }
        public float winRatio => numWins / numRolls;

        public int NumRolls(int idx)
        {
            int count = 0;
            foreach (var child in childs[idx])
                count += child.numRolls;
            return count;
        }


        // Construct root
        public MCTS_Stochastic(TGame state, float eGreedy)
        {
            this.state = DeepCopier.Copy(state);
            this.state.chanceHash = 0;
            this.parent = null;
            this.eGreedy = eGreedy;
            rootPlayer = state.activePlayer;

            actions = this.state.GetPossibleActions(SORT_ACTIONS);
            childs = new List<MCTS_Stochastic<TGame, TMove>>[actions.Count];
            for (int i = 0; i < actions.Count; i++)
                childs[i] = new();
            GameMath.Shuffle(RandomSeed, actions);
        }

        // construct child
        public MCTS_Stochastic(MCTS_Stochastic<TGame, TMove> parent, int actionIdx)
        {
            this.state = DeepCopier.Copy(parent.state);
            this.actionIdx = actionIdx;
            this.parent = parent;
            // this.rewardMap = parent.rewardMap;
            this.eGreedy = parent.eGreedy;
            this.rootPlayer = parent.rootPlayer;

            var action = parent.actions[actionIdx];
            bool isChance = this.state.Play(action);

            if (this.state.IsGameOver() == false)
            {
                actions = this.state.GetPossibleActions(SORT_ACTIONS);
                childs = new List<MCTS_Stochastic<TGame, TMove>>[actions.Count];
                for (int i = 0; i < actions.Count; i++)
                    childs[i] = new();
                GameMath.Shuffle(RandomSeed, actions);
            }
        }

        public MCTS_Stochastic(MCTS_Stochastic<TGame, TMove> old, TGame state) :
            this(state, old.eGreedy)
        { }

        public int NumChilds
        {
            get
            {
                int count = 0;
                for (int i = 0; i < childs.Length; i++)
                    count += childs[i].Count;
                return count;
            }
        }


        public bool IsLeaf() => actions.Count == 0;
        public bool IsRoot() => parent == null;
        // public bool IsFullyExpanded() => childs.Count == actions.Count; // a leaf => fully expanded
        public bool IsChance() => state.chanceHash > 0;
        public UInt64 GetChanceHash() => state.chanceHash;

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
                var action = stateNext.GetEGreedyMove(eGreedy);
                // Debug.Assert(stateNext.IsValid(action), "bad move!");
                stateNext.Play(action);
            }
            var reward = stateNext.GetRewards();
            return reward;
        }

        public MCTS_Stochastic<TGame, TMove> Expand()
        {   // allow for expanding previously explored nodes
            // int idx = IsRoot() ? numRolls : numRolls - 1;
            // childs[idx].Add(new(this, idx));
            // return childs[idx][0];

            for (int i = 0; i < childs.Length; i++)
                if (childs[i].Count == 0)
                {
                    // Debug.Assert(i == numRolls - 1, "Should be last node");
                    childs[i].Add(new(this, i));
                    return childs[i][0];
                }
            Debug.Assert(false, "Should not reach here");
            return childs[0][0];

            // childs[numRolls].Add(new(this, numRolls));
            // return childs[numRolls][0];  // never chance nodes
        }


        public MCTS_Stochastic<TGame, TMove> Select()
        {
            // return node to expand, these nodes can be:
            //   - not fully,
            //   - fully expanded but terminal
            //   - or fully expanded but parent of the best node, which is chance node
            var node = this;
            int count = 0;
            while (node.IsChance() || (node.IsLeaf() == false && node.NumChilds >= node.actions.Count))
            {
                count++;
                Debug.Assert(count < 300, "Inf loop");
                bool found = false;
                // if node is chance, select child randonly 
                if (node.IsChance())
                {
                    Debug.Assert(node.parent != null, "Should have parent");
                    MCTS_Stochastic<TGame, TMove> nodeNew = new(node.parent, node.actionIdx);
                    // search siblings for already existing chance node

                    // foreach (var n in node.parent.childs[node.actionIdx])
                    for (int i = 0; i < node.parent.childs[node.actionIdx].Count; i++)
                        if (node.parent.childs[node.actionIdx][i].GetChanceHash() == nodeNew.GetChanceHash())
                        {
                            node = node.parent.childs[node.actionIdx][i];
                            found = true;
                            break;  // next, search best child of this node
                        }
                    // if (found) continue;
                    if (found == false)  // if new chance roll
                    {
                        // Console.WriteLine($"{nodeNew.GetChanceHash()} <-");
                        node.parent.childs[node.actionIdx].Add(nodeNew);
                        node = node.parent.childs[node.actionIdx].Last();  // will exit loop
                        break;
                    }
                }

                // regular search ====================
                float k = c * MathF.Sqrt(MathF.Log(node.numRolls));

                float valBest = -100.0f;
                var nodeBest = node;
                for (int i = 0; i < node.childs.Length; i++)
                {
                    int NumRolls = 0;
                    float NumWins = 0;
                    foreach (var child in node.childs[i])
                    {
                        if (child.numRolls == 0) return child; // unexplored node
                        NumRolls += child.numRolls;
                        NumWins += child.numWins;
                    }
                    // var child = node.childs[i][0];
                    float val = NumWins / NumRolls + k / MathF.Sqrt(NumRolls);
                    if (val > valBest)
                    {
                        valBest = val;
                        nodeBest = node.childs[i][0];
                    }
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
                // if not terminal, (ie unexplored or chance node)
                if (node.IsLeaf() == false)
                    node = node.Expand();
                var reward = node.Rollout();
                node.Backpropagate(reward);
            }
        }

        public void GrowWhile(float seconds, int maxRolls = 300_000)
        {
            Stopwatch stopWatch = new();
            stopWatch.Start();
            while (stopWatch.Elapsed.TotalSeconds < seconds && numRolls < maxRolls)
                Grow();
        }

        public int GetParanoidActionIdx(float winRatioThresh = 0.999f)
        {   // get action index to maximize score using paranoid search
            float TIMEOUT = 0.1f;
            int DEPTH_MAX = 30;

            int idxBest = -1;
            float valBest;

            Stopwatch stopWatch = new();
            stopWatch.Start();
            for (int d = 1; d <= DEPTH_MAX; d++)
            {
                valBest = -10_000f;
                foreach (var nodes in childs)
                {
                    int count = 0;
                    float score = 0f;
                    foreach (var nodeNext in nodes)
                    {
                        if (nodeNext.winRatio < winRatioThresh) continue;
                        count++;
                        score += -GetGameValue(nodeNext, d - 1, -1000f, -valBest, state.activePlayer);

                        if (stopWatch.Elapsed.TotalSeconds > TIMEOUT)
                            return idxBest;
                    }
                    score /= count;
                    if (score > valBest || (score == valBest && RandomSeed.NextDouble() < 0.5))
                    {
                        valBest = score;
                        idxBest = nodes[0].actionIdx;
                    }
                }

                // insert best move to front 
                // var actionBest = actions[idxBest];
                // childs.RemoveAt(idxBest);
                // actions.Insert(0, actionBest);
            }
            return idxBest;

            static float GetGameValue(MCTS_Stochastic<TGame, TMove> node, int depth, float alpha, float beta, int rootPlayer)
            {
                bool isGameOver = node.IsLeaf();
                if (depth < 0 || isGameOver)
                {
                    float[] scores = node.state.GetScores(); // 0.0 - 1.0
                    float scoreSum = 0f, scoreMax = -100f;
                    for (int i = 0; i < scores.Length; i++)
                    {
                        scoreSum += scores[i];
                        if (scores[i] > scoreMax)
                            scoreMax = scores[i];
                    }
                    return (2 * scores[rootPlayer] - scoreSum) *
                        (node.state.activePlayer == rootPlayer ? 1 : -1);
                }

                foreach (var nodes in node.childs)
                {
                    float val = 0f;
                    foreach (var nodeNext in nodes)
                    {
                        if ((nodeNext.state.activePlayer == rootPlayer && node.state.activePlayer != rootPlayer) ||
                            (nodeNext.state.activePlayer != rootPlayer && node.state.activePlayer == rootPlayer))
                            val -= GetGameValue(nodeNext, depth - 1, -beta, -alpha, rootPlayer);
                        else
                            val += GetGameValue(nodeNext, depth - 1, alpha, beta, rootPlayer);
                    }
                    val /= nodes.Count;

                    if (val > alpha) alpha = val;
                    if (alpha >= beta)
                        return beta;
                }
                return alpha;
            }
        }

        public int GetBestActionIdx()
        {
            Debug.Assert(actions.Count > 0);  // not terminal node

            float valBest = -100.0f;
            int idxBest = 0;
            for (int i = 0; i < childs.Length; i++)
            {
                float val = WinRatio(i);
                if (val > valBest)
                {
                    valBest = val;
                    idxBest = i;
                }
            }
            return idxBest;
        }
        public TMove GetBestAction() => actions[GetBestActionIdx()];


        public (MCTS_Stochastic<TGame, TMove>, bool) SearchNode(TGame state, int maxDepth = 4)
        {
            if (this.state.Equals(state))
            {
                parent = null;
                rootPlayer = state.activePlayer;
                this.state.chanceHash = 0;
                return (this, true);
            }
            if (maxDepth > 0 && childs != null)
                // search breath first to find node
                for (int i = 0; i < childs.Length; i++)
                {
                    foreach (var child in childs[i])
                    {
                        var (node, found) = child.SearchNode(state, maxDepth - 1);
                        if (found)
                            return (node, true);
                    }
                }
            // return copy of root, but with new state
            // Console.WriteLine("Node not found");
            return (new(this, state), false);
        }

        // convert Print to ToString
        public string ToString(int counts = 10)
        {
            if (childs == null) return $"Active player {state.activePlayer}, numRolls 0, numChilds 0";
            var idxSort = Enumerable.Range(0, childs.Length).OrderByDescending(i => WinRatio(i));
            string s = $"Active player {state.activePlayer}, numRolls {numRolls}, numChilds {childs.Length}";
            foreach (var i in idxSort.Take(counts))
            {
                s += $"\nChild {i:D3}) {actions[i]},\tRolls: {NumRolls(i)},\twinRatio: {WinRatio(i):F8}";
                foreach (var child in childs[i].Take(3))
                    if (child.IsChance())
                        s += $"\n  Rolls: {child.numRolls},\twRatio: {child.winRatio:F3},\thash: {child.GetChanceHash()}";
                if (childs[i].Count > 0 && childs[i][0].IsChance() && childs[i].Count > 3)
                    s += $"\n  ...";

            }
            s += "\n...\n";
            return s;
        }
        public override string ToString() => ToString(10);
    }



    public class MCTS<TGame, TMove>
        where TGame : GameUtils.Game<TMove>
    {
        public List<MCTS<TGame, TMove>> childs = new();
        public MCTS<TGame, TMove>? parent;
        public TGame state;
        public List<TMove> actions = new();
        public int actionIdx;
        public int numRolls = 0;
        public float numWins = 0;
        public static float c = MathF.Sqrt(2.0f);
        // Func<float[], float[]> rewardMap = RewardMap.Passthrough;
        float eGreedy;
        public int rootPlayer;
        const bool SORT_ACTIONS = true;
        Random RandomSeed { get => state.RandomSeed; }

        public float WinRatio(int idx) => childs[idx].numWins / childs[idx].numRolls;
        public int NumRolls(int idx) => childs[idx].numRolls;


        // Construct root
        public MCTS(TGame state, float eGreedy)
        {
            this.state = DeepCopier.Copy(state);
            this.parent = null;
            this.eGreedy = eGreedy;
            rootPlayer = state.activePlayer;

            actions = this.state.GetPossibleActions(SORT_ACTIONS);
            GameMath.Shuffle(RandomSeed, actions);
        }

        public MCTS(MCTS<TGame, TMove> old, TGame state) :
            this(state, old.eGreedy)
        { }

        // construct child
        public MCTS(MCTS<TGame, TMove> parent, int actionIdx)
        {
            this.state = DeepCopier.Copy(parent.state);
            this.actionIdx = actionIdx;
            this.parent = parent;
            this.eGreedy = parent.eGreedy;
            this.rootPlayer = parent.rootPlayer;

            var action = parent.actions[actionIdx];
            bool isChance = this.state.Play(action);

            if (isChance == false && this.state.IsGameOver() == false)
            {
                actions = this.state.GetPossibleActions(SORT_ACTIONS);
                GameMath.Shuffle(RandomSeed, actions);
            }
        }

        // chance node have 0 actions => is seen as leaf and fully expanded
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
                TMove action;
                if (stateNext.RandomSeed.NextDouble() < eGreedy)
                    action = stateNext.GetRandomMove();
                else
                    action = stateNext.GetGreedyMove();
                // action = stateNext.GetEGreedyMove(eGreedy);
                stateNext.Play(action);
            }
            var reward = stateNext.GetRewards();
            return reward;
        }

        public MCTS<TGame, TMove> Expand()
        {
            childs.Add(new(this, childs.Count));
            return childs.Last();
        }

        public MCTS<TGame, TMove> Select()
        {
            // return node to expand
            var node = this;
            // stop searching when reach explored or chance nodes 
            // childs.Count == actions.Count && actions.Count > 0
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
        public TMove GetBestAction() => actions[GetBestActionIdx()];

        // public M GetBestMove(G state, float timeout = 0.1f)
        // {
        //     Stopwatch stopWatch = new();
        //     stopWatch.Start();
        //     (this, var found) = SearchNode(state);
        //     while (stopWatch.Elapsed.TotalSeconds < timeout)
        //         Grow();
        //     return GetBestAction();
        // }

        public (MCTS<TGame, TMove>, bool) SearchNode(TGame state, int maxDepth = 4)
        {
            if (this.state.Equals(state))
            {
                parent = null;
                return (this, true);
            }
            if (maxDepth > 0)
                // search breath first to find node
                for (int i = 0; i < childs.Count; i++)
                {
                    var (node, found) = childs[i].SearchNode(state, maxDepth - 1);
                    if (found)
                        return (node, true);
                }
            return (new(this, state), false);
        }

        // convert Print to ToString
        public string ToString(int nLines = 5)
        {
            var idxSort = Enumerable.Range(0, childs.Count).OrderByDescending(i => WinRatio(i));
            string s = $"Active player {state.activePlayer},\tnumRolls {numRolls},\tnumChilds {childs.Count}\n";
            foreach (var i in idxSort.Take(nLines))
                s += $"Child {i:D3}) {actions[i]}" +
                    $",\tnRolls: {NumRolls(i)},\twRatio: {WinRatio(i):F03}\n";
            s += "...\n";
            return s;
        }
        public override string ToString() => ToString(5);

    }

}