namespace TicTacToe;
using DeepCopy;
using System.Data.Common;
using System.Diagnostics;

public class Move
{
    public int i;
    public int j;
    public int activePlayer;
    public Move(int i, int j, int activePlayer)
    {
        this.i = i;
        this.j = j;
        this.activePlayer = activePlayer;
    }
    public override string ToString()
    {
        return $"({activePlayer}: {i},{j})";
    }
}

public class Game : GameUtils.Game<Move>
{
    public int[,] grid;
    const int ROWS = 4;
    const int COLS = 4;
    const int NUM_SET = 3;
    const int EMPTY = -1;
    const int TIE = -1;
    const int NOT_OVER = -2;

    public Game(int numPlayers = 2)
    {
        this.numPlayers = numPlayers;
        activePlayer = 0;
        step = 0;

        grid = new int[ROWS, COLS];
        for (int row = 0; row < ROWS; row++)
            for (int col = 0; col < COLS; col++)
                grid[row, col] = EMPTY;
    }
    public Game() : this(2)
    { }


    public override Game Reset(int numPlayers) => new(numPlayers);

    public override bool IsGameOver() => GetWinner() != NOT_OVER;

    // public override bool Compare(Game a)
    // {
    //     return true;
    //     if (a.numPlayers != numPlayers ||
    //         a.activePlayer != activePlayer ||
    //         a.step != step) return false;
    //     for (int row = 0; row < ROWS; row++)
    //         for (int col = 0; col < COLS; col++)
    //             if (a.grid[row, col] != grid[row, col])
    //                 return false;
    //     return true;
    // }

    public override bool Equals(GameUtils.Game<Move> other)
    {
        var a = other as Game;
        if (a.numPlayers != numPlayers ||
            a.activePlayer != activePlayer ||
            a.step != step) return false;
        for (int row = 0; row < ROWS; row++)
            for (int col = 0; col < COLS; col++)
                if (a.grid[row, col] != grid[row, col])
                    return false;
        return true;
    }

    // public static bool operator !=(Game a, Game b) => !(a == b);

    public static void UnitTest()
    {
        var game = new Game(2);
        game.grid = new int[,] {
            {  1,  0, -1 },
            { -1,  0, -1 },
            { -1, -1, -1 },
        };
        Move move = new(0, 1, 0);
        Debug.Assert(game.GetWinner() == Game.NOT_OVER);
        Debug.Assert(game.IsGameOver() == false);
        Debug.Assert(game.IsValid(move) == false);
        move = new(2, 1, 0);
        Debug.Assert(game.IsValid(move) == true);
        var reward = game.GetHeuristics();
        Debug.Assert(reward[0] == 0.4f && reward[1] == 0.1f);
        Debug.Assert(game.Play(move) == false);
        Debug.Assert(game.GetWinner() == 0);
        reward = game.GetHeuristics();
        Debug.Assert(reward[0] == 1f && reward[1] == 0f);

        game.grid = new int[,] {
            {  1,  0, -1 },
            { -1,  1, -1 },
            { -1, -1,  1 },
        };
        Debug.Assert(game.GetWinner() == 1);
        Debug.Assert(game.IsGameOver() == true);
        reward = game.GetHeuristics();
        Debug.Assert(reward[0] == 0f && reward[1] == 1f);

        game.grid = new int[,] {
            { 1, 0, 1 },
            { 1, 1, 0 },
            { 0, 1, 0 },
        };
        Debug.Assert(game.GetWinner() == Game.TIE);
        Debug.Assert(game.IsGameOver() == true);
        reward = game.GetHeuristics();
        Debug.Assert(reward[0] == 0.5f && reward[1] == 0.5f);
    }

    // public override bool IsTerminal() => GetWinner() != NOT_OVER;
    public int GetWinner()
    {
        int row, col, p;
        int Count(int rowDir, int colDir)
        {
            for (int i = 1; i <= NUM_SET; i++)
            {
                int r = row + i * rowDir;
                int c = col + i * colDir;
                if (
                    r < 0
                    || c < 0
                    || r >= ROWS
                    || c >= COLS
                    || grid[r, c] != p
                )
                    return i - 1;
            }
            return NUM_SET;
        }

        int numMarks = 0;
        for (row = 0; row < ROWS; row++)
            for (col = 0; col < COLS; col++)
            {
                for (p = 0; p < numPlayers; p++)
                    if (grid[row, col] == p && (
                            Count(1, 0) + Count(-1, 0) + 1 >= NUM_SET
                            || Count(0, 1) + Count(0, -1) + 1 >= NUM_SET
                            || Count(1, 1) + Count(-1, -1) + 1 >= NUM_SET
                            || Count(1, -1) + Count(-1, 1) + 1 >= NUM_SET)
                            )
                        return p;
                if (grid[row, col] != EMPTY) numMarks++;
            }

        return (numMarks < ROWS * COLS) ? NOT_OVER : TIE;
    }
    public override bool IsValid(Move action) => grid[action.i, action.j] == EMPTY;

    public override bool Play(Move action)
    {
        grid[action.i, action.j] = action.activePlayer;
        activePlayer = (activePlayer + 1) % numPlayers;
        step++;
        return false;
    }
    public override List<Move> GetPossibleActions(bool sort = false)
    {
        List<Move> actions = new();
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                if (grid[i, j] == EMPTY)
                    actions.Add(new(i, j, activePlayer));
        return actions;
    }
    public override Move GetRandomMove()
    {
        List<Move> actions = GetPossibleActions();
        return actions[RandomSeed.Next(actions.Count)];
    }
    public override Move GetGreedyMove()
    {
        var actions = GameUtils.GameMath.Shuffled<Move>(RandomSeed, GetPossibleActions());
        int bestIdx = 0;
        float scoreBest = float.MinValue;
        for (int i = 0; i < actions.Count; i++)
        {
            Game g = DeepCopier.Copy(this);
            g.Play(actions[i]);
            var score = (g.IsGameOver() ? g.GetRewards() : g.GetHeuristics())[activePlayer];
            if (score > scoreBest)
            {
                bestIdx = i;
                scoreBest = score;
            }
        }
        return actions[bestIdx];
    }


    public override float[] GetRewards()
    {
        int winnerIdx = GetWinner();
        Debug.Assert(winnerIdx != NOT_OVER, "GetReward should be called on game over");

        if (winnerIdx == TIE)
            return Enumerable.Repeat(1f / numPlayers, numPlayers).ToArray();
        else
        {
            float[] reward = Enumerable.Repeat(0f, numPlayers).ToArray();
            reward[winnerIdx] = 1f;
            return reward;
        }
    }
    public override float[] GetScores() => GetRewards();

    public override float[] GetHeuristics()
    {   // loss = 0, tie = 1/numPlayers, win = 1
        // if (IsGameOver())
        //     return GetRewards();

        // return Enumerable.Repeat(1f / numPlayers, numPlayers).ToArray();

        int row, col, p;
        int Count(int rowDir, int colDir)
        {
            for (int i = 1; i <= NUM_SET; i++)
            {
                int r = row + i * rowDir;
                int c = col + i * colDir;
                if (
                    r < 0
                    || c < 0
                    || r >= ROWS
                    || c >= COLS
                    || (grid[r, c] != p && grid[r, c] != EMPTY)
                )
                    return i - 1;
            }
            return NUM_SET;
        }

        float v = 1.0f / (ROWS * COLS * ROWS * COLS);
        var reward = new float[numPlayers];
        for (row = 0; row < ROWS; row++)
            for (col = 0; col < COLS; col++)
                for (p = 0; p < numPlayers; p++)
                    if (grid[row, col] == p)
                        reward[p] += (
                            ((Count(0, 1) + Count(0, -1) + 1) >= NUM_SET ? v : 0.0f) +
                            ((Count(1, 0) + Count(-1, 0) + 1) >= NUM_SET ? v : 0.0f) +
                            ((Count(1, 1) + Count(-1, -1) + 1) >= NUM_SET ? v : 0.0f) +
                            ((Count(-1, 1) + Count(1, -1) + 1) >= NUM_SET ? v : 0.0f));

        return reward;
    }

    public override string ToString()
    {
        string s;
        s = IsGameOver() ? $"Game over, winner={GetWinner()}\n" : "";
        s += $"Player={activePlayer}/{numPlayers}, step={step}\n";

        for (int row = 0; row < ROWS; row++)
        {
            s += $"{row}   ";
            for (int col = 0; col < COLS; col++)
            {
                s += $"{(grid[row, col] == EMPTY ? " " : grid[row, col])}";
                s += col < COLS - 1 ? " | " : " \n   ";
            }
            if (row < ROWS - 1)
                for (int col = 0; col < COLS; col++)
                {
                    s += col < COLS - 1 ? "---+" : "---\n";
                }
        }
        return s;
    }

    public Move GetUserMove()
    {
        Console.WriteLine(this);
        Console.WriteLine($"Player {activePlayer}, enter your move (row, col): ");
        string[] input = Console.ReadLine().Split(' ');
        int i = int.Parse(input?[0]);
        int j = int.Parse(input?[1]);
        return new(i, j, activePlayer);
    }



}
