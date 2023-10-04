namespace Azul;
using System.Diagnostics;
using System.Globalization;
using System.Reflection.Metadata;
using DeepCopy;
using GameMath = GameUtils.GameMath;
using System.Diagnostics;


public class Game : GameUtils.Game<Move>
{
    public int numFactories;  // not counting center (factories[-1])
    public int roundIdx = 0;
    // public int step = 0;  // player steps
    const int MAX_STEPS = 1_000;  // max number of game steps before stalemate
    const int NUM_COLORS = 5;
    const int ROWS = 5;
    const int COLS = 5;
    public Player[] players;
    // center == factories[0]
    public int[][] factories;  // factoryIdx, colorIdx: numTiles
    public int[] bag = new int[NUM_COLORS];  // numTiles per color
    public int[] discarded = new int[NUM_COLORS];  // numTiles per color
    public int[] factoryIdxArray;
    static int[] rowIdxArray = new int[ROWS + 1];
    static int[] colorIdxArray = new int[NUM_COLORS];
    public int CENTER;

    // public Game DeepCopy() 
    // {
    //     // var game = new Game(numPlayers);
    //     // efficiently copy object into new object

    // }

    public Game(int numPlayers)
    {
        this.numPlayers = numPlayers;
        numFactories = FactoriesVsPlayer(numPlayers);
        CENTER = numFactories;
        players = new Player[numPlayers];

        factories = new int[numFactories + 1][];
        for (int i = 0; i < numFactories; i++)
            factories[i] = new int[Constants.numColors];
        factories[CENTER] = new int[Constants.numColors + 1];

        for (int i = 0; i < numPlayers; i++)
            players[i] = new Player();

        // initialize bag
        for (int i = 0; i < Constants.numColors; i++)
            bag[i] = Constants.totalTilesPerColor;

        FillFactories();
        chanceHash = 0;  // root is never random

        // intialize randomized idx arrays
        factoryIdxArray = new int[numFactories + 1];  // update size
        for (int i = 0; i < numFactories + 1; i++)
            factoryIdxArray[i] = i;
        for (int i = 0; i < Constants.numRows + 1; i++)
            rowIdxArray[i] = i;
        for (int i = 0; i < Constants.numColors; i++)
            colorIdxArray[i] = i;
    }

    public Game() : this(2) { }

    public override Game Reset(int numPlayers) => new Game(numPlayers);

    public static Move DefaultPolicy(Game g)
    {
        GameMath.Shuffle<int>(rng, g.factoryIdxArray);
        GameMath.Shuffle<int>(rng, Game.colorIdxArray);
        GameMath.Shuffle<int>(rng, Game.rowIdxArray);

        var actionBest = new Move();
        int searchTries = 15;
        int scoreMax = -1;
        foreach (var factoryIdx in g.factoryIdxArray)
            foreach (var color in colorIdxArray)
                foreach (var row in rowIdxArray)
                    if (g.IsValid(factoryIdx, color, row))
                    {
                        int numTiles = g.factories[factoryIdx][color];
                        bool isFirst = factoryIdx == g.CENTER && g.factories[g.CENTER][(int)Tiles.FIRST_MOVE] > 0;

                        Player p = new(g.players[g.activePlayer]);
                        var a = new Move(factoryIdx, color, row, numTiles, isFirst, g.activePlayer);

                        p.Play(a);
                        p.UpdateRound();
                        p.UpdateGame();

                        if (p.score > scoreMax)
                        {
                            scoreMax = p.score;
                            actionBest = a;
                        }
                        searchTries--;
                        if (searchTries < 0 && scoreMax > 0)
                            return actionBest;
                    }
        return actionBest;
    }

    public override float[] GetRewards()
    {
        // assumes game is over 
        float[] scores = new float[numPlayers];
        float scoreMax = -1f;
        int numTies = 0;
        for (int i = 0; i < numPlayers; i++)
        {
            scores[i] = (float)players[i].score;
            if (scores[i] > scoreMax)
            {
                scoreMax = scores[i];
                numTies = 1;
            }
            else if (scores[i] == scoreMax)
                numTies++;
        }
        for (int i = 0; i < numPlayers; i++)
            scores[i] = (scores[i] == scoreMax) ? 1f / numTies : 0f;
        return scores;
    }

    public override float[] GetHeuristics()
    {
        // if (IsGameOver()) return GetRewards();
        float scoreMax = -100f;
        int numTies = 0;
        float[] scores = new float[numPlayers];
        for (int i = 0; i < numPlayers; i++)
        {
            Player p = new(players[i]);
            p.UpdateRound();
            p.UpdateGame();
            scores[i] = (float)p.score;
            if (scores[i] > scoreMax)
            {
                scoreMax = scores[i];
                numTies = 1;
            }
            else if (scores[i] == scoreMax)
                numTies++;
        }
        float timeGain = step < 100 ? 1f / (100 - step) : 1f;
        for (int i = 0; i < numPlayers; i++)
            scores[i] = (scores[i] == scoreMax) ? timeGain / numTies : 0f;
        return scores;
    }

    public int Heuristic(Move m)
    {
        Player p = new(players[m.playerIdx]);
        p.Play(m);
        p.UpdateRound();
        p.UpdateGame();
        return p.score;
    }

    public int Heuristic2(Player p, Move m)
    {
        // player p already moved completed lines to grid
        // var p = players[m.playerIdx];
        int row = m.row;
        int col = Player.ColorToCol(m.row, m.color);
        int score = 0;

        int numFloor = 0;
        for (int i = 0; i < Constants.numColors + 1; i++)
            numFloor += p.floor[i];

        // tiles left to fill row (0: complete, +: overfilled, -: not filled)
        if (row == (int)Rows.FLOOR)
            numFloor += m.numTiles;
        else
        {
            int tileFill = p.line[row, m.color] + m.numTiles - row - 1;
            if (tileFill >= 0)
            {
                score += p.ScoreTile(row, col);
                numFloor += tileFill;
            }
            // add points for each column
            bool countCol = true, countRow = true, colorCount = true;
            for (int i = 0; i < Constants.numColors; i++)
            {
                if (p.grid[i, col] == 0) countCol = false;
                if (p.grid[row, i] == 0) countRow = false;
                if (p.grid[i, Player.ColorToCol(i, m.color)] == 0) colorCount = false;
            }
            if (countCol)
                score += Constants.pointsPerColumn;
            if (countRow)
                score += Constants.pointsPerRow;
            if (colorCount)
                score += Constants.pointsPerColor;
        }
        score += Player.FloorToScore(numFloor);

        return score;
    }

    // e==0: greedy, e==1: random
    public override Move GetGreedyMove2()
    {
        var actionBest = new Move();
        int scoreMax = -1;
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            for (int color = 0; color < Constants.numColors; color++)
                for (int row = 0; row < Constants.numRows + 1; row++)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
                        var a = new Move(factoryIdx, color, row, numTiles, isFirst, activePlayer);

                        var score = Heuristic(a);
                        if (score > scoreMax)
                        {
                            scoreMax = score;
                            actionBest = a;
                        }
                    }
        if (scoreMax == -1 && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
            return new Move(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, 1, true, activePlayer);

        Debug.Assert(scoreMax > -1, "No valid action found");
        return actionBest;
    }

    public override Move GetGreedyMove()
    {
        var actionBest = new Move();
        int scoreMax = -1;
        var p = new Player(players[activePlayer]);
        p.UpdateRound();
        int scoreBaseline = Heuristic2(p, new Move(0, 0, (int)Rows.FLOOR, 0, false, activePlayer));

        int nFloorTiles = 0;
        for (int i = 0; i < Constants.numColors + 1; i++)
            nFloorTiles += players[activePlayer].floor[i];

        for (int row = 0; row < Constants.numRows + 1; row++)
            for (int color = 0; color < Constants.numColors; color++)
            {
                int tileFill = row == (int)Rows.FLOOR ? 0 : row + 1 - players[activePlayer].line[row, color];  // required tiles to fill row
                int scoreComplete = 0;
                bool isCalculated = false;
                for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;

                        // calculate score if row/color is complete: 
                        if (isCalculated == false)
                        {
                            scoreComplete = Heuristic2(p, new Move(factoryIdx, color, row, tileFill, isFirst, activePlayer));
                            isCalculated = true;
                        }

                        int score;
                        if (numTiles < tileFill)
                            score = scoreBaseline;
                        else if (numTiles == tileFill)
                            score = scoreComplete;
                        else  // subtract extra floor tiles
                        {
                            score = scoreComplete + Player.FloorToScore(nFloorTiles + numTiles - tileFill) - Player.FloorToScore(nFloorTiles);
                            if (score < 0) score = 0;
                        }

                        if (score > scoreMax)
                        {
                            scoreMax = score;
                            actionBest = new Move(factoryIdx, color, row, numTiles, isFirst, activePlayer);
                        }
                    }
            }
        if (scoreMax == -1 && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
            return new Move(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, 1, true, activePlayer);

        Debug.Assert(scoreMax > -1, "No valid action found");
        return actionBest;
    }

    public UInt64 GetHash()
    {
        // hash of the factory state in the full state
        UInt64 hash = 0;
        UInt64 iPow = 1;
        for (int f = 0; f < numFactories; f++)
            for (int color = 0; color < Constants.numColors; color++)
                for (int i = 0; i < factories[f][color]; i++)
                {
                    hash += (UInt64)color * iPow;
                    iPow *= Constants.numTilesPerFactory;
                }
        return hash;
    }

    public override bool IsEqual(GameUtils.Game<Move> other)
    {
        var a = other as Game;
        // var a = this;
        if (a.step != step ||
            a.activePlayer != activePlayer ||
            a.roundIdx != roundIdx ||
            a.numPlayers != numPlayers ||
            a.numFactories != numFactories)
            return false;

        for (int i = 0; i < a.numFactories + 1; i++)
            for (int j = 0; j < Constants.numColors; j++)
                if (a.factories[i][j] != factories[i][j])
                    return false;
        if (a.factories[a.numFactories][(int)Tiles.FIRST_MOVE] !=
            factories[numFactories][(int)Tiles.FIRST_MOVE])
            return false;  // compare first token on factory

        for (int i = 0; i < a.numPlayers; i++)
        {
            if (a.players[i].score != players[i].score)
                return false;
            for (int row = 0; row < Constants.numRows; row++)
                for (int col = 0; col < Constants.numCols; col++)
                    if (a.players[i].grid[row, col] != players[i].grid[row, col] ||
                        a.players[i].line[row, col] != players[i].line[row, col])
                        return false;
            for (int j = 0; j < Constants.numColors + 1; j++)
                if (a.players[i].floor[j] != players[i].floor[j])
                    return false;
        }
        return true;
    }

    public static Game GenerateLastMoveGame()
    {
        Game game = new(2);
        game.factories = new int[][]{
            new int[]{ 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0 },
            new int[]{ 1, 0, 0, 0, 0, 0 },
        };
        game.bag = new int[] { 18, 2, 0, 0, 0 };
        game.discarded = new int[] { 26, 2, 0, 0, 0 };

        // last round move. player 0 has 2 options
        // 1) move with high reward now, but much worse  reward later
        // 2) move with low  reward now, but much better reward later

        // MCTS stochastic can estimate the expected reward better
        // MCTS baseline will assume first roll always happen

        game.players[0].score = 100;
        game.players[0].line = new int[,] {
            {1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0},};
        game.players[0].grid = new int[,] {
            { 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0},
            { 1, 1, 1, 1, 0},};
        game.players[0].floor = new int[] { 0, 0, 0, 0, 0, 1 };
        // for (int i = 0; i < Constants.numColors; i++)
        //     game.players[0].grid[4, i] = 1;

        game.players[1].score = 100;
        game.players[1].floor = new int[] { 0, 0, 0, 0, 0, 0 };

        return game;
    }

    public static Game LoadCustomGame()
    {
        var game = new Game(2);
        game.bag = new int[] { 0, 0, 0, 0, 0, };
        game.discarded = new int[] { 9, 8, 10, 8, 12, };
        game.factories = new int[][]{
            new int[]{ 0, 0, 1, 1, 2, },
            new int[]{ 0, 2, 1, 0, 1, },
            new int[]{ 2, 0, 1, 0, 1, },
            new int[]{ 0, 2, 0, 2, 0, },
            new int[]{ 0, 1, 2, 1, 0, },
            new int[]{ 0, 0, 0, 0, 0, 1 },
        };
        game.players = new Player[]{
            new Player(){
                score = 27,
                line = new int[,] {
                    { 0, 0, 0, 0, 0, },
                    { 0, 0, 0, 0, 0, },
                    { 2, 0, 0, 0, 0, },
                    { 0, 0, 0, 0, 0, },
                    { 0, 0, 0, 3, 0, },
                },
                grid = new int[,] {
                    { 1, 1, 1, 1, 0, },
                    { 1, 1, 1, 0, 0, },
                    { 0, 1, 0, 1, 0, },
                    { 1, 1, 1, 0, 0, },
                    { 1, 0, 0, 0, 0, },
                },
                floor = new int[] { 0, 0, 0, 0, 0, 0, },
            },
            new Player(){
                score = 29,
                line = new int[,] {
                    { 0, 0, 0, 0, 0, },
                    { 0, 0, 0, 0, 0, },
                    { 0, 1, 0, 0, 0, },
                    { 2, 0, 0, 0, 0, },
                    { 0, 0, 0, 0, 0, },
                },
                grid = new int[,] {
                    { 1, 1, 1, 1, 0, },
                    { 0, 1, 1, 1, 1, },
                    { 0, 1, 1, 0, 0, },
                    { 0, 1, 0, 0, 0, },
                    { 0, 1, 0, 0, 0, },
                },
                floor = new int[] { 0, 0, 0, 0, 0, 0, },
            },
        }; return game;
    }


    public static string ToScript(Game g)
    {
        string str = $"Game game = new({g.numPlayers});\n" +
        "game.bag = new int[] { ";
        for (int i = 0; i < Constants.numColors; i++)
            str += $"{g.bag[i]}, ";
        str += "};\n" + "game.discarded = new int[] { ";
        for (int i = 0; i < Constants.numColors; i++)
            str += $"{g.discarded[i]}, ";
        str += "};\n" + "game.factories = new int[][]{\n";
        for (int i = 0; i < g.numFactories + 1; i++)
        {
            str += "    new int[]{ ";
            for (int j = 0; j < Constants.numColors; j++)
                str += $"{g.factories[i][j]}, ";
            if (i == g.numFactories)
                str += $"{g.factories[i][(int)Tiles.FIRST_MOVE]}";
            str += "},\n";
        }
        str += "};\n" + "game.players = new Player[]{\n";
        for (int i = 0; i < g.numPlayers; i++)
        {
            str += "    new Player(){\n";
            str += $"        score = {g.players[i].score},\n";
            str += "        line = new int[,] {\n";
            for (int row = 0; row < Constants.numRows; row++)
            {
                str += "            { ";
                for (int col = 0; col < Constants.numCols; col++)
                    str += $"{g.players[i].line[row, col]}, ";
                str += "},\n";
            }
            str += "        },\n";
            str += "        grid = new int[,] {\n";
            for (int row = 0; row < Constants.numRows; row++)
            {
                str += "            { ";
                for (int col = 0; col < Constants.numCols; col++)
                    str += $"{g.players[i].grid[row, col]}, ";
                str += "},\n";
            }
            str += "        },\n";
            str += "        floor = new int[] { ";
            for (int j = 0; j < Constants.numColors + 1; j++)
                str += $"{g.players[i].floor[j]}, ";
            str += "},\n";
            str += "    },\n";
        }
        str += "};\n";

        return str;
    }


    public static Game GenerateEndedMoveGame()
    {
        Game game = new(2);
        game.bag = new int[] { 0, 0, 0, 0, 0 };
        game.discarded = new int[] { 15, 15, 15, 15, 15 };
        game.factories = new int[][]{
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
        };


        // game.players[0].line[1] = new int[] { 0, 0, 0, 0, 0 };
        // game.players[0].score = 100;
        game.players[0].grid[1, 0] = 1;
        game.players[0].floor[(int)Tiles.FIRST_MOVE] = 1;
        for (int i = 0; i < Constants.numColors; i++)
            game.players[0].grid[4, i] = 1;

        return game;
    }

    public int[] CountTotalTiles()
    {
        int[] total = new int[Constants.numColors];
        // from factories
        for (int i = 0; i < numFactories + 1; i++)
            for (int j = 0; j < Constants.numColors; j++)
                total[j] += factories[i][j];
        // from bag
        for (int i = 0; i < Constants.numColors; i++)
            total[i] += bag[i];
        // from discard
        for (int i = 0; i < Constants.numColors; i++)
            total[i] += discarded[i];
        // from players
        foreach (var player in players)
        {
            int[] ptotal = player.CountTotalTiles();
            for (int color = 0; color < Constants.numColors; color++)
                total[color] += ptotal[color];
        }
        return total;
    }

    public override string ToString()
    {
        // string str = $"Round: {roundIdx}-{step}\tActive player: {activePlayer}\tBag: ";
        string str = $"Round: {roundIdx}-{step}\t\tBag: ";
        // print bag
        for (int i = 0; i < Constants.numColors; i++)
            str += $"{bag[i]} ";
        // print discarded
        str += "\t\tDiscarded: ";
        for (int i = 0; i < Constants.numColors; i++)
            str += $"{discarded[i]} ";
        // print factories
        List<int>[] tiles = new List<int>[numFactories];
        for (int factoryIdx = 0; factoryIdx < numFactories; factoryIdx++)
        {
            tiles[factoryIdx] = new List<int>();
            for (int color = 0; color < Constants.numColors; color++)
                for (int i = 0; i < factories[factoryIdx][color]; i++)
                    tiles[factoryIdx].Add(color);
        }
        str += "\n";
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            str += $"{factoryIdx})       ";
        for (int row = 0; row < 2; row++)
        {
            str += "\n  ";
            for (int factoryIdx = 0; factoryIdx < numFactories; factoryIdx++)
            {
                // str += row == 0 ? $"{factoryIdx}) " : "   ";
                str += tiles[factoryIdx].Count > 0 ? $"{tiles[factoryIdx][row * 2]} {tiles[factoryIdx][row * 2 + 1]}      " : "         ";
            }

            if (row == 0)
            {
                str += (factories[CENTER][(int)Tiles.FIRST_MOVE] > 0 ? $"{Constants.numColors} " : "  ");
                for (int color = 0; color < Constants.numColors; color++)
                {
                    for (int i = 0; i < factories[CENTER][color]; i++)
                        str += $"{color} ";
                    str += " ";
                }
            }
        }

        // print players
        for (int i = 0; i < numPlayers; i++)
        {
            str += i == activePlayer ? $"\n=> Player {i} " : $"\n   Player {i} ";
            str += players[i].ToString();
        }
        return str;
    }

    public override List<Move> GetPossibleActions()
    {
        List<Move> actions = new();
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            for (int color = 0; color < Constants.numColors; color++)
                for (int row = 0; row < Constants.numRows + 1; row++)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
                        actions.Add(new(factoryIdx, color, row, numTiles, isFirst, activePlayer));
                    }
        if (factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
            actions.Add(new(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, 1, true, activePlayer));

        actions = actions.OrderByDescending(o =>
        {
            if (o.row == (int)Rows.FLOOR) return o.numTiles;
            var drops = o.numTiles + players[o.playerIdx].line[o.row, o.color] - o.row;
            return drops < 0 ? -drops : drops;
        }).ToList();

        return actions;
    }

    public override Move GetRandomMove()
    {
        GameMath.Shuffle<int>(rng, factoryIdxArray);
        GameMath.Shuffle<int>(rng, colorIdxArray);
        GameMath.Shuffle<int>(rng, rowIdxArray);

        foreach (var factoryIdx in factoryIdxArray)
            foreach (var color in colorIdxArray)
                foreach (var row in rowIdxArray)
                    // for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
                    //     for (int color = 0; color < Constants.numColors; color++)
                    //         for (int row = 0; row < Constants.numRows + 1; row++)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
                        return new(factoryIdx, color, row, numTiles, isFirst, activePlayer);
                    }
        if (factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
            return new(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, 1, true, activePlayer);

        Debug.Assert(false, "No valid action found");
        return null;
    }

    public void FillFactories()
    {
        // add first player tile to center
        factories[CENTER][(int)Tiles.FIRST_MOVE] = 1;

        // fill factories with random tiles from bag
        int tilesLeft = bag.Sum();
        bool isDone = false;
        for (int factoryIdx = 0; factoryIdx < numFactories; factoryIdx++)
        {
            for (int i = 0; i < Constants.numTilesPerFactory; i++)
            {
                // transfer discard to bag if bag is incomplete
                if (tilesLeft == 0)
                {
                    tilesLeft += discarded.Sum();
                    for (int j = 0; j < Constants.numColors; j++)
                    {
                        bag[j] += discarded[j];
                        discarded[j] = 0;
                    }
                    if (tilesLeft == 0)  // dicarded bag was empty, leave factories incomplete
                    {
                        isDone = true;
                        break;
                    }
                }
                int color = GameMath.SampleWeightedDiscrete(rng, bag);
                bag[color]--;
                factories[factoryIdx][color]++;
                tilesLeft--;
            }
            if (isDone) break;
        }

        // sort factories
        int[] keys = new int[numFactories + 1];
        for (int i = 0; i < numFactories + 1; i++)
        {
            int jPow = 1;
            for (int j = 0; j < Constants.numColors; j++)
            {
                keys[i] -= factories[i][j] * jPow;
                jPow *= Constants.numTilesPerFactory;
            }
        }
        Array.Sort(keys, factories);

        // update hash
        // identifies the random unique state of the factories
        // chanceHash = roundIdx == 0 ? 0 : GetHash();
        chanceHash = GetHash();
    }

    public override bool Play(Move action)
    {   // returns true if play creates random outcome
        // get number of tiles in factory
        ref var factory = ref factories[action.factoryIdx];
        bool isCenter = action.factoryIdx == numFactories;
        ref int color = ref action.color;
        ref int row = ref action.row;
        ref int numTiles = ref action.numTiles;
        ref bool isFirst = ref action.isFirst;
        ref var player = ref players[action.playerIdx];

        // Debug.Assert(IsValid(action.factoryIdx, action.color, action.row), "Invalid action");
        // Debug.Assert(false, "Invalid action");

        step++;
        chanceHash = 0;

        // update player -------------------
        player.Play(action);  // added tiles here            

        // update factories ----------------
        factory[color] = 0;  // reset factory
        if (isCenter == false)  // move other tiles to center
            for (int i = 0; i < Constants.numColors; i++)
            {
                factories[CENTER][i] += factory[i];
                factory[i] = 0;
            }

        if (isFirst)
            factory[(int)Tiles.FIRST_MOVE] = 0;

        // update active player 
        activePlayer = (activePlayer + 1) % numPlayers;

        if (IsFactoryEmpty())
        {
            roundIdx++;
            // for each player, update score
            for (int i = 0; i < numPlayers; i++)
            {
                int[] discard = players[i].UpdateRound();
                for (int j = 0; j < Constants.numColors; j++)
                    discarded[j] += discard[j];

                // update active player
                if (discard[(int)Tiles.FIRST_MOVE] > 0)
                    activePlayer = i;
            }

            if (IsGameOver())  // update player scores
            {
                for (int i = 0; i < numPlayers; i++)
                    players[i].UpdateGame();
                return false;
            }
            else  // reset factories with random tiles
                FillFactories();

            return true;
        }
        return false;
    }

    // public bool IsTerminal() => IsRoundOver();

    bool IsFactoryEmpty()
    {
        // check if factories are empty
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            for (int color = 0; color < Constants.numColors; color++)
                if (factories[factoryIdx][color] > 0)
                    return false;
        return factories[CENTER][(int)Tiles.FIRST_MOVE] == 0;
    }

    public override bool IsGameOver()
    {
        if (step > MAX_STEPS)
        {
            Debug.Write("Game ended in stalemate");
            return true;
        }
        // check if any player has a full row
        if (IsFactoryEmpty() == false)
            return false;

        foreach (Player player in players)
            for (int row = 0; row < Constants.numRows; row++)
            {
                int colorCount = 0;
                for (int col = 0; col < Constants.numColors; col++)
                    colorCount += (int)player.grid[row, col];
                if (colorCount >= Constants.numColors)
                    return true;
            }
        return false;
    }

    public bool IsValid(int factoryIdx, int color, int row)
    {
        ref var factory = ref factories[factoryIdx];
        ref var player = ref players[activePlayer];
        // int color = T.color;
        // int row = T.row;

        // check if factory is empty
        if (color == (int)Tiles.FIRST_MOVE && factoryIdx != CENTER)
            return false;
        if (factory[color] <= 0)
            return false;
        if (row == (int)Rows.FLOOR)  // to floor
            return true;
        if (color == (int)Tiles.FIRST_MOVE)  // must go to floor
            return false;
        // check if line already has any another color 
        for (int i = 0; i < Constants.numColors; i++)
            if ((i != color) && (player.line[row, i] > 0))
                return false;
        // check if row is full
        if (player.line[row, color] > row)
            return false;
        // check if grid is full
        if (player.grid[row, Player.ColorToCol(row, color)] > 0)
            return false;
        return true;
    }
    public override bool IsValid(Move action) => IsValid(action.factoryIdx, action.color, action.row);

    static int FactoriesVsPlayer(int numPlayers)
    {
        switch (numPlayers)
        {
            case 2: return 5;
            case 3: return 7;
            default: return 9;
        }
    }

    public Move GetUserMove()
    {
        Console.WriteLine(this);
        Console.WriteLine($"Player {activePlayer}, enter your move (factoryIdx, color, row): ");
        string input = Console.ReadLine();
        string[] tokens = input.Split(' ');
        int factoryIdx = int.Parse(tokens[0]);
        int color = int.Parse(tokens[1]);
        int row = int.Parse(tokens[2]);
        return new(factoryIdx, color, row, this);
    }
}

public class Move
{
    public int factoryIdx = 0;
    public int color = 0;
    public int row = 0;
    // non-essential fields --------------
    public int numTiles = 0;
    public bool isFirst = false;
    public int playerIdx = 0;

    public Move(int factoryIdx, int color, int row,
        int numTiles, bool isFirst, int playerIdx)
    {
        this.factoryIdx = factoryIdx;
        this.color = color;
        this.row = row;
        this.numTiles = numTiles;
        this.isFirst = isFirst;
        this.playerIdx = playerIdx;
    }

    public Move(int factoryIdx, int color, int row, Game state)
    {
        this.factoryIdx = factoryIdx;
        this.color = color;
        this.row = row;
        this.numTiles = state.factories[factoryIdx][color];
        this.isFirst = factoryIdx == (int)state.CENTER && state.factories[factoryIdx][(int)Tiles.FIRST_MOVE] > 0;
        this.playerIdx = state.activePlayer;
    }

    public Move() { }

    public override string ToString()
    {
        return $"Player {playerIdx}) Factory={factoryIdx}, " +
            $"Color={Enum.GetName(typeof(Tiles), color)}:{color}, Row={Enum.GetName(typeof(Rows), row)}, " +
            $"NumTiles={numTiles}, IsFirst={isFirst}";
    }
}

public class Player
{
    public int score = 0;
    // q: how to define a int matrix?
    public int[,] grid = new int[Constants.numColors, Constants.numColors];
    public int[,] line = new int[Constants.numRows, Constants.numColors];
    public int[] floor = new int[Constants.numColors + 1]; // first = NONE = 5


    public Player()
    { }

    public Player(Player player)
    {
        score = player.score;
        grid = (int[,])player.grid.Clone();
        line = (int[,])player.line.Clone();
        floor = (int[])player.floor.Clone();
    }

    public bool SanityCheck()  // true if valid state
    {
        // check if grid is full
        for (int row = 0; row < Constants.numRows; row++)
            for (int col = 0; col < Constants.numCols; col++)
            {
                if ((grid[row, col] < 0) || (grid[row, col] > 1))
                    return false;
                // pretend col is color
                if (grid[row, ColorToCol(row, col)] > 0 && line[row, col] > 0)
                    return false;
            }
        // check if line is full
        for (int row = 0; row < Constants.numRows; row++)
        {
            bool isFull = false;
            for (int col = 0; col < Constants.numCols; col++)
                if (line[row, col] > 1)
                {
                    if (isFull) return false;  // more than one color
                    isFull = true;
                }
        }
        // check if floor is full
        foreach (int v in floor)
            if (v < 0) return false;

        if (floor[(int)Tiles.FIRST_MOVE] > 1) return false;

        return true;
    }

    // convert Print() method to ToString()
    public string ToString2()
    {
        string str = $"Score: {score}\n" +
            "Grid:\n";
        for (int row = 0; row < Constants.numRows; row++)
        {
            for (int col = 0; col < Constants.numCols; col++)
                str += $"{grid[row, col]} ";
            str += "\n";
        }
        str += "Line:\n";
        for (int row = 0; row < Constants.numRows; row++)
        {
            for (int col = 0; col < Constants.numCols; col++)
                str += $"{line[row, col]} ";
            str += "\n";
        }
        str += "Floor:\n";
        for (int j = 0; j < Constants.numColors + 1; j++)
            str += $"{floor[j]} ";
        return str;
    }

    public override string ToString()
    {
        const int ROWS = 5,
                  COLS = 5,
                  NUM_COLORS = 5;
        string str = $"({score}) -----------------";// Floor: " + (floor[NUM_COLORS] > 0 ? "-1 " : "");

        for (int row = 0; row < ROWS; row++)
        {
            int color;
            for (color = 0; color < NUM_COLORS; color++)
                if (line[row, color] > 0)
                    break;
            int numTiles = color >= NUM_COLORS ? 0 : line[row, color];
            str += $"\n{row}  ";
            for (int col = 0; col < COLS; col++)
                str += (col < COLS - row - 1) ? "  " : ((col + numTiles >= COLS && numTiles > 0) ? $"{color} " : "- ");

            str += "\t";
            for (int col = 0; col < COLS; col++)
                str += grid[row, col] > 0 ? $"{ColToColor(row, col)} " : "- ";
        }
        str += $"\n{ROWS}  FLOOR: " + (floor[NUM_COLORS] > 0 ? $"{NUM_COLORS} " : "");
        for (int i = 0; i < NUM_COLORS; i++)
            for (int j = 0; j < floor[i]; j++)
                str += $"{i} ";

        // str += "\n";
        return str;
    }

    public int[] UpdateRound()
    {
        int[] discard = new int[Constants.numColors + 1];
        // transfer completed lines to grid
        for (int row = 0; row < Constants.numRows; row++)
        {
            for (int color = 0; color < Constants.numColors; color++)
                if (line[row, color] > row)
                {
                    int col = ColorToCol(row, color);
                    grid[row, col] = 1;
                    line[row, color] = 0;
                    discard[color] += row;
                    score += ScoreTile(row, col);
                    break;
                }
        }

        // subtract floor
        score += FloorToScore(floor.Sum());
        if (score < 0) score = 0;  // constraint to >= 0
        for (int i = 0; i < Constants.numColors + 1; i++)
        {
            discard[i] += floor[i];
            floor[i] = 0;
        }
        return discard;
    }

    public void UpdateGame()
    {   // update scores from grid patterns
        for (int i = 0; i < Constants.numColors; i++)
        {
            // add points for each column
            int countCol = 0, countRow = 0, colorCount = 0;
            for (int j = 0; j < Constants.numColors; j++)
            {
                countCol += (int)grid[j, i];
                countRow += (int)grid[i, j];
                colorCount += (int)grid[j, ColorToCol(j, i)];
            }
            if (countCol >= Constants.numColors)
                score += Constants.pointsPerColumn;
            if (countRow >= Constants.numColors)
                score += Constants.pointsPerRow;
            if (colorCount >= Constants.numColors)
                score += Constants.pointsPerColor;
        }
    }

    public void Play(Move action)
    {
        int color = action.color;
        int row = action.row;
        int numTiles = action.numTiles;
        bool isFirst = action.isFirst;

        if (isFirst)
            floor[(int)Tiles.FIRST_MOVE] = 1;
        if (color == (int)Tiles.FIRST_MOVE)
            return;

        if (row >= Constants.numRows)  // to floor
        {
            floor[color] += numTiles;
            return;
        }

        int dropTiles = numTiles - (row + 1 - line[row, color]);
        if (dropTiles > 0)
        {   // fill line and drop excess tiles to floor
            floor[color] += dropTiles;
            line[row, color] = row + 1;
        }
        else
            line[row, color] += numTiles;
    }

    public int[] CountTotalTiles()
    {
        int[] total = new int[Constants.numColors];
        // from floor
        for (int i = 0; i < Constants.numColors; i++)
            total[i] += floor[i];
        // from grid
        for (int row = 0; row < Constants.numRows; row++)
            for (int col = 0; col < Constants.numCols; col++)
                total[col] += (int)grid[row, ColorToCol(row, col)];
        // from line
        for (int row = 0; row < Constants.numRows; row++)
            for (int col = 0; col < Constants.numCols; col++)
                total[col] += line[row, col];
        return total;
    }

    public int ScoreTile(int row, int col)
    {
        int CountJointTiles(int rowDir, int colDir)
        {
            int count = 0;
            int rowIdx = row + rowDir;
            int colIdx = col + colDir;
            while (rowIdx >= 0 && rowIdx < Constants.numRows &&
                   colIdx >= 0 && colIdx < Constants.numColors &&
                   grid[rowIdx, colIdx] == 1)
            {
                count++;
                rowIdx += rowDir;
                colIdx += colDir;
            }
            return count;
        }

        int horzPoints = CountJointTiles(0, -1) + CountJointTiles(0, 1);
        int vertPoints = CountJointTiles(-1, 0) + CountJointTiles(1, 0);
        int points = horzPoints + vertPoints + 1;
        if (horzPoints > 0 && vertPoints > 0)
            points++;
        return points;
        // score += points;
    }

    public static int ColorToCol(int row, int color) { return (row + color) % Constants.numCols; }
    public static int ColToColor(int row, int col) { return (Constants.numCols - row + col) % Constants.numCols; }


    public static int FloorToScore(int numTiles)
    {
        switch (numTiles)
        {
            case 0: return 0;
            case 1: return -1;
            case 2: return -2;
            case 3: return -4;
            case 4: return -6;
            case 5: return -8;
            case 6: return -11;
            default: return -14;
        }
    }
}

static class Constants
{
    public const int numColors = 5;
    public const int numTiles = 20;
    public const int maxFloorLen = 7;
    public const int numTilesPerFactory = 4;
    public const int numRows = numColors;
    public const int numCols = numColors;
    public const int totalTilesPerColor = 20;
    public const int pointsPerColumn = 7;
    public const int pointsPerRow = 2;
    public const int pointsPerColor = 10;
}


enum Tiles : ushort
{
    BLUE = 0,
    YELLOW,
    RED,
    BLACK,
    WHITE,
    EMPTY = 5,
    FIRST_MOVE = 5
}

enum Rows : int
{
    SINGLE = 0,
    TWOS,
    THREES,
    FOURS,
    FIVES,
    FLOOR = 5,
}
