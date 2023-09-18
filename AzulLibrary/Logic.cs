namespace Azul;
using System.Diagnostics;
using System.Globalization;
using System.Reflection.Metadata;
using Utils;

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

public class Game
{
    public static Random rng = new();
    public int numPlayers;
    public int numFactories;  // not counting center (factories[-1])
    public int activePlayer = 0;
    public int roundIdx = 0;
    public int step = 0;
    const int MAX_STEPS = 1_000;
    public Player[] players;
    // center == factories[0]
    public int[][] factories;  // factoryIdx, colorIdx: numTiles
    public int[] bag = new int[Constants.numColors];  // numTiles per color
    public int[] discarded = new int[Constants.numColors];  // numTiles per color
    int[] factoryIdxArray;
    int[] rowIdxArray;
    int[] colorIdxArray;
    UInt64 factoryHash = 0;
    public int CENTER;

    public Game(int numPlayers_ = 2)
    {
        numPlayers = numPlayers_;
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

        // intialize randomized idx arrays
        factoryIdxArray = new int[numFactories + 1];
        for (int i = 0; i < numFactories + 1; i++)
            factoryIdxArray[i] = i;
        rowIdxArray = new int[Constants.numRows + 1];
        for (int i = 0; i < Constants.numRows + 1; i++)
            rowIdxArray[i] = i;
        colorIdxArray = new int[Constants.numColors + 1];
        for (int i = 0; i < Constants.numColors + 1; i++)
            colorIdxArray[i] = i;
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

    public static bool operator ==(Game a, Game b)
    {
        if (a.step != b.step ||
            a.activePlayer != b.activePlayer ||
            a.roundIdx != b.roundIdx ||
            a.numPlayers != b.numPlayers ||
            a.numFactories != b.numFactories)
            return false;

        for (int i = 0; i < a.numFactories + 1; i++)
            for (int j = 0; j < Constants.numColors; j++)
                if (a.factories[i][j] != b.factories[i][j])
                    return false;
        if (a.factories[a.numFactories][(int)Tiles.FIRST_MOVE] !=
            b.factories[b.numFactories][(int)Tiles.FIRST_MOVE])
            return false;  // compare first token on factory

        for (int i = 0; i < a.numPlayers; i++)
        {
            if (a.players[i].score != b.players[i].score)
                return false;
            for (int row = 0; row < Constants.numRows; row++)
                for (int col = 0; col < Constants.numCols; col++)
                    if (a.players[i].grid[row, col] != b.players[i].grid[row, col] ||
                        a.players[i].line[row][col] != b.players[i].line[row][col])
                        return false;
            for (int j = 0; j < Constants.numColors + 1; j++)
                if (a.players[i].floor[j] != b.players[i].floor[j])
                    return false;
        }
        return true;
    }
    public static bool operator !=(Game a, Game b) => !(a == b);

    public static Game GenerateLastMoveGame()
    {
        Game game = new(3);
        game.factories = new int[][]{
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 1, 0, 0, 0, 0, 0 },
        };
        game.bag = new int[] { 0, 0, 0, 0, 0 };
        game.discarded = new int[] { 15, 15, 15, 15, 15 };

        // game.players[0].line[1] = new int[] { 0, 0, 0, 0, 0 };

        game.players[0].score = 100;
        game.players[0].grid[1, 0] = 1;
        game.players[0].floor[(int)Tiles.FIRST_MOVE] = 1;
        game.players[0].floor = new int[] { 0, 0, 0, 0, 0, 1 };
        for (int i = 0; i < Constants.numColors; i++)
            game.players[0].grid[4, i] = 1;

        game.players[1].score = 100;
        game.players[1].floor = new int[] { 0, 0, 0, 0, 0, 0 };

        return game;
    }

    public static Game GenerateEndedMoveGame()
    {
        Game game = new(2);
        game.factories = new int[][]{
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
            new int[]{ 0, 0, 0, 0, 0, 0 },
        };
        game.bag = new int[] { 0, 0, 0, 0, 0 };
        game.discarded = new int[] { 15, 15, 15, 15, 15 };

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
            // for (int row = 0; row < Constants.numRows; row++)
            // {
            //     total[color] += (int)player.grid[row, Player.ColorToCol(row, color)];
            //     total[color] += player.line[row][color];
            // }
        }
        return total;
        // foreach (var count in total)
        //     Console.WriteLine(count);
    }
    public void Print()
    {
        Console.WriteLine($"Round: {roundIdx} ({step})");
        Console.Write("Active player: " + activePlayer);
        // print factories
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
        {
            Console.Write("\nFactory " + factoryIdx + ": ");
            for (int color = 0; color < Constants.numColors; color++)
                Console.Write(factories[factoryIdx][color] + " ");
        }
        Console.WriteLine(factories[CENTER][(int)Tiles.FIRST_MOVE]);

        // print bag
        Console.Write("Bag: ");
        for (int i = 0; i < Constants.numColors; i++)
            Console.Write(bag[i] + " ");
        Console.WriteLine();
        // print discarded
        Console.Write("Discarded: ");
        for (int i = 0; i < Constants.numColors; i++)
            Console.Write(discarded[i] + " ");
        Console.WriteLine();
        // print players
        for (int i = 0; i < numPlayers; i++)
        {
            Console.WriteLine("Player " + i + ": -----------------");
            players[i].Print();
        }
    }

    public List<GameAction> GetPossibleActions()
    {
        List<GameAction> actions = new();
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            for (int color = 0; color < Constants.numColors; color++)
                for (int row = 0; row < Constants.numRows + 1; row++)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
                        actions.Add(new GameAction(factoryIdx, color, row, numTiles, isFirst, activePlayer));
                    }
        if (factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
            actions.Add(new GameAction(CENTER, (int)Tiles.FIRST_MOVE,
                (int)Rows.FLOOR, 1, true, activePlayer));

        return actions;
    }


    public GameAction GetRandomAction()
    {
        int tries;
        for (tries = 0; tries < 100_000; tries++)
        {
            int factoryIdx = rng.Next(0, numFactories + 1);
            int color = rng.Next(0, Constants.numColors + 1);
            int row = rng.Next(0, Constants.numRows + 1);

            if (IsValid(factoryIdx, color, row))
            {
                int numTiles = factories[factoryIdx][color];
                bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
                return new GameAction(factoryIdx, color, row, numTiles, isFirst, activePlayer);
            }
        }
        Debug.Assert(true, "No valid action found");
        return null;
    }

    public GameAction GetRandomAction2()
    {
        GameUtils.Shuffle<int>(rng, factoryIdxArray);
        GameUtils.Shuffle<int>(rng, colorIdxArray);
        GameUtils.Shuffle<int>(rng, rowIdxArray);

        foreach (var factoryIdx in factoryIdxArray)
            foreach (var color in colorIdxArray)
                foreach (var row in rowIdxArray)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
                        return new GameAction(factoryIdx, color, row, numTiles, isFirst, activePlayer);
                    }
        Debug.Assert(true, "No valid action found");
        return null;
    }

    public float[] GetReward()
    {
        // assumes game is over 
        float[] scores = new float[numPlayers];
        for (int i = 0; i < numPlayers; i++)
            scores[i] = players[i].score;
        return scores;
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
                int color = GameUtils.SampleWeightedDiscrete(rng, bag);
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
        factoryHash = GetHash();
    }

    public bool Play(ref GameAction action)
    {   // returns true if play creates random outcome
        // get number of tiles in factory
        ref var factory = ref factories[action.factoryIdx];
        bool isCenter = action.factoryIdx == numFactories;
        ref int color = ref action.color;
        ref int row = ref action.row;
        ref int numTiles = ref action.numTiles;
        ref bool isFirst = ref action.isFirst;
        ref var player = ref players[action.playerIdx];

        Debug.Assert(IsValid(action.factoryIdx, action.color, action.row), "Invalid action");
        // Debug.Assert(false, "Invalid action");

        step++;

        // update player -------------------
        player.Play(ref action);  // added tiles here            

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

        if (IsRoundOver())
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

    public bool IsTerminal() => IsRoundOver();

    public bool IsRoundOver()
    {
        if (step > MAX_STEPS)  // stalemate
            return true;

        // check if factories are empty
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            for (int color = 0; color < Constants.numColors; color++)
                if (factories[factoryIdx][color] > 0)
                    return false;
        return (factories[CENTER][(int)Tiles.FIRST_MOVE] == 0);
    }

    public bool IsGameOver()
    {
        // check if any player has a full row
        if (IsRoundOver() == false)
            return false;
        if (step > MAX_STEPS)
            return true;

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
        // int color = GameAction.color;
        // int row = GameAction.row;

        // check if factory is empty
        if (color == (int)Tiles.FIRST_MOVE && factoryIdx < numFactories)
            return false;
        if (factory[color] <= 0)
            return false;
        if (row >= Constants.numRows)  // to floor
            return true;
        if (color == (int)Tiles.FIRST_MOVE)  // must go to floor
            return false;
        // check if line already has any another color 
        for (int i = 0; i < Constants.numColors; i++)
            if ((i != color) && (player.line[row][i] > 0))
                return false;
        // check if row is full
        if (player.line[row][color] > row)
            return false;
        // check if grid is full
        if (player.grid[row, Player.ColorToCol(row, color)] > 0)
            return false;
        return true;
    }
    public bool IsValid(GameAction action) => IsValid(action.factoryIdx, action.color, action.row);

    static int FactoriesVsPlayer(int numPlayers)
    {
        switch (numPlayers)
        {
            case 2: return 5;
            case 3: return 7;
            default: return 9;
        }
    }
}

public class GameAction
{
    public int factoryIdx = 0;
    public int color = 0;
    public int row = 0;
    // non-essential fields --------------
    public int numTiles = 0;
    public bool isFirst = false;
    public int playerIdx = 0;

    public GameAction(int factoryIdx_, int color_, int row_,
        int numTiles_, bool isFirst_, int playerIdx_)
    {
        factoryIdx = factoryIdx_;
        color = color_;
        row = row_;
        numTiles = numTiles_;
        isFirst = isFirst_;
        playerIdx = playerIdx_;
    }

    public GameAction(int factoryIdx_, int color_, int row_, Game state)
    {
        factoryIdx = factoryIdx_;
        color = color_;
        row = row_;
        numTiles = state.factories[factoryIdx][color];
        isFirst = factoryIdx == (int)state.CENTER && state.factories[factoryIdx][(int)Tiles.FIRST_MOVE] > 0;
        playerIdx = state.activePlayer;
    }

    public GameAction() { }

    public string Print(string prefix = "", bool toScreen = true)
    {
        string str = $"Player {playerIdx}) Factory={factoryIdx}, " +
            $"Color={Enum.GetName(typeof(Tiles), color)}:{color}, Row={Enum.GetName(typeof(Rows), row)}, " +
            $"NumTiles={numTiles}, IsFirst={isFirst}" + prefix;

        if (toScreen)
            Console.WriteLine(str);
        return str;
    }
}

public class Player
{

    public int score = 0;
    // q: how to define a int matrix?
    public int[,] grid = new int[Constants.numColors, Constants.numColors];
    public int[][] line = new int[Constants.numRows + 1][];
    public int[] floor = new int[Constants.numColors + 1]; // first = NONE = 5


    public Player()
    {
        for (int i = 0; i < Constants.numRows; i++)
            line[i] = new int[Constants.numColors];
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
                if (grid[row, ColorToCol(row, col)] > 0 && line[row][col] > 0)
                    return false;
            }
        // check if line is full
        foreach (var lineRow in line)
        {
            bool isFull = false;
            for (int col = 0; col < Constants.numCols; col++)
                if (lineRow[col] > 1)
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
    public void Print()
    {
        Console.WriteLine("Score: " + score);
        Console.WriteLine("Grid:");
        for (int row = 0; row < Constants.numRows; row++)
        {
            for (int col = 0; col < Constants.numCols; col++)
                Console.Write(grid[row, col] + " ");
            Console.WriteLine();
        }
        Console.WriteLine("Line:");
        for (int row = 0; row < Constants.numRows; row++)
        {
            for (int col = 0; col < Constants.numCols; col++)
                Console.Write(line[row][col] + " ");
            Console.WriteLine();
        }
        Console.WriteLine("Floor:");
        for (int j = 0; j < Constants.numColors + 1; j++)
            Console.Write(floor[j] + " ");
        Console.WriteLine();
    }

    public int[] UpdateRound()
    {
        int[] discard = new int[Constants.numColors + 1];
        // transfer completed lines to grid
        for (int row = 0; row < Constants.numRows; row++)
        {
            for (int color = 0; color < Constants.numColors; color++)
                if (line[row][color] > row)
                {
                    int col = ColorToCol(row, color);
                    grid[row, col] = 1;
                    line[row][color] = 0;
                    discard[color] += row;
                    ScoreTile(row, col);
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

    public void Play(ref GameAction GameAction)
    {
        int color = GameAction.color;
        int row = GameAction.row;
        int numTiles = GameAction.numTiles;
        bool isFirst = GameAction.isFirst;

        if (isFirst)
            floor[(int)Tiles.FIRST_MOVE] = 1;
        if (color == (int)Tiles.FIRST_MOVE)
            return;

        if (row >= Constants.numRows)  // to floor
        {
            floor[color] += numTiles;
            return;
        }

        int dropTiles = numTiles - (row + 1 - line[row][color]);
        if (dropTiles > 0)
        {   // fill line and drop excess tiles to floor
            floor[color] += dropTiles;
            line[row][color] = row + 1;
        }
        else
            line[row][color] += numTiles;
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
                total[col] += line[row][col];
        return total;
    }

    void ScoreTile(int row, int col)
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
        score += points;
    }

    public static int ColorToCol(int row, int color) { return (row + color) % Constants.numCols; }

    static int FloorToScore(int numTiles)
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



