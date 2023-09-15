namespace Azul;
using System.Diagnostics;
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

public class Game
{
    public int numPlayers;
    public int numFactories;  // not counting center (factories[0])
    public int activePlayer = 0;
    public int roundIdx = 0;
    int step = 0;
    public Player[] players;
    // center == factories[0]
    public int[][] factories;  // factoryIdx, colorIdx: numTiles
    public int[] bag = new int[Constants.numColors];  // numTiles per color
    public int[] discarded = new int[Constants.numColors];  // numTiles per color
    int[] factoryIdxArray;
    int[] rowIdxArray;
    int[] colorIdxArray;
    Random rng = new();

    public Game(int numPlayers_ = 2)
    {
        numPlayers = numPlayers_;
        numFactories = FactoriesVsPlayer(numPlayers);
        players = new Player[numPlayers];

        factories = new int[numFactories + 1][];
        for (int i = 0; i < numFactories + 1; i++)
            factories[i] = new int[Constants.numColors + 1];

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
            rowIdxArray[i] = i - 1;
        colorIdxArray = new int[Constants.numColors + 1];
        for (int i = 0; i < Constants.numColors + 1; i++)
            colorIdxArray[i] = i;

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
            for (int color = 0; color < Constants.numColors; color++)
            {
                total[color] += player.floor[color];
                for (int row = 0; row < Constants.numRows; row++)
                {
                    total[color] += (int)player.grid[row, Player.ColorToCol(row, color)];
                    total[color] += player.line[row][color];
                }
            }
        }
        return total;
        // foreach (var count in total)
        //     Console.WriteLine(count);
    }
    public string GetYAML()
    {
        string yaml = "";
        yaml += "numPlayers: " + numPlayers + "\n";
        yaml += "numFactories: " + numFactories + "\n";
        yaml += "activePlayer: " + activePlayer + "\n";
        yaml += "roundIdx: " + roundIdx + "\n";
        yaml += "players:\n";
        for (int i = 0; i < numPlayers; i++)
        {
            yaml += "  - " + i + ":\n";
            yaml += players[i].GetYAML();
        }
        yaml += "factories:\n";
        for (int i = 0; i < numFactories + 1; i++)
        {
            yaml += "  - " + i + ":\n";
            yaml += "    - ";
            for (int j = 0; j < Constants.numColors + 1; j++)
                yaml += factories[i][j] + " ";
            yaml += "\n";
        }
        yaml += "bag:\n";
        yaml += "  - ";
        for (int i = 0; i < Constants.numColors; i++)
            yaml += bag[i] + " ";
        yaml += "\n";
        yaml += "discarded:\n";
        yaml += "  - ";
        for (int i = 0; i < Constants.numColors; i++)
            yaml += discarded[i] + " ";
        yaml += "\n";
        return yaml;
    }

    public void Print()
    {
        Console.WriteLine("Round: " + roundIdx);
        Console.WriteLine("Active player: " + activePlayer);
        // print factories
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
        {
            Console.Write("Factory " + factoryIdx + ": ");
            for (int color = 0; color < Constants.numColors + 1; color++)
                Console.Write(factories[factoryIdx][color] + " ");
            Console.WriteLine();
        }
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
            for (int color = 0; color < Constants.numColors + 1; color++)
                for (int row = -1; row < Constants.numRows; row++)
                    if (IsValid(factoryIdx, color, row))
                    {
                        int numTiles = factories[factoryIdx][color];
                        bool isFirst = factoryIdx == 0 && factories[0][(int)Tiles.FIRST_MOVE] > 0;
                        actions.Add(new GameAction(factoryIdx, color, row, numTiles, isFirst, activePlayer));
                    }

        return actions;
    }


    public GameAction GetRandomAction2()
    {
        int tries;
        for (tries = 0; tries < 100_000; tries++)
        {
            int factoryIdx = rng.Next(0, numFactories + 1);
            int color = rng.Next(0, Constants.numColors + 1);
            int row = rng.Next(-1, Constants.numRows);

            if (IsValid(factoryIdx, color, row))
            {
                int numTiles = factories[factoryIdx][color];
                bool isFirst = factories[0][(int)Tiles.FIRST_MOVE] > 0;
                return new GameAction(factoryIdx, color, row, numTiles, isFirst, activePlayer);
            }
        }
        return null;
    }

    public GameAction GetRandomAction()
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
                        bool isFirst = factories[factoryIdx][(int)Tiles.FIRST_MOVE] > 0;
                        return new GameAction(factoryIdx, color, row, numTiles, isFirst, activePlayer);
                    }
        return null;
    }

    public float[] GetReward()
    {
        // assumes game is over 
        float[] reward = new float[numPlayers];
        float scoreMax = -1;
        int ties = 1;
        for (int i = 0; i < numPlayers; i++)
        {
            reward[i] = players[i].score;
            if (reward[i] > scoreMax)
            {
                scoreMax = reward[i];
                ties = 1;
            }
            else if (reward[i] == scoreMax)
                ties++;
        }

        for (int i = 0; i < numPlayers; i++)
        {
            if (reward[i] == scoreMax)
                reward[i] = 1.0f / ties;
            else
                reward[i] = 0.0f;
        }

        return reward;
    }


    public void FillFactories()
    {
        // transfer discard to bag if bag is incomplete
        if (bag.Sum() < Constants.numTilesPerFactory * numFactories)
            for (int i = 0; i < Constants.numColors; i++)
            {
                bag[i] += discarded[i];
                discarded[i] = 0;
            }
        int numTilesInBag = bag.Sum();
        // CountTotalTiles();

        // add first player tile to center
        factories[0][(int)Tiles.FIRST_MOVE] = 1;

        // fill factories with random tiles from bag
        for (int factoryIdx = 1; factoryIdx < numFactories + 1; factoryIdx++)
        {
            for (int i = 0; i < Constants.numTilesPerFactory; i++)
            {
                int color = GameUtils.SampleWeightedDiscrete(rng, bag);
                bag[color]--;
                factories[factoryIdx][color]++;
            }
        }
    }

    public bool Play(ref GameAction GameAction)
    {
        // get number of tiles in factory
        ref var factory = ref factories[GameAction.factoryIdx];
        bool isCenter = GameAction.factoryIdx == 0;
        ref int color = ref GameAction.color;
        ref int row = ref GameAction.row;
        ref int numTiles = ref GameAction.numTiles;
        ref bool isFirst = ref GameAction.isFirst;
        ref var player = ref players[GameAction.playerIdx];

        step++;

        // update player -------------------
        player.Play(ref GameAction);  // added tiles here            

        // update factories ----------------
        factory[color] = 0;  // reset factory
                             // move other tiles to center
        if (isCenter == false)
            for (int i = 0; i < Constants.numColors; i++)
            {
                factories[0][i] += factory[i];
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
                for (int i = 0; i < numPlayers; i++)
                    players[i].UpdateGame();
            else  // reset factories with random tiles
                FillFactories();

            return true;
        }
        return false;
    }

    public bool IsTerminal() { return IsRoundOver(); }

    bool IsRoundOver()
    {
        // check if factories are empty
        for (int factoryIdx = 0; factoryIdx < numFactories + 1; factoryIdx++)
            for (int color = 0; color < Constants.numColors + 1; color++)
                if (factories[factoryIdx][color] > 0)
                    return false;
        return true;
    }

    public bool IsGameOver()
    {   // Assumes round is over
        // check if any player has a full row
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

    bool IsValid(int factoryIdx, int color, int row)
    {
        ref var factory = ref factories[factoryIdx];
        ref var player = ref players[activePlayer];
        // int color = GameAction.color;
        // int row = GameAction.row;

        // check if factory is empty
        if (factory[color] <= 0)
            return false;
        if (row < 0)  // to floor
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

    static int FactoriesVsPlayer(int numPlayers)
    {
        switch (numPlayers)
        {
            case 2: return 5;
            case 3: return 7;
            case 4: return 9;
            default: return 0;
        }
    }
}

public class GameAction
{
    public int factoryIdx;
    public int color;
    public int row;
    // non-essential fields --------------
    public int numTiles;
    public bool isFirst;
    public int playerIdx;

    public GameAction(int factoryIdx_, int color_, int row_, int numTiles_, bool isFirst_, int playerIdx_)
    {
        factoryIdx = factoryIdx_;
        color = color_;
        row = row_;
        numTiles = numTiles_;
        isFirst = isFirst_;
        playerIdx = playerIdx_;
    }

    public void Print()
    {
        Console.WriteLine("Factory: " + factoryIdx);
        Console.WriteLine("Color: " + color);
        Console.WriteLine("Row: " + row);
        Console.WriteLine("NumTiles: " + numTiles);
        Console.WriteLine("IsFirst: " + isFirst);
        Console.WriteLine("PlayerIdx: " + playerIdx);
    }
}

public class Player
{

    public int score = 0;
    // q: how to define a int matrix?
    public float[,] grid = new float[Constants.numColors, Constants.numColors];
    public int[][] line = new int[Constants.numRows][];
    public int[] floor = new int[Constants.numColors + 1]; // first = NONE = 5


    public Player()
    {
        for (int i = 0; i < Constants.numRows; i++)
            line[i] = new int[Constants.numColors];
    }

    bool SanityCheck()
    {
        // check if grid is full
        for (int row = 0; row < Constants.numRows; row++)
            for (int col = 0; col < Constants.numCols; col++)
                if ((grid[row, col] < 0) || (grid[row, col] > 1))
                    return false;
        // check if line is full
        foreach (int[] lineRow in line)
        {
            bool isFull = false;
            for (int col = 0; col < Constants.numCols; col++)
                if (lineRow[col] > 1)
                {
                    if (isFull) return false;
                    isFull = true;
                }
        }
        // check if floor is full
        foreach (int v in floor)
            if (v < 0) return false;

        if (floor[(int)Tiles.FIRST_MOVE] > 1) return false;

        return true;
    }
    public string GetYAML()
    {   // Display object contents as yaml
        string yaml = "";
        yaml += "score: " + score + "\n";
        yaml += "grid:\n";
        for (int row = 0; row < Constants.numRows; row++)
        {
            yaml += "  - ";
            for (int col = 0; col < Constants.numCols; col++)
                yaml += grid[row, col] + " ";
            yaml += "\n";
        }
        yaml += "line:\n";
        for (int row = 0; row < Constants.numRows; row++)
        {
            yaml += "  - ";
            for (int col = 0; col < Constants.numCols; col++)
                yaml += line[row][col] + " ";
            yaml += "\n";
        }
        yaml += "floor:\n";
        yaml += "  - ";
        for (int j = 0; j < Constants.numColors + 1; j++)
            yaml += floor[j] + " ";
        yaml += "\n";
        return yaml;
    }

    public static Player FromYAML(string yaml)
    {
        Player player = new();
        string[] lines = yaml.Split("\n");
        foreach (string line in lines)
        {
            string[] tokens = line.Split(" ");
            if (tokens[0] == "score:")
                player.score = int.Parse(tokens[1]);
            else if (tokens[0] == "grid:")
            {
                for (int row = 0; row < Constants.numRows; row++)
                {
                    for (int col = 0; col < Constants.numCols; col++)
                        player.grid[row, col] = int.Parse(tokens[2 + row * Constants.numCols + col]);
                }
            }
            else if (tokens[0] == "line:")
            {
                for (int row = 0; row < Constants.numRows; row++)
                {
                    for (int col = 0; col < Constants.numCols; col++)
                        player.line[row][col] = int.Parse(tokens[2 + row * Constants.numCols + col]);
                }
            }
            else if (tokens[0] == "floor:")
            {
                for (int j = 0; j < Constants.numColors + 1; j++)
                    player.floor[j] = int.Parse(tokens[2 + j]);
                break;
            }
        }
        return player;
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
                countCol += (int)grid[i, j];
                countRow += (int)grid[j, i];
                colorCount += (int)grid[j, ColorToCol(j, i)];
            }
            if (countCol > Constants.numColors)
                score += Constants.pointsPerColumn;
            if (countRow > Constants.numColors)
                score += Constants.pointsPerRow;
            if (colorCount > Constants.numColors)
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

        if (row < 0)
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
            case 2: return -1;
            case 3: return -2;
            case 4: return -2;
            case 5: return -2;
            case 6: return -3;
            case 7: return -3;
            default: return -3;
        }
    }
}



