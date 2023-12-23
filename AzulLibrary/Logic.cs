namespace Azul
{
    using System.Diagnostics;
    using System.Collections.Generic;
    using GameMath = GameUtils.GameMath;
    using System;
    using System.Linq;  // Sum, Max, Min
    using DeepCopy;
    using System.Drawing;

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
        public const int EMPTY_TILE = -1;
        public bool isRegularPhase = true;
        public int newRoundPlayer = 0;
        public int countPlayerClearedRound = 0;

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

        public override GameUtils.Game<Move> Reset(int numPlayers) => new Game(numPlayers);

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

        public IEnumerable<Move> GetColIdxMoves()
        // public List<Move> GetColIdxMoves()
        {
            var p = players[activePlayer];
            int[] colors = new int[Constants.numRows];
            List<int>[] opens = new List<int>[Constants.numRows];
            for (int row = 0; row < Constants.numRows; row++)
            {
                colors[row] = -1;
                bool isFilled = false;
                opens[row] = new List<int>();
                for (int color = 0; color < Constants.numColors; color++)
                    if (p.line[row, color] > row)  // line is full, find available col
                    {
                        colors[row] = color;
                        isFilled = true;
                        // find available cols
                        for (int ic = 0; ic < Constants.numCols; ic++)
                        {
                            if (p.grid[row, ic] != EMPTY_TILE)
                                continue;
                            bool isGoodCol = true;
                            for (int i = 0; i < Constants.numRows; i++)
                                if (p.grid[i, ic] == color)  // bad col
                                {
                                    isGoodCol = false;
                                    break;
                                }
                            if (isGoodCol)
                                opens[row].Add(ic);
                        }
                        break;
                    }
                if (isFilled) opens[row].Add(Constants.numColors);  // floor
                else opens[row].Add(-1);  // na
                GameMath.Shuffle<int>(rng, opens[row]);
            }

            // var moves = new List<Move>();
            foreach (var i0 in opens[0])
                foreach (var i1 in opens[1])
                    foreach (var i2 in opens[2])
                        foreach (var i3 in opens[3])
                            foreach (var i4 in opens[4])
                            {
                                var move = new Move(new int[] { i0, i1, i2, i3, i4 }, colors, activePlayer);
                                bool badMove = false;
                                for (int i = 0; i < Constants.numRows; i++)
                                {
                                    for (int j = i - 1; j >= 0; j--)
                                        if (move.colIdx[i] == move.colIdx[j] && move.colors[i] == move.colors[j] && move.colIdx[i] != -1 && move.colIdx[i] != (int)Rows.FLOOR)
                                        {
                                            badMove = true;
                                            break;
                                        }
                                    if (badMove) break;
                                }
                                if (badMove) continue;

                                // if (IsValid(move) == false)
                                //     throw new Exception("Invalid move");
                                yield return move;
                            }
        }

        public override float[] GetScores() => players.Select(p => (float)p.score).ToArray();

        public override float[] GetHeuristics()
        {
            // TODO fix this for non MCTS AIs
            Debug.Assert(false, "GetHeuristics() not implemented");
            // on middle of round, compute expected points
            // compute score by transferring lines to grid, then scale to [0, 1]
            float scoreMax = -100f;
            int numTies = 0;
            float[] scores = new float[numPlayers];
            for (int i = 0; i < numPlayers; i++)
            {
                Player p = new(players[i]);
                // p.UpdateRound();
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

        public (Move, int) GetGreedyColIdxMove()
        {
            var p = new Player(players[activePlayer]);
            // point earned by filling row-col
            int[] idxCol = new int[Constants.numRows];
            int[] colors = new int[Constants.numRows];
            for (int row = 0; row < Constants.numRows; row++)
            {
                idxCol[row] = -1;
                colors[row] = -1;
                for (int color = 0; color < Constants.numCols; color++)
                    if (players[activePlayer].line[row, color] > row)
                    {
                        colors[row] = color;
                        break;
                    }
                if (colors[row] == -1) continue;

                idxCol[row] = (int)Rows.FLOOR;  // to floor if no col available
                int vmax = int.MinValue;
                for (int col = 0; col < Constants.numCols; col++)
                {
                    if (p.grid[row, col] != EMPTY_TILE) continue;
                    bool badCol = false;
                    for (int i = 0; i < Constants.numRows; i++)
                        if (p.grid[i, col] == colors[row])  // bad col
                        {
                            badCol = true;
                            break;
                        }
                    if (badCol) continue;

                    p.grid[row, col] = colors[row];
                    int pts = p.ScoreTile(row, col) + p.ScoreCombos();
                    p.grid[row, col] = EMPTY_TILE;
                    if (pts > vmax || (pts == vmax && RandomSeed.NextDouble() < 0.25))
                    {
                        vmax = pts;
                        idxCol[row] = col;
                    }
                }
                if (vmax > int.MinValue)
                    p.grid[row, idxCol[row]] = colors[row];
            }
            // int score = 0;
            // for (int row = 0; row < Constants.numRows; row++)
            //     if (idxCol[row] != -1 && idxCol[row] != (int)Rows.FLOOR)
            //         score += p.ScoreTile(row, idxCol[row]);

            return (new Move(idxCol, colors, activePlayer), -1);
            // var p = new Player(players[activePlayer]);

            // point earned by filling row-col
            // int[,] scoreTile = new int[Constants.numRows, Constants.numCols];
            // for (int row = 0; row < Constants.numRows; row++)
            //     for (int col = 0; col < Constants.numCols; col++)
            //     {
            //         scoreTile[row, col] = p.ScoreTile(row, col);
            //         if (p.grid[row, col] != EMPTY_TILE)
            //             p.grid[row, col] = 100;
            //         scoreTile[row, col] += p.ScoreCombos();
            //         p.grid[row, col] = players[activePlayer].grid[row, col];
            //     }

            // int scoreMax = int.MinValue;
            // Move moveMax = new();
            // var moves = GameMath.Shuffled(RandomSeed, GetColIdxMoves().ToList());
            // foreach (var m in moves)
            // {
            //     // value = -10 * floor 
            //     int pts = 0;
            //     for (int row = 0; row < Constants.numRows; row++)
            //     {
            //         if (m.colIdx[row] == -1) continue;
            //         if (m.colIdx[row] == (int)Rows.FLOOR)
            //             pts += -10 * (row + 1);
            //         else
            //         {
            //             // pts += p.ScoreTile(row, m.colIdx[row]);
            //             // p.grid[row, m.colIdx[row]] = m.colors[row];
            //             pts += scoreTile[row, m.colIdx[row]];
            //         }
            //     }
            //     // pts += p.ScoreCombos();
            //     // for (int row = 0; row < Constants.numRows; row++)
            //     //     if (m.colIdx[row] != -1 && m.colIdx[row] != (int)Rows.FLOOR)
            //     //         p.grid[row, m.colIdx[row]] = EMPTY_TILE;

            //     if (pts > scoreMax)
            //     {
            //         scoreMax = pts;
            //         moveMax = m;
            //     }
            // }
            // return (moveMax, scoreMax);
        }

        // public Move GetGreedyMoveFull()
        // {
        //     if (isRegularPhase == false)
        //         return GetGreedyColIdxMove().Item1;

        //     var p = players[activePlayer];

        public override Move GetGreedyMove()
        {
            if (isRegularPhase == false)
                return GetGreedyColIdxMove().Item1;

            var p = players[activePlayer];

            // faster implementation ===================
            int valueMax = int.MinValue;
            int fMax = 0, cMax = 0, rMax = 0;
            GameMath.Shuffle<int>(rng, colorIdxArray);
            foreach (var color in colorIdxArray)
            {
                var factoryArry = new List<int>();
                var rowArry = new List<int>();

                for (int fId = 0; fId < numFactories + 1; fId++)
                    if (factories[fId][color] > 0)
                        factoryArry.Add(fId);  // factory has color

                for (int row = 0; row < Constants.numRows; row++)
                {
                    if (p.line[row, color] > 0 && p.line[row, color] < row + 1)
                    {
                        rowArry.Add(row);  // row has color
                        continue;
                    }
                    bool goodRow = true;
                    for (int col = 0; col < Constants.numCols; col++)
                        if (p.line[row, col] > 0 || p.grid[row, col] == color)
                        {
                            goodRow = false;
                            break;
                        }
                    if (goodRow)
                        rowArry.Add(row);
                }

                foreach (var row in rowArry)
                    foreach (var factoryIdx in factoryArry)
                    {
                        int value = int.MinValue;
                        int numTiles = factories[factoryIdx][color];
                        if (row == (int)Rows.FLOOR)
                            value = -10 * numTiles;
                        else
                        {
                            int tilesToFill = row + 1 - p.line[row, color];  // needed
                            if (tilesToFill > numTiles)
                                value = -1 * (tilesToFill - numTiles);
                            else
                                value = 100 - 10 * (numTiles - tilesToFill) + numTiles;
                        }

                        if (value > valueMax || (value == valueMax && RandomSeed.NextDouble() < 0.25))
                        {
                            valueMax = value;
                            fMax = factoryIdx;
                            cMax = color;
                            rMax = row;
                            if (valueMax >= 103)
                                return new Move(fMax, cMax, rMax, this);
                        }
                    }
            }
            if (valueMax == int.MinValue)
            {
                // no moves found, play single tile to floor
                if (factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
                    return new Move(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, this);
                // (fMax, cMax, rMax) = (CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR);

                for (int fId = 0; fId < numFactories + 1; fId++)
                    for (int color = 0; color < Constants.numColors; color++)
                        if (factories[fId][color] > 0 && -factories[fId][color] > valueMax)
                        {
                            valueMax = -factories[fId][color];
                            (fMax, cMax, rMax) = (fId, color, (int)Rows.FLOOR);
                            if (valueMax == -1)
                                return new Move(fMax, cMax, rMax, this);
                        }

            }
            Debug.Assert(valueMax != int.MinValue, "No moves found");
            return new Move(fMax, cMax, rMax, this);

            // ====================

            // // value: 100*completeLine - 1*missingLineTile -10*floorTile
            // // GameMath.Shuffle<int>(rng, factoryIdxArray);
            // GameMath.Shuffle<int>(rng, colorIdxArray);
            // // GameMath.Shuffle<int>(rng, rowIdxArray);

            // int fMax = 0, cMax = 0, rMax = 0;

            // int valueMax = int.MinValue;
            // // Move moveMax = new();
            // foreach (var factoryIdx in factoryIdxArray)
            //     foreach (var color in colorIdxArray)
            //         foreach (var row in rowIdxArray)
            //             if (IsValid(factoryIdx, color, row))
            //             {
            //                 int value;
            //                 int numTiles = factories[factoryIdx][color];
            //                 if (row == (int)Rows.FLOOR)
            //                     value = -10 * numTiles;
            //                 else
            //                 {
            //                     int tilesToFill = row + 1 - p.line[row, color];  // needed
            //                     if (tilesToFill > numTiles)
            //                         value = -1 * (tilesToFill - numTiles);
            //                     else
            //                         value = 100 - 10 * (numTiles - tilesToFill);
            //                 }

            //                 if (value > valueMax || (value == valueMax && RandomSeed.NextDouble() < 0.25))
            //                 {
            //                     valueMax = value;
            //                     // bool isFirst = factoryIdx == CENTER && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0;
            //                     // moveMax = new Move(factoryIdx, color, row, numTiles, isFirst, activePlayer);
            //                     fMax = factoryIdx;
            //                     cMax = color;
            //                     rMax = row;
            //                     if (valueMax >= 100)
            //                         return new Move(fMax, cMax, rMax, this);
            //                     // return moveMax;
            //                 }
            //             }
            // if (valueMax == int.MinValue && factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
            //     return new Move(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, 1, true, activePlayer);
            // Debug.Assert(valueMax != int.MinValue, "No moves found");

            // return new Move(fMax, cMax, rMax, this);
            // // return moveMax;
        }

        public Move GetGreedyMove2()
        {
            // p.UpdateRound();
            // int scoreBaseline = Heuristic(p, new Move(0, 0, (int)Rows.FLOOR, 0, false, activePlayer));
            var (actionBest, scoreBaseline) = GetGreedyColIdxMove();
            if (isRegularPhase == false)
                return actionBest;

            // var actionBest = new Move();
            int scoreMax = -1;
            var p0 = new Player(players[activePlayer]);

            int nFloorTiles = 0;
            for (int i = 0; i < Constants.numColors + 1; i++)
                nFloorTiles += players[activePlayer].floor[i];

            for (int row = 0; row < Constants.numRows + 1; row++)
                for (int color = 0; color < Constants.numColors; color++)
                {
                    // required tiles to fill row
                    int tileFill = row == (int)Rows.FLOOR ? 0 : row + 1 - players[activePlayer].line[row, color];
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
                                // scoreComplete = Heuristic(p, new Move(factoryIdx, color, row, tileFill, isFirst, activePlayer));
                                players[activePlayer].Play(new Move(factoryIdx, color, row, tileFill, false, activePlayer));
                                (_, scoreComplete) = GetGreedyColIdxMove();
                                players[activePlayer] = new Player(p0);  // undo play
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

                            if (score > scoreMax || (score == scoreMax && RandomSeed.NextDouble() < 0.25))
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


        Move GetRandomColIdxMove()
        {
            var cols = Enumerable.Range(0, Constants.numCols).ToList();
            GameMath.Shuffle<int>(rng, cols);
            var p = players[activePlayer];
            var colors = new int[Constants.numRows];
            var colIdx = new int[Constants.numRows];
            for (int row = 0; row < Constants.numRows; row++)
            {
                colors[row] = -1;
                colIdx[row] = -1;
                int color;
                for (color = 0; color < Constants.numColors; color++)
                    if (p.line[row, color] > row)  // line is full, find available col
                    {
                        colors[row] = color;
                        colIdx[row] = (int)Rows.FLOOR;  // to floor if no col available
                        break;
                    }
                if (colors[row] == -1) continue;  // row is not full
                bool isGoodCol = true;
                foreach (var ic in cols)
                {
                    if (p.grid[row, ic] != EMPTY_TILE)
                        continue;
                    for (int i = 0; i < Constants.numRows; i++)
                        if (p.grid[i, ic] == color)  // bad col
                        {
                            isGoodCol = false;
                            break;
                        }
                    if (isGoodCol == false) continue;
                    for (int i = row - 1; i >= 0; i--)  // previous row already claimed col
                        if (colIdx[i] == ic && colors[i] == color)  // bad col
                        {
                            isGoodCol = false;
                            break;
                        }
                    if (isGoodCol)
                    {
                        colIdx[row] = ic;
                        GameMath.Shuffle<int>(rng, cols);
                        break;
                    }
                }
            }
            return new Move(colIdx, colors, activePlayer);
        }

        public override Move GetRandomMove()
        {
            if (isRegularPhase == false)
                return GetRandomColIdxMove();

            var p = players[activePlayer];

            // faster implementation ===================
            GameMath.Shuffle<int>(rng, colorIdxArray);
            foreach (var color in colorIdxArray)
            {
                var factoryArry = new List<int>();
                var rowArry = new List<int> { (int)Rows.FLOOR };

                for (int fId = 0; fId < numFactories + 1; fId++)
                    if (factories[fId][color] > 0)
                        factoryArry.Add(fId);  // factory has color

                for (int row = 0; row < Constants.numRows; row++)
                {
                    if (p.line[row, color] > 0 && p.line[row, color] <= row)
                    {
                        rowArry.Add(row);  // row has color
                        continue;
                    }
                    bool goodRow = true;
                    for (int col = 0; col < Constants.numCols; col++)
                        if (p.line[row, col] > 0 || p.grid[row, col] == color)
                        {
                            goodRow = false;
                            break;
                        }
                    if (goodRow)
                        rowArry.Add(row);
                }

                if (factoryArry.Count == 0)  // only floor available
                    continue;
                return new Move(factoryArry[rng.Next(factoryArry.Count)], color, rowArry[rng.Next(rowArry.Count)], this);
            }
            // no moves found, play single tile to floor
            if (factories[CENTER][(int)Tiles.FIRST_MOVE] > 0)
                Debug.Assert(factories[CENTER][(int)Tiles.FIRST_MOVE] > 0, "No moves found");
            return new Move(CENTER, (int)Tiles.FIRST_MOVE, (int)Rows.FLOOR, this);
        }
        public Move GetRandomMoveSlow()
        {
            if (isRegularPhase == false)
            {
                var cols = Enumerable.Range(0, Constants.numCols).ToList();
                GameMath.Shuffle<int>(rng, cols);
                var p = players[activePlayer];
                var colors = new int[Constants.numRows];
                var colIdx = new int[Constants.numRows];
                for (int row = 0; row < Constants.numRows; row++)
                {
                    colors[row] = -1;
                    colIdx[row] = -1;
                    int color;
                    for (color = 0; color < Constants.numColors; color++)
                        if (p.line[row, color] > row)  // line is full, find available col
                        {
                            colors[row] = color;
                            colIdx[row] = (int)Rows.FLOOR;  // to floor if no col available
                            break;
                        }
                    if (color == Constants.numColors) continue;
                    bool isGoodCol = true;
                    foreach (var ic in cols)
                    {
                        for (int i = 0; i < Constants.numRows; i++)
                            if (p.grid[i, ic] == color)  // bad col
                            {
                                isGoodCol = false;
                                break;
                            }
                        if (isGoodCol == false) continue;
                        for (int i = row - 1; i >= 0; i--)  // previous row already claimed col
                            if (colIdx[i] == ic && colors[i] == color)  // bad col
                            {
                                isGoodCol = false;
                                break;
                            }
                        if (isGoodCol)
                        {
                            colIdx[row] = ic;
                            GameMath.Shuffle<int>(rng, cols);
                            break;
                        }
                    }
                    break;
                }
                return new Move(colIdx, colors, activePlayer);
            }

            GameMath.Shuffle<int>(rng, factoryIdxArray);
            GameMath.Shuffle<int>(rng, colorIdxArray);
            GameMath.Shuffle<int>(rng, rowIdxArray);

            foreach (var factoryIdx in factoryIdxArray)
                foreach (var color in colorIdxArray)
                    foreach (var row in rowIdxArray)
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

        public ulong GetHash()
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

        public override bool Equals(GameUtils.Game<Move> other)
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
            new int[]{ 0, 0, 0, 0, 0, 0 },
        };
            game.isRegularPhase = false;
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
            {0, 3, 0, 0, 0},
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
            // explain this move:
            // Child 3) Player=2, Factory=7, Color=WHITE:4, Row=FLOOR:5, NumTiles=2, IsFirst=False, Rolls: 38295, winRatio: 0.51298255
            Game game = new(3);
            game.isRegularPhase = false;
            game.bag = new int[] { 2, 2, 3, 3, 1, };
            game.discarded = new int[] { 2, 6, 1, 6, 6, };
            game.factories = new int[][]{
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, },
    new int[]{ 0, 0, 0, 0, 0, 0},
};
            game.players = new Player[]{
    new Player(){
        score = 31,
        line = new int[,] {
            { 0, 0, 0, 1, 0, },
            { 0, 0, 0, 0, 0, },
            { 0, 3, 0, 0, 0, },
            { 2, 0, 0, 0, 0, },
            { 5, 0, 0, 0, 0, },
        },
        grid = new int[,] {
            { 2, 1, 4, -1, -1, },
            { 3, 2, 1, 4, -1, },
            { 0, 3, 2, -1, -1, },
            { -1, -1, -1, 2, -1, },
            { -1, -1, -1, -1, 2, },
        },
        floor = new int[] { 1, 0, 0, 2, 1, 1, },
    },
    new Player(){
        score = 61,
        line = new int[,] {
            { 0, 0, 0, 0, 0, },
            { 0, 0, 0, 0, 0, },
            { 0, 0, 0, 0, 0, },
            { 0, 0, 0, 0, 0, },
            { 0, 0, 0, 0, 0, },
        },
        grid = new int[,] {
            { 2, 4, 3, 0, 1, },
            { 4, 1, 0, 2, 3, },
            { -1, -1, 4, 1, 0, },
            { -1, -1, 1, 4, -1, },
            { -1, -1, -1, 3, -1, },
        },
        floor = new int[] { 0, 0, 0, 0, 0, 0, },
    },
    new Player(){
        score = 39,
        line = new int[,] {
            { 1, 0, 0, 0, 0, },
            { 0, 0, 2, 0, 0, },
            { 0, 0, 1, 0, 0, },
            { 0, 0, 4, 0, 0, },
            { 0, 0, 0, 0, 2, },
        },
        grid = new int[,] {
            { -1, 4, 1, 3, 2, },
            { -1, 0, 4, 1, 3, },
            { -1, -1, 0, 4, -1, },
            { -1, -1, -1, 0, 4, },
            { -1, 1, 3, -1, -1, },
        },
        floor = new int[] { 0, 0, 1, 0, 0, 0, },
    },
};
            return game;
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
            str += "\t\tMode: " + (isRegularPhase ? "Regular" : $"Pick col idx ({countPlayerClearedRound})");
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

        public List<Move> GetPossibleActions2()
        {
            List<Move> actions = new();
            int NUM_ROWS = 5;

            if (isRegularPhase == false)
            {
                var p = players[activePlayer];
                var colIdx = new int[NUM_ROWS];
                var colors = new int[NUM_ROWS];

                for (int row = 0; row < NUM_ROWS; row++)
                {
                    colIdx[row] = colors[row] = -1;
                    for (int color = 0; color < NUM_COLORS; color++)
                        if (p.line[row, color] > row)
                        {
                            colIdx[row] = 0;
                            colors[row] = color;
                            break;
                        }
                }

                for (int i0 = -1; i0 <= NUM_ROWS; i0++)
                {
                    if (i0 != -1 && colors[0] == -1) continue;
                    colIdx[0] = i0;
                    for (int i1 = -1; i1 <= NUM_ROWS; i1++)
                    {
                        if (i1 != -1 && colors[1] == -1) continue;
                        colIdx[1] = i1;
                        for (int i2 = -1; i2 <= NUM_ROWS; i2++)
                        {
                            if (i2 != -1 && colors[2] == -1) continue;
                            colIdx[2] = i2;
                            for (int i3 = -1; i3 <= NUM_ROWS; i3++)
                            {
                                if (i3 != -1 && colors[3] == -1) continue;
                                colIdx[3] = i3;
                                for (int i4 = -1; i4 <= NUM_ROWS; i4++)
                                {
                                    if (i4 != -1 && colors[4] == -1) continue;
                                    colIdx[4] = i4;
                                    if (IsValid(colIdx, colors))
                                        actions.Add(new Move(colIdx, colors, activePlayer));
                                }
                            }
                        }
                    }
                }
            }
            else
            {
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
            }
            return actions;
        }

        public override List<Move> GetPossibleActions(bool sort = false)
        {
            List<Move> actions = new();

            if (isRegularPhase == false)
            {
                actions = GetColIdxMoves().ToList();
                // actions = GetColIdxMoves();
                if (sort)
                    actions = actions.OrderByDescending(m =>
                    {
                        int drops = 0;
                        // int score = 0;
                        for (int row = 0; row < Constants.numRows; row++)
                            if (m.colIdx[row] != -1)
                            {
                                if (m.colIdx[row] == (int)Rows.FLOOR)
                                    drops -= row + 1;
                                else
                                    for (int i = row - 1; i >= 0; i--)
                                        if (m.colIdx[i] == m.colIdx[row] && m.colors[i] == m.colors[row])
                                        {
                                            drops += row + 1;
                                            break;
                                        }
                            }
                        return drops;
                    }).ToList();
                return actions;
            }

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

            if (sort)
                actions = actions.OrderByDescending(m =>
                {
                    if (m.row == (int)Rows.FLOOR) return -m.numTiles;
                    var drops = m.numTiles + players[m.playerIdx].line[m.row, m.color] - (m.row + 1);
                    drops = drops < 0 ? 0 : drops;
                    var fits = m.numTiles - drops;
                    return fits - drops;
                }).ToList();

            return actions;
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
                    // transfer discard to bactionsag if bag is incomplete
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

        public override bool Play(Move m)
        {
            step++;
            chanceHash = 0;

            // update player -------------------
            players[m.playerIdx].Play(m);  // added tiles here            

            if (isRegularPhase)  // regular play
            {
                var factory = factories[m.factoryIdx];
                bool isCenter = m.factoryIdx == numFactories;

                // update factories ----------------
                factory[m.color] = 0;  // reset factory
                if (isCenter == false)  // move other tiles to center
                    for (int i = 0; i < Constants.numColors; i++)
                    {
                        factories[CENTER][i] += factory[i];
                        factory[i] = 0;
                    }

                if (m.isFirst)
                    factory[(int)Tiles.FIRST_MOVE] = 0;

                // update active player 
                activePlayer = (activePlayer + 1) % numPlayers;

                if (IsFactoryEmpty())
                {
                    isRegularPhase = false;
                    countPlayerClearedRound = 0;
                }
            }
            else
            {
                Debug.Assert(m.playerIdx == activePlayer, "Invalid player index");
                // int[] discard = players[m.playerIdx].UpdateRound(m);
                int[] discard = players[activePlayer].UpdateRound(m);
                countPlayerClearedRound++;
                for (int j = 0; j < Constants.numColors; j++)
                    discarded[j] += discard[j];

                // update active player
                if (discard[(int)Tiles.FIRST_MOVE] > 0)
                    newRoundPlayer = m.playerIdx;

                // check if all players have completed their lines
                if (countPlayerClearedRound >= numPlayers)  // reset factories with random tiles
                {
                    if (IsGameOver())  // update player scores
                    {
                        for (int i = 0; i < numPlayers; i++)
                            players[i].UpdateGame();
                        return false;
                    }
                    activePlayer = newRoundPlayer;
                    isRegularPhase = true;
                    roundIdx++;
                    FillFactories();
                    return true;
                }

                // update active player 
                activePlayer = (activePlayer + 1) % numPlayers;
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
            if (isRegularPhase == true)
                return false;
            // clear board of all players
            if (countPlayerClearedRound < numPlayers)
                return false;

            if (step > MAX_STEPS)
            {
                // Debug.Write("Game ended in stalemate");
                return true;
            }

            if (IsFactoryEmpty() == false)
                return false;

            // check if any player has a full row
            foreach (var player in players)
                for (int row = 0; row < Constants.numRows; row++)
                {
                    bool isRowFull = true;
                    for (int col = 0; col < Constants.numColors; col++)
                        if (player.grid[row, col] == EMPTY_TILE)
                        {
                            isRowFull = false;
                            break;
                        }
                    if (isRowFull)
                        return true;
                }
            return false;
        }

        public bool IsValid(int[] colIdx, int[] colors)
        {
            var p = players[activePlayer];
            for (int row = 0; row < Constants.numRows; row++)
            {
                int color = colors[row];
                if (color == -1) continue;  // empty row
                int numTiles = p.line[row, color];

                if (numTiles < row + 1)  // incomplete line
                {
                    if (colIdx[row] != -1)
                        return false;  // line incomplete, but idx set
                    continue;  // check next row
                }
                if (colIdx[row] >= Constants.numCols) continue; // floor
                // line complete, but idx set out of bounds
                if (numTiles == row + 1 && colIdx[row] < 0)
                    return false;
                // grid already filled
                if (p.grid[row, colIdx[row]] != EMPTY_TILE)
                    return false;
                // color already in same row or col
                for (int i = 0; i < Constants.numCols; i++)
                    if (p.grid[row, i] == color || p.grid[i, colIdx[row]] == color)
                        return false;
                // grid[row, colIdx[row]] = color;
                // previous line already filled this column
                for (int i = row - 1; i >= 0; i--)
                    if (colIdx[i] == colIdx[row] && colors[i] == colors[row])
                        return false;
            }
            return true;
        }

        public bool IsValid(int factoryIdx, int color, int row)
        {
            ref var factory = ref factories[factoryIdx];
            ref var player = ref players[activePlayer];
            // int color = T.color;
            // int row = T.row;

            if (color == (int)Tiles.FIRST_MOVE && factoryIdx != CENTER)
                return false;
            // check if factory is empty
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
            for (int i = 0; i < Constants.numColors; i++)
                if (player.grid[row, i] == color)
                    return false;
            // if (player.grid[row, Player.ColorToCol(row, color)] > 0)
            //     return false;
            return true;
        }
        public override bool IsValid(Move action) //=> IsValid(action.factoryIdx, action.color, action.row);
        {
            if (action.colIdx[0] == Move.NOT_SET)
                return IsValid(action.factoryIdx, action.color, action.row);
            else
                return IsValid(action.colIdx, action.colors);
        }

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
            if (isRegularPhase == false)
            {
                while (true)
                {
                    Console.WriteLine($"Player {activePlayer}, enter your move (colIdx[5]): ");
                    string? input = Console.ReadLine();
                    if (input == null)
                        continue;
                    string[] tokens = input.Split(' ');
                    if (tokens.Length != Constants.numRows)
                        continue;
                    int[] colIdx = new int[Constants.numRows];
                    for (int i = 0; i < Constants.numRows; i++)
                        colIdx[i] = int.Parse(tokens[i]);
                    var move = new Move(colIdx, this);
                    if (IsValid(move))
                        return move;
                    Console.WriteLine("Invalid move");
                }
            }
            while (true)
            {
                Console.WriteLine($"Player {activePlayer}, enter your move (factoryIdx, color, row): ");
                string? input = Console.ReadLine();
                if (input == null)
                    continue;
                string[] tokens = input.Split(' ');
                if (tokens.Length != 3)
                    continue;
                // return new(int.Parse(tokens[0]), int.Parse(tokens[1]), int.Parse(tokens[2]), this);
                var move = new Move(int.Parse(tokens[0]), int.Parse(tokens[1]), int.Parse(tokens[2]), this);
                if (IsValid(move))
                    return move;
                Console.WriteLine("Invalid move");
            }
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
        public int[] colIdx = new int[Constants.numRows];
        public int[] colors = new int[Constants.numRows];
        public const int NOT_SET = -2;


        public Move(int factoryIdx, int color, int row,
            int numTiles, bool isFirst, int playerIdx)
        {
            this.factoryIdx = factoryIdx;
            this.color = color;
            this.row = row;
            this.numTiles = numTiles;
            this.isFirst = isFirst;
            this.playerIdx = playerIdx;
            this.colIdx[0] = NOT_SET;
        }

        public Move(int[] colIdx, int[] colors, int activePlayer)
        {
            this.playerIdx = activePlayer;
            for (int i = 0; i < Constants.numRows; i++)
            {
                this.colIdx[i] = colIdx[i];
                this.colors[i] = colors[i];
            }
            // this.colIdx = colIdx;
        }

        public Move(int[] colIdx, Game state) :
            this(colIdx,
            colIdx.Select((col, idx) =>
            {
                if (col == -1) return -1;
                for (int color = 0; color < Constants.numColors; color++)
                    if (state.players[state.activePlayer].line[idx, color] > idx)
                        return color;
                return -1;
            }).ToArray(),
            state.activePlayer)
        { }

        public Move(Move move, Game state)  // corrects entries
        {
            if (state.isRegularPhase)
            {
                factoryIdx = move.factoryIdx;
                color = move.color;
                row = move.row;
                numTiles = state.factories[factoryIdx][color];
                isFirst = factoryIdx == state.CENTER && state.factories[factoryIdx][(int)Tiles.FIRST_MOVE] > 0;
                playerIdx = state.activePlayer;
                colIdx[0] = NOT_SET;
            }
            else
            {
                playerIdx = state.activePlayer;
                for (int i = 0; i < Constants.numRows; i++)
                {
                    colIdx[i] = move.colIdx[i];
                    colors[i] = -1;
                    for (int c = 0; c < Constants.numColors; c++)
                        if (state.players[state.activePlayer].line[i, c] > i)
                        {
                            colors[i] = c;
                            break;
                        }
                    if (colors[i] == -1)
                        colIdx[i] = -1;
                }
            }
        }

        public Move(int factoryIdx, int color, int row, Game state) :
            this(factoryIdx, color, row,
            state.factories[factoryIdx][color],
            factoryIdx == (int)state.CENTER && state.factories[factoryIdx][(int)Tiles.FIRST_MOVE] > 0,
            state.activePlayer)
        { }

        public Move() { }

        public override string ToString()
        {
            if (colIdx[0] == NOT_SET)
                return $"Player={playerIdx},\tFactory={factoryIdx},\t" +
                    $"Color={Enum.GetName(typeof(Tiles), color)}:{color},\tRow={Enum.GetName(typeof(Rows), row)}:{row},\t" +
                    $"NumTiles={numTiles},\tIsFirst={isFirst}";
            else
                return $"Player {playerIdx})\tColIdx=[{string.Join(", ", colIdx)}]";
        }
    }

    public class Player
    {
        public int score = 0;
        // grid == -1 for empty, tile index otherwise
        public int[,] grid = new int[Constants.numColors, Constants.numColors];
        public int[,] line = new int[Constants.numRows, Constants.numColors];
        public int[] floor = new int[Constants.numColors + 1]; // first = NONE = 5


        public Player()
        {
            for (int row = 0; row < Constants.numRows; row++)
                for (int col = 0; col < Constants.numCols; col++)
                    grid[row, col] = -1;
        }

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
                    // if (grid[row, ColorToCol(row, col)] > 0 && line[row, col] > 0)
                    //     return false;
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
                // print line
                for (int col = 0; col < COLS; col++)
                    str += (col < COLS - row - 1) ? "  " : ((col + numTiles >= COLS && numTiles > 0) ? $"{color} " : "- ");

                // print grid
                str += "\t";
                for (int col = 0; col < COLS; col++)
                    str += grid[row, col] == Game.EMPTY_TILE ? "- " : $"{grid[row, col]} ";
                // str += grid[row, col] > 0 ? $"{ColToColor(row, col)} " : "- ";
            }
            str += $"\n{ROWS}  FLOOR: " + (floor[NUM_COLORS] > 0 ? $"{NUM_COLORS} " : "");
            for (int i = 0; i < NUM_COLORS; i++)
                for (int j = 0; j < floor[i]; j++)
                    str += $"{i} ";

            // str += "\n";
            return str;
        }

        public int[] UpdateRound(Move move)
        {
            int[] discard = new int[Constants.numColors + 1];
            // transfer completed lines to grid
            for (int row = 0; row < Constants.numRows; row++)
            {
                int col = move.colIdx[row];
                if (col == -1) continue;  // incomplete line
                int color = move.colors[row];
                // for (int color = 0; color < Constants.numColors; color++)
                //     if (line[row, color] > row)  // completed line
                //     {
                line[row, color] = 0;
                if (col == Constants.numCols)  // floor
                    floor[color] += row + 1;
                else
                {
                    // bool isColFull = false;
                    // for (int i = 0; i < row; i++)
                    //     if (grid[i, col] == color)
                    //     {
                    //         isColFull = true;
                    //         break;
                    //     }
                    // if (isColFull)
                    //     floor[color] += row + 1;
                    // else
                    // {
                    grid[row, col] = color;
                    score += ScoreTile(row, col);
                    discard[color] += row;
                    // }
                }
                // break;
                // }
            }

            // subtract floor
            score += FloorToScore(floor.Sum());
            for (int i = 0; i < Constants.numColors + 1; i++)
            {
                discard[i] += floor[i];
                floor[i] = 0;
            }
            if (score < 0) score = 0;  // constraint to >= 0
            return discard;
        }

        public void UpdateGame()
        {   // update scores from grid patterns
            for (int i = 0; i < Constants.numColors; i++)
            {
                // add points for each column
                bool countCol = true, countRow = true;
                int colorCount = 0;
                for (int j = 0; j < Constants.numColors; j++)
                {
                    if (grid[j, i] == Game.EMPTY_TILE) countCol = false;
                    if (grid[i, j] == Game.EMPTY_TILE) countRow = false;
                    // for color i, search all (j, k) for grid[j, k] == i
                    for (int k = 0; k < Constants.numColors; k++)
                        if (grid[j, k] == i)
                            colorCount += 1;
                }

                if (countCol)
                    score += Constants.pointsPerColumn;
                if (countRow)
                    score += Constants.pointsPerRow;
                if (colorCount >= Constants.numRows)
                    score += Constants.pointsPerColor;
            }
        }

        public int ScoreCombos()
        {   // update scores from grid patterns
            int pts = 0;
            for (int i = 0; i < Constants.numColors; i++)
            {
                // add points for each column
                bool countCol = true, countRow = true;
                int colorCount = 0;
                for (int j = 0; j < Constants.numColors; j++)
                {
                    if (grid[j, i] == Game.EMPTY_TILE) countCol = false;
                    if (grid[i, j] == Game.EMPTY_TILE) countRow = false;
                    // for color i, search all (j, k) for grid[j, k] == i
                    for (int k = 0; k < Constants.numColors; k++)
                        if (grid[j, k] == i)
                            colorCount += 1;
                }

                if (countCol)
                    pts += Constants.pointsPerColumn;
                if (countRow)
                    pts += Constants.pointsPerRow;
                if (colorCount >= Constants.numRows)
                    pts += Constants.pointsPerColor;
            }
            return pts;
        }

        public void Play(Move action)  // regular play (not column selection)
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

        public int ScoreTile(int row, int col)
        {
            int CountJointTiles(int rowDir, int colDir)
            {
                int count = 0;
                int rowIdx = row + rowDir;
                int colIdx = col + colDir;
                while (rowIdx >= 0 && rowIdx < Constants.numRows &&
                       colIdx >= 0 && colIdx < Constants.numColors &&
                       grid[rowIdx, colIdx] != Game.EMPTY_TILE)
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
        }

        // public static int ColorToCol(int row, int color) { return (row + color) % Constants.numCols; }
        // public static int ColToColor(int row, int col) { return (Constants.numCols - row + col) % Constants.numCols; }


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
        FIRST_MOVE = 5,
        EMPTY = 5,
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
}