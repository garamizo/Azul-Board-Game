namespace Azul
{
    using System.Collections.Generic;
    using System.Linq;

    /// A completed pattern line's colour and where it may go: wall columns
    /// (ascending) and 5 = floor, always last.
    public sealed record WallRowOption(int Color, int[] Targets);

    public partial class Game
    {
        /// Every legal take for the active player, deterministic and without
        /// side effects (no RNG, no shuffling). Same set as GetPossibleActions.
        public List<(int Factory, int Color, int Row)> LegalTakes()
        {
            var takes = new List<(int, int, int)>();
            if (!isRegularPhase || IsFinished)
                return takes;
            for (int f = 0; f <= numFactories; f++)
                for (int color = 0; color < Constants.numColors; color++)
                    for (int row = 0; row <= Constants.numRows; row++)
                        if (IsValid(f, color, row))
                            takes.Add((f, color, row));
            if (factories[CENTER][FIRST] > 0)
                takes.Add((CENTER, FIRST, (int)Rows.FLOOR));
            return takes;
        }

        /// Per-row wall choices for the active player; null outside the wall
        /// phase. Same per-row rule as GetColIdxMoves, without its shuffling.
        public WallRowOption?[]? WallOptions()
        {
            if (isRegularPhase || IsFinished)
                return null;
            var p = players[activePlayer];
            var result = new WallRowOption?[Constants.numRows];
            for (int row = 0; row < Constants.numRows; row++)
            {
                for (int color = 0; color < Constants.numColors; color++)
                {
                    if (p.line[row, color] <= row)
                        continue;
                    var targets = new List<int>();
                    for (int col = 0; col < Constants.numCols; col++)
                    {
                        if (p.grid[row, col] != EMPTY_TILE)
                            continue;
                        bool colorInColumn = false;
                        for (int r = 0; r < Constants.numRows; r++)
                            if (p.grid[r, col] == color) { colorInColumn = true; break; }
                        if (!colorInColumn)
                            targets.Add(col);
                    }
                    targets.Add(Constants.numCols);  // floor
                    result[row] = new WallRowOption(color, targets.ToArray());
                    break;
                }
            }
            return result;
        }

        /// The active player's only legal move, or null when there is a choice
        /// (or the game is over). Floor is always open to a completed line, so
        /// a wall turn is forced only when every completed line can go nowhere
        /// but the floor (or there is none).
        public Move? ForcedMove()
        {
            if (IsFinished)
                return null;
            if (isRegularPhase)
            {
                var takes = LegalTakes();
                if (takes.Count != 1)
                    return null;
                var t = takes[0];
                var take = new Move(t.Factory, t.Color, t.Row, this);
                return IsValid(take) ? take : null;
            }
            var options = WallOptions()!;
            if (options.Any(o => o is not null && o.Targets.Length != 1))
                return null;
            var cols = options.Select(o => o is null ? -1 : o.Targets[0]).ToArray();
            var wall = new Move(cols, this);
            return IsValid(wall) ? wall : null;
        }
    }
}
