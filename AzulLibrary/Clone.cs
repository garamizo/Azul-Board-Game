namespace Azul
{
    using System;
    using System.Linq;

    public partial class Game
    {
        sealed class CloneTag
        {
            public static readonly CloneTag Instance = new();
        }

        // Allocates nothing game-specific and deals nothing; Clone and
        // FromSnapshot fill every field themselves.
        Game(CloneTag _) { }

        /// Deep copy; the copy gets its own fresh Random, so simulations on it
        /// never advance this game's randomness.
        public override Game Clone() => Clone(new Random());

        public Game Clone(Random random)
        {
            var g = new Game(CloneTag.Instance)
            {
                rng = random,
                activePlayer = activePlayer,
                numPlayers = numPlayers,
                step = step,
                chanceHash = chanceHash,
                numFactories = numFactories,
                CENTER = CENTER,
                roundIdx = roundIdx,
                isRegularPhase = isRegularPhase,
                newRoundPlayer = newRoundPlayer,
                countPlayerClearedRound = countPlayerClearedRound,
                factories = factories.Select(f => (int[])f.Clone()).ToArray(),
                bag = (int[])bag.Clone(),
                discarded = (int[])discarded.Clone(),
                factoryIdxArray = (int[])factoryIdxArray.Clone(),
                rowIdxArray = (int[])rowIdxArray.Clone(),
                colorIdxArray = (int[])colorIdxArray.Clone(),
                players = players.Select(p => new Player(p)).ToArray(),
            };
            return g;
        }
    }
}
