using Azul;

namespace AzulServer.Games;

public interface IBotBrain
{
    Move ChooseMove(Game game, CancellationToken ct);
}

/// Deterministic and instant; used by tests.
public sealed class GreedyBrain : IBotBrain
{
    public Move ChooseMove(Game game, CancellationToken ct) => game.GetGreedyMove();
}
