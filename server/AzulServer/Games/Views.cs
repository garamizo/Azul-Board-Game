namespace AzulServer.Games;

public static class Status
{
    public const string Lobby = "lobby", Playing = "playing", Finished = "finished";
}

public static class SeatKind
{
    public const string Open = "open", Human = "human", Bot = "bot";
}

public sealed record ViewerInfo(string Email, int? Seat);
public sealed record SeatView(int Idx, string Kind, string? Email);
public sealed record PlayerView(int Score, int[]?[] Lines, int[][] Wall, int[] Floor, bool HasFirst);
public sealed record BoardView(int Round, string Phase, int ActiveSeat, int[][] Factories, int[] Center,
    bool CenterHasFirst, int[] Bag, int[] Discard, PlayerView[] Players);
public sealed record WallRowView(int Color, int[] Targets);
public sealed record LegalView(int[][]? Takes, WallRowView?[]? Wall);
public sealed record LastMoveView(long Version, int Seat, string Kind, int? Factory, int? Color, int? Row, int? Tiles, int[]? Columns);
public sealed record ResultView(int[] Scores, int[] Winners, string Reason);
public sealed record GameView(string Id, string Status, long Version, int NumPlayers, string Creator, ViewerInfo You,
    SeatView[] Seats, BoardView? Board, LegalView? Legal, LastMoveView? LastMove, ResultView? Result);
public sealed record GameSummary(string Id, string Status, int NumPlayers, string Creator, SeatView[] Seats, int? Round, string UpdatedAt);

/// What a move did, as stored in moves.move_json.
public sealed record MoveJson(string Kind, int? Factory, int? Color, int? Row, int? Tiles, int[]? Columns);
