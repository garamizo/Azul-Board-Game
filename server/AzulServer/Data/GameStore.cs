using Microsoft.Data.Sqlite;

namespace AzulServer.Data;

public sealed record SeatRecord(int Idx, string Kind, string? Email);

public sealed record GameRecord(
    string Id, string Creator, string Status, int NumPlayers, long Version,
    string? StateJson, string? FinishReason, string CreatedAt, string UpdatedAt,
    IReadOnlyList<SeatRecord> Seats)
{
    /// Hub reporting (schema 2): stamped by Start and by the finishing commit.
    public string? StartedAt { get; init; }
    public string? FinishedAt { get; init; }
    public bool HubTracked { get; init; }
    public string? BotKey { get; init; }
}

public sealed record MoveRecord(
    string GameId, long Version, int Seat, string Actor, string MoveJson,
    string? RequestId, string? RequestHash, string? ResultJson, string At);

public sealed class ConcurrencyException(string message) : Exception(message);

public static class GameStore
{
    const string GameColumns = "id, creator, status, num_players, version, state_json, finish_reason, created_at, updated_at, started_at, finished_at, hub_tracked, bot_key";
    const string MoveColumns = "game_id, version, seat, actor, move_json, request_id, request_hash, result_json, at";

    static SqliteCommand Command(SqliteConnection c, SqliteTransaction? tx, string sql, params (string Name, object? Value)[] args)
    {
        var cmd = c.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        foreach (var (name, value) in args)
            cmd.Parameters.AddWithValue(name, value ?? DBNull.Value);
        return cmd;
    }

    static string? Str(SqliteDataReader r, int i) => r.IsDBNull(i) ? null : r.GetString(i);

    static GameRecord ReadGame(SqliteDataReader r) => new(
        r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), r.GetInt64(4),
        Str(r, 5), Str(r, 6), r.GetString(7), r.GetString(8), [])
    {
        StartedAt = Str(r, 9),
        FinishedAt = Str(r, 10),
        HubTracked = r.GetInt64(11) != 0,
        BotKey = Str(r, 12),
    };

    static MoveRecord ReadMove(SqliteDataReader r) => new(
        r.GetString(0), r.GetInt64(1), r.GetInt32(2), r.GetString(3), r.GetString(4),
        Str(r, 5), Str(r, 6), Str(r, 7), r.GetString(8));

    static List<SeatRecord> Seats(SqliteConnection c, SqliteTransaction? tx, string id)
    {
        using var cmd = Command(c, tx, "SELECT idx, kind, email FROM seats WHERE game_id = $id ORDER BY idx", ("$id", id));
        using var r = cmd.ExecuteReader();
        var seats = new List<SeatRecord>();
        while (r.Read()) seats.Add(new SeatRecord(r.GetInt32(0), r.GetString(1), Str(r, 2)));
        return seats;
    }

    public static GameRecord? Load(SqliteConnection c, string id, SqliteTransaction? tx = null)
    {
        GameRecord g;
        using (var cmd = Command(c, tx, $"SELECT {GameColumns} FROM games WHERE id = $id", ("$id", id)))
        using (var r = cmd.ExecuteReader())
        {
            if (!r.Read()) return null;
            g = ReadGame(r);
        }
        return g with { Seats = Seats(c, tx, id) };
    }

    public static List<GameRecord> List(SqliteConnection c)
    {
        var games = new List<GameRecord>();
        foreach (var sql in new[]
        {
            $"SELECT {GameColumns} FROM games WHERE status <> 'finished' ORDER BY updated_at DESC",
            $"SELECT {GameColumns} FROM games WHERE status = 'finished' ORDER BY updated_at DESC LIMIT 20",
        })
        {
            using var cmd = Command(c, null, sql);
            using var r = cmd.ExecuteReader();
            while (r.Read()) games.Add(ReadGame(r));
        }
        return games.Select(g => g with { Seats = Seats(c, null, g.Id) }).ToList();
    }

    public static List<string> PlayingIds(SqliteConnection c)
    {
        using var cmd = Command(c, null, "SELECT id FROM games WHERE status = 'playing'");
        using var r = cmd.ExecuteReader();
        var ids = new List<string>();
        while (r.Read()) ids.Add(r.GetString(0));
        return ids;
    }

    public static void Insert(SqliteConnection c, SqliteTransaction tx, GameRecord g)
    {
        using (var cmd = Command(c, tx,
            $"INSERT INTO games({GameColumns}) VALUES ($id, $creator, $status, $n, $v, $state, $finish, $created, $updated, $started, $finished, $tracked, $botkey)",
            ("$id", g.Id), ("$creator", g.Creator), ("$status", g.Status), ("$n", g.NumPlayers), ("$v", g.Version),
            ("$state", g.StateJson), ("$finish", g.FinishReason), ("$created", g.CreatedAt), ("$updated", g.UpdatedAt),
            ("$started", g.StartedAt), ("$finished", g.FinishedAt), ("$tracked", g.HubTracked ? 1 : 0), ("$botkey", g.BotKey)))
            cmd.ExecuteNonQuery();
        InsertSeats(c, tx, g);
    }

    public static void Update(SqliteConnection c, SqliteTransaction tx, GameRecord g, long expectedVersion)
    {
        using (var cmd = Command(c, tx,
            "UPDATE games SET status = $status, version = $v, state_json = $state, finish_reason = $finish, updated_at = $updated, " +
            "started_at = $started, finished_at = $finished, hub_tracked = $tracked, bot_key = $botkey " +
            "WHERE id = $id AND version = $expected",
            ("$status", g.Status), ("$v", g.Version), ("$state", g.StateJson), ("$finish", g.FinishReason),
            ("$updated", g.UpdatedAt), ("$started", g.StartedAt), ("$finished", g.FinishedAt),
            ("$tracked", g.HubTracked ? 1 : 0), ("$botkey", g.BotKey), ("$id", g.Id), ("$expected", expectedVersion)))
        {
            if (cmd.ExecuteNonQuery() != 1)
                throw new ConcurrencyException($"game {g.Id} is not at version {expectedVersion}");
        }
        using (var del = Command(c, tx, "DELETE FROM seats WHERE game_id = $id", ("$id", g.Id)))
            del.ExecuteNonQuery();
        InsertSeats(c, tx, g);
    }

    static void InsertSeats(SqliteConnection c, SqliteTransaction tx, GameRecord g)
    {
        foreach (var s in g.Seats)
        {
            using var cmd = Command(c, tx, "INSERT INTO seats(game_id, idx, kind, email) VALUES ($id, $idx, $kind, $email)",
                ("$id", g.Id), ("$idx", s.Idx), ("$kind", s.Kind), ("$email", s.Email));
            cmd.ExecuteNonQuery();
        }
    }

    public static void Delete(SqliteConnection c, SqliteTransaction tx, string id)
    {
        using var cmd = Command(c, tx, "DELETE FROM games WHERE id = $id", ("$id", id));
        cmd.ExecuteNonQuery();
    }

    public static void InsertMove(SqliteConnection c, SqliteTransaction tx, MoveRecord m)
    {
        using var cmd = Command(c, tx,
            $"INSERT INTO moves({MoveColumns}) VALUES ($g, $v, $seat, $actor, $move, $req, $hash, $result, $at)",
            ("$g", m.GameId), ("$v", m.Version), ("$seat", m.Seat), ("$actor", m.Actor), ("$move", m.MoveJson),
            ("$req", m.RequestId), ("$hash", m.RequestHash), ("$result", m.ResultJson), ("$at", m.At));
        cmd.ExecuteNonQuery();
    }

    public static MoveRecord? FindMove(SqliteConnection c, string gameId, string requestId)
    {
        using var cmd = Command(c, null, $"SELECT {MoveColumns} FROM moves WHERE game_id = $g AND request_id = $r",
            ("$g", gameId), ("$r", requestId));
        using var r = cmd.ExecuteReader();
        return r.Read() ? ReadMove(r) : null;
    }

    public static MoveRecord? LastMove(SqliteConnection c, string gameId, SqliteTransaction? tx = null)
    {
        using var cmd = Command(c, tx, $"SELECT {MoveColumns} FROM moves WHERE game_id = $g ORDER BY version DESC LIMIT 1",
            ("$g", gameId));
        using var r = cmd.ExecuteReader();
        return r.Read() ? ReadMove(r) : null;
    }
}
