using AzulServer.Data;
using Microsoft.Data.Sqlite;

namespace AzulServer.Tests;

public sealed class HubSchemaTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("azul-schema-").FullName;
    public void Dispose() => SqliteConnection.ClearAllPools();

    static void Exec(SqliteConnection c, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    static object? Scalar(SqliteConnection c, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        return cmd.ExecuteScalar();
    }

    [Fact]
    public void SchemaOneUpgradesAndItsGamesAreNotTracked()
    {
        var db = new Db(new AzulOptions { DataDir = dir });
        using (var c = db.Open())
        {
            Exec(c, "CREATE TABLE schema_version(v INTEGER NOT NULL);");
            Exec(c, Db.Migrations[0]);
            Exec(c, "INSERT INTO schema_version(v) VALUES (1);");
            Exec(c, "INSERT INTO games(id, creator, status, num_players, version, created_at, updated_at) " +
                    "VALUES ('old', 'a@x.com', 'finished', 2, 9, 't', 't');");
        }
        db.Migrate();
        using var c2 = db.Open();
        Assert.Equal(2L, Scalar(c2, "SELECT MAX(v) FROM schema_version"));
        Assert.Equal(0L, Scalar(c2, "SELECT COUNT(*) FROM hub_reports"));
        var g = GameStore.Load(c2, "old")!;
        Assert.False(g.HubTracked);
        Assert.Null(g.StartedAt);
        Assert.Null(g.FinishedAt);
        Assert.Null(g.BotKey);
    }

    [Fact]
    public void HubColumnsRoundTripThroughInsertAndUpdate()
    {
        var db = new Db(new AzulOptions { DataDir = dir });
        db.Migrate();
        var g = new GameRecord("g1", "a@x.com", "playing", 2, 2, "{}", null, "c", "u",
            [new SeatRecord(0, "human", "a@x.com"), new SeatRecord(1, "bot", null)])
        { StartedAt = "s", HubTracked = true, BotKey = "mcts@0123456789ab" };
        using var c = db.Open();
        using (var tx = c.BeginTransaction()) { GameStore.Insert(c, tx, g); tx.Commit(); }
        var loaded = GameStore.Load(c, "g1")!;
        Assert.Equal(("s", (string?)null, true, "mcts@0123456789ab"),
            (loaded.StartedAt, loaded.FinishedAt, loaded.HubTracked, loaded.BotKey));
        using (var tx = c.BeginTransaction())
        {
            GameStore.Update(c, tx, loaded with { Version = 3, Status = "finished", FinishedAt = "f" }, 2);
            tx.Commit();
        }
        Assert.Equal("f", GameStore.Load(c, "g1")!.FinishedAt);
    }

    [Fact]
    public void HubReportsHasNoForeignKeyToGames()
    {
        var db = new Db(new AzulOptions { DataDir = dir });
        db.Migrate();
        using var c = db.Open();
        Exec(c, "INSERT INTO hub_reports(game_id, body, status, next_attempt_at, created_at) VALUES ('gone', '{}', 'pending', 'n', 'n');");
        Assert.Equal(1L, Scalar(c, "SELECT COUNT(*) FROM hub_reports"));
    }
}
