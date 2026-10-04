using AzulServer.Data;
using Microsoft.Data.Sqlite;

namespace AzulServer.Tests;

public sealed class StoreTests : IDisposable
{
    readonly Db db = new(new AzulOptions { DataDir = Directory.CreateTempSubdirectory("azul-db-").FullName });

    public StoreTests() => db.Migrate();

    public void Dispose() => SqliteConnection.ClearAllPools();

    static GameRecord Lobby(string id, string creator = "a@x") => new(id, creator, "lobby", 3, 1, null, null,
        "2026-10-03T00:00:00Z", "2026-10-03T00:00:00Z",
        [new SeatRecord(0, "human", creator), new SeatRecord(1, "open", null), new SeatRecord(2, "bot", null)]);

    void Insert(GameRecord g)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        GameStore.Insert(c, tx, g);
        tx.Commit();
    }

    [Fact]
    public void MigrateTwiceIsHarmless()
    {
        db.Migrate();
        using var c = db.Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = "SELECT MAX(v) FROM schema_version";
        Assert.Equal(1L, (long)cmd.ExecuteScalar()!);
    }

    [Fact]
    public void InsertLoadRoundTrip()
    {
        Insert(Lobby("g1"));
        using var c = db.Open();
        var g = GameStore.Load(c, "g1")!;
        Assert.Equal("lobby", g.Status);
        Assert.Equal(3, g.Seats.Count);
        Assert.Equal(new SeatRecord(2, "bot", null), g.Seats[2]);
        Assert.Null(GameStore.Load(c, "nope"));
    }

    [Fact]
    public void UpdateChecksTheVersion()
    {
        Insert(Lobby("g2"));
        using var c = db.Open();
        var g = GameStore.Load(c, "g2")!;
        using (var tx = c.BeginTransaction())
        {
            GameStore.Update(c, tx, g with { Version = 2, Seats = [g.Seats[0], new SeatRecord(1, "human", "b@x"), g.Seats[2]] }, 1);
            tx.Commit();
        }
        Assert.Equal("b@x", GameStore.Load(c, "g2")!.Seats[1].Email);
        using var tx2 = c.BeginTransaction();
        Assert.Throws<ConcurrencyException>(() => GameStore.Update(c, tx2, g with { Version = 3 }, 1));
    }

    [Fact]
    public void OneSeatPerPersonPerGame()
    {
        var bad = Lobby("g3") with { Seats = [new SeatRecord(0, "human", "a@x"), new SeatRecord(1, "human", "a@x"), new SeatRecord(2, "open", null)] };
        Assert.Throws<SqliteException>(() => Insert(bad));
    }

    [Fact]
    public void SeatInvariantsAreEnforced()
    {
        Assert.Throws<SqliteException>(() => Insert(Lobby("g4") with { Seats = [new SeatRecord(0, "open", "a@x")] }));
        Assert.Throws<SqliteException>(() => Insert(Lobby("g5") with { Seats = [new SeatRecord(0, "human", null)] }));
    }

    [Fact]
    public void MovesAndDeleteCascade()
    {
        Insert(Lobby("g6"));
        using var c = db.Open();
        using (var tx = c.BeginTransaction())
        {
            GameStore.InsertMove(c, tx, new MoveRecord("g6", 2, 0, "a@x", "{}", "r1", "h1", "{\"v\":2}", "t"));
            GameStore.InsertMove(c, tx, new MoveRecord("g6", 3, 1, "bot", "{}", null, null, "{\"v\":3}", "t"));
            tx.Commit();
        }
        Assert.Equal(2, GameStore.FindMove(c, "g6", "r1")!.Version);
        Assert.Equal(3, GameStore.LastMove(c, "g6")!.Version);
        using (var tx = c.BeginTransaction())
        {
            GameStore.Delete(c, tx, "g6");
            tx.Commit();
        }
        Assert.Null(GameStore.Load(c, "g6"));
        Assert.Null(GameStore.LastMove(c, "g6"));
        using var cmd = c.CreateCommand();
        cmd.CommandText = "SELECT (SELECT COUNT(*) FROM seats WHERE game_id='g6') + (SELECT COUNT(*) FROM moves WHERE game_id='g6')";
        Assert.Equal(0L, (long)cmd.ExecuteScalar()!);
    }

    [Fact]
    public void ListShowsActiveGamesAndTheLast20Finished()
    {
        Insert(Lobby("active"));
        for (int i = 0; i < 25; i++)
            Insert(Lobby($"f{i:D2}") with { Status = "finished", UpdatedAt = $"2026-10-03T00:00:{i:D2}Z" });
        using var c = db.Open();
        var list = GameStore.List(c);
        Assert.Contains(list, g => g.Id == "active");
        Assert.Equal(20, list.Count(g => g.Status == "finished"));
        Assert.Contains(list, g => g.Id == "f24");
        Assert.DoesNotContain(list, g => g.Id == "f00");
        Assert.All(list, g => Assert.NotEmpty(g.Seats));
    }
}
