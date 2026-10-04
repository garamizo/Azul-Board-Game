using Microsoft.Data.Sqlite;

namespace AzulServer.Data;

public sealed class Db(AzulOptions options)
{
    // Append only; each entry runs once, in a transaction.
    static readonly string[] Migrations =
    [
        """
        CREATE TABLE games(
          id TEXT PRIMARY KEY,
          creator TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('lobby','playing','finished')),
          num_players INTEGER NOT NULL CHECK(num_players BETWEEN 2 AND 4),
          version INTEGER NOT NULL,
          state_json TEXT,
          finish_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL);
        CREATE TABLE seats(
          game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
          idx INTEGER NOT NULL,
          kind TEXT NOT NULL CHECK(kind IN ('open','human','bot')),
          email TEXT,
          PRIMARY KEY(game_id, idx),
          CHECK((kind = 'open' AND email IS NULL) OR (kind = 'human' AND email IS NOT NULL) OR kind = 'bot'));
        CREATE UNIQUE INDEX seats_one_per_person ON seats(game_id, email) WHERE email IS NOT NULL;
        CREATE TABLE moves(
          game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
          version INTEGER NOT NULL,
          seat INTEGER NOT NULL,
          actor TEXT NOT NULL,
          move_json TEXT NOT NULL,
          request_id TEXT,
          request_hash TEXT,
          result_json TEXT,
          at TEXT NOT NULL,
          PRIMARY KEY(game_id, version));
        CREATE UNIQUE INDEX moves_request ON moves(game_id, request_id) WHERE request_id IS NOT NULL;
        CREATE INDEX games_status ON games(status, updated_at);
        """,
    ];

    public string ConnectionString { get; } = new SqliteConnectionStringBuilder
    {
        DataSource = Path.Combine(options.DataDir, "azul.db"),
        ForeignKeys = true,
        DefaultTimeout = 30,
        Pooling = true,
    }.ToString();

    public SqliteConnection Open()
    {
        var c = new SqliteConnection(ConnectionString);
        c.Open();
        return c;
    }

    public void Migrate()
    {
        Directory.CreateDirectory(options.DataDir);
        using var c = Open();
        Exec(c, null, "PRAGMA journal_mode=WAL;");
        Exec(c, null, "CREATE TABLE IF NOT EXISTS schema_version(v INTEGER NOT NULL);");
        using var q = c.CreateCommand();
        q.CommandText = "SELECT COALESCE(MAX(v), 0) FROM schema_version";
        var current = Convert.ToInt32(q.ExecuteScalar());
        for (int i = current; i < Migrations.Length; i++)
        {
            using var tx = c.BeginTransaction();
            Exec(c, tx, Migrations[i]);
            Exec(c, tx, $"INSERT INTO schema_version(v) VALUES ({i + 1});");
            tx.Commit();
        }
    }

    static void Exec(SqliteConnection c, SqliteTransaction? tx, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }
}
