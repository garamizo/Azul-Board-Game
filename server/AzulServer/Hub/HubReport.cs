using System.Globalization;
using System.Text.Json.Nodes;
using AzulServer.Data;
using AzulServer.Games;
using Microsoft.Data.Sqlite;

namespace AzulServer.Hub;

/// `Body` is the JSON to send, or null with `Skip` saying why there is none.
public sealed record HubBuild(string? Body, string? Skip);

/// The hub 5.1 body for a finished game (spec section 4). Pure: everything
/// comes from the record (its StateJson is the final snapshot) and the seats a
/// bot moved for.
public static class HubReport
{
    public const string StalemateReason = "stalemate";
    public const string BotPlayedReason = "a bot played for a person";

    /// Seats a bot made a move for: every stored bot move, plus `pending`, the
    /// finishing move, which Mutate has not inserted yet when it builds.
    public static ISet<int> BotPlayedSeats(SqliteConnection c, SqliteTransaction? tx, string gameId, MoveRecord? pending)
    {
        var seats = new HashSet<int>();
        using (var cmd = c.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "SELECT DISTINCT seat FROM moves WHERE game_id = $g AND actor = 'bot'";
            cmd.Parameters.AddWithValue("$g", gameId);
            using var r = cmd.ExecuteReader();
            while (r.Read()) seats.Add(r.GetInt32(0));
        }
        if (pending is { Actor: "bot" }) seats.Add(pending.Seat);
        return seats;
    }

    public static HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin)
    {
        if (!g.HubTracked) return new(null, "not tracked");
        if (g.Status != Status.Finished || g.StateJson is null || g.FinishedAt is null) return new(null, "not finished");
        var snap = Projection.Snapshot(g);
        var keys = snap.Players.Select(p => (p.Score, Rows: p.Grid.Count(row => row.All(v => v >= 0)))).ToArray();
        var players = new JsonArray();
        foreach (var seat in g.Seats.OrderBy(s => s.Idx))
        {
            int i = seat.Idx;
            JsonObject p;
            if (seat.Email is { } email)
                p = new JsonObject
                {
                    ["seat"] = $"P{i + 1}",
                    ["kind"] = "human",
                    ["email"] = HubContract.Email(email),
                    ["name"] = HubContract.Name(email.Split('@')[0], "Player"),
                };
            else if (seat.Kind == SeatKind.Bot && g.BotKey is { } botKey)
                p = new JsonObject
                {
                    ["seat"] = $"P{i + 1}",
                    ["kind"] = "bot",
                    ["bot_key"] = botKey,
                    ["name"] = BotIdentity.DisplayName,
                };
            else
                return new(null, seat.Kind == SeatKind.Bot ? "no bot key" : $"seat {i} is {seat.Kind}");
            p["rank"] = 1 + keys.Count(k => k.Score > keys[i].Score || (k.Score == keys[i].Score && k.Rows > keys[i].Rows));
            p["score"] = keys[i].Score;
            players.Add(p);
        }
        string? reason = g.FinishReason != "normal" ? StalemateReason
            : g.Seats.Any(s => s.Email is not null && botPlayedSeats.Contains(s.Idx)) ? BotPlayedReason
            : null;
        var body = new JsonObject
        {
            ["external_id"] = g.Id,
            ["variant"] = $"free-wall-{g.NumPlayers}p",
            ["rated"] = reason is null,
            ["unrated_reason"] = reason,
            ["started_at"] = g.StartedAt is null ? null : Utc(g.StartedAt),
            ["finished_at"] = Utc(g.FinishedAt),
            ["replay_url"] = publicOrigin is null ? null : $"{publicOrigin}/g/{g.Id}",
            ["players"] = players,
        };
        return new(body.ToJsonString(), null);
    }

    /// "O" UTC string -> yyyy-MM-ddTHH:mm:ss.fffZ. The custom format truncates
    /// (it never rounds into the next second).
    static string Utc(string stamp) =>
        DateTimeOffset.Parse(stamp, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal)
            .UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);
}
