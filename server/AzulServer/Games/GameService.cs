using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Azul;
using AzulServer.Api;
using AzulServer.Data;
using Microsoft.Data.Sqlite;

namespace AzulServer.Games;

public sealed record ErrorBody(string Error, object? View);

public sealed record ApiResult(int Status, object? Body)
{
    public static ApiResult Ok(object body) => new(200, body);
    public static readonly ApiResult NoContent = new(204, null);
    public static ApiResult Error(int status, string code, object? view = null) => new(status, new ErrorBody(code, view));

    public IResult ToHttp() => Status == 204
        ? Results.NoContent()
        : Results.Json(Body, Json.Options, statusCode: Status);
}

public interface IFaultInjector
{
    /// Runs right after a commit, before notifications. Tests throw here to
    /// simulate a crash between the commit and the announcements.
    void AfterCommit(string gameId);
}

public sealed class NoFaults : IFaultInjector
{
    public void AfterCommit(string gameId) { }
}

public sealed class GameService(Db db, EventHub hub, ServerMoveQueue queue, IFaultInjector faults,
    TimeProvider time, ILogger<GameService> log)
{
    abstract record Decision;
    sealed record Respond(ApiResult Result) : Decision;
    sealed record Commit(GameRecord Next, MoveRecord? Move = null) : Decision;

    readonly ConcurrentDictionary<string, SemaphoreSlim> locks = new();

    SemaphoreSlim LockFor(string id) => locks.GetOrAdd(id, _ => new SemaphoreSlim(1, 1));

    string Now() => time.GetUtcNow().UtcDateTime.ToString("O");

    static Respond Reject(int status, string code) => new(ApiResult.Error(status, code));

    // ---------- reads ----------

    public IReadOnlyList<GameSummary> List()
    {
        using var c = db.Open();
        var list = new List<GameSummary>();
        foreach (var g in GameStore.List(c))
        {
            try { list.Add(Projection.Summary(g)); }
            catch (Exception e) when (e is InvalidSnapshotException or JsonException)
            {
                log.LogError(e, "game {Game} has an unreadable state", g.Id);
            }
        }
        return list;
    }

    /// Null if the game does not exist. Throws for a game whose stored state
    /// is invalid (Program's exception handler answers 500 for that request).
    public GameView? GetView(string id, string viewer) =>
        Current(id) is { } cur ? Projection.Project(cur.Game, cur.Last, viewer) : null;

    /// Game, seats and last move from one read transaction, so they belong to
    /// the same committed version. No cache: it would race with deletes.
    (GameRecord Game, MoveRecord? Last)? Current(string id)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction(deferred: true);
        var g = GameStore.Load(c, id, tx);
        return g is null ? null : (g, GameStore.LastMove(c, id, tx));
    }

    // ---------- lobby ----------

    public ApiResult Create(string viewer, int players)
    {
        if (players is < 2 or > 4)
            return ApiResult.Error(400, "players-must-be-2-to-4");
        var now = Now();
        var seats = Enumerable.Range(0, players)
            .Select(i => i == 0 ? new SeatRecord(0, SeatKind.Human, viewer) : new SeatRecord(i, SeatKind.Open, null))
            .ToList();
        var g = new GameRecord(NewId(), viewer, Status.Lobby, players, 1, null, null, now, now, seats);
        using var c = db.Open();
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, g);
            tx.Commit();
        }
        return new ApiResult(201, Projection.Project(g, null, viewer));
    }

    static string NewId()
    {
        const string alphabet = "abcdefghijklmnopqrstuvwxyz234567";
        return new string(Enumerable.Range(0, 10).Select(_ => alphabet[RandomNumberGenerator.GetInt32(alphabet.Length)]).ToArray());
    }

    static SeatRecord? Seat(GameRecord g, int idx) => idx >= 0 && idx < g.Seats.Count ? g.Seats[idx] : null;

    static GameRecord WithSeat(GameRecord g, int idx, string kind, string? email) =>
        g with { Seats = g.Seats.Select(s => s.Idx == idx ? new SeatRecord(idx, kind, email) : s).ToList() };

    public Task<ApiResult> Claim(string id, int idx, string viewer) => Mutate(id, viewer, g =>
    {
        if (g.Status != Status.Lobby) return Reject(409, "not-in-lobby");
        if (Seat(g, idx) is not { } seat) return Reject(404, "no-such-seat");
        if (g.Seats.Any(s => s.Email == viewer)) return Reject(409, "already-seated");
        if (seat.Kind != SeatKind.Open) return Reject(409, "seat-taken");
        return new Commit(WithSeat(g, idx, SeatKind.Human, viewer));
    });

    public Task<ApiResult> Release(string id, int idx, string viewer) => Mutate(id, viewer, g =>
    {
        if (g.Status != Status.Lobby) return Reject(409, "not-in-lobby");
        if (Seat(g, idx) is not { } seat) return Reject(404, "no-such-seat");
        if (seat.Kind != SeatKind.Human) return Reject(409, "seat-not-human");
        if (seat.Email != viewer && g.Creator != viewer) return Reject(403, "not-your-seat");
        return new Commit(WithSeat(g, idx, SeatKind.Open, null));
    });

    public Task<ApiResult> SetKind(string id, int idx, string viewer, string? kind) => Mutate(id, viewer, g =>
    {
        if (g.Status != Status.Lobby) return Reject(409, "not-in-lobby");
        if (g.Creator != viewer) return Reject(403, "creator-only");
        if (kind is not (SeatKind.Open or SeatKind.Bot)) return Reject(400, "bad-kind");
        if (Seat(g, idx) is not { } seat) return Reject(404, "no-such-seat");
        if (seat.Kind == SeatKind.Human) return Reject(409, "seat-human");
        return new Commit(WithSeat(g, idx, kind, null));
    });

    public Task<ApiResult> Start(string id, string viewer) => Mutate(id, viewer, g =>
    {
        if (g.Status != Status.Lobby) return Reject(409, "not-in-lobby");
        if (g.Creator != viewer) return Reject(403, "creator-only");
        if (!g.Seats.Any(s => s.Kind == SeatKind.Human)) return Reject(409, "no-human");
        var game = new Game(g.NumPlayers);
        return new Commit(g with
        {
            Status = Status.Playing,
            StateJson = Json.Serialize(game.ToSnapshot()),
            Seats = g.Seats.Select(s => s.Kind == SeatKind.Open ? s with { Kind = SeatKind.Bot } : s).ToList(),
        });
    });

    public Task<ApiResult> ToBot(string id, int idx, string viewer) => Mutate(id, viewer, g =>
    {
        if (g.Status != Status.Playing) return Reject(409, "not-playing");
        if (Seat(g, idx) is not { } seat) return Reject(404, "no-such-seat");
        if (seat.Kind != SeatKind.Human) return Reject(409, "seat-not-human");
        if (seat.Email != viewer && g.Creator != viewer) return Reject(403, "not-your-seat");
        return new Commit(WithSeat(g, idx, SeatKind.Bot, seat.Email));  // email kept: may take it back
    });

    public Task<ApiResult> TakeBack(string id, int idx, string viewer) => Mutate(id, viewer, g =>
    {
        if (g.Status != Status.Playing) return Reject(409, "not-playing");
        if (Seat(g, idx) is not { } seat) return Reject(404, "no-such-seat");
        if (seat.Kind != SeatKind.Bot || seat.Email != viewer) return Reject(403, "not-yours");
        return new Commit(WithSeat(g, idx, SeatKind.Human, viewer));
    });

    public async Task<ApiResult> Delete(string id, string viewer)
    {
        var gate = LockFor(id);
        await gate.WaitAsync();
        try
        {
            using var c = db.Open();
            var g = GameStore.Load(c, id);
            if (g is null) return ApiResult.Error(404, "not-found");
            if (g.Creator != viewer) return ApiResult.Error(403, "creator-only");
            using (var tx = c.BeginTransaction())
            {
                GameStore.Delete(c, tx, id);
                tx.Commit();
            }
            try
            {
                queue.CancelSearch(id);
                hub.PublishDeleted(id);
            }
            catch (Exception e)
            {
                log.LogError(e, "post-delete step failed for game {Game}", id);
            }
            return ApiResult.NoContent;
        }
        finally
        {
            gate.Release();
        }
    }

    // ---------- the one write path ----------

    Task<ApiResult> Mutate(string id, string viewer, Func<GameRecord, Decision> decide) =>
        Mutate(id, viewer, (g, _) => decide(g));

    async Task<ApiResult> Mutate(string id, string viewer, Func<GameRecord, SqliteConnection, Decision> decide)
    {
        var gate = LockFor(id);
        await gate.WaitAsync();
        try
        {
            using var c = db.Open();
            var g = GameStore.Load(c, id);
            if (g is null) return ApiResult.Error(404, "not-found");
            switch (decide(g, c))
            {
                case Respond r:
                    return r.Result;
                case Commit commit:
                    var next = commit.Next with { Version = g.Version + 1, UpdatedAt = Now() };
                    var move = commit.Move is null ? null : commit.Move with { Version = next.Version };
                    var last = move ?? GameStore.LastMove(c, id);
                    var view = Projection.Project(next, last, viewer);
                    if (move is not null) move = move with { ResultJson = Json.Serialize(view) };
                    using (var tx = c.BeginTransaction())
                    {
                        GameStore.Update(c, tx, next, g.Version);
                        if (move is not null) GameStore.InsertMove(c, tx, move);
                        tx.Commit();
                    }
                    AfterCommit(id, next.Version);
                    return ApiResult.Ok(view);
                default:
                    throw new InvalidOperationException("unknown decision");
            }
        }
        finally
        {
            gate.Release();
        }
    }

    void AfterCommit(string id, long version)
    {
        try
        {
            faults.AfterCommit(id);
            queue.CancelSearch(id);  // a bot thinking on the old version stops early
            hub.Publish(id, version);
            queue.Poke(id);
        }
        catch (Exception e)
        {
            // Committed already: the sweep (Task 14) repairs a missed poke and
            // the SSE heartbeat (Task 13) a missed notification.
            log.LogError(e, "post-commit step failed for game {Game}", id);
        }
    }

    // ---------- server-made moves (Task 14 uses these) ----------

    public IReadOnlyList<string> GamesNeedingServerMove()
    {
        List<string> ids;
        using (var c = db.Open()) ids = GameStore.PlayingIds(c);
        var result = new List<string>();
        foreach (var id in ids)
        {
            try
            {
                if (GetServerTurn(id) is not null) result.Add(id);
            }
            catch (Exception e) when (e is InvalidSnapshotException or JsonException)
            {
                log.LogError(e, "game {Game} has an unreadable state", id);
            }
            catch (Exception e)
            {
                // Any other bad row (e.g. a state that does not fit its seats)
                // must not stop the sweep for every other game.
                log.LogError(e, "game {Game} could not be checked for a server move", id);
            }
        }
        return result;
    }

    public sealed record ServerTurn(string GameId, long Version, Game Game, string Actor, Move? Forced);

    public ServerTurn? GetServerTurn(string id)
    {
        if (Current(id) is not { } cur) return null;
        var g = cur.Game;
        if (g.Status != Status.Playing || g.StateJson is null) return null;
        var game = Game.FromSnapshot(Projection.Snapshot(g));
        var seat = g.Seats[game.activePlayer];
        if (seat.Kind == SeatKind.Bot) return new ServerTurn(id, g.Version, game, "bot", null);
        if (seat.Kind == SeatKind.Human && game.ForcedMove() is { } forced)
            return new ServerTurn(id, g.Version, game, "auto", forced);
        return null;
    }

    // ---------- moves ----------

    public Task<ApiResult> Move(string id, string viewer, MoveRequest req)
    {
        if (MoveShape(req) is { } bad)
            return Task.FromResult(bad);
        var hash = RequestHash(req);
        return Mutate(id, viewer, (g, c) =>
        {
            // 1. A request already applied answers with its stored result,
            //    whatever happened to the game since.
            if (GameStore.FindMove(c, id, req.RequestId!) is { } prior)
                return prior.Actor == viewer && prior.RequestHash == hash && prior.ResultJson is not null
                    ? new Respond(ApiResult.Ok(JsonDocument.Parse(prior.ResultJson).RootElement.Clone()))
                    : Reject(409, "request-id-reused");
            // 2-3. Playing, and at the version the client saw. Version before
            //      ownership: a double submit from two devices is stale, not 403.
            if (g.Status != Status.Playing) return Reject(409, "not-playing");
            if (req.Version != g.Version)
                return new Respond(ApiResult.Error(409, "stale", Projection.Project(g, GameStore.LastMove(c, id), viewer)));
            var game = Game.FromSnapshot(Projection.Snapshot(g));
            var seat = g.Seats[game.activePlayer];
            // 4. Only the person in the active human seat.
            if (seat.Kind != SeatKind.Human || seat.Email != viewer) return Reject(403, "not-your-turn");
            // 5. The phase comes from the game.
            bool isTake = req.Kind == "take";
            if (isTake != game.isRegularPhase) return Reject(400, "wrong-phase");
            Move move;
            if (isTake)
            {
                int f = req.Factory!.Value, color = req.Color!.Value, row = req.Row!.Value;
                if (f > game.numFactories) return Reject(400, "bad-factory");
                if (color == 5 && (f != game.numFactories || row != 5)) return Reject(400, "bad-first-take");
                // The correcting constructor fills count, FIRST flag and player
                // from the game; nothing of that comes from the client.
                move = new Move(new Move { factoryIdx = f, color = color, row = row }, game);
            }
            else
            {
                move = new Move(new Move { colIdx = (int[])req.Columns!.Clone() }, game);
            }
            // 6. The engine's rules.
            if (!game.IsValid(move)) return Reject(400, "illegal");
            return Apply(g, game, move, seat.Idx, viewer, req.RequestId, hash);
        });
    }

    static ApiResult? MoveShape(MoveRequest r)
    {
        static ApiResult Bad(string code) => ApiResult.Error(400, code);
        if (r.RequestId is null || !Guid.TryParse(r.RequestId, out _)) return Bad("bad-request-id");
        if (r.Version < 1) return Bad("bad-version");
        switch (r.Kind)
        {
            case "take":
                if (r.Factory is not { } f || f < 0) return Bad("bad-factory");  // upper bound needs the game
                if (r.Color is not { } c || c < 0 || c > 5) return Bad("bad-color");
                if (r.Row is not { } row || row < 0 || row > 5) return Bad("bad-row");
                if (r.Columns is not null) return Bad("unexpected-columns");
                return null;
            case "wall":
                if (r.Columns is not { Length: 5 } cols || cols.Any(x => x < -1 || x > 5)) return Bad("bad-columns");
                if (r.Factory is not null || r.Color is not null || r.Row is not null) return Bad("unexpected-take-fields");
                return null;
            default:
                return Bad("bad-kind");
        }
    }

    static string RequestHash(MoveRequest r)
    {
        var canonical = $"{r.Version}|{r.Kind}|{r.Factory}|{r.Color}|{r.Row}|{string.Join(',', r.Columns ?? [])}";
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical)));
    }

    Decision Apply(GameRecord g, Game game, Move move, int seat, string actor, string? requestId, string? hash)
    {
        var record = new MoveRecord(g.Id, 0, seat, actor, Json.Serialize(Projection.Describe(move)), requestId, hash, null, Now());
        game.Play(move);
        bool finished = game.IsFinished;
        return new Commit(g with
        {
            StateJson = Json.Serialize(game.ToSnapshot()),
            Status = finished ? Status.Finished : Status.Playing,
            FinishReason = finished ? (AnyFullWallRow(game) ? "normal" : "stalemate") : null,
        }, record);
    }

    static bool AnyFullWallRow(Game game) =>
        game.players.Any(p => Enumerable.Range(0, 5).Any(r => Enumerable.Range(0, 5).All(c => p.grid[r, c] >= 0)));

    /// A bot's move, or a human's forced move ("auto"), computed outside the
    /// lock against `version`; refused if anything changed since.
    public Task<ApiResult> ApplyServerMove(string id, long version, Move move, string actor) => Mutate(id, actor, g =>
    {
        if (g.Status != Status.Playing) return Reject(409, "not-playing");
        if (g.Version != version) return Reject(409, "stale");
        var game = Game.FromSnapshot(Projection.Snapshot(g));
        var seat = g.Seats[game.activePlayer];
        bool allowed = seat.Kind == SeatKind.Bot
                       || (seat.Kind == SeatKind.Human && actor == "auto" && game.ForcedMove() is not null);
        if (!allowed) return Reject(409, "not-a-server-turn");
        if (!game.IsValid(move))
        {
            log.LogError("server move {Move} is invalid for game {Game}", move, id);
            return Reject(409, "illegal");
        }
        return Apply(g, game, move, seat.Idx, actor, null, null);
    });
}
