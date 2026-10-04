using AzulServer.Auth;
using AzulServer.Games;

namespace AzulServer.Api;

public sealed record CreateGameRequest(int Players);
public sealed record SetKindRequest(string? Kind);
public sealed record MoveRequest(long Version, string? RequestId, string? Kind, int? Factory, int? Color, int? Row, int[]? Columns);

public static class ApiEndpoints
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/api/health", () => Results.Json(new { ok = true }));

        var api = app.MapGroup("/api");
        api.MapGet("/me", (HttpContext c) => Results.Json(new { email = c.Email() }));
        api.MapGet("/games/{id}/events", EventStream.Handle);
        api.MapGet("/games", (GameService s) => Results.Json(s.List(), Json.Options));
        api.MapPost("/games", (HttpContext c, GameService s, CreateGameRequest r) => s.Create(c.Email(), r.Players).ToHttp());
        api.MapGet("/games/{id}", (HttpContext c, GameService s, string id) =>
            s.GetView(id, c.Email()) is { } v ? Results.Json(v, Json.Options) : ApiResult.Error(404, "not-found").ToHttp());
        api.MapPost("/games/{id}/seats/{idx:int}/claim", async (HttpContext c, GameService s, string id, int idx) =>
            (await s.Claim(id, idx, c.Email())).ToHttp());
        api.MapPost("/games/{id}/seats/{idx:int}/release", async (HttpContext c, GameService s, string id, int idx) =>
            (await s.Release(id, idx, c.Email())).ToHttp());
        api.MapPost("/games/{id}/seats/{idx:int}/kind", async (HttpContext c, GameService s, string id, int idx, SetKindRequest r) =>
            (await s.SetKind(id, idx, c.Email(), r.Kind)).ToHttp());
        api.MapPost("/games/{id}/start", async (HttpContext c, GameService s, string id) =>
            (await s.Start(id, c.Email())).ToHttp());
        api.MapPost("/games/{id}/seats/{idx:int}/to-bot", async (HttpContext c, GameService s, string id, int idx) =>
            (await s.ToBot(id, idx, c.Email())).ToHttp());
        api.MapPost("/games/{id}/seats/{idx:int}/take-back", async (HttpContext c, GameService s, string id, int idx) =>
            (await s.TakeBack(id, idx, c.Email())).ToHttp());
        api.MapPost("/games/{id}/moves", async (HttpContext c, GameService s, string id, MoveRequest r) =>
            (await s.Move(id, c.Email(), r)).ToHttp());
        api.MapDelete("/games/{id}", async (HttpContext c, GameService s, string id) =>
            (await s.Delete(id, c.Email())).ToHttp());
    }
}
