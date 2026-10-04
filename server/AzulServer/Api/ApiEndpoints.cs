using AzulServer.Auth;

namespace AzulServer.Api;

public static class ApiEndpoints
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/api/health", () => Results.Json(new { ok = true }));
        app.MapGet("/api/me", (HttpContext c) => Results.Json(new { email = c.Email() }));
    }
}
