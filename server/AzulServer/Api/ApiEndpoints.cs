namespace AzulServer.Api;

public static class ApiEndpoints
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/api/health", () => Results.Json(new { ok = true }));
    }
}
