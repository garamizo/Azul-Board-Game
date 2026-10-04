using System.Globalization;

namespace AzulServer;

public sealed record AzulOptions
{
    public string? TeamDomain { get; init; }
    public string? Aud { get; init; }
    public string? PublicOrigin { get; init; }
    public string DataDir { get; init; } = "/data";
    public string? WebRoot { get; init; }
    public int BotWorkers { get; init; } = 2;
    public double BotThinkSeconds { get; init; } = 3.0;
    public double MinMoveDelaySeconds { get; init; } = 1.0;
    public double SweepSeconds { get; init; } = 30;
    public double SseHeartbeatSeconds { get; init; } = 20;
    public double SseMaxMinutes { get; init; } = 30;

    /// No Access configuration: identity comes from X-Dev-User / azul_dev_user.
    public bool DevMode => TeamDomain is null;

    public static AzulOptions FromEnvironment(Func<string, string?> env)
    {
        string? Get(string key) => env(key) is { } v && !string.IsNullOrWhiteSpace(v) ? v.Trim() : null;
        double Num(string key, double fallback) =>
            Get(key) is { } v ? double.Parse(v, CultureInfo.InvariantCulture) : fallback;

        var team = Get("AZUL_ACCESS_TEAM_DOMAIN");
        var aud = Get("AZUL_ACCESS_AUD");
        if ((team is null) != (aud is null))
            throw new InvalidOperationException(
                "Set both AZUL_ACCESS_TEAM_DOMAIN and AZUL_ACCESS_AUD, or neither (dev mode).");
        if (team is not null)
        {
            if (team.StartsWith("https://", StringComparison.OrdinalIgnoreCase)) team = team[8..];
            team = team.TrimEnd('/');
        }
        return new AzulOptions
        {
            TeamDomain = team,
            Aud = aud,
            PublicOrigin = Get("AZUL_PUBLIC_ORIGIN")?.TrimEnd('/'),
            DataDir = Get("AZUL_DATA_DIR") ?? "/data",
            WebRoot = Get("AZUL_WEB_ROOT"),
            BotWorkers = (int)Num("AZUL_BOT_WORKERS", 2),
            BotThinkSeconds = Num("AZUL_BOT_THINK_SECONDS", 3),
            MinMoveDelaySeconds = Num("AZUL_MIN_MOVE_DELAY_SECONDS", 1),
            SweepSeconds = Num("AZUL_SWEEP_SECONDS", 30),
            SseHeartbeatSeconds = Num("AZUL_SSE_HEARTBEAT_SECONDS", 20),
            SseMaxMinutes = Num("AZUL_SSE_MAX_MINUTES", 30),
        };
    }
}
