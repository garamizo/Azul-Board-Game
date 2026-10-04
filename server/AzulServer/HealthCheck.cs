namespace AzulServer;

/// `AzulServer --healthcheck`: the runtime image has no curl.
public static class HealthCheck
{
    public static async Task<int> RunAsync()
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
            var port = Environment.GetEnvironmentVariable("ASPNETCORE_HTTP_PORTS")?.Split(';')[0] ?? "8080";
            var res = await http.GetAsync($"http://127.0.0.1:{port}/api/health");
            return res.IsSuccessStatusCode ? 0 : 1;
        }
        catch
        {
            return 1;
        }
    }
}
