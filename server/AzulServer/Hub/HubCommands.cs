using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using AzulServer.Data;

namespace AzulServer.Hub;

/// `/app/AzulServer hub <command>` (spec 6.4): opens the SQLite database
/// directly (WAL lets it run beside the server), prints one JSON document.
public static class HubCommands
{
    const string Usage = "AzulServer hub status | bot-key | ping | retry <game_id>|--all-failed [--rebuild]";

    public static async Task<int> RunAsync(string[] args, TextWriter output, Func<string, string?> env, HttpMessageHandler? handler = null)
    {
        void Print(object value) => output.WriteLine(JsonSerializer.Serialize(value, Json.Options));
        try
        {
            var options = AzulOptions.FromEnvironment(env);
            var now = DateTime.UtcNow.ToString("O");
            switch (args)
            {
                case ["bot-key"]:
                    Print(new { botKey = BotIdentity.For(options).Key });
                    return 0;
                case ["status"]:
                {
                    using var c = new Db(options).Open();
                    var s = HubOutbox.Summary(c);
                    Print(new { botKey = BotIdentity.For(options).Key, counts = s.Counts, failed = s.Failed, oldestPendingCreatedAt = s.OldestPendingCreatedAt });
                    return 0;
                }
                case ["retry", var target, .. var rest] when rest is [] or ["--rebuild"]:
                {
                    var db = new Db(options);
                    bool rebuild = rest is ["--rebuild"];
                    List<string> ids;
                    if (target == "--all-failed") { using var c = db.Open(); ids = HubOutbox.FailedIds(c); }
                    else ids = [target];
                    var results = new List<object>();
                    foreach (var id in ids)
                    {
                        string outcome;
                        if (rebuild) outcome = HubOutbox.Rebuild(db, id, options.PublicOrigin, now);
                        else { using var c = db.Open(); outcome = HubOutbox.ResetForRetry(c, id, now); }
                        results.Add(new { gameId = id, outcome });
                    }
                    Print(new { results });
                    return 0;
                }
                case ["ping"]:
                {
                    if (!options.Hub.SenderConfigured)
                    {
                        Print(new { ok = false, status = (int?)null, message = "AZUL_HUB_URL and AZUL_HUB_KEY must both be set" });
                        return 1;
                    }
                    using var http = handler is null ? new HttpClient(HubSender.PrimaryHandler()) : new HttpClient(handler, disposeHandler: false);
                    http.Timeout = TimeSpan.FromSeconds(10);
                    using var req = new HttpRequestMessage(HttpMethod.Post, options.Hub.Url!.TrimEnd('/') + "/api/v1/results")
                    {
                        Content = new StringContent("{}", Encoding.UTF8, "application/json"),
                    };
                    req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", options.Hub.Key);
                    using var res = await http.SendAsync(req);
                    int status = (int)res.StatusCode;
                    // 422 = reached, key accepted, content type passed, empty report refused.
                    Print(new { ok = status == 422, status });
                    return status == 422 ? 0 : 1;
                }
                default:
                    Print(new { error = "usage", usage = Usage });
                    return 2;
            }
        }
        catch (Exception e)
        {
            Print(new { error = "internal", message = e.Message });
            return 1;
        }
    }
}
