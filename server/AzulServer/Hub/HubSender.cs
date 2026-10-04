using System.Net.Http.Headers;
using System.Text;
using AzulServer.Data;

namespace AzulServer.Hub;

/// Delivers hub_reports (spec 6.3). Nothing escapes ExecuteAsync: a
/// BackgroundService exception would stop the host and gameplay with it.
public sealed class HubSender(Db db, AzulOptions options, HubSignal signal, IHttpClientFactory http,
    TimeProvider time, ILogger<HubSender> log) : BackgroundService
{
    public const string HttpName = "hub";
    static readonly TimeSpan Lease = TimeSpan.FromMinutes(5);
    static readonly TimeSpan Stuck = TimeSpan.FromHours(1);
    /// Games whose report reconcile has already logged as an error.
    readonly HashSet<string> reconcileErrors = new();

    /// How long an idle sender waits for a wake before its next cycle.
    public TimeSpan IdleWait { get; init; } = TimeSpan.FromSeconds(60);

    public enum Outcome { Sent, Failed, RetryKey, RetryTransient, RetryConfig }

    /// hub 5.3: any 2xx delivered; 409/422 permanent; 401 keep and alert; 5xx
    /// and network errors keep; anything else is a configuration fault, kept.
    public static Outcome Classify(int? status) => status switch
    {
        >= 200 and < 300 => Outcome.Sent,
        409 or 422 => Outcome.Failed,
        401 => Outcome.RetryKey,
        null or >= 500 => Outcome.RetryTransient,
        _ => Outcome.RetryConfig,
    };

    public static TimeSpan Backoff(int attempts, double jitter01) =>
        TimeSpan.FromSeconds(Math.Min(60 * Math.Pow(2, attempts - 1), 3600) * (1 + 0.1 * jitter01));

    /// The "hub" client's handler. Redirects are not followed: Cloudflare
    /// Access answers a public-hostname URL with 302 to its login page, which
    /// a following client would fetch with GET and see as a 200 "sent".
    public static SocketsHttpHandler PrimaryHandler() => new() { AllowAutoRedirect = false };

    static string Stamp(DateTimeOffset t) => t.UtcDateTime.ToString("O");

    static string? Cut(string? s) => s is null ? null : s.Length <= 500 ? s : s[..500];

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        if (!options.Hub.SenderConfigured)
        {
            if (options.Hub.HalfConfigured)
                log.LogWarning("hub reporting is off: set both AZUL_HUB_URL and AZUL_HUB_KEY");
            return;
        }
        await Task.Yield();   // the first cycle's database work does not hold up host startup
        while (!ct.IsCancellationRequested)
        {
            try { await RunCycleAsync(ct); }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { return; }
            catch (Exception e) { log.LogError(e, "hub sender cycle failed"); }
            try { await signal.WaitAsync(IdleWait, ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    /// One pass: reconcile, send every due row, alert on stuck rows.
    public async Task<int> RunCycleAsync(CancellationToken ct)
    {
        var now = time.GetUtcNow();
        HubOutbox.QueueMissing(db, options.PublicOrigin, Stamp(now), log, reconcileErrors);
        List<string> due;
        using (var c = db.Open()) due = HubOutbox.DueIds(c, Stamp(now), 50);
        int sent = 0;
        foreach (var id in due)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                if (await SendOne(id, ct)) sent++;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
            catch (Exception e)
            {
                // The lease expires and the row is retried.
                log.LogError(e, "hub report for game {Game} could not be sent", id);
            }
        }
        AlertStuck();
        return sent;
    }

    async Task<bool> SendOne(string id, CancellationToken ct)
    {
        var start = time.GetUtcNow();
        HubClaim? claim;
        using (var c = db.Open()) claim = HubOutbox.TryClaim(c, id, Stamp(start), Stamp(start + Lease));
        if (claim is null) return false;

        int? status = null;
        string? detail = null;
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Post, options.Hub.Url!.TrimEnd('/') + "/api/v1/results")
            {
                Content = new StringContent(claim.Body, Encoding.UTF8, "application/json"),
            };
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", options.Hub.Key);
            using var res = await http.CreateClient(HttpName).SendAsync(req, ct);
            status = (int)res.StatusCode;
            detail = await res.Content.ReadAsStringAsync(ct);
        }
        catch (Exception e) when (e is HttpRequestException || (e is TaskCanceledException && !ct.IsCancellationRequested))
        {
            detail = e.Message;   // a network error, or the client's 10 s timeout
        }

        var outcome = Classify(status);
        var after = time.GetUtcNow();
        using var c2 = db.Open();
        // Recorded only under this claim's lease: a retry or rebuild while the
        // request was in flight cleared it, and then this answer is stale.
        bool owned = outcome switch
        {
            Outcome.Sent => HubOutbox.MarkSent(c2, id, claim.LeaseId, status!.Value, Stamp(after)),
            Outcome.Failed => HubOutbox.MarkFailed(c2, id, claim.LeaseId, status!.Value, Cut(detail)),
            _ => HubOutbox.MarkRetry(c2, id, claim.LeaseId, status, Cut(detail),
                Stamp(after + Backoff(claim.Attempts + 1, Random.Shared.NextDouble()))),
        };
        if (!owned)
        {
            log.LogInformation("stale hub outcome for game {Game} dropped ({Status}): the report was reset or rebuilt meanwhile", id, status);
            return false;
        }
        switch (outcome)
        {
            case Outcome.Sent:
                log.LogInformation("hub accepted the report of game {Game} ({Status})", id, status);
                return true;
            case Outcome.Failed:
                log.LogError("hub refused the report of game {Game} permanently ({Status}): {Body}", id, status, Cut(detail));
                return false;
            case Outcome.RetryKey:
                log.LogError("hub rejected the game key (401) for game {Game}; reports stay pending until AZUL_HUB_KEY is fixed and the server restarted", id);
                return false;
            case Outcome.RetryTransient:
                log.LogWarning("hub unavailable for game {Game} ({Status}): {Detail}", id, status, Cut(detail));
                return false;
            default:
                log.LogError("hub answered {Status} for game {Game}; check AZUL_HUB_URL", status, id);
                return false;
        }
    }

    void AlertStuck()
    {
        var now = time.GetUtcNow();
        using var c = db.Open();
        foreach (var id in HubOutbox.StuckIds(c, Stamp(now - Stuck), Stamp(now - Stuck)))
        {
            log.LogError("hub report for game {Game} has been pending for over an hour", id);
            HubOutbox.MarkAlerted(c, id, Stamp(now));
        }
    }
}
