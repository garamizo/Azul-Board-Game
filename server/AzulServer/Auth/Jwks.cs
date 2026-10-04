using Microsoft.IdentityModel.Tokens;

namespace AzulServer.Auth;

public interface IJwksFetcher
{
    Task<string> FetchAsync(string teamDomain, CancellationToken ct);
}

public sealed class HttpJwksFetcher(IHttpClientFactory clients) : IJwksFetcher
{
    public async Task<string> FetchAsync(string teamDomain, CancellationToken ct)
    {
        var http = clients.CreateClient();
        http.Timeout = TimeSpan.FromSeconds(5);
        return await http.GetStringAsync($"https://{teamDomain}/cdn-cgi/access/certs", ct);
    }
}

public enum KeyLookupResult { Found, Unknown, Unavailable }

/// Access signing keys. Cached for an hour; an unknown kid triggers a refresh
/// at most every 30 s. Never fails open: no keys means Unavailable (503).
public sealed class JwksCache(IJwksFetcher fetcher, AzulOptions options, TimeProvider time, ILogger<JwksCache> log)
{
    public static readonly TimeSpan MaxAge = TimeSpan.FromHours(1);
    public static readonly TimeSpan MinRefreshInterval = TimeSpan.FromSeconds(30);
    static readonly TimeSpan FetchTimeout = TimeSpan.FromSeconds(5);

    readonly SemaphoreSlim gate = new(1, 1);
    IReadOnlyList<SecurityKey> keys = [];
    DateTimeOffset fetchedAt = DateTimeOffset.MinValue;
    DateTimeOffset lastAttempt = DateTimeOffset.MinValue;
    bool lastOk;

    public IReadOnlyList<SecurityKey> Keys => keys;

    public async Task<KeyLookupResult> LookupAsync(string? kid, CancellationToken ct)
    {
        if (kid is not null && Has(kid))
        {
            if (time.GetUtcNow() - fetchedAt > MaxAge)
                _ = Task.Run(() => RefreshAsync(CancellationToken.None));
            return KeyLookupResult.Found;
        }
        bool ok = await RefreshAsync(ct);
        if (kid is not null && Has(kid)) return KeyLookupResult.Found;
        // A kid we cannot find after a failed refresh may be a rotated key:
        // keys unavailable (503), not a bad token (401).
        return ok ? KeyLookupResult.Unknown : KeyLookupResult.Unavailable;
    }

    bool Has(string kid) => keys.Any(k => k.KeyId == kid);

    async Task<bool> RefreshAsync(CancellationToken ct)
    {
        await gate.WaitAsync(ct);
        try
        {
            var now = time.GetUtcNow();
            if (lastAttempt != DateTimeOffset.MinValue && now - lastAttempt < MinRefreshInterval)
                return lastOk;
            lastAttempt = now;
            try
            {
                // Not linked to the caller: an aborted request must not leave the cache
                // with a failed attempt recorded for the next 30 s.
                using var timeout = new CancellationTokenSource(FetchTimeout);
                var json = await fetcher.FetchAsync(options.TeamDomain!, timeout.Token);
                keys = new JsonWebKeySet(json).GetSigningKeys().ToList();
                fetchedAt = now;
                lastOk = true;
            }
            catch (Exception e)
            {
                log.LogWarning(e, "Fetching Access signing keys failed");
                lastOk = false;
            }
            return lastOk;
        }
        finally
        {
            gate.Release();
        }
    }
}
