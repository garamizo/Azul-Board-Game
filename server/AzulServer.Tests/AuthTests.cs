using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using AzulServer.Auth;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Time.Testing;
using Microsoft.IdentityModel.Tokens;

namespace AzulServer.Tests;

public sealed class AuthTests : IDisposable
{
    readonly RSA k1 = RSA.Create(2048);
    readonly RSA k2 = RSA.Create(2048);
    readonly FakeJwks jwks = new();
    readonly FakeTimeProvider clock = new(DateTimeOffset.UtcNow);
    readonly TestApp app;

    public AuthTests()
    {
        jwks.Json = Jwt.JwksJson((k1, "k1"));
        var root = Directory.CreateTempSubdirectory("azul-web-").FullName;
        File.WriteAllText(Path.Combine(root, "index.html"), "<!doctype html>");
        app = new TestApp(
            new AzulOptions { BotWorkers = 0, TeamDomain = Jwt.Team, Aud = Jwt.Aud, WebRoot = root, PublicOrigin = "https://azul.example" },
            s => { s.AddSingleton<IJwksFetcher>(jwks); s.AddSingleton<TimeProvider>(clock); });
    }

    public void Dispose() => app.Dispose();

    async Task<HttpResponseMessage> Me(string? token, string path = "/api/me")
    {
        var c = app.Client(email: null);
        var req = new HttpRequestMessage(HttpMethod.Get, path);
        if (token is not null) req.Headers.Add("Cf-Access-Jwt-Assertion", token);
        return await c.SendAsync(req);
    }

    [Fact]
    public async Task ValidTokenGivesTheEmail()
    {
        var res = await Me(Jwt.Token(k1, "k1"));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        Assert.Contains("alice@example.com", await res.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task EmailIsLowercased()
    {
        var res = await Me(Jwt.Token(k1, "k1", email: "Alice@Example.COM"));
        Assert.Contains("\"alice@example.com\"", await res.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("no token")]
    [InlineData("garbage")]
    [InlineData("wrong aud")]
    [InlineData("wrong iss")]
    [InlineData("expired")]
    [InlineData("wrong alg")]
    [InlineData("no email")]
    [InlineData("unknown key")]
    public async Task BadTokensGet401(string kind)
    {
        string? token = kind switch
        {
            "no token" => null,
            "garbage" => "not.a.jwt",
            "wrong aud" => Jwt.Token(k1, "k1", aud: "other"),
            "wrong iss" => Jwt.Token(k1, "k1", iss: "https://evil.example.com"),
            "expired" => Jwt.Token(k1, "k1", expires: DateTime.UtcNow.AddMinutes(-5)),
            "wrong alg" => Jwt.Token(k1, "k1", alg: SecurityAlgorithms.RsaSha384),
            "no email" => Jwt.Token(k1, "k1", email: null),
            "unknown key" => Jwt.Token(k2, "k2"),
            _ => throw new ArgumentException(kind),
        };
        Assert.Equal(HttpStatusCode.Unauthorized, (await Me(token)).StatusCode);
    }

    [Fact]
    public async Task HeaderEmailIsIgnored()
    {
        var c = app.Client(email: null);
        var req = new HttpRequestMessage(HttpMethod.Get, "/api/me");
        req.Headers.Add("Cf-Access-Authenticated-User-Email", "alice@example.com");
        Assert.Equal(HttpStatusCode.Unauthorized, (await c.SendAsync(req)).StatusCode);
    }

    [Fact]
    public async Task StaticFilesAreGuardedToo()
    {
        Assert.Equal(HttpStatusCode.Unauthorized, (await Me(null, "/")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k1, "k1"), "/")).StatusCode);
    }

    [Fact]
    public async Task RotatedKeyIsPickedUpAfterTheRefreshInterval()
    {
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
        jwks.Json = Jwt.JwksJson((k1, "k1"), (k2, "k2"));
        // Within 30 s of the last fetch the unknown kid is not refetched.
        Assert.Equal(HttpStatusCode.Unauthorized, (await Me(Jwt.Token(k2, "k2"))).StatusCode);
        clock.Advance(TimeSpan.FromSeconds(31));
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k2, "k2"))).StatusCode);
    }

    [Fact]
    public async Task KeysUnavailableGives503()
    {
        jwks.Fail = true;
        Assert.Equal(HttpStatusCode.ServiceUnavailable, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
    }

    [Fact]
    public async Task CachedKeysKeepWorkingWhenRefreshFails()
    {
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
        jwks.Fail = true;
        clock.Advance(TimeSpan.FromHours(2));  // stale: triggers a background refresh that fails
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
    }

    [Fact]
    public async Task RotationDuringAnOutageGives503()
    {
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
        jwks.Fail = true;
        clock.Advance(TimeSpan.FromSeconds(31));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, (await Me(Jwt.Token(k2, "k2"))).StatusCode);
    }

    [Fact]
    public async Task CallerCancellingTheFetchDoesNotPoisonTheNextRequest()
    {
        var cache = app.Service<JwksCache>();  // starts the server before the clock below
        jwks.Delay = TimeSpan.FromMilliseconds(300);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        try { await cache.LookupAsync("k1", cts.Token); }
        catch (OperationCanceledException) { }
        jwks.Delay = TimeSpan.Zero;
        await Task.Delay(500);
        Assert.Equal(HttpStatusCode.OK, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
    }

    [Fact]
    public async Task HangingJwksGives503()
    {
        jwks.Delay = TimeSpan.FromSeconds(30);
        var started = DateTime.UtcNow;
        Assert.Equal(HttpStatusCode.ServiceUnavailable, (await Me(Jwt.Token(k1, "k1"))).StatusCode);
        Assert.True(DateTime.UtcNow - started < TimeSpan.FromSeconds(10));
    }

    async Task<HttpResponseMessage> PostMe(string? contentType, string? origin)
    {
        var c = app.Client(email: null);
        var req = new HttpRequestMessage(HttpMethod.Post, "/api/me");
        req.Headers.Add("Cf-Access-Jwt-Assertion", Jwt.Token(k1, "k1"));
        if (origin is not null) req.Headers.Add("Origin", origin);
        req.Content = contentType is null ? null : new StringContent("{}", Encoding.UTF8, contentType);
        return await c.SendAsync(req);
    }

    [Fact]
    public async Task NonJsonPostIsRefused()
    {
        Assert.Equal(HttpStatusCode.Forbidden, (await PostMe("text/plain", null)).StatusCode);
        Assert.Equal(HttpStatusCode.Forbidden, (await PostMe(null, null)).StatusCode);
    }

    [Fact]
    public async Task ForeignOriginIsRefused()
    {
        Assert.Equal(HttpStatusCode.Forbidden, (await PostMe("application/json", "https://evil.example")).StatusCode);
    }

    [Fact]
    public async Task PublicOriginPassesCsrfWhateverTheHost()
    {
        // Host is "localhost" in TestServer, like "app:8080" behind cloudflared.
        var res = await PostMe("application/json", "https://azul.example");
        Assert.NotEqual(HttpStatusCode.Forbidden, res.StatusCode);
    }

    [Fact]
    public async Task DevModeIdentityFromHeaderOrCookie()
    {
        using var dev = new TestApp();
        var c = dev.Client(email: null);
        Assert.Contains("dev@localhost", await c.GetStringAsync("/api/me"));
        var req = new HttpRequestMessage(HttpMethod.Get, "/api/me");
        req.Headers.Add("Cookie", "azul_dev_user=Bob@Example.com");
        Assert.Contains("bob@example.com", await (await c.SendAsync(req)).Content.ReadAsStringAsync());
        Assert.Contains("alice@example.com", await dev.Client("alice@example.com").GetStringAsync("/api/me"));
    }
}
