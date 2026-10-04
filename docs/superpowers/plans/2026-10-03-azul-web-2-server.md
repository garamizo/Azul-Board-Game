# Azul Web — Plan 2 of 4: Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An ASP.NET Core server that hosts the engine: Access JWT authentication, lobby and seat rules, transactional moves with idempotent retries, per-viewer live updates over SSE, and MCTS bots plus forced moves driven from the database.

**Architecture:** One minimal-API app (`server/AzulServer`). `GameService` is the only writer: one lock per game, every change applied to a fresh engine object and committed in one SQLite transaction, then announced as `(gameId, version)` to the `EventHub` and the `ServerMoveQueue`. SSE streams project the view for their own viewer. `BotScheduler` workers take game ids from a deduplicated queue that a 30 s sweep refills from the database.

**Tech Stack:** .NET 10, ASP.NET Core minimal APIs, Microsoft.Data.Sqlite 10.0.12, Microsoft.IdentityModel.JsonWebTokens 8.23.0, xUnit + Microsoft.AspNetCore.Mvc.Testing 10.0.12, Microsoft.Extensions.TimeProvider.Testing.

**Spec:** `docs/superpowers/specs/2026-10-03-azul-web-design.md` (§4). Requires Plan 1 (`docs/superpowers/plans/2026-10-03-azul-web-1-engine.md`) to be done.

## Global Constraints

Everything in Plan 1's Global Constraints applies. In addition:

- Environment variables (exact names): `AZUL_ACCESS_TEAM_DOMAIN`, `AZUL_ACCESS_AUD` (both or neither; neither = dev mode), `AZUL_PUBLIC_ORIGIN` (e.g. `https://azul.signalwave.dev`), `AZUL_DATA_DIR` (default `/data`), `AZUL_WEB_ROOT` (built SPA), `AZUL_BOT_WORKERS` (default 2; 0 = no server moves, tests only), `AZUL_BOT_THINK_SECONDS` (default 3), `AZUL_MIN_MOVE_DELAY_SECONDS` (default 1), `AZUL_SWEEP_SECONDS` (default 30), `AZUL_SSE_HEARTBEAT_SECONDS` (default 20), `AZUL_SSE_MAX_MINUTES` (default 30).
- JWT: RS256 only; `aud` = AUD; `iss` = `https://<team domain>`; `exp` required; `nbf` if present; 60 s leeway; email claim lowercased; keys from `https://<team domain>/cdn-cgi/access/certs`, cached 1 h, refresh on unknown `kid` at most every 30 s; bad token 401, keys unavailable 503. `Cf-Access-Authenticated-User-Email` is never read.
- Dev mode identity: header `X-Dev-User`, else cookie `azul_dev_user`, else `dev@localhost`.
- Every route except `GET /api/health` requires identity (spec A6).
- JSON is `System.Text.Json` web defaults (camelCase) everywhere (`AzulServer.Json.Options`).
- Error bodies are `{"error": "<code>", "view": <GameView or null>}`.
- Dev server port 5080 on 127.0.0.1 (`make dev-server`). It is fixed: two worktrees running it at once collide.

### Deviations from the spec found while planning

1. `AZUL_PUBLIC_ORIGIN` is new: the CSRF `Origin` check accepts it in addition to the request's `Host`, because the Host header cloudflared forwards was not verified (Review Focus 2).
2. Dev identity can also come from the cookie `azul_dev_user`, because `EventSource` cannot send headers (needed by Plan 3's e2e tests).
3. `board.phase` is `"over"` once the game is finished (spec lists `take` / `wall` only).
4. Test-only knobs: `AZUL_BOT_WORKERS=0`, `AZUL_SWEEP_SECONDS`, `AZUL_MIN_MOVE_DELAY_SECONDS`, `AZUL_SSE_HEARTBEAT_SECONDS`, `AZUL_SSE_MAX_MINUTES`.
5. No in-memory view cache (spec §4.6 said "cached per version"): every read loads the game, seats and last move in one read transaction. A cache raced with deletes (Codex plan review), and at four players the extra reads cost nothing.
6. Each SSE heartbeat also compares the stored version with the last one sent and sends the newer state, so a notification lost after a commit is repaired within one heartbeat.

## Review Focus

1. A player double-taps Confirm, or retries after a dropped connection: the second request must not move twice or fail confusingly → Task 12 `SameRequestTwiceAtOnceMovesOnce`.
2. Behind cloudflared the `Host` the origin sees may not match the browser's `Origin`: a real POST must still pass the CSRF check when `Origin` equals `AZUL_PUBLIC_ORIGIN` → Task 8 `PublicOriginPassesCsrfWhateverTheHost`.
3. A game row whose `state_json` fails snapshot validation (a hand-edited or corrupted DB): reads must answer 500 for that game only, not crash list or the sweep → Task 11 `CorruptGameDoesNotBreakTheListOrTheSweep`.
4. The JWKS endpoint is slow or hangs: requests must not pile up forever; the fetch has a 5 s timeout and a hung fetch gives 503 → Task 8 `HangingJwksGives503`.
5. A seat claimed by an email with different letter case (`Alice@X.com` vs `alice@x.com`): it is the same person → Task 8 `EmailIsLowercased` and Task 11 `ClaimIsCaseInsensitiveInDevMode`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `server/AzulServer/AzulServer.csproj` | Web project referencing `AzulLibrary`. |
| `server/AzulServer/Program.cs` | Composition root, middleware order, static files, `--healthcheck`. |
| `server/AzulServer/AzulOptions.cs` | Options record and `FromEnvironment`. |
| `server/AzulServer/Json.cs` | Shared JSON options and helpers. |
| `server/AzulServer/HealthCheck.cs` | `--healthcheck` client. |
| `server/AzulServer/Auth/Jwks.cs` | `IJwksFetcher`, `HttpJwksFetcher`, `JwksCache`. |
| `server/AzulServer/Auth/AccessVerifier.cs` | JWT validation. |
| `server/AzulServer/Auth/AccessAuthMiddleware.cs` | Identity for every request; `HttpContext.Email()`. |
| `server/AzulServer/Auth/CsrfMiddleware.cs` | JSON content type and Origin checks. |
| `server/AzulServer/Data/Db.cs` | Connections, migrations. |
| `server/AzulServer/Data/GameStore.cs` | Records and SQL for games, seats, moves. |
| `server/AzulServer/Games/Views.cs` | View records (JSON contract with the client). |
| `server/AzulServer/Games/Projection.cs` | Engine state + seats → per-viewer `GameView`. |
| `server/AzulServer/Games/GameService.cs` | Lobby, moves, server moves, deletion. |
| `server/AzulServer/Games/EventHub.cs` | Version notices per game. |
| `server/AzulServer/Games/ServerMoveQueue.cs` | Deduplicated work queue + search cancellation. |
| `server/AzulServer/Games/BotScheduler.cs` | Workers, sweep, `IBotBrain`, `MctsBrain`, `GreedyBrain`, `IFaultInjector`. |
| `server/AzulServer/Api/ApiEndpoints.cs` | Route table and request records. |
| `server/AzulServer/Api/EventStream.cs` | SSE endpoint. |
| `server/AzulServer.Tests/` | `TestApp.cs`, `Jwt.cs`, `Play.cs` helpers; one test file per task. |

---

### Task 7: Server skeleton, options, health, static files

**Files:**
- Create: `server/AzulServer/AzulServer.csproj`, `Program.cs`, `AzulOptions.cs`, `Json.cs`, `HealthCheck.cs`, `Api/ApiEndpoints.cs` (health only for now); `server/AzulServer.Tests/` (xunit template), `server/AzulServer.Tests/TestApp.cs`, `server/AzulServer.Tests/SkeletonTests.cs`
- Modify: `Azul.slnx`, `Makefile` (add `dev-server`)

**Interfaces:**
- Produces: `public sealed record AzulOptions` (properties below) with `static AzulOptions FromEnvironment(Func<string, string?> env)` and `bool DevMode`; `AzulServer.Json` (`Options`, `Serialize`, `Deserialize<T>`); `public partial class Program`; test helper `TestApp`.

- [ ] **Step 1: Create the projects**

```bash
cd /home/garamizo/Azul-Board-Game-web
mkdir -p server/AzulServer/Api server/AzulServer/Auth server/AzulServer/Data server/AzulServer/Games
make dotnet ARGS="new xunit -o server/AzulServer.Tests"
rm -f server/AzulServer.Tests/UnitTest1.cs
```

`server/AzulServer/AzulServer.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk.Web">

  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <InvariantGlobalization>true</InvariantGlobalization>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="../../AzulLibrary/AzulLibrary.csproj" />
    <PackageReference Include="Microsoft.Data.Sqlite" Version="10.0.12" />
    <PackageReference Include="Microsoft.IdentityModel.JsonWebTokens" Version="8.23.0" />
  </ItemGroup>

  <ItemGroup>
    <InternalsVisibleTo Include="AzulServer.Tests" />
  </ItemGroup>

</Project>
```

```bash
make dotnet ARGS="add server/AzulServer.Tests/AzulServer.Tests.csproj reference server/AzulServer/AzulServer.csproj"
make dotnet ARGS="add server/AzulServer.Tests/AzulServer.Tests.csproj package Microsoft.AspNetCore.Mvc.Testing --version 10.0.12"
make dotnet ARGS="add server/AzulServer.Tests/AzulServer.Tests.csproj package Microsoft.Extensions.TimeProvider.Testing"
make dotnet ARGS="sln Azul.slnx add server/AzulServer/AzulServer.csproj server/AzulServer.Tests/AzulServer.Tests.csproj"
grep -n '<Using Include="Xunit"' server/AzulServer.Tests/AzulServer.Tests.csproj
```

If the grep prints nothing, add `<ItemGroup><Using Include="Xunit" /></ItemGroup>` to the test csproj.

- [ ] **Step 2: Write the test helper and failing tests**

`server/AzulServer.Tests/TestApp.cs`:

```csharp
using System.Net.Http.Json;
using AzulServer.Games;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;

namespace AzulServer.Tests;

/// One server instance on its own temporary data directory. Bots are off
/// (BotWorkers = 0) unless a test turns them on; the brain is GreedyBrain.
public sealed class TestApp : IDisposable
{
    public string DataDir { get; }
    public AzulOptions Options { get; }
    readonly WebApplicationFactory<Program> factory;

    public TestApp(AzulOptions? options = null, Action<IServiceCollection>? services = null, string? dataDir = null)
    {
        DataDir = dataDir ?? Directory.CreateTempSubdirectory("azul-test-").FullName;
        Options = (options ?? new AzulOptions { BotWorkers = 0 }) with { DataDir = DataDir };
        factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseEnvironment("Testing");
            b.ConfigureTestServices(s =>
            {
                s.AddSingleton(Options);
                s.AddSingleton<IBotBrain, GreedyBrain>();
                services?.Invoke(s);
            });
        });
    }

    public HttpClient Client(string? email = "alice@example.com")
    {
        var c = factory.CreateClient(new WebApplicationFactoryClientOptions { AllowAutoRedirect = false });
        if (email is not null) c.DefaultRequestHeaders.Add("X-Dev-User", email);
        return c;
    }

    public T Service<T>() where T : notnull => factory.Services.GetRequiredService<T>();

    public void Dispose()
    {
        factory.Dispose();
        SqliteConnection.ClearAllPools();
    }
}

public static class HttpExtensions
{
    public static Task<HttpResponseMessage> Post(this HttpClient c, string url, object? body = null) =>
        c.PostAsJsonAsync(url, body ?? new { }, Json.Options);

    public static async Task<T> Read<T>(this HttpResponseMessage r) =>
        (await r.Content.ReadFromJsonAsync<T>(Json.Options))!;
}
```

(`GreedyBrain` and `IBotBrain` come in Task 14. Until then, create `server/AzulServer/Games/BotScheduler.cs` with just these two so the helper compiles:)

```csharp
using Azul;

namespace AzulServer.Games;

public interface IBotBrain
{
    Move ChooseMove(Game game, CancellationToken ct);
}

/// Deterministic and instant; used by tests.
public sealed class GreedyBrain : IBotBrain
{
    public Move ChooseMove(Game game, CancellationToken ct) => game.GetGreedyMove();
}
```

`server/AzulServer.Tests/SkeletonTests.cs`:

```csharp
using System.Net;

namespace AzulServer.Tests;

public class SkeletonTests
{
    [Fact]
    public async Task HealthAnswersWithoutIdentity()
    {
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, TeamDomain = "team.example.com", Aud = "aud" });
        var res = await app.Client(email: null).GetAsync("/api/health");
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
    }

    [Fact]
    public void OptionsNeedBothAccessValuesOrNeither()
    {
        Assert.Throws<InvalidOperationException>(() => AzulOptions.FromEnvironment(k => k == "AZUL_ACCESS_AUD" ? "x" : null));
        Assert.Throws<InvalidOperationException>(() => AzulOptions.FromEnvironment(k => k == "AZUL_ACCESS_TEAM_DOMAIN" ? "x" : null));
        Assert.True(AzulOptions.FromEnvironment(_ => null).DevMode);
    }

    [Fact]
    public void TeamDomainIsNormalised()
    {
        var o = AzulOptions.FromEnvironment(k => k switch
        {
            "AZUL_ACCESS_TEAM_DOMAIN" => " https://team.cloudflareaccess.com/ ",
            "AZUL_ACCESS_AUD" => "aud",
            _ => null,
        });
        Assert.Equal("team.cloudflareaccess.com", o.TeamDomain);
        Assert.False(o.DevMode);
    }

    [Fact]
    public async Task SpaFilesAndFallback()
    {
        var root = Directory.CreateTempSubdirectory("azul-web-").FullName;
        File.WriteAllText(Path.Combine(root, "index.html"), "<!doctype html><title>Azul</title>");
        File.WriteAllText(Path.Combine(root, "app.js"), "console.log(1)");
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, WebRoot = root });
        var c = app.Client();
        Assert.Equal("console.log(1)", await c.GetStringAsync("/app.js"));
        Assert.Contains("<title>Azul</title>", await c.GetStringAsync("/g/abc123"));
        Assert.Contains("<title>Azul</title>", await c.GetStringAsync("/"));
        Assert.Equal(HttpStatusCode.NotFound, (await c.GetAsync("/api/nope")).StatusCode);
    }
}
```

- [ ] **Step 3: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~SkeletonTests`
Expected: compile errors (`AzulOptions`, `Program` missing).

- [ ] **Step 4: Implement**

`server/AzulServer/AzulOptions.cs`:

```csharp
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
```

`server/AzulServer/Json.cs`:

```csharp
using System.Text.Json;

namespace AzulServer;

public static class Json
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);

    public static string Serialize<T>(T value) => JsonSerializer.Serialize(value, Options);

    public static T Deserialize<T>(string json) =>
        JsonSerializer.Deserialize<T>(json, Options) ?? throw new JsonException("null JSON");
}
```

`server/AzulServer/HealthCheck.cs`:

```csharp
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
```

`server/AzulServer/Api/ApiEndpoints.cs` (grows in later tasks):

```csharp
namespace AzulServer.Api;

public static class ApiEndpoints
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/api/health", () => Results.Json(new { ok = true }));
    }
}
```

`server/AzulServer/Program.cs`:

```csharp
using AzulServer;
using AzulServer.Api;
using Microsoft.Extensions.FileProviders;

if (args.Contains("--healthcheck"))
    return await HealthCheck.RunAsync();

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton(sp =>
    AzulOptions.FromEnvironment(key => sp.GetRequiredService<IConfiguration>()[key]));
builder.Services.AddSingleton(TimeProvider.System);
builder.Services.AddHttpClient();

var app = builder.Build();
var options = app.Services.GetRequiredService<AzulOptions>();  // fail fast on bad configuration

// Unhandled errors (a corrupt stored game, say) answer 500 for that request
// only, as JSON; TestServer would otherwise rethrow them into the test.
app.UseExceptionHandler(errors => errors.Run(async ctx =>
{
    ctx.Response.StatusCode = StatusCodes.Status500InternalServerError;
    await ctx.Response.WriteAsJsonAsync(new { error = "internal" });
}));

// (Task 8 adds the identity and CSRF middleware here, before static files.)

PhysicalFileProvider? files = options.WebRoot is { } webRoot && Directory.Exists(webRoot)
    ? new PhysicalFileProvider(Path.GetFullPath(webRoot))
    : null;
if (files is not null)
{
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = files });
    app.UseStaticFiles(new StaticFileOptions { FileProvider = files });
}
// Explicit, and after the static files: StaticFileMiddleware does nothing for
// a request that routing already matched to an endpoint (the fallback below
// matches everything).
app.UseRouting();

ApiEndpoints.Map(app);

if (files is not null)
{
    // Client-side routes (/g/<id>) get the SPA shell; unknown /api paths stay 404.
    app.MapFallback(async ctx =>
    {
        if (ctx.Request.Path.StartsWithSegments("/api"))
        {
            ctx.Response.StatusCode = StatusCodes.Status404NotFound;
            return;
        }
        ctx.Response.ContentType = "text/html; charset=utf-8";
        await ctx.Response.SendFileAsync(files.GetFileInfo("index.html"));
    });
}

app.Run();
return 0;

public partial class Program { }
```

Add to the `Makefile` (after `desktop-smoke`), and add `dev-server` to `.PHONY`:

```make
# The server in dev mode (no Access; identity from X-Dev-User or the
# azul_dev_user cookie) on http://127.0.0.1:5080, serving web/dist if built.
dev-server: | $(NUGET_DIR)
	mkdir -p .data/dev
	docker run --rm -i --user $(UID):$(GID) -p 127.0.0.1:5080:8080 \
		-e HOME=/tmp -e DOTNET_CLI_HOME=/tmp -e NUGET_PACKAGES=/nuget -e DOTNET_NOLOGO=1 \
		-e ASPNETCORE_URLS=http://0.0.0.0:8080 -e AZUL_DATA_DIR=/src/.data/dev \
		-e AZUL_WEB_ROOT=/src/web/dist -e AZUL_BOT_THINK_SECONDS=$${AZUL_BOT_THINK_SECONDS:-1} \
		-v $(CURDIR):/src -v $(NUGET_DIR):/nuget -w /src $(SDK_IMAGE) \
		dotnet run --project server/AzulServer --no-launch-profile
```

- [ ] **Step 5: Run the tests**

Run: `make test FILTER=FullyQualifiedName~SkeletonTests`
Expected: 4 pass. (`HealthAnswersWithoutIdentity` passes now trivially; Task 8 makes it meaningful.)

- [ ] **Step 6: Commit**

```bash
git add -A server Azul.slnx Makefile
git commit -m "server: skeleton, options, health, SPA hosting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Access authentication and CSRF (spec §4.1, §4.4 CSRF)

**Files:**
- Create: `server/AzulServer/Auth/Jwks.cs`, `Auth/AccessVerifier.cs`, `Auth/AccessAuthMiddleware.cs`, `Auth/CsrfMiddleware.cs`; `server/AzulServer.Tests/Jwt.cs`, `server/AzulServer.Tests/AuthTests.cs`
- Modify: `server/AzulServer/Program.cs`, `server/AzulServer/Api/ApiEndpoints.cs` (`GET /api/me`). The CSRF tests POST to `/api/me`: a request that passes the middleware gets 405 from routing, one that fails gets 403.

**Interfaces:**
- Produces: `IJwksFetcher { Task<string> FetchAsync(string teamDomain, CancellationToken ct) }`, `JwksCache`, `AccessVerifier.VerifyAsync(string? token, CancellationToken) → Task<AuthResult>`, `enum AuthOutcome { Ok, Unauthorized, Unavailable }`, `AccessAuthMiddleware`, `CsrfMiddleware`, extension `string HttpContext.Email()`.

- [ ] **Step 1: Write the JWT helper and failing tests**

`server/AzulServer.Tests/Jwt.cs`:

```csharp
using System.Security.Cryptography;
using AzulServer.Auth;
using Microsoft.IdentityModel.JsonWebTokens;
using Microsoft.IdentityModel.Tokens;

namespace AzulServer.Tests;

public sealed class FakeJwks : IJwksFetcher
{
    public string Json = "{\"keys\":[]}";
    public bool Fail;
    public TimeSpan Delay = TimeSpan.Zero;
    public int Calls;

    public async Task<string> FetchAsync(string teamDomain, CancellationToken ct)
    {
        Interlocked.Increment(ref Calls);
        if (Delay > TimeSpan.Zero) await Task.Delay(Delay, ct);
        if (Fail) throw new HttpRequestException("JWKS down");
        return Json;
    }
}

public static class Jwt
{
    public const string Team = "team.example.com";
    public const string Aud = "aud-1";

    public static string JwksJson(params (RSA Rsa, string Kid)[] keys) =>
        "{\"keys\":[" + string.Join(",", keys.Select(k =>
        {
            var p = k.Rsa.ExportParameters(false);
            return $"{{\"kty\":\"RSA\",\"kid\":\"{k.Kid}\",\"use\":\"sig\",\"alg\":\"RS256\"," +
                   $"\"n\":\"{Base64UrlEncoder.Encode(p.Modulus)}\",\"e\":\"{Base64UrlEncoder.Encode(p.Exponent)}\"}}";
        })) + "]}";

    public static string Token(RSA rsa, string kid, string? email = "alice@example.com",
        string iss = "https://" + Team, string aud = Aud, DateTime? expires = null,
        string alg = SecurityAlgorithms.RsaSha256)
    {
        var exp = expires ?? DateTime.UtcNow.AddMinutes(10);
        var claims = new Dictionary<string, object>();
        if (email is not null) claims["email"] = email;
        return new JsonWebTokenHandler().CreateToken(new SecurityTokenDescriptor
        {
            Issuer = iss,
            Audience = aud,
            Claims = claims,
            NotBefore = exp.AddHours(-1),
            IssuedAt = exp.AddHours(-1),
            Expires = exp,
            SigningCredentials = new SigningCredentials(new RsaSecurityKey(rsa) { KeyId = kid }, alg),
        });
    }
}
```

`server/AzulServer.Tests/AuthTests.cs`:

```csharp
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~AuthTests`
Expected: compile errors (`IJwksFetcher` missing).

- [ ] **Step 3: Implement**

`server/AzulServer/Auth/Jwks.cs`:

```csharp
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
                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
                timeout.CancelAfter(FetchTimeout);
                var json = await fetcher.FetchAsync(options.TeamDomain!, timeout.Token);
                keys = new JsonWebKeySet(json).GetSigningKeys().ToList();
                fetchedAt = now;
                lastOk = true;
            }
            catch (Exception e) when (e is not OperationCanceledException || !ct.IsCancellationRequested)
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
```

`server/AzulServer/Auth/AccessVerifier.cs`:

```csharp
using Microsoft.IdentityModel.JsonWebTokens;
using Microsoft.IdentityModel.Tokens;

namespace AzulServer.Auth;

public enum AuthOutcome { Ok, Unauthorized, Unavailable }

public sealed record AuthResult(AuthOutcome Outcome, string? Email);

public sealed class AccessVerifier(AzulOptions options, JwksCache cache)
{
    readonly JsonWebTokenHandler handler = new();

    public async Task<AuthResult> VerifyAsync(string? token, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(token))
            return new(AuthOutcome.Unauthorized, null);
        string? kid;
        try { kid = new JsonWebToken(token).Kid; }
        catch (Exception) { return new(AuthOutcome.Unauthorized, null); }

        var lookup = await cache.LookupAsync(kid, ct);
        if (lookup == KeyLookupResult.Unavailable)
            return new(AuthOutcome.Unavailable, null);

        var result = await handler.ValidateTokenAsync(token, new TokenValidationParameters
        {
            ValidIssuer = $"https://{options.TeamDomain}",
            ValidAudience = options.Aud,
            IssuerSigningKeys = cache.Keys,
            ValidAlgorithms = [SecurityAlgorithms.RsaSha256],
            RequireSignedTokens = true,
            RequireExpirationTime = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ClockSkew = TimeSpan.FromSeconds(60),
        });
        if (!result.IsValid)
            return new(AuthOutcome.Unauthorized, null);
        var email = result.Claims.TryGetValue("email", out var e) ? e as string : null;
        return string.IsNullOrWhiteSpace(email)
            ? new(AuthOutcome.Unauthorized, null)
            : new(AuthOutcome.Ok, email.Trim().ToLowerInvariant());
    }
}
```

`server/AzulServer/Auth/AccessAuthMiddleware.cs`:

```csharp
namespace AzulServer.Auth;

/// Identity for every request except /api/health (spec A6). In Access mode
/// it comes only from the verified JWT; in dev mode from X-Dev-User or the
/// azul_dev_user cookie.
public sealed class AccessAuthMiddleware(RequestDelegate next, AzulOptions options, AccessVerifier verifier)
{
    public const string EmailKey = "azul.email";

    public async Task InvokeAsync(HttpContext ctx)
    {
        if (ctx.Request.Path.Equals("/api/health", StringComparison.OrdinalIgnoreCase))
        {
            await next(ctx);
            return;
        }
        string email;
        if (options.DevMode)
        {
            email = (ctx.Request.Headers["X-Dev-User"].FirstOrDefault()
                     ?? ctx.Request.Cookies["azul_dev_user"]
                     ?? "dev@localhost").Trim().ToLowerInvariant();
        }
        else
        {
            var result = await verifier.VerifyAsync(ctx.Request.Headers["Cf-Access-Jwt-Assertion"].FirstOrDefault(), ctx.RequestAborted);
            if (result.Outcome != AuthOutcome.Ok)
            {
                ctx.Response.StatusCode = result.Outcome == AuthOutcome.Unavailable ? 503 : 401;
                await ctx.Response.WriteAsJsonAsync(new { error = result.Outcome == AuthOutcome.Unavailable ? "auth-unavailable" : "unauthorized" });
                return;
            }
            email = result.Email!;
        }
        ctx.Items[EmailKey] = email;
        await next(ctx);
    }
}

public static class HttpContextIdentity
{
    public static string Email(this HttpContext ctx) =>
        ctx.Items[AccessAuthMiddleware.EmailKey] as string
        ?? throw new InvalidOperationException("no identity on this request");
}
```

`server/AzulServer/Auth/CsrfMiddleware.cs`:

```csharp
namespace AzulServer.Auth;

/// State-changing requests must be JSON (forces a CORS preflight, which this
/// server never grants) and, when the browser sends Origin, come from us.
public sealed class CsrfMiddleware(RequestDelegate next, AzulOptions options)
{
    public async Task InvokeAsync(HttpContext ctx)
    {
        var method = ctx.Request.Method;
        bool changes = HttpMethods.IsPost(method) || HttpMethods.IsPut(method)
                       || HttpMethods.IsPatch(method) || HttpMethods.IsDelete(method);
        if (changes)
        {
            if (!HttpMethods.IsDelete(method)
                && ctx.Request.ContentType?.StartsWith("application/json", StringComparison.OrdinalIgnoreCase) != true)
            {
                await Refuse(ctx, "json-required");
                return;
            }
            var origin = ctx.Request.Headers.Origin.FirstOrDefault();
            if (origin is not null && !Allowed(origin, ctx))
            {
                await Refuse(ctx, "bad-origin");
                return;
            }
        }
        await next(ctx);
    }

    bool Allowed(string origin, HttpContext ctx)
    {
        if (options.PublicOrigin is { } pub && string.Equals(origin.TrimEnd('/'), pub, StringComparison.OrdinalIgnoreCase))
            return true;
        return Uri.TryCreate(origin, UriKind.Absolute, out var uri)
               && string.Equals(uri.Authority, ctx.Request.Host.Value, StringComparison.OrdinalIgnoreCase);
    }

    static async Task Refuse(HttpContext ctx, string code)
    {
        ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
        await ctx.Response.WriteAsJsonAsync(new { error = code });
    }
}
```

`Program.cs`: after `builder.Services.AddHttpClient();` add

```csharp
builder.Services.AddSingleton<IJwksFetcher, HttpJwksFetcher>();
builder.Services.AddSingleton<JwksCache>();
builder.Services.AddSingleton<AccessVerifier>();
```

(with `using AzulServer.Auth;`), and replace the comment line `// (Task 8 adds the identity and CSRF middleware here, before static files.)` with:

```csharp
if (options.DevMode)
    app.Logger.LogWarning("DEV MODE: no Access JWT check; identity comes from X-Dev-User or the azul_dev_user cookie.");

app.UseMiddleware<AccessAuthMiddleware>();
app.UseMiddleware<CsrfMiddleware>();
```

`ApiEndpoints.Map`: add `app.MapGet("/api/me", (HttpContext c) => Results.Json(new { email = c.Email() }));` (with `using AzulServer.Auth;`).

Note the order: the middleware comes before `UseDefaultFiles`/`UseStaticFiles` and `UseRouting`, so static files are guarded too, and it runs for `/api/health` as well, which it skips explicitly.

- [ ] **Step 4: Run the tests**

Run: `make test FILTER="FullyQualifiedName~AuthTests|FullyQualifiedName~SkeletonTests"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A server
git commit -m "server: Access JWT verification on every path, CSRF checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: SQLite storage (spec §4.2)

**Files:**
- Create: `server/AzulServer/Data/Db.cs`, `server/AzulServer/Data/GameStore.cs`, `server/AzulServer.Tests/StoreTests.cs`
- Modify: `server/AzulServer/Program.cs` (register `Db`, run `Migrate()`)

**Interfaces:**
- Produces:
  - `public sealed record SeatRecord(int Idx, string Kind, string? Email)`
  - `public sealed record GameRecord(string Id, string Creator, string Status, int NumPlayers, long Version, string? StateJson, string? FinishReason, string CreatedAt, string UpdatedAt, IReadOnlyList<SeatRecord> Seats)`
  - `public sealed record MoveRecord(string GameId, long Version, int Seat, string Actor, string MoveJson, string? RequestId, string? RequestHash, string? ResultJson, string At)`
  - `public sealed class ConcurrencyException : Exception`
  - `Db.Open() → SqliteConnection`, `Db.Migrate()`
  - `GameStore.Load(c, id)`, `List(c)`, `PlayingIds(c)`, `Insert(c, tx, g)`, `Update(c, tx, g, expectedVersion)`, `Delete(c, tx, id)`, `InsertMove(c, tx, m)`, `FindMove(c, gameId, requestId)`, `LastMove(c, gameId)`

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/StoreTests.cs`:

```csharp
using AzulServer.Data;
using Microsoft.Data.Sqlite;

namespace AzulServer.Tests;

public sealed class StoreTests : IDisposable
{
    readonly Db db = new(new AzulOptions { DataDir = Directory.CreateTempSubdirectory("azul-db-").FullName });

    public StoreTests() => db.Migrate();

    public void Dispose() => SqliteConnection.ClearAllPools();

    static GameRecord Lobby(string id, string creator = "a@x") => new(id, creator, "lobby", 3, 1, null, null,
        "2026-10-03T00:00:00Z", "2026-10-03T00:00:00Z",
        [new SeatRecord(0, "human", creator), new SeatRecord(1, "open", null), new SeatRecord(2, "bot", null)]);

    void Insert(GameRecord g)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        GameStore.Insert(c, tx, g);
        tx.Commit();
    }

    [Fact]
    public void MigrateTwiceIsHarmless()
    {
        db.Migrate();
        using var c = db.Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = "SELECT MAX(v) FROM schema_version";
        Assert.Equal(1L, (long)cmd.ExecuteScalar()!);
    }

    [Fact]
    public void InsertLoadRoundTrip()
    {
        Insert(Lobby("g1"));
        using var c = db.Open();
        var g = GameStore.Load(c, "g1")!;
        Assert.Equal("lobby", g.Status);
        Assert.Equal(3, g.Seats.Count);
        Assert.Equal(new SeatRecord(2, "bot", null), g.Seats[2]);
        Assert.Null(GameStore.Load(c, "nope"));
    }

    [Fact]
    public void UpdateChecksTheVersion()
    {
        Insert(Lobby("g2"));
        using var c = db.Open();
        var g = GameStore.Load(c, "g2")!;
        using (var tx = c.BeginTransaction())
        {
            GameStore.Update(c, tx, g with { Version = 2, Seats = [g.Seats[0], new SeatRecord(1, "human", "b@x"), g.Seats[2]] }, 1);
            tx.Commit();
        }
        Assert.Equal("b@x", GameStore.Load(c, "g2")!.Seats[1].Email);
        using var tx2 = c.BeginTransaction();
        Assert.Throws<ConcurrencyException>(() => GameStore.Update(c, tx2, g with { Version = 3 }, 1));
    }

    [Fact]
    public void OneSeatPerPersonPerGame()
    {
        var bad = Lobby("g3") with { Seats = [new SeatRecord(0, "human", "a@x"), new SeatRecord(1, "human", "a@x"), new SeatRecord(2, "open", null)] };
        Assert.Throws<SqliteException>(() => Insert(bad));
    }

    [Fact]
    public void SeatInvariantsAreEnforced()
    {
        Assert.Throws<SqliteException>(() => Insert(Lobby("g4") with { Seats = [new SeatRecord(0, "open", "a@x")] }));
        Assert.Throws<SqliteException>(() => Insert(Lobby("g5") with { Seats = [new SeatRecord(0, "human", null)] }));
    }

    [Fact]
    public void MovesAndDeleteCascade()
    {
        Insert(Lobby("g6"));
        using var c = db.Open();
        using (var tx = c.BeginTransaction())
        {
            GameStore.InsertMove(c, tx, new MoveRecord("g6", 2, 0, "a@x", "{}", "r1", "h1", "{\"v\":2}", "t"));
            GameStore.InsertMove(c, tx, new MoveRecord("g6", 3, 1, "bot", "{}", null, null, "{\"v\":3}", "t"));
            tx.Commit();
        }
        Assert.Equal(2, GameStore.FindMove(c, "g6", "r1")!.Version);
        Assert.Equal(3, GameStore.LastMove(c, "g6")!.Version);
        using (var tx = c.BeginTransaction())
        {
            GameStore.Delete(c, tx, "g6");
            tx.Commit();
        }
        Assert.Null(GameStore.Load(c, "g6"));
        Assert.Null(GameStore.LastMove(c, "g6"));
        using var cmd = c.CreateCommand();
        cmd.CommandText = "SELECT (SELECT COUNT(*) FROM seats WHERE game_id='g6') + (SELECT COUNT(*) FROM moves WHERE game_id='g6')";
        Assert.Equal(0L, (long)cmd.ExecuteScalar()!);
    }

    [Fact]
    public void ListShowsActiveGamesAndTheLast20Finished()
    {
        Insert(Lobby("active"));
        for (int i = 0; i < 25; i++)
            Insert(Lobby($"f{i:D2}") with { Status = "finished", UpdatedAt = $"2026-10-03T00:00:{i:D2}Z" });
        using var c = db.Open();
        var list = GameStore.List(c);
        Assert.Contains(list, g => g.Id == "active");
        Assert.Equal(20, list.Count(g => g.Status == "finished"));
        Assert.Contains(list, g => g.Id == "f24");
        Assert.DoesNotContain(list, g => g.Id == "f00");
        Assert.All(list, g => Assert.NotEmpty(g.Seats));
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~StoreTests`
Expected: compile errors (`Db` missing).

- [ ] **Step 3: Implement**

`server/AzulServer/Data/Db.cs`:

```csharp
using Microsoft.Data.Sqlite;

namespace AzulServer.Data;

public sealed class Db(AzulOptions options)
{
    // Append only; each entry runs once, in a transaction.
    static readonly string[] Migrations =
    [
        """
        CREATE TABLE games(
          id TEXT PRIMARY KEY,
          creator TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('lobby','playing','finished')),
          num_players INTEGER NOT NULL CHECK(num_players BETWEEN 2 AND 4),
          version INTEGER NOT NULL,
          state_json TEXT,
          finish_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL);
        CREATE TABLE seats(
          game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
          idx INTEGER NOT NULL,
          kind TEXT NOT NULL CHECK(kind IN ('open','human','bot')),
          email TEXT,
          PRIMARY KEY(game_id, idx),
          CHECK((kind = 'open' AND email IS NULL) OR (kind = 'human' AND email IS NOT NULL) OR kind = 'bot'));
        CREATE UNIQUE INDEX seats_one_per_person ON seats(game_id, email) WHERE email IS NOT NULL;
        CREATE TABLE moves(
          game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
          version INTEGER NOT NULL,
          seat INTEGER NOT NULL,
          actor TEXT NOT NULL,
          move_json TEXT NOT NULL,
          request_id TEXT,
          request_hash TEXT,
          result_json TEXT,
          at TEXT NOT NULL,
          PRIMARY KEY(game_id, version));
        CREATE UNIQUE INDEX moves_request ON moves(game_id, request_id) WHERE request_id IS NOT NULL;
        CREATE INDEX games_status ON games(status, updated_at);
        """,
    ];

    public string ConnectionString { get; } = new SqliteConnectionStringBuilder
    {
        DataSource = Path.Combine(options.DataDir, "azul.db"),
        ForeignKeys = true,
        DefaultTimeout = 30,
        Pooling = true,
    }.ToString();

    public SqliteConnection Open()
    {
        var c = new SqliteConnection(ConnectionString);
        c.Open();
        return c;
    }

    public void Migrate()
    {
        Directory.CreateDirectory(options.DataDir);
        using var c = Open();
        Exec(c, null, "PRAGMA journal_mode=WAL;");
        Exec(c, null, "CREATE TABLE IF NOT EXISTS schema_version(v INTEGER NOT NULL);");
        using var q = c.CreateCommand();
        q.CommandText = "SELECT COALESCE(MAX(v), 0) FROM schema_version";
        var current = Convert.ToInt32(q.ExecuteScalar());
        for (int i = current; i < Migrations.Length; i++)
        {
            using var tx = c.BeginTransaction();
            Exec(c, tx, Migrations[i]);
            Exec(c, tx, $"INSERT INTO schema_version(v) VALUES ({i + 1});");
            tx.Commit();
        }
    }

    static void Exec(SqliteConnection c, SqliteTransaction? tx, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }
}
```

`server/AzulServer/Data/GameStore.cs`:

```csharp
using Microsoft.Data.Sqlite;

namespace AzulServer.Data;

public sealed record SeatRecord(int Idx, string Kind, string? Email);

public sealed record GameRecord(
    string Id, string Creator, string Status, int NumPlayers, long Version,
    string? StateJson, string? FinishReason, string CreatedAt, string UpdatedAt,
    IReadOnlyList<SeatRecord> Seats);

public sealed record MoveRecord(
    string GameId, long Version, int Seat, string Actor, string MoveJson,
    string? RequestId, string? RequestHash, string? ResultJson, string At);

public sealed class ConcurrencyException(string message) : Exception(message);

public static class GameStore
{
    const string GameColumns = "id, creator, status, num_players, version, state_json, finish_reason, created_at, updated_at";
    const string MoveColumns = "game_id, version, seat, actor, move_json, request_id, request_hash, result_json, at";

    static SqliteCommand Command(SqliteConnection c, SqliteTransaction? tx, string sql, params (string Name, object? Value)[] args)
    {
        var cmd = c.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        foreach (var (name, value) in args)
            cmd.Parameters.AddWithValue(name, value ?? DBNull.Value);
        return cmd;
    }

    static string? Str(SqliteDataReader r, int i) => r.IsDBNull(i) ? null : r.GetString(i);

    static GameRecord ReadGame(SqliteDataReader r) => new(
        r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), r.GetInt64(4),
        Str(r, 5), Str(r, 6), r.GetString(7), r.GetString(8), []);

    static MoveRecord ReadMove(SqliteDataReader r) => new(
        r.GetString(0), r.GetInt64(1), r.GetInt32(2), r.GetString(3), r.GetString(4),
        Str(r, 5), Str(r, 6), Str(r, 7), r.GetString(8));

    static List<SeatRecord> Seats(SqliteConnection c, SqliteTransaction? tx, string id)
    {
        using var cmd = Command(c, tx, "SELECT idx, kind, email FROM seats WHERE game_id = $id ORDER BY idx", ("$id", id));
        using var r = cmd.ExecuteReader();
        var seats = new List<SeatRecord>();
        while (r.Read()) seats.Add(new SeatRecord(r.GetInt32(0), r.GetString(1), Str(r, 2)));
        return seats;
    }

    public static GameRecord? Load(SqliteConnection c, string id, SqliteTransaction? tx = null)
    {
        GameRecord g;
        using (var cmd = Command(c, tx, $"SELECT {GameColumns} FROM games WHERE id = $id", ("$id", id)))
        using (var r = cmd.ExecuteReader())
        {
            if (!r.Read()) return null;
            g = ReadGame(r);
        }
        return g with { Seats = Seats(c, tx, id) };
    }

    public static List<GameRecord> List(SqliteConnection c)
    {
        var games = new List<GameRecord>();
        foreach (var sql in new[]
        {
            $"SELECT {GameColumns} FROM games WHERE status <> 'finished' ORDER BY updated_at DESC",
            $"SELECT {GameColumns} FROM games WHERE status = 'finished' ORDER BY updated_at DESC LIMIT 20",
        })
        {
            using var cmd = Command(c, null, sql);
            using var r = cmd.ExecuteReader();
            while (r.Read()) games.Add(ReadGame(r));
        }
        return games.Select(g => g with { Seats = Seats(c, null, g.Id) }).ToList();
    }

    public static List<string> PlayingIds(SqliteConnection c)
    {
        using var cmd = Command(c, null, "SELECT id FROM games WHERE status = 'playing'");
        using var r = cmd.ExecuteReader();
        var ids = new List<string>();
        while (r.Read()) ids.Add(r.GetString(0));
        return ids;
    }

    public static void Insert(SqliteConnection c, SqliteTransaction tx, GameRecord g)
    {
        using (var cmd = Command(c, tx,
            $"INSERT INTO games({GameColumns}) VALUES ($id, $creator, $status, $n, $v, $state, $finish, $created, $updated)",
            ("$id", g.Id), ("$creator", g.Creator), ("$status", g.Status), ("$n", g.NumPlayers), ("$v", g.Version),
            ("$state", g.StateJson), ("$finish", g.FinishReason), ("$created", g.CreatedAt), ("$updated", g.UpdatedAt)))
            cmd.ExecuteNonQuery();
        InsertSeats(c, tx, g);
    }

    public static void Update(SqliteConnection c, SqliteTransaction tx, GameRecord g, long expectedVersion)
    {
        using (var cmd = Command(c, tx,
            "UPDATE games SET status = $status, version = $v, state_json = $state, finish_reason = $finish, updated_at = $updated " +
            "WHERE id = $id AND version = $expected",
            ("$status", g.Status), ("$v", g.Version), ("$state", g.StateJson), ("$finish", g.FinishReason),
            ("$updated", g.UpdatedAt), ("$id", g.Id), ("$expected", expectedVersion)))
        {
            if (cmd.ExecuteNonQuery() != 1)
                throw new ConcurrencyException($"game {g.Id} is not at version {expectedVersion}");
        }
        using (var del = Command(c, tx, "DELETE FROM seats WHERE game_id = $id", ("$id", g.Id)))
            del.ExecuteNonQuery();
        InsertSeats(c, tx, g);
    }

    static void InsertSeats(SqliteConnection c, SqliteTransaction tx, GameRecord g)
    {
        foreach (var s in g.Seats)
        {
            using var cmd = Command(c, tx, "INSERT INTO seats(game_id, idx, kind, email) VALUES ($id, $idx, $kind, $email)",
                ("$id", g.Id), ("$idx", s.Idx), ("$kind", s.Kind), ("$email", s.Email));
            cmd.ExecuteNonQuery();
        }
    }

    public static void Delete(SqliteConnection c, SqliteTransaction tx, string id)
    {
        using var cmd = Command(c, tx, "DELETE FROM games WHERE id = $id", ("$id", id));
        cmd.ExecuteNonQuery();
    }

    public static void InsertMove(SqliteConnection c, SqliteTransaction tx, MoveRecord m)
    {
        using var cmd = Command(c, tx,
            $"INSERT INTO moves({MoveColumns}) VALUES ($g, $v, $seat, $actor, $move, $req, $hash, $result, $at)",
            ("$g", m.GameId), ("$v", m.Version), ("$seat", m.Seat), ("$actor", m.Actor), ("$move", m.MoveJson),
            ("$req", m.RequestId), ("$hash", m.RequestHash), ("$result", m.ResultJson), ("$at", m.At));
        cmd.ExecuteNonQuery();
    }

    public static MoveRecord? FindMove(SqliteConnection c, string gameId, string requestId)
    {
        using var cmd = Command(c, null, $"SELECT {MoveColumns} FROM moves WHERE game_id = $g AND request_id = $r",
            ("$g", gameId), ("$r", requestId));
        using var r = cmd.ExecuteReader();
        return r.Read() ? ReadMove(r) : null;
    }

    public static MoveRecord? LastMove(SqliteConnection c, string gameId, SqliteTransaction? tx = null)
    {
        using var cmd = Command(c, tx, $"SELECT {MoveColumns} FROM moves WHERE game_id = $g ORDER BY version DESC LIMIT 1",
            ("$g", gameId));
        using var r = cmd.ExecuteReader();
        return r.Read() ? ReadMove(r) : null;
    }
}
```

`Program.cs`: register `builder.Services.AddSingleton<Db>();` (with `using AzulServer.Data;`) and after the `options` line call `app.Services.GetRequiredService<Db>().Migrate();`.

- [ ] **Step 4: Run the tests**

Run: `make test FILTER=FullyQualifiedName~StoreTests`
Expected: 7 pass.

- [ ] **Step 5: Commit**

```bash
git add -A server
git commit -m "server: SQLite schema, migrations and game store

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Views and per-viewer projection (spec §4.4 game view)

**Files:**
- Create: `server/AzulServer/Games/Views.cs`, `server/AzulServer/Games/Projection.cs`, `server/AzulServer.Tests/ProjectionTests.cs`

**Interfaces:**
- Consumes: `Game.FromSnapshot`, `LegalTakes`, `WallOptions`, `IsFinished` (Plan 1); `GameRecord`, `MoveRecord`, `SeatRecord` (Task 9).
- Produces (the JSON contract Plan 3 mirrors in `web/src/lib/types.ts`):

```csharp
public sealed record ViewerInfo(string Email, int? Seat);
public sealed record SeatView(int Idx, string Kind, string? Email);
public sealed record PlayerView(int Score, int[]?[] Lines, int[][] Wall, int[] Floor, bool HasFirst);
public sealed record BoardView(int Round, string Phase, int ActiveSeat, int[][] Factories, int[] Center,
    bool CenterHasFirst, int[] Bag, int[] Discard, PlayerView[] Players);
public sealed record WallRowView(int Color, int[] Targets);
public sealed record LegalView(int[][]? Takes, WallRowView?[]? Wall);
public sealed record LastMoveView(long Version, int Seat, string Kind, int? Factory, int? Color, int? Row, int? Tiles, int[]? Columns);
public sealed record ResultView(int[] Scores, int[] Winners, string Reason);
public sealed record GameView(string Id, string Status, long Version, int NumPlayers, string Creator, ViewerInfo You,
    SeatView[] Seats, BoardView? Board, LegalView? Legal, LastMoveView? LastMove, ResultView? Result);
public sealed record GameSummary(string Id, string Status, int NumPlayers, string Creator, SeatView[] Seats, int? Round, string UpdatedAt);
public sealed record MoveJson(string Kind, int? Factory, int? Color, int? Row, int? Tiles, int[]? Columns);
public static class Projection { GameView Project(GameRecord, MoveRecord?, string viewer); GameSummary Summary(GameRecord); MoveJson Describe(Move); }
public static class Status { const string Lobby, Playing, Finished; }  public static class SeatKind { const string Open, Human, Bot; }
```

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/ProjectionTests.cs`:

```csharp
using Azul;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

public class ProjectionTests
{
    static GameRecord Playing(Game game, params SeatRecord[] seats) => new(
        "g", "a@x", Status.Playing, game.numPlayers, 7, Json.Serialize(game.ToSnapshot()), null, "t", "t", seats);

    static readonly SeatRecord[] ThreeSeats =
        [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Human, "b@x"), new(2, SeatKind.Bot, "c@x")];

    [Fact]
    public void BoardMirrorsTheEngine()
    {
        var game = new Game(3, new Random(1));
        var s = game.ToSnapshot();
        // Player 1: two red on line 2, one blue and the FIRST marker on the floor.
        s.Players[1].Line[2][2] = 2; s.Bag[2] -= 2;
        s.Players[1].Floor[0] = 1; s.Bag[0] -= 1;
        s.Players[1].Floor[5] = 1; s.Factories[^1][5] = 0;
        var restored = Game.FromSnapshot(s);
        var v = Projection.Project(Playing(restored, ThreeSeats), null, "b@x");

        Assert.Equal(restored.numFactories, v.Board!.Factories.Length);
        Assert.Equal(5, v.Board.Center.Length);
        Assert.False(v.Board.CenterHasFirst);
        var p1 = v.Board.Players[1];
        Assert.Equal(new[] { 2, 2 }, p1.Lines[2]);
        Assert.Null(p1.Lines[0]);
        Assert.Equal(new[] { 0 }, p1.Floor);
        Assert.True(p1.HasFirst);
        Assert.Equal("take", v.Board.Phase);
        Assert.Equal(1, v.Board.Round);
    }

    [Fact]
    public void LegalOnlyForTheActiveHuman()
    {
        var game = new Game(3, new Random(1));  // seat 0 to move
        var rec = Playing(game, ThreeSeats);
        var a = Projection.Project(rec, null, "a@x");
        Assert.Equal(0, a.You.Seat);
        Assert.NotNull(a.Legal!.Takes);
        Assert.Null(a.Legal.Wall);
        Assert.Null(Projection.Project(rec, null, "b@x").Legal);
        var spectator = Projection.Project(rec, null, "z@x");
        Assert.Null(spectator.You.Seat);
        Assert.Null(spectator.Legal);
    }

    [Fact]
    public void HandedToBotSeatStillBelongsToItsEmail()
    {
        var game = new Game(3, new Random(1));
        game.Play(game.GetGreedyMove());
        game.Play(game.GetGreedyMove());  // seat 2 (the bot seat) to move
        var v = Projection.Project(Playing(game, ThreeSeats), null, "c@x");
        Assert.Equal(2, v.You.Seat);
        Assert.Null(v.Legal);  // a bot plays that seat until it is taken back
        Assert.Equal("c@x", v.Seats[2].Email);
    }

    [Fact]
    public void FinishedGameHasAResultAndTies()
    {
        var game = new Game(2, new Random(4));
        while (!game.IsGameOver()) game.Play(game.GetGreedyMove());
        var rec = Playing(game, [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Bot, null)])
            with { Status = Status.Finished, FinishReason = "normal" };
        var v = Projection.Project(rec, null, "a@x");
        Assert.Equal("over", v.Board!.Phase);
        Assert.Equal(game.players.Select(p => p.score), v.Result!.Scores);
        int max = v.Result.Scores.Max();
        Assert.Equal(Enumerable.Range(0, 2).Where(i => v.Result.Scores[i] == max), v.Result.Winners);
        Assert.Null(v.Legal);
    }

    [Fact]
    public void LobbyHasNoBoard()
    {
        var rec = new GameRecord("g", "a@x", Status.Lobby, 2, 1, null, null, "t", "t",
            [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Open, null)]);
        var v = Projection.Project(rec, null, "a@x");
        Assert.Null(v.Board);
        Assert.Equal(0, v.You.Seat);
        Assert.Null(Projection.Summary(rec).Round);
    }

    [Fact]
    public void LastMoveAndDescribe()
    {
        var game = new Game(2, new Random(1));
        var move = new Move(new Move { factoryIdx = game.numFactories, color = 5, row = 5 }, game);
        var json = Json.Serialize(Projection.Describe(move));
        var last = new MoveRecord("g", 8, 0, "a@x", json, null, null, null, "t");
        var v = Projection.Project(Playing(game, [new(0, SeatKind.Human, "a@x"), new(1, SeatKind.Bot, null)]), last, "a@x");
        Assert.Equal(new LastMoveView(8, 0, "take", game.numFactories, 5, 5, 1, null), v.LastMove);
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~ProjectionTests`
Expected: compile errors (`Projection` missing).

- [ ] **Step 3: Implement**

`server/AzulServer/Games/Views.cs`:

```csharp
namespace AzulServer.Games;

public static class Status
{
    public const string Lobby = "lobby", Playing = "playing", Finished = "finished";
}

public static class SeatKind
{
    public const string Open = "open", Human = "human", Bot = "bot";
}

public sealed record ViewerInfo(string Email, int? Seat);
public sealed record SeatView(int Idx, string Kind, string? Email);
public sealed record PlayerView(int Score, int[]?[] Lines, int[][] Wall, int[] Floor, bool HasFirst);
public sealed record BoardView(int Round, string Phase, int ActiveSeat, int[][] Factories, int[] Center,
    bool CenterHasFirst, int[] Bag, int[] Discard, PlayerView[] Players);
public sealed record WallRowView(int Color, int[] Targets);
public sealed record LegalView(int[][]? Takes, WallRowView?[]? Wall);
public sealed record LastMoveView(long Version, int Seat, string Kind, int? Factory, int? Color, int? Row, int? Tiles, int[]? Columns);
public sealed record ResultView(int[] Scores, int[] Winners, string Reason);
public sealed record GameView(string Id, string Status, long Version, int NumPlayers, string Creator, ViewerInfo You,
    SeatView[] Seats, BoardView? Board, LegalView? Legal, LastMoveView? LastMove, ResultView? Result);
public sealed record GameSummary(string Id, string Status, int NumPlayers, string Creator, SeatView[] Seats, int? Round, string UpdatedAt);

/// What a move did, as stored in moves.move_json.
public sealed record MoveJson(string Kind, int? Factory, int? Color, int? Row, int? Tiles, int[]? Columns);
```

`server/AzulServer/Games/Projection.cs`:

```csharp
using Azul;
using AzulServer.Data;

namespace AzulServer.Games;

public static class Projection
{
    const int First = 5;

    public static GameSnapshot Snapshot(GameRecord g) => Json.Deserialize<GameSnapshot>(g.StateJson!);

    public static GameSummary Summary(GameRecord g) => new(
        g.Id, g.Status, g.NumPlayers, g.Creator, SeatViews(g),
        g.StateJson is null ? null : Snapshot(g).RoundIdx + 1, g.UpdatedAt);

    public static GameView Project(GameRecord g, MoveRecord? last, string viewer)
    {
        int? seat = g.Seats.FirstOrDefault(s => s.Email == viewer)?.Idx;
        BoardView? board = null;
        LegalView? legal = null;
        ResultView? result = null;
        if (g.StateJson is not null)
        {
            var game = Game.FromSnapshot(Snapshot(g));
            board = Board(game, g.Status);
            if (g.Status == Status.Playing && seat is { } s && g.Seats[s].Kind == SeatKind.Human && game.activePlayer == s)
                legal = Legal(game);
            if (g.Status == Status.Finished)
                result = Result(game, g.FinishReason ?? "normal");
        }
        return new GameView(g.Id, g.Status, g.Version, g.NumPlayers, g.Creator, new ViewerInfo(viewer, seat),
            SeatViews(g), board, legal, LastMove(last), result);
    }

    public static MoveJson Describe(Move m) => m.colIdx[0] == Move.NOT_SET
        ? new MoveJson("take", m.factoryIdx, m.color, m.row, m.numTiles, null)
        : new MoveJson("wall", null, null, null, null, (int[])m.colIdx.Clone());

    static SeatView[] SeatViews(GameRecord g) => g.Seats.Select(s => new SeatView(s.Idx, s.Kind, s.Email)).ToArray();

    static BoardView Board(Game game, string status)
    {
        int nf = game.numFactories;
        var center = game.factories[nf];
        return new BoardView(
            game.roundIdx + 1,
            status == Status.Finished ? "over" : game.isRegularPhase ? "take" : "wall",
            game.activePlayer,
            game.factories.Take(nf).Select(f => (int[])f.Clone()).ToArray(),
            center.Take(5).ToArray(),
            center[First] > 0,
            (int[])game.bag.Clone(),
            (int[])game.discarded.Clone(),
            game.players.Select(PlayerOf).ToArray());
    }

    static PlayerView PlayerOf(Player p)
    {
        var lines = new int[]?[5];
        for (int row = 0; row < 5; row++)
            for (int c = 0; c < 5; c++)
                if (p.line[row, c] > 0)
                {
                    lines[row] = [c, p.line[row, c]];
                    break;
                }
        var wall = Enumerable.Range(0, 5)
            .Select(r => Enumerable.Range(0, 5).Select(col => p.grid[r, col]).ToArray()).ToArray();
        // The engine keeps counts, not arrival order: expand in colour order.
        var floor = new List<int>();
        for (int c = 0; c < 5; c++)
            for (int i = 0; i < p.floor[c]; i++)
                floor.Add(c);
        return new PlayerView(p.score, lines, wall, floor.ToArray(), p.floor[First] > 0);
    }

    static LegalView Legal(Game game) => game.isRegularPhase
        ? new LegalView(game.LegalTakes().Select(t => new[] { t.Factory, t.Color, t.Row }).ToArray(), null)
        : new LegalView(null, game.WallOptions()!.Select(o => o is null ? null : new WallRowView(o.Color, o.Targets)).ToArray());

    static ResultView Result(Game game, string reason)
    {
        var scores = game.players.Select(p => p.score).ToArray();
        int max = scores.Max();
        return new ResultView(scores, Enumerable.Range(0, scores.Length).Where(i => scores[i] == max).ToArray(), reason);
    }

    static LastMoveView? LastMove(MoveRecord? m)
    {
        if (m is null) return null;
        var j = Json.Deserialize<MoveJson>(m.MoveJson);
        return new LastMoveView(m.Version, m.Seat, j.Kind, j.Factory, j.Color, j.Row, j.Tiles, j.Columns);
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `make test FILTER=FullyQualifiedName~ProjectionTests`
Expected: 6 pass.

- [ ] **Step 5: Commit**

```bash
git add -A server
git commit -m "server: game views projected per viewer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: GameService lobby and seat rules, endpoints (spec §4.3, §4.4)

**Files:**
- Create: `server/AzulServer/Games/GameService.cs`, `server/AzulServer/Games/EventHub.cs` (minimal: `Publish`, `PublishDeleted`; Task 13 adds subscriptions), `server/AzulServer/Games/ServerMoveQueue.cs` (minimal: `Poke`, `CancelSearch`; Task 14 completes it), `server/AzulServer.Tests/Play.cs`, `server/AzulServer.Tests/LobbyTests.cs`
- Modify: `server/AzulServer/Api/ApiEndpoints.cs`, `server/AzulServer/Program.cs`

**Interfaces:**
- Consumes: Tasks 8–10.
- Produces: `GameService` with `List()`, `GetView(id, viewer)`, `Create(viewer, players)`, `Claim`, `Release`, `SetKind`, `Start`, `ToBot`, `TakeBack`, `Delete` (all seat/lifecycle methods `Task<ApiResult>`), internal `Mutate`; `public sealed record ApiResult(int Status, object? Body)` with `ToHttp()`; `public sealed record ErrorBody(string Error, object? View)`; `IFaultInjector` (`void AfterCommit(string gameId)`) with `NoFaults`; request records `CreateGameRequest(int Players)`, `SetKindRequest(string? Kind)`; test helper `Play` (below).

- [ ] **Step 1: Write the test helper and failing tests**

`server/AzulServer.Tests/Play.cs`:

```csharp
using System.Net;
using AzulServer.Games;

namespace AzulServer.Tests;

/// API-level helpers shared by the lobby, move, SSE and bot tests.
public static class Play
{
    public static async Task<GameView> Create(HttpClient c, int players)
    {
        var res = await c.Post("/api/games", new { players });
        Assert.Equal(HttpStatusCode.Created, res.StatusCode);
        return await res.Read<GameView>();
    }

    public static async Task<GameView> Get(HttpClient c, string id) =>
        await (await c.GetAsync($"/api/games/{id}")).Read<GameView>();

    /// Creator = clients[0] in seat 0; clients[i] claims seat i; remaining
    /// seats are bots; started.
    public static async Task<string> Started(int players, params HttpClient[] clients)
    {
        var g = await Create(clients[0], players);
        for (int i = 1; i < clients.Length; i++)
            Assert.Equal(HttpStatusCode.OK, (await clients[i].Post($"/api/games/{g.Id}/seats/{i}/claim")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await clients[0].Post($"/api/games/{g.Id}/start")).StatusCode);
        return g.Id;
    }

    public static object Take(GameView v, int[] take, string? requestId = null) => new
    {
        version = v.Version,
        requestId = requestId ?? Guid.NewGuid().ToString(),
        kind = "take",
        factory = take[0],
        color = take[1],
        row = take[2],
    };

    public static object Wall(GameView v, int[] columns, string? requestId = null) => new
    {
        version = v.Version,
        requestId = requestId ?? Guid.NewGuid().ToString(),
        kind = "wall",
        columns,
    };

    /// A legal move for the viewer: the first take, or every completed line to
    /// the floor (always legal; first targets could clash on a wall column).
    public static object AnyLegal(GameView v) => v.Legal!.Takes is { } takes
        ? Take(v, takes[0])
        : Wall(v, v.Legal.Wall!.Select(r => r is null ? -1 : 5).ToArray());

    public static async Task<GameView> WaitFor(HttpClient c, string id, Func<GameView, bool> done, int seconds = 30)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (true)
        {
            var v = await Get(c, id);
            if (done(v)) return v;
            if (DateTime.UtcNow > until) throw new TimeoutException($"game {id} stuck at version {v.Version}");
            await Task.Delay(50);
        }
    }
}
```

`server/AzulServer.Tests/LobbyTests.cs`:

```csharp
using System.Net;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

public sealed class LobbyTests : IDisposable
{
    readonly TestApp app = new();
    readonly HttpClient alice, bob, carol;

    public LobbyTests()
    {
        alice = app.Client("alice@x.com");
        bob = app.Client("bob@x.com");
        carol = app.Client("carol@x.com");
    }

    public void Dispose() => app.Dispose();

    static async Task<HttpStatusCode> S(Task<HttpResponseMessage> r) => (await r).StatusCode;

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public async Task CreateSeatsTheCreator(int players)
    {
        var g = await Play.Create(alice, players);
        Assert.Equal(Status.Lobby, g.Status);
        Assert.Equal(players, g.Seats.Length);
        Assert.Equal(new SeatView(0, SeatKind.Human, "alice@x.com"), g.Seats[0]);
        Assert.All(g.Seats.Skip(1), s => Assert.Equal(SeatKind.Open, s.Kind));
        Assert.Equal(0, g.You.Seat);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public async Task CreateRejectsBadPlayerCounts(int players) =>
        Assert.Equal(HttpStatusCode.BadRequest, await S(alice.Post("/api/games", new { players })));

    [Fact]
    public async Task ClaimRules()
    {
        var g = await Play.Create(alice, 3);
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{g.Id}/seats/1/claim")));
        Assert.Equal(HttpStatusCode.Conflict, await S(carol.Post($"/api/games/{g.Id}/seats/1/claim")));  // taken
        Assert.Equal(HttpStatusCode.Conflict, await S(bob.Post($"/api/games/{g.Id}/seats/2/claim")));    // already seated
        Assert.Equal(HttpStatusCode.NotFound, await S(carol.Post($"/api/games/{g.Id}/seats/7/claim")));
        Assert.Equal(HttpStatusCode.NotFound, await S(carol.Post("/api/games/nope/seats/1/claim")));
    }

    [Fact]
    public async Task ClaimIsCaseInsensitiveInDevMode()
    {
        var g = await Play.Create(alice, 3);
        var shouty = app.Client("ALICE@X.COM");
        Assert.Equal(HttpStatusCode.Conflict, await S(shouty.Post($"/api/games/{g.Id}/seats/1/claim")));
    }

    [Fact]
    public async Task ReleaseRules()
    {
        var g = await Play.Create(alice, 3);
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        await carol.Post($"/api/games/{g.Id}/seats/2/claim");
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{g.Id}/seats/1/release")));
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{g.Id}/seats/1/release")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/2/release")));  // creator removes carol
        var v = await Play.Get(alice, g.Id);
        Assert.Equal(SeatKind.Open, v.Seats[1].Kind);
        Assert.Null(v.Seats[2].Email);
    }

    [Fact]
    public async Task KindRules()
    {
        var g = await Play.Create(alice, 3);
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        Assert.Equal(HttpStatusCode.Forbidden, await S(bob.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "bot" })));
        Assert.Equal(HttpStatusCode.Conflict, await S(alice.Post($"/api/games/{g.Id}/seats/1/kind", new { kind = "bot" })));
        Assert.Equal(HttpStatusCode.BadRequest, await S(alice.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "human" })));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "bot" })));
        Assert.Equal(SeatKind.Bot, (await Play.Get(alice, g.Id)).Seats[2].Kind);
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/2/kind", new { kind = "open" })));
    }

    [Fact]
    public async Task StartRules()
    {
        var g = await Play.Create(alice, 3);
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        Assert.Equal(HttpStatusCode.Forbidden, await S(bob.Post($"/api/games/{g.Id}/start")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/start")));
        var v = await Play.Get(alice, g.Id);
        Assert.Equal(Status.Playing, v.Status);
        Assert.Equal(SeatKind.Bot, v.Seats[2].Kind);
        Assert.NotNull(v.Board);
        Assert.Equal(0, v.Board!.ActiveSeat);
        Assert.Equal(HttpStatusCode.Conflict, await S(alice.Post($"/api/games/{g.Id}/start")));
        Assert.Equal(HttpStatusCode.Conflict, await S(carol.Post($"/api/games/{g.Id}/seats/2/claim")));
    }

    [Fact]
    public async Task UnseatedCreatorCanStartButNotWithoutHumans()
    {
        var g = await Play.Create(alice, 2);
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/seats/0/release")));
        Assert.Equal(HttpStatusCode.Conflict, await S(alice.Post($"/api/games/{g.Id}/start")));  // no human
        await bob.Post($"/api/games/{g.Id}/seats/1/claim");
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{g.Id}/start")));
        Assert.Null((await Play.Get(alice, g.Id)).You.Seat);
    }

    [Fact]
    public async Task HandToBotAndTakeBack()
    {
        var id = await Play.Started(3, alice, bob, carol);
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{id}/seats/1/to-bot")));
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{id}/seats/1/to-bot")));
        var v = await Play.Get(bob, id);
        Assert.Equal(new SeatView(1, SeatKind.Bot, "bob@x.com"), v.Seats[1]);
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{id}/seats/1/take-back")));
        Assert.Equal(HttpStatusCode.OK, await S(bob.Post($"/api/games/{id}/seats/1/take-back")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.Post($"/api/games/{id}/seats/2/to-bot")));  // creator hands carol's seat
        Assert.Equal(HttpStatusCode.OK, await S(carol.Post($"/api/games/{id}/seats/2/take-back")));
    }

    [Fact]
    public async Task NeverHumanBotCannotBeTaken()
    {
        var id = await Play.Started(3, alice, bob);
        Assert.Equal(HttpStatusCode.Forbidden, await S(carol.Post($"/api/games/{id}/seats/2/take-back")));
        Assert.Equal(HttpStatusCode.Forbidden, await S(alice.Post($"/api/games/{id}/seats/2/take-back")));
    }

    [Fact]
    public async Task DeleteRules()
    {
        var id = await Play.Started(2, alice, bob);
        Assert.Equal(HttpStatusCode.Forbidden, await S(bob.SendAsync(new HttpRequestMessage(HttpMethod.Delete, $"/api/games/{id}"))));
        Assert.Equal(HttpStatusCode.NoContent, await S(alice.SendAsync(new HttpRequestMessage(HttpMethod.Delete, $"/api/games/{id}"))));
        Assert.Equal(HttpStatusCode.NotFound, await S(alice.GetAsync($"/api/games/{id}")));
    }

    [Fact]
    public async Task ListAndSpectating()
    {
        var id = await Play.Started(2, alice, bob);
        var list = await (await carol.GetAsync("/api/games")).Read<GameSummary[]>();
        Assert.Contains(list, s => s.Id == id && s.Round == 1);
        var watch = await Play.Get(carol, id);
        Assert.Null(watch.You.Seat);
        Assert.NotNull(watch.Board);
    }

    [Fact]
    public async Task CorruptGameDoesNotBreakTheListOrTheSweep()
    {
        var good = await Play.Started(2, alice, bob);
        var bad = await Play.Started(2, alice, bob);
        using (var c = app.Service<Db>().Open())
        using (var cmd = c.CreateCommand())
        {
            cmd.CommandText = "UPDATE games SET state_json = '{\"format\":1}' WHERE id = $id";
            cmd.Parameters.AddWithValue("$id", bad);
            cmd.ExecuteNonQuery();
        }
        Assert.Equal(HttpStatusCode.OK, await S(alice.GetAsync("/api/games")));
        Assert.Equal(HttpStatusCode.InternalServerError, await S(alice.GetAsync($"/api/games/{bad}")));
        Assert.Equal(HttpStatusCode.OK, await S(alice.GetAsync($"/api/games/{good}")));
        Assert.DoesNotContain(bad, app.Service<GameService>().GamesNeedingServerMove());
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~LobbyTests`
Expected: compile errors (`GameService` missing).

- [ ] **Step 3: Implement**

`server/AzulServer/Games/EventHub.cs` (minimal for now):

```csharp
namespace AzulServer.Games;

public sealed class EventHub
{
    public const long Deleted = -1;

    public void Publish(string gameId, long version) { }

    public void PublishDeleted(string gameId) => Publish(gameId, Deleted);
}
```

`server/AzulServer/Games/ServerMoveQueue.cs` (minimal for now):

```csharp
namespace AzulServer.Games;

public sealed class ServerMoveQueue
{
    public void Poke(string gameId) { }

    public void CancelSearch(string gameId) { }
}
```

`server/AzulServer/Games/GameService.cs`:

```csharp
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Azul;
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
}
```

`server/AzulServer/Api/ApiEndpoints.cs` (replace):

```csharp
using AzulServer.Auth;
using AzulServer.Games;

namespace AzulServer.Api;

public sealed record CreateGameRequest(int Players);
public sealed record SetKindRequest(string? Kind);

public static class ApiEndpoints
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/api/health", () => Results.Json(new { ok = true }));

        var api = app.MapGroup("/api");
        api.MapGet("/me", (HttpContext c) => Results.Json(new { email = c.Email() }));
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
        api.MapDelete("/games/{id}", async (HttpContext c, GameService s, string id) =>
            (await s.Delete(id, c.Email())).ToHttp());
    }
}
```

`Program.cs`: register (with `using AzulServer.Games;`)

```csharp
builder.Services.AddSingleton<EventHub>();
builder.Services.AddSingleton<ServerMoveQueue>();
builder.Services.AddSingleton<IFaultInjector, NoFaults>();
builder.Services.AddSingleton<GameService>();
```

A corrupt game's `GET` throws `InvalidSnapshotException` inside the endpoint; the exception handler from Task 7 answers 500 for that request only, which is what `CorruptGameDoesNotBreakTheListOrTheSweep` expects.

- [ ] **Step 4: Run the tests**

Run: `make test FILTER="FullyQualifiedName~LobbyTests"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A server
git commit -m "server: lobby, seats, start, hand to bot, delete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Moves with idempotent retries (spec §4.4 moves, §4.5)

**Files:**
- Modify: `server/AzulServer/Games/GameService.cs` (add `Move`, `ApplyServerMove`, helpers), `server/AzulServer/Api/ApiEndpoints.cs` (route + `MoveRequest`)
- Create: `server/AzulServer.Tests/MoveTests.cs`

**Interfaces:**
- Produces: `public sealed record MoveRequest(long Version, string? RequestId, string? Kind, int? Factory, int? Color, int? Row, int[]? Columns)`; `GameService.Move(string id, string viewer, MoveRequest req) → Task<ApiResult>`; `GameService.ApplyServerMove(string id, long version, Move move, string actor) → Task<ApiResult>`.

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/MoveTests.cs`:

```csharp
using System.Net;
using System.Text.Json;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

public sealed class MoveTests : IDisposable
{
    readonly TestApp app = new();
    readonly HttpClient alice, bob;

    public MoveTests()
    {
        alice = app.Client("alice@x.com");
        bob = app.Client("bob@x.com");
    }

    public void Dispose() => app.Dispose();

    async Task<(string Id, GameView View)> TwoHumans()
    {
        var id = await Play.Started(2, alice, bob);
        return (id, await Play.Get(alice, id));
    }

    Task<HttpResponseMessage> Move(HttpClient c, string id, object body) => c.Post($"/api/games/{id}/moves", body);

    [Fact]
    public async Task LegalMoveAdvancesTheGame()
    {
        var (id, v) = await TwoHumans();
        var res = await Move(alice, id, Play.AnyLegal(v));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var after = await res.Read<GameView>();
        Assert.Equal(v.Version + 1, after.Version);
        Assert.Equal(1, after.Board!.ActiveSeat);
        Assert.Equal("take", after.LastMove!.Kind);
        Assert.NotNull((await Play.Get(bob, id)).Legal);
    }

    [Fact]
    public async Task NotYourTurnIs403()
    {
        var (id, v) = await TwoHumans();
        Assert.Equal(HttpStatusCode.Forbidden, (await Move(bob, id, Play.Take(v, v.Legal!.Takes![0]))).StatusCode);
    }

    [Fact]
    public async Task StaleVersionIs409WithTheCurrentView()
    {
        var (id, v) = await TwoHumans();
        await Move(alice, id, Play.AnyLegal(v));
        var res = await Move(alice, id, Play.AnyLegal(v));
        Assert.Equal(HttpStatusCode.Conflict, res.StatusCode);
        var body = await res.Read<JsonElement>();
        Assert.Equal("stale", body.GetProperty("error").GetString());
        Assert.Equal(v.Version + 1, body.GetProperty("view").GetProperty("version").GetInt64());
    }

    [Fact]
    public async Task IllegalAndWrongPhaseAre400()
    {
        var (id, v) = await TwoHumans();
        // A colour the factory does not hold.
        int f = 0, color = Enumerable.Range(0, 5).First(c => v.Board!.Factories[0][c] == 0);
        Assert.Equal(HttpStatusCode.BadRequest, (await Move(alice, id, Play.Take(v, [f, color, 5]))).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await Move(alice, id, Play.Wall(v, [-1, -1, -1, -1, -1]))).StatusCode);
    }

    // VERSION is replaced with the game's current version, so the checks that
    // need the game (factory upper bound, FIRST only from the centre) are reached.
    public static TheoryData<string> BadShapes => new()
    {
        """{"version":VERSION,"kind":"take","factory":0,"color":0,"row":0}""",
        """{"version":VERSION,"requestId":"nope","kind":"take","factory":0,"color":0,"row":0}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":0,"color":7,"row":0}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":0,"color":0,"row":6}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":99,"color":0,"row":0}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"take","factory":0,"color":5,"row":5}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"wall","columns":[1,2,3,4]}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"wall","columns":[6,-1,-1,-1,-1]}""",
        """{"version":VERSION,"requestId":"00000000-0000-0000-0000-000000000001","kind":"dance"}""",
        """{"version":"two"}""",
    };

    [Theory]
    [MemberData(nameof(BadShapes))]
    public async Task BadShapesAre400(string json)
    {
        var (id, v) = await TwoHumans();
        var body = json.Replace("VERSION", v.Version.ToString());
        var res = await alice.PostAsync($"/api/games/{id}/moves", new StringContent(body, System.Text.Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Equal(v.Version, (await Play.Get(alice, id)).Version);
    }

    [Fact]
    public async Task TakingOnlyTheFirstMarker()
    {
        var (id, v) = await TwoHumans();
        int centre = v.Board!.Factories.Length;
        Assert.Contains(v.Legal!.Takes!, t => t.SequenceEqual(new[] { centre, 5, 5 }));
        var res = await Move(alice, id, Play.Take(v, [centre, 5, 5]));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var after = await res.Read<GameView>();
        Assert.True(after.Board!.Players[0].HasFirst);
        Assert.False(after.Board.CenterHasFirst);
    }

    [Fact]
    public async Task RetryAfterTheTurnMovedOnReturnsTheStoredResult()
    {
        var (id, v) = await TwoHumans();
        var rid = Guid.NewGuid().ToString();
        var first = await (await Move(alice, id, Play.Take(v, v.Legal!.Takes![0], rid))).Read<GameView>();
        var bobView = await Play.Get(bob, id);
        Assert.Equal(HttpStatusCode.OK, (await Move(bob, id, Play.AnyLegal(bobView))).StatusCode);
        var retry = await Move(alice, id, Play.Take(v, v.Legal!.Takes![0], rid));
        Assert.Equal(HttpStatusCode.OK, retry.StatusCode);
        Assert.Equal(first.Version, (await retry.Read<GameView>()).Version);
        Assert.Equal(v.Version + 2, (await Play.Get(alice, id)).Version);  // nothing applied twice
    }

    [Fact]
    public async Task ReusedRequestIdWithOtherContentIs409()
    {
        var (id, v) = await TwoHumans();
        var rid = Guid.NewGuid().ToString();
        await Move(alice, id, Play.Take(v, v.Legal!.Takes![0], rid));
        var res = await Move(alice, id, Play.Take(v, v.Legal!.Takes![1], rid));
        Assert.Equal(HttpStatusCode.Conflict, res.StatusCode);
        Assert.Equal("request-id-reused", (await res.Read<JsonElement>()).GetProperty("error").GetString());
        var bobView = await Play.Get(bob, id);
        var asBob = await Move(bob, id, Play.Take(bobView, bobView.Legal!.Takes![0], rid));
        Assert.Equal(HttpStatusCode.Conflict, asBob.StatusCode);
    }

    [Fact]
    public async Task SameRequestTwiceAtOnceMovesOnce()
    {
        var (id, v) = await TwoHumans();
        var body = Play.Take(v, v.Legal!.Takes![0], Guid.NewGuid().ToString());
        var results = await Task.WhenAll(Move(alice, id, body), Move(alice, id, body));
        Assert.All(results, r => Assert.Equal(HttpStatusCode.OK, r.StatusCode));
        Assert.Equal(v.Version + 1, (await Play.Get(alice, id)).Version);
    }

    [Fact]
    public async Task TwoDevicesOneVersionOneWins()
    {
        var (id, v) = await TwoHumans();
        var phone = app.Client("alice@x.com");
        var results = await Task.WhenAll(
            Move(alice, id, Play.Take(v, v.Legal!.Takes![0])),
            Move(phone, id, Play.Take(v, v.Legal!.Takes![0])));
        Assert.Equal(new[] { HttpStatusCode.OK, HttpStatusCode.Conflict }, results.Select(r => r.StatusCode).Order());
    }

    [Fact]
    public async Task FinishedGameAcceptsOnlyRetries()
    {
        // alice vs greedy bot, with bots on and no delay.
        using var bots = new TestApp(new AzulOptions { BotWorkers = 1, MinMoveDelaySeconds = 0, SweepSeconds = 0.2 });
        var a = bots.Client("alice@x.com");
        var id = await Play.Started(2, a);
        object? lastAccepted = null;
        GameView v;
        while (true)
        {
            v = await Play.WaitFor(a, id, x => x.Status == Status.Finished || x.Legal is not null);
            if (v.Status == Status.Finished) break;
            var body = Play.AnyLegal(v);
            var r = await a.Post($"/api/games/{id}/moves", body);
            // 409 when a forced move was auto-played first.
            Assert.True(r.StatusCode is HttpStatusCode.OK or HttpStatusCode.Conflict, $"{r.StatusCode}");
            if (r.StatusCode == HttpStatusCode.OK) lastAccepted = body;
        }
        Assert.NotNull(v.Result);
        Assert.NotNull(lastAccepted);
        var retry = await a.Post($"/api/games/{id}/moves", lastAccepted);
        Assert.Equal(HttpStatusCode.OK, retry.StatusCode);
        var fresh = await a.Post($"/api/games/{id}/moves", Play.Wall(v, [-1, -1, -1, -1, -1]));
        Assert.Equal(HttpStatusCode.Conflict, fresh.StatusCode);
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~MoveTests`
Expected: most tests fail with 404/405 (no `/moves` route).

- [ ] **Step 3: Implement**

In `GameService.cs`, add (inside the class):

```csharp
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
```

Put `MoveRequest` in `ApiEndpoints.cs` next to the other request records (namespace `AzulServer.Api`) and add `using AzulServer.Api;` to `GameService.cs`:

```csharp
public sealed record MoveRequest(long Version, string? RequestId, string? Kind, int? Factory, int? Color, int? Row, int[]? Columns);
```

Route in `ApiEndpoints.Map`:

```csharp
        api.MapPost("/games/{id}/moves", async (HttpContext c, GameService s, string id, MoveRequest r) =>
            (await s.Move(id, c.Email(), r)).ToHttp());
```

`FinishedGameAcceptsOnlyRetries` needs bots, which arrive in Task 14. Mark it `[Fact(Skip = "needs BotScheduler (Task 14)")]` now; Task 14 removes the `Skip`.

- [ ] **Step 4: Run the tests**

Run: `make test FILTER=FullyQualifiedName~MoveTests`
Expected: all pass except the skipped one.

- [ ] **Step 5: Commit**

```bash
git add -A server
git commit -m "server: moves with ordered checks and idempotent retries

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Live updates over SSE (spec §4.6)

**Files:**
- Modify: `server/AzulServer/Games/EventHub.cs` (full), `server/AzulServer/Api/ApiEndpoints.cs` (route)
- Create: `server/AzulServer/Api/EventStream.cs`, `server/AzulServer.Tests/SseTests.cs`

**Interfaces:**
- Produces: `EventHub.Subscribe(string gameId) → (Guid Id, ChannelReader<long> Reader)`, `Unsubscribe(gameId, id)`, `Publish`, `PublishDeleted`, `SubscriberCount(gameId)`; `GET /api/games/{id}/events` with events `state` (data = `GameView`, `id:` = version) and `deleted`, and `: ping` comments.

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/SseTests.cs`:

```csharp
using System.Net;
using System.Text;
using AzulServer.Games;
using Microsoft.Extensions.DependencyInjection;

namespace AzulServer.Tests;

public sealed class SseTests : IDisposable
{
    readonly TestApp app = new(new AzulOptions { BotWorkers = 0, SseHeartbeatSeconds = 0.5 });

    public void Dispose() => app.Dispose();

    sealed class Stream(StreamReader reader, HttpResponseMessage response) : IDisposable
    {
        public HttpResponseMessage Response { get; } = response;

        /// Next event, skipping pings unless asked for.
        public async Task<(string Event, string Data)> Next(bool pings = false, int seconds = 10)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
            string? ev = null;
            var data = new StringBuilder();
            while (true)
            {
                var line = await reader.ReadLineAsync(cts.Token) ?? throw new EndOfStreamException();
                if (line.StartsWith(':'))
                {
                    if (pings) return (":", line);
                    continue;
                }
                if (line.Length == 0)
                {
                    if (ev is not null) return (ev, data.ToString());
                    continue;
                }
                if (line.StartsWith("event: ")) ev = line[7..];
                else if (line.StartsWith("data: ")) data.Append(line[6..]);
            }
        }

        public async Task<bool> Ended(int seconds = 5)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
            try { return await reader.ReadLineAsync(cts.Token) is null; }
            catch (OperationCanceledException) { return false; }
        }

        public void Dispose() { reader.Dispose(); Response.Dispose(); }
    }

    async Task<Stream> Open(HttpClient c, string id)
    {
        var res = await c.GetAsync($"/api/games/{id}/events", HttpCompletionOption.ResponseHeadersRead);
        return new Stream(new StreamReader(await res.Content.ReadAsStreamAsync()), res);
    }

    [Fact]
    public async Task FirstEventIsTheSnapshotThenHeartbeats()
    {
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice, app.Client("bob@x.com"));
        using var s = await Open(alice, id);
        Assert.Equal("text/event-stream", s.Response.Content.Headers.ContentType!.MediaType);
        var (ev, data) = await s.Next();
        Assert.Equal("state", ev);
        Assert.Equal(id, Json.Deserialize<GameView>(data).Id);
        Assert.Equal(":", (await s.Next(pings: true)).Event);
    }

    [Fact]
    public async Task EachViewerGetsTheirOwnView()
    {
        var alice = app.Client("alice@x.com");
        var bob = app.Client("bob@x.com");
        var carol = app.Client("carol@x.com");  // spectator
        var dave = app.Client("dave@x.com");
        var id = await Play.Started(3, alice, bob, dave);
        await dave.Post($"/api/games/{id}/seats/2/to-bot");
        var streams = new Dictionary<string, Stream>();
        foreach (var (name, c) in new[] { ("alice", alice), ("bob", bob), ("carol", carol), ("dave", dave) })
        {
            streams[name] = await Open(c, id);
            await streams[name].Next();  // snapshot
        }
        var v = await Play.Get(alice, id);
        Assert.Equal(HttpStatusCode.OK, (await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v))).StatusCode);

        var views = new Dictionary<string, GameView>();
        foreach (var (name, s) in streams)
        {
            var (ev, data) = await s.Next();
            Assert.Equal("state", ev);
            views[name] = Json.Deserialize<GameView>(data);
            Assert.Equal(v.Version + 1, views[name].Version);
        }
        Assert.Equal(0, views["alice"].You.Seat);
        Assert.Null(views["alice"].Legal);
        Assert.Equal(1, views["bob"].You.Seat);
        Assert.NotNull(views["bob"].Legal);  // bob is next
        Assert.Null(views["carol"].You.Seat);
        Assert.Null(views["carol"].Legal);
        Assert.Equal(2, views["dave"].You.Seat);
        Assert.Null(views["dave"].Legal);
        foreach (var s in streams.Values) s.Dispose();
    }

    sealed class LoseThirdNotification : IFaultInjector
    {
        int commits;
        public void AfterCommit(string gameId)
        {
            // claim = 1, start = 2, alice's move = 3
            if (Interlocked.Increment(ref commits) == 3)
                throw new InvalidOperationException("notification lost");
        }
    }

    [Fact]
    public async Task HeartbeatRepairsALostNotification()
    {
        using var faulty = new TestApp(new AzulOptions { BotWorkers = 0, SseHeartbeatSeconds = 0.5 },
            s => s.AddSingleton<IFaultInjector, LoseThirdNotification>());
        var alice = faulty.Client("alice@x.com");
        var bob = faulty.Client("bob@x.com");
        var id = await Play.Started(2, alice, bob);
        using var s = await Open(bob, id);
        var first = Json.Deserialize<GameView>((await s.Next()).Data);
        await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(await Play.Get(alice, id)));
        var (ev, data) = await s.Next(seconds: 5);
        Assert.Equal("state", ev);
        var next = Json.Deserialize<GameView>(data);
        Assert.Equal(first.Version + 1, next.Version);
        Assert.NotNull(next.Legal);  // bob's turn, an ordinary human turn
    }

    [Fact]
    public async Task DeleteEndsTheStream()
    {
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice, app.Client("bob@x.com"));
        using var s = await Open(alice, id);
        await s.Next();
        await alice.SendAsync(new HttpRequestMessage(HttpMethod.Delete, $"/api/games/{id}"));
        Assert.Equal("deleted", (await s.Next()).Event);
        Assert.True(await s.Ended());
    }

    [Fact]
    public async Task UnknownGameIs404()
    {
        var res = await app.Client().GetAsync("/api/games/nope/events", HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.NotFound, res.StatusCode);
    }

    [Fact]
    public void SlowSubscriberSeesOnlyTheLatestVersion()
    {
        var hub = new EventHub();
        var (_, reader) = hub.Subscribe("g");
        hub.Publish("g", 5);
        hub.Publish("g", 6);
        hub.Publish("g", 7);
        Assert.True(reader.TryRead(out var v));
        Assert.Equal(7, v);
        Assert.False(reader.TryRead(out _));
    }

    [Fact]
    public async Task StreamsUnsubscribeWhenClientsLeave()
    {
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice, app.Client("bob@x.com"));
        var s = await Open(alice, id);
        await s.Next();
        Assert.Equal(1, app.Service<EventHub>().SubscriberCount(id));
        s.Dispose();
        var until = DateTime.UtcNow.AddSeconds(5);
        while (app.Service<EventHub>().SubscriberCount(id) > 0 && DateTime.UtcNow < until)
            await Task.Delay(50);
        Assert.Equal(0, app.Service<EventHub>().SubscriberCount(id));
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~SseTests`
Expected: compile errors (`Subscribe` missing).

- [ ] **Step 3: Implement**

`server/AzulServer/Games/EventHub.cs` (replace):

```csharp
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace AzulServer.Games;

/// Version notices per game. Each subscriber holds at most one pending
/// notice (latest wins): every SSE event carries the full state.
public sealed class EventHub
{
    public const long Deleted = -1;

    readonly ConcurrentDictionary<string, ConcurrentDictionary<Guid, Channel<long>>> subscribers = new();

    public (Guid Id, ChannelReader<long> Reader) Subscribe(string gameId)
    {
        var channel = Channel.CreateBounded<long>(new BoundedChannelOptions(1)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
        });
        var id = Guid.NewGuid();
        subscribers.GetOrAdd(gameId, _ => new())[id] = channel;
        return (id, channel.Reader);
    }

    public void Unsubscribe(string gameId, Guid id)
    {
        if (subscribers.TryGetValue(gameId, out var subs))
            subs.TryRemove(id, out _);
    }

    public int SubscriberCount(string gameId) =>
        subscribers.TryGetValue(gameId, out var subs) ? subs.Count : 0;

    public void Publish(string gameId, long version)
    {
        if (!subscribers.TryGetValue(gameId, out var subs)) return;
        foreach (var channel in subs.Values)
            channel.Writer.TryWrite(version);
    }

    public void PublishDeleted(string gameId) => Publish(gameId, Deleted);
}
```

`server/AzulServer/Api/EventStream.cs`:

```csharp
using AzulServer.Auth;
using AzulServer.Games;

namespace AzulServer.Api;

public static class EventStream
{
    public static async Task Handle(HttpContext ctx, string id, GameService games, EventHub hub, AzulOptions options, TimeProvider time)
    {
        var viewer = ctx.Email();
        // Subscribe before reading the first snapshot, so no version can slip
        // between the two; the client ignores versions it already has.
        var (subscription, reader) = hub.Subscribe(id);
        try
        {
            var first = games.GetView(id, viewer);
            if (first is null)
            {
                ctx.Response.StatusCode = StatusCodes.Status404NotFound;
                return;
            }
            ctx.Response.Headers.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            ctx.Response.Headers["X-Accel-Buffering"] = "no";
            long sent = first.Version;
            await WriteState(ctx, first);

            var ct = ctx.RequestAborted;
            var until = time.GetUtcNow() + TimeSpan.FromMinutes(options.SseMaxMinutes);
            var heartbeat = TimeSpan.FromSeconds(options.SseHeartbeatSeconds);
            while (!ct.IsCancellationRequested && time.GetUtcNow() < until)
            {
                bool hasNotice;
                using (var wait = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    wait.CancelAfter(heartbeat);
                    try
                    {
                        hasNotice = await reader.WaitToReadAsync(wait.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        // Heartbeat. Also catch up if a notification was lost
                        // after a commit (GameService.AfterCommit failed).
                        var current = games.GetView(id, viewer);
                        if (current is null)
                        {
                            await WriteDeleted(ctx);
                            return;
                        }
                        if (current.Version > sent)
                        {
                            sent = current.Version;
                            await WriteState(ctx, current);
                        }
                        else
                        {
                            await ctx.Response.WriteAsync(": ping\n\n", ct);
                            await ctx.Response.Body.FlushAsync(ct);
                        }
                        continue;
                    }
                }
                if (!hasNotice) return;
                while (reader.TryRead(out var version))
                {
                    if (version == EventHub.Deleted)
                    {
                        await WriteDeleted(ctx);
                        return;
                    }
                    if (version <= sent) continue;
                    var view = games.GetView(id, viewer);
                    if (view is null)
                    {
                        await WriteDeleted(ctx);
                        return;
                    }
                    if (view.Version > sent)
                    {
                        sent = view.Version;
                        await WriteState(ctx, view);
                    }
                }
            }
        }
        catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
        {
            // client went away
        }
        finally
        {
            hub.Unsubscribe(id, subscription);
        }
    }

    static async Task WriteState(HttpContext ctx, GameView view)
    {
        await ctx.Response.WriteAsync($"event: state\nid: {view.Version}\ndata: {Json.Serialize(view)}\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }

    static async Task WriteDeleted(HttpContext ctx)
    {
        await ctx.Response.WriteAsync("event: deleted\ndata: {}\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }
}
```

Route in `ApiEndpoints.Map`: `api.MapGet("/games/{id}/events", EventStream.Handle);`

- [ ] **Step 4: Run the tests**

Run: `make test FILTER=FullyQualifiedName~SseTests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A server
git commit -m "server: per-viewer SSE with heartbeats and deletion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Bots, forced moves, sweep, restart (spec §4.7)

**Files:**
- Modify: `server/AzulServer/Games/ServerMoveQueue.cs` (full), `server/AzulServer/Games/BotScheduler.cs` (full), `server/AzulServer/Program.cs`, `server/AzulServer.Tests/MoveTests.cs` (remove the `Skip`)
- Create: `server/AzulServer.Tests/BotTests.cs`

**Interfaces:**
- Produces: `ServerMoveQueue.Poke(id)`, `TakeAsync(ct)`, `BeginSearch(id, ct) → CancellationTokenSource`, `EndSearch(id, cts)`, `CancelSearch(id)`; `BotScheduler : BackgroundService`; `MctsBrain : IBotBrain`.

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/BotTests.cs`:

```csharp
using System.Net;
using Azul;
using AzulServer.Data;
using AzulServer.Games;
using Microsoft.Extensions.DependencyInjection;

namespace AzulServer.Tests;

public class BotTests
{
    static AzulOptions Fast(int workers = 1) =>
        new() { BotWorkers = workers, MinMoveDelaySeconds = 0, SweepSeconds = 0.2 };

    static long Count(TestApp app, string sql)
    {
        using var c = app.Service<Db>().Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        return (long)cmd.ExecuteScalar()!;
    }

    [Fact]
    public async Task BotsAnswerAfterAHumanMove()
    {
        using var app = new TestApp(Fast());
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(3, alice);
        var v = await Play.Get(alice, id);
        await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v));
        var back = await Play.WaitFor(alice, id, x => x.Board!.ActiveSeat == 0 || x.Status == Status.Finished);
        Assert.True(back.Version >= v.Version + 3);
        Assert.True(Count(app, $"SELECT COUNT(*) FROM moves WHERE game_id='{id}' AND actor='bot'") >= 2);
    }

    [Fact]
    public async Task ForcedWallTurnsArePlayedForTheHuman()
    {
        using var app = new TestApp(Fast());
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(2, alice);
        // alice always sends everything to the floor, so she never completes a
        // line and every wall turn of hers is forced.
        while (true)
        {
            // Wait for a take turn of alice's; her (forced) wall turns are the server's.
            var v = await Play.WaitFor(alice, id, x => x.Legal?.Takes is not null || x.Board!.Round >= 2 || x.Status == Status.Finished);
            if (v.Board!.Round >= 2 || v.Status == Status.Finished) break;
            var floorTake = v.Legal!.Takes!.First(t => t[2] == 5);
            await alice.Post($"/api/games/{id}/moves", Play.Take(v, floorTake));  // 409 if a forced take beat her
        }
        Assert.True(Count(app, $"SELECT COUNT(*) FROM moves WHERE game_id='{id}' AND actor='auto'") >= 1);
    }

    sealed class BlockingBrain : IBotBrain
    {
        public readonly SemaphoreSlim Started = new(0);
        public Move ChooseMove(Game game, CancellationToken ct)
        {
            Started.Release();
            ct.WaitHandle.WaitOne(TimeSpan.FromSeconds(10));
            return game.GetGreedyMove();
        }
    }

    [Fact]
    public async Task TakeBackWhileTheBotThinks()
    {
        var brain = new BlockingBrain();
        using var app = new TestApp(Fast(), s => s.AddSingleton<IBotBrain>(brain));
        var alice = app.Client("alice@x.com");
        var bob = app.Client("bob@x.com");
        var id = await Play.Started(2, alice, bob);
        await bob.Post($"/api/games/{id}/seats/1/to-bot");
        var v = await Play.Get(alice, id);
        await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v));
        Assert.True(await brain.Started.WaitAsync(TimeSpan.FromSeconds(10)), "bot never started thinking");
        Assert.Equal(HttpStatusCode.OK, (await bob.Post($"/api/games/{id}/seats/1/take-back")).StatusCode);
        var mine = await Play.WaitFor(bob, id, x => x.Legal is not null, 10);
        Assert.Equal(1, mine.Board!.ActiveSeat);
        await Task.Delay(300);  // give a stale bot result time to (wrongly) land
        Assert.Equal(0L, Count(app, $"SELECT COUNT(*) FROM moves WHERE game_id='{id}' AND actor='bot'"));
        Assert.Equal(mine.Version, (await Play.Get(bob, id)).Version);
    }

    /// Throws after the second commit of the test (the start is the first,
    /// alice's move the second): the bot's turn is committed but never poked.
    sealed class CrashAfterSecondCommit : IFaultInjector
    {
        int commits;
        public void AfterCommit(string gameId)
        {
            if (Interlocked.Increment(ref commits) == 2)
                throw new InvalidOperationException("simulated crash after commit");
        }
    }

    [Fact]
    public async Task SweepRepairsALostPoke()
    {
        using var app = new TestApp(Fast(), s => s.AddSingleton<IFaultInjector, CrashAfterSecondCommit>());
        var alice = app.Client("alice@x.com");
        var g = await Play.Create(alice, 2);
        await alice.Post($"/api/games/{g.Id}/start");
        var v = await Play.Get(alice, g.Id);
        await alice.Post($"/api/games/{g.Id}/moves", Play.AnyLegal(v));
        await Play.WaitFor(alice, g.Id, x => x.Board!.ActiveSeat == 0 && x.Version >= v.Version + 2);
    }

    [Fact]
    public async Task EveryGameKeepsMovingWithOneWorker()
    {
        using var app = new TestApp(Fast(workers: 1));
        var alice = app.Client("alice@x.com");
        var ids = new List<string>();
        for (int i = 0; i < 6; i++)
        {
            var id = await Play.Started(4, alice);
            await alice.Post($"/api/games/{id}/seats/0/to-bot");  // all bots now
            ids.Add(id);
        }
        var start = new Dictionary<string, long>();
        foreach (var id in ids) start[id] = (await Play.Get(alice, id)).Version;
        await Task.Delay(TimeSpan.FromSeconds(3));
        foreach (var id in ids)
        {
            var v = await Play.Get(alice, id);
            Assert.True(v.Version >= start[id] + 3 || v.Status == Status.Finished, $"game {id} stalled at {v.Version}");
        }
    }

    sealed class CountingBrain : IBotBrain
    {
        int current, max;
        public int Max => max;
        public Move ChooseMove(Game game, CancellationToken ct)
        {
            int n = Interlocked.Increment(ref current);
            int seen;
            do { seen = max; } while (n > seen && Interlocked.CompareExchange(ref max, n, seen) != seen);
            Thread.Sleep(150);
            Interlocked.Decrement(ref current);
            return game.GetGreedyMove();
        }
    }

    [Fact]
    public async Task OneSearchPerGameEvenWithEagerSweeps()
    {
        var brain = new CountingBrain();
        using var app = new TestApp(new AzulOptions { BotWorkers = 3, MinMoveDelaySeconds = 0, SweepSeconds = 0.02 },
            s => s.AddSingleton<IBotBrain>(brain));
        var alice = app.Client("alice@x.com");
        var id = await Play.Started(4, alice);
        await alice.Post($"/api/games/{id}/seats/0/to-bot");
        var start = (await Play.Get(alice, id)).Version;
        await Task.Delay(TimeSpan.FromSeconds(2));
        Assert.Equal(1, brain.Max);
        Assert.True((await Play.Get(alice, id)).Version >= start + 3);
    }

    [Fact]
    public async Task PendingBotTurnResumesAfterARestart()
    {
        var dir = Directory.CreateTempSubdirectory("azul-restart-").FullName;
        string id;
        long version;
        using (var first = new TestApp(new AzulOptions { BotWorkers = 0 }, dataDir: dir))
        {
            var alice = first.Client("alice@x.com");
            id = await Play.Started(2, alice);
            var v = await Play.Get(alice, id);
            await alice.Post($"/api/games/{id}/moves", Play.AnyLegal(v));  // now the bot's turn; no workers
            version = (await Play.Get(alice, id)).Version;
        }
        using var second = new TestApp(Fast(), dataDir: dir);
        var after = await Play.WaitFor(second.Client("alice@x.com"), id, x => x.Version > version);
        Assert.Equal(Status.Playing, after.Status);
    }

    [Fact]
    public void MctsBrainProposesALegalMove()
    {
        var brain = new MctsBrain(new AzulOptions { BotThinkSeconds = 0.3 });
        var game = new Game(3, new Random(1));
        Assert.True(game.IsValid(brain.ChooseMove(game, CancellationToken.None)));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.True(game.IsValid(brain.ChooseMove(game, cancelled.Token)));
    }
}
```

In `MoveTests.cs`, remove `Skip = "needs BotScheduler (Task 14)"` from `FinishedGameAcceptsOnlyRetries`.

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~BotTests`
Expected: compile error (`MctsBrain` missing).

- [ ] **Step 3: Implement**

`server/AzulServer/Games/ServerMoveQueue.cs` (replace):

```csharp
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace AzulServer.Games;

/// Game ids that may need a server-made move. Deduplicated (bounded by the
/// number of games), never blocks the caller. The database stays the source
/// of truth: the sweep refills this from it.
public sealed class ServerMoveQueue
{
    readonly ConcurrentDictionary<string, byte> pending = new();
    readonly Channel<string> channel = Channel.CreateUnbounded<string>();
    readonly ConcurrentDictionary<string, CancellationTokenSource> searches = new();
    readonly ConcurrentDictionary<string, byte> active = new();

    /// One worker per game at a time; a second take of a game in progress is
    /// skipped (the running worker's commit pokes the game again, and the
    /// sweep covers a move that ended in 409).
    public bool TryStart(string gameId) => active.TryAdd(gameId, 0);

    public void Finish(string gameId) => active.TryRemove(gameId, out _);

    public void Poke(string gameId)
    {
        if (pending.TryAdd(gameId, 0))
            channel.Writer.TryWrite(gameId);
    }

    public async ValueTask<string> TakeAsync(CancellationToken ct)
    {
        var id = await channel.Reader.ReadAsync(ct);
        pending.TryRemove(id, out _);
        return id;
    }

    public CancellationTokenSource BeginSearch(string gameId, CancellationToken shutdown)
    {
        var cts = CancellationTokenSource.CreateLinkedTokenSource(shutdown);
        searches[gameId] = cts;
        return cts;
    }

    public void EndSearch(string gameId, CancellationTokenSource cts)
    {
        searches.TryRemove(KeyValuePair.Create(gameId, cts));
        cts.Dispose();
    }

    /// Any change to the game (take-back, deletion) stops a search on the old version.
    public void CancelSearch(string gameId)
    {
        if (searches.TryGetValue(gameId, out var cts))
        {
            try { cts.Cancel(); }
            catch (ObjectDisposedException) { }
        }
    }
}
```

`server/AzulServer/Games/BotScheduler.cs` (replace):

```csharp
using System.Diagnostics;
using Ai;
using Azul;

namespace AzulServer.Games;

public interface IBotBrain
{
    Move ChooseMove(Game game, CancellationToken ct);
}

/// Deterministic and instant; used by tests.
public sealed class GreedyBrain : IBotBrain
{
    public Move ChooseMove(Game game, CancellationToken ct) => game.GetGreedyMove();
}

/// The desktop game's bot: MCTS_Stochastic for AZUL_BOT_THINK_SECONDS on its
/// own clone, greedy when the search found nothing (game.py:261).
public sealed class MctsBrain(AzulOptions options) : IBotBrain
{
    public Move ChooseMove(Game game, CancellationToken ct)
    {
        var root = new MCTS_Stochastic<Game, Move>(game, 0.0f);  // clones the game
        var clock = Stopwatch.StartNew();
        while (!ct.IsCancellationRequested
               && clock.Elapsed.TotalSeconds < options.BotThinkSeconds
               && root.numRolls < 300_000)
            root.Grow();
        if (root.actions.Count == 0) return game.GetGreedyMove();
        int best = root.GetBestActionIdx();
        var move = root.NumRolls(best) > 0 && root.WinRatio(best) > 0 ? root.actions[best] : game.GetGreedyMove();
        return game.IsValid(move) ? move : game.GetGreedyMove();
    }
}

public sealed class BotScheduler(ServerMoveQueue queue, GameService games, IBotBrain brain,
    AzulOptions options, ILogger<BotScheduler> log) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken ct)
    {
        var loops = Enumerable.Range(0, Math.Max(0, options.BotWorkers))
            .Select(_ => Task.Run(() => Worker(ct), ct))
            .Append(Sweep(ct));
        return Task.WhenAll(loops);
    }

    async Task Sweep(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                foreach (var id in games.GamesNeedingServerMove())
                    queue.Poke(id);
            }
            catch (Exception e)
            {
                log.LogError(e, "sweep failed");
            }
            try { await Task.Delay(TimeSpan.FromSeconds(options.SweepSeconds), ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    async Task Worker(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            string id;
            try { id = await queue.TakeAsync(ct); }
            catch (OperationCanceledException) { return; }
            if (!queue.TryStart(id)) continue;
            try
            {
                await PlayOne(id, ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                return;
            }
            catch (Exception e)
            {
                // Dropped; the sweep brings the game back.
                log.LogError(e, "server move failed for game {Game}", id);
            }
            finally
            {
                queue.Finish(id);
            }
        }
    }

    async Task PlayOne(string id, CancellationToken ct)
    {
        var turn = games.GetServerTurn(id);
        if (turn is null) return;
        var clock = Stopwatch.StartNew();
        Move move;
        if (turn.Forced is { } forced)
        {
            move = forced;
        }
        else
        {
            var search = queue.BeginSearch(id, ct);
            try { move = brain.ChooseMove(turn.Game, search.Token); }
            finally { queue.EndSearch(id, search); }
        }
        var wait = TimeSpan.FromSeconds(options.MinMoveDelaySeconds) - clock.Elapsed;
        if (wait > TimeSpan.Zero) await Task.Delay(wait, ct);
        await games.ApplyServerMove(id, turn.Version, move, turn.Actor);  // 409 = something changed; dropped
    }
}
```

`Program.cs`: register

```csharp
builder.Services.AddSingleton<IBotBrain, MctsBrain>();
builder.Services.AddHostedService<BotScheduler>();
```

- [ ] **Step 4: Run the whole server suite**

Run: `make test`
Expected: all engine and server tests pass, including `FinishedGameAcceptsOnlyRetries` (no longer skipped).

- [ ] **Step 5: Smoke-run the dev server**

Run (two terminals, or background the first):

```bash
make dev-server   # leave running
curl -s http://127.0.0.1:5080/api/health
curl -s -H 'X-Dev-User: a@x' -H 'Content-Type: application/json' -d '{"players":2}' http://127.0.0.1:5080/api/games
```

Expected: `{"ok":true}`, then a game view JSON with `"status":"lobby"`. Stop the server with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add -A server
git commit -m "server: bot workers, forced moves, sweep and restart recovery

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
