using AzulServer;
using AzulServer.Api;
using AzulServer.Auth;
using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Extensions.FileProviders;

if (args.Contains("--healthcheck"))
    return await HealthCheck.RunAsync();

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton(sp =>
    AzulOptions.FromEnvironment(key => sp.GetRequiredService<IConfiguration>()[key]));
builder.Services.AddSingleton(TimeProvider.System);
builder.Services.AddHttpClient();
builder.Services.AddSingleton<IJwksFetcher, HttpJwksFetcher>();
builder.Services.AddSingleton<JwksCache>();
builder.Services.AddSingleton<AccessVerifier>();
builder.Services.AddSingleton<Db>();
builder.Services.AddSingleton<EventHub>();
builder.Services.AddSingleton<ServerMoveQueue>();
builder.Services.AddSingleton<IFaultInjector, NoFaults>();
builder.Services.AddSingleton<HubSignal>();
builder.Services.AddSingleton<IHubReportBuilder, HubReportBuilder>();
builder.Services.AddSingleton<GameService>();
builder.Services.AddSingleton<IBotBrain, MctsBrain>();
builder.Services.AddSingleton(sp => BotIdentity.For(sp.GetRequiredService<AzulOptions>()));
builder.Services.AddHostedService<BotScheduler>();
builder.Services.AddHttpClient(HubSender.HttpName, c => c.Timeout = TimeSpan.FromSeconds(10))
    .ConfigurePrimaryHttpMessageHandler(HubSender.PrimaryHandler);
builder.Services.AddHostedService<HubSender>();

var app = builder.Build();
var options = app.Services.GetRequiredService<AzulOptions>();  // fail fast on bad configuration
app.Services.GetRequiredService<Db>().Migrate();
app.Logger.LogInformation("bot key {Key}", app.Services.GetRequiredService<BotIdentity>().Key);

// Unhandled errors (a corrupt stored game, say) answer 500 for that request
// only, as JSON; TestServer would otherwise rethrow them into the test.
app.UseExceptionHandler(errors => errors.Run(async ctx =>
{
    ctx.Response.StatusCode = StatusCodes.Status500InternalServerError;
    await ctx.Response.WriteAsJsonAsync(new { error = "internal" });
}));

if (options.DevMode)
    app.Logger.LogWarning("DEV MODE: no Access JWT check; identity comes from X-Dev-User or the azul_dev_user cookie.");

app.UseMiddleware<AccessAuthMiddleware>();
app.UseMiddleware<CsrfMiddleware>();

PhysicalFileProvider? files = options.WebRoot is { } webRoot && Directory.Exists(webRoot)
    ? new PhysicalFileProvider(Path.GetFullPath(webRoot))
    : null;
if (files is not null)
{
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = files });
    app.UseStaticFiles(new StaticFileOptions
    {
        FileProvider = files,
        // The shell names the hashed bundles of the current build: revalidate it every time.
        OnPrepareResponse = ctx =>
        {
            if (ctx.File.Name == "index.html")
                ctx.Context.Response.Headers.CacheControl = "no-cache";
        },
    });
}
// Explicit, and after the static files: StaticFileMiddleware does nothing for
// a request that routing already matched to an endpoint (the fallback below
// matches everything).
app.UseRouting();

ApiEndpoints.Map(app);

if (files is not null)
{
    // Client-side routes (/g/<id>) get the SPA shell; unknown /api paths and
    // missing static assets (an old bundle after a redeploy) stay 404.
    app.MapFallback(async ctx =>
    {
        if (ctx.Request.Path.StartsWithSegments("/api") || ctx.Request.Path.StartsWithSegments("/assets"))
        {
            ctx.Response.StatusCode = StatusCodes.Status404NotFound;
            return;
        }
        ctx.Response.ContentType = "text/html; charset=utf-8";
        ctx.Response.Headers.CacheControl = "no-cache";
        await ctx.Response.SendFileAsync(files.GetFileInfo("index.html"));
    });
}

app.Run();
return 0;

public partial class Program { }
